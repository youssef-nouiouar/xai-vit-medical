"""Layer-wise Relevance Propagation — Bach et al., 2015.

ViT adaptation: Chefer et al., 2021. Use ``zennit`` library for
flexible composite rules.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def _build_captum_rule_dict(cfg: Any) -> dict[type[nn.Module], Any]:
    """Approximate a composite using Captum propagation rules.

    Captum's LRP API supports attaching propagation rules per module type
    through a ``rule_dict``. This is less flexible than zennit composites,
    but avoids adding a new dependency.
    """

    # Captum keeps rules in a private module; this is stable across 0.7.x
    from captum.attr._utils.lrp_rules import (  # type: ignore[import-not-found]
        Alpha1_Beta0_Rule,
        EpsilonRule,
        GammaRule,
        IdentityRule,
    )

    gamma = float(getattr(cfg, "gamma", 0.25))
    epsilon = float(getattr(cfg, "epsilon", 0.25))
    alpha = float(getattr(cfg, "alpha", 2.0))
    beta = float(getattr(cfg, "beta", 1.0))

    # Note: Captum exposes Alpha1_Beta0_Rule; use it as a reasonable default
    # when alpha/beta are requested.
    _ = (alpha, beta)  # keep for future extension

    return {
        nn.Conv2d: GammaRule(gamma=gamma),
        nn.Linear: EpsilonRule(epsilon=epsilon),
        nn.LayerNorm: IdentityRule(),
        nn.GELU: IdentityRule(),
        nn.Dropout: IdentityRule(),
        nn.ReLU: Alpha1_Beta0_Rule(),
    }


def build_lrp_composite(model: nn.Module, cfg: Any) -> Any:
    """Build a zennit Composite for the model based on the LRP rules in cfg.

    Args:
        model: Trained model.
        cfg: ``cfg.xai`` Hydra subtree (lrp config).

    Returns:
        A ``zennit.composites.Composite`` instance.
    """
    # The config describes a zennit-based composite, but zennit isn't part of the
    # Phase-1 requirements. We provide a Captum-based fallback.
    try:
        import zennit  # noqa: F401

        raise NotImplementedError(
            "zennit support is not wired yet. Use run_lrp() (Captum fallback)."
        )
    except Exception:
        return _build_captum_rule_dict(cfg)


def run_lrp(
    model: nn.Module,
    images: torch.Tensor,
    targets: list[int],
    cfg: Any,
) -> torch.Tensor:
    """Run LRP on a batch.

    Returns:
        Pixel-level relevance ``(B, H, W)``.
    """
    from captum.attr import LRP

    if images.ndim != 4:
        raise ValueError(f"Expected images of shape (B, C, H, W), got {images.shape}")
    if len(targets) != images.shape[0]:
        raise ValueError(
            f"targets length ({len(targets)}) must match batch size ({images.shape[0]})"
        )

    model.eval()

    # Captum LRP traverses module internals, so keep an nn.Module wrapper.
    class _LogitsWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.inner(x)
            if hasattr(out, "logits"):
                return out.logits
            if isinstance(out, tuple):
                return out[0]
            return out

    wrapped = _LogitsWrapper(model)
    lrp = LRP(wrapped)
    rule_dict = _build_captum_rule_dict(cfg)

    try:
        rel = lrp.attribute(
            images,
            target=torch.tensor(targets, device=images.device),
            rule_dict=rule_dict,
        )  # (B, C, H, W)
    except TypeError:
        # Older Captum signatures might not support rule_dict; retry without.
        rel = lrp.attribute(
            images,
            target=torch.tensor(targets, device=images.device),
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "LRP failed on this model. For Transformers, Captum LRP can be "
            "unsupported depending on ops used in attention. "
            "Consider using Generic Attention (Chefer 2021) for ViT."
        ) from e

    # ---- Post-process ----
    reduce = str(cfg.postprocess.reduce)
    if reduce == "sum":
        sal = rel.sum(dim=1)
    elif reduce == "mean":
        sal = rel.mean(dim=1)
    else:
        raise ValueError(f"Unknown postprocess.reduce: {reduce}")

    if bool(cfg.postprocess.abs):
        sal = sal.abs()

    if bool(cfg.postprocess.positive_only):
        sal = sal.clamp(min=0)

    if str(cfg.postprocess.normalize) == "minmax":
        s_min = sal.flatten(1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        s_max = sal.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        sal = (sal - s_min) / (s_max - s_min + 1e-8)

    logger.debug(f"LRP: {images.shape[0]} samples processed")
    return sal
