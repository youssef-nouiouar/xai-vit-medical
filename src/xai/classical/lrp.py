"""Layer-wise Relevance Propagation — Bach et al., 2015.

ViT adaptation: Chefer et al., 2021. Use ``zennit`` library for
flexible composite rules.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def _apply_lrp_rules(model: nn.Module, cfg: Any) -> None:
    """Attach Captum LRP rules directly to model modules (in-place).

    Captum's LRP reads a ``.rule`` attribute from each module during the
    backward relevance pass.  Rules must cover every layer type in the model
    or Captum will leave those layers' relevance as ``None``, producing the
    ``AttributeError: 'NoneType' object has no attribute 'relevance_input'``
    error at runtime.
    """
    from captum.attr._utils.lrp_rules import (  # type: ignore[import-not-found]
        Alpha1_Beta0_Rule,
        EpsilonRule,
        GammaRule,
        IdentityRule,
    )

    gamma = float(getattr(cfg, "gamma", 0.25))
    epsilon = float(getattr(cfg, "epsilon", 0.25))

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.rule = GammaRule(gamma=gamma)  # type: ignore[attr-defined]
        elif isinstance(module, nn.Linear):
            module.rule = EpsilonRule(epsilon=epsilon)  # type: ignore[attr-defined]
        elif isinstance(module, (nn.ReLU,)):
            module.rule = Alpha1_Beta0_Rule()  # type: ignore[attr-defined]
        elif isinstance(
            module,
            (
                nn.BatchNorm2d,
                nn.LayerNorm,
                nn.GELU,
                nn.Dropout,
                nn.Identity,
                nn.AdaptiveAvgPool2d,
                nn.MaxPool2d,
                nn.AvgPool2d,
                nn.Flatten,
            ),
        ):
            module.rule = IdentityRule()  # type: ignore[attr-defined]



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

    model = model.to(images.device).eval()

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
    # Rules must be set on modules before constructing LRP.
    _apply_lrp_rules(wrapped, cfg)
    lrp = LRP(wrapped)

    try:
        rel = lrp.attribute(
            images,
            target=torch.tensor(targets, device=images.device),
        )  # (B, C, H, W)
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
