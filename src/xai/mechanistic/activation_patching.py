"""Activation Patching for ViT — Contribution C2.

Reference: Heimersheim & Nanda, 2024 — "How to use and interpret
activation patching" (https://arxiv.org/abs/2404.15255).

Adapted for ViT in medical imaging: first protocol with clinically
valid corruptions (lesion masking, ruler removal, etc.) for shortcut
detection.

Two main modes:
    - **denoising**: clean activation patched into corrupt forward pass
      (identifies SUFFICIENT components)
    - **noising**: corrupt activation patched into clean forward pass
      (identifies NECESSARY components)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


# ---- Corruption strategies ----


def corrupt_lesion_masking(
    image: torch.Tensor,
    mask: torch.Tensor,
    fill: str = "gaussian_noise",
) -> torch.Tensor:
    """Replace ROI inside ``mask`` with noise.

    Args:
        image: ``(C, H, W)``.
        mask: Binary mask ``(H, W)``.
        fill: ``"gaussian_noise"`` | ``"black"`` | ``"mean"``.

    Returns:
        Corrupted image.
    """
    raise NotImplementedError


def corrupt_artifact_removal(image: torch.Tensor, method: str = "telea") -> torch.Tensor:
    """Inpaint dermoscopic ruler / artifact.

    Tests if the model relies on a SHORTCUT (ruler presence).

    Args:
        image: ``(C, H, W)``.
        method: ``"telea"`` | ``"ns"`` | ``"lama"``.
    """
    raise NotImplementedError


def corrupt_null_baseline(image: torch.Tensor, dataset_mean: torch.Tensor) -> torch.Tensor:
    """Replace by dataset-mean image — reproducible baseline."""
    raise NotImplementedError


def corrupt_patch_shuffle(image: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Randomly shuffle a fraction of patches (tests spatial structure)."""
    raise NotImplementedError


# ---- Patching engine ----


def activation_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    target_layer: str,
    target_component: str = "resid_post",
    direction: str = "denoising",
    metric: Callable[[torch.Tensor], float] | None = None,
) -> float:
    """Single-component activation patching.

    Args:
        model: Trained ViT.
        clean_input: ``(1, C, H, W)``.
        corrupt_input: ``(1, C, H, W)``.
        target_layer: e.g. ``"blocks.6"``.
        target_component: ``"resid_post"`` | ``"attn_output"`` | ``"mlp_output"``.
        direction: ``"denoising"`` (clean→corrupt) | ``"noising"`` (corrupt→clean).
        metric: Function ``logits -> float``. Default = logit-difference for binary task.

    Returns:
        **Indirect Effect (IE)** — normalized metric difference.
    """
    raise NotImplementedError("Implement in this file.")


def attribution_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    layers: list[str],
    metric: Callable[[torch.Tensor], float] | None = None,
) -> dict[str, float]:
    """Fast linear approximation of activation patching (~100× faster).

    Use for initial screening before exact patching.
    """
    raise NotImplementedError


def detect_shortcuts(
    model: nn.Module,
    dataloader: Any,
    cfg: Any,
) -> dict[str, dict[str, float]]:
    """Run shortcut detection across artifact corruptions.

    Returns:
        ``{artifact_name: {layer: IE}}`` mapping. Layers with
        ``IE > cfg.thresholds.shortcut_detected`` (default 0.3) flag a shortcut.
    """
    raise NotImplementedError("Implement in this file.")
