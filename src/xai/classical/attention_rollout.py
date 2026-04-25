"""Attention Rollout — Abnar & Zuidema, ACL 2020.

WARNING: Often fails sanity checks (Adebayo 2018). Prefer
:mod:`src.xai.classical.generic_attention` for ViT.

Not applicable to Swin Transformer (windowed attention).
"""

from __future__ import annotations

from typing import Any

import torch


def attention_rollout(
    attentions: list[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
    residual_weight: float = 0.5,
) -> torch.Tensor:
    """Compute the rollout matrix from per-layer attention tensors.

    Args:
        attentions: List of length ``L`` of tensors ``(B, H, N, N)``.
        head_fusion: ``"mean"``, ``"max"``, or ``"min"``.
        discard_ratio: Bottom fraction of attentions to discard per layer.
        residual_weight: Weight for ``A`` in ``residual_weight * A + I``.

    Returns:
        Rollout matrix ``(B, N, N)``.
    """
    raise NotImplementedError("Implement in this file.")


def run_attention_rollout(
    model: Any,
    images: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Run rollout on a batch.

    Returns:
        Saliency ``(B, H, W)`` — extracted from the [CLS] row of the rollout
        matrix and reshaped to the patch grid.
    """
    raise NotImplementedError("Implement in this file.")
