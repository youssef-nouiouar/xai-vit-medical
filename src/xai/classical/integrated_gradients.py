"""Integrated Gradients — Sundararajan et al., ICML 2017.

Wraps Captum's ``IntegratedGradients`` with a baseline strategy
adapted to medical imaging.

For X-ray imagery, prefer ``baseline=blurred_original`` — black is informative.
"""

from __future__ import annotations

from typing import Any

import torch


def make_baseline(images: torch.Tensor, baseline_type: str, **kwargs: Any) -> torch.Tensor:
    """Build a baseline tensor for IG.

    Args:
        images: Input batch ``(B, C, H, W)``.
        baseline_type: ``"black_image"``, ``"white_image"``, ``"random_noise"``,
            ``"blurred_original"``, ``"mean_image"``.

    Returns:
        Baseline tensor of same shape as ``images``.
    """
    raise NotImplementedError("Implement in this file.")


def run_integrated_gradients(
    model: Any,
    images: torch.Tensor,
    targets: list[int],
    cfg: Any,
) -> torch.Tensor:
    """Run IG on a batch.

    Returns:
        Pixel-level attributions ``(B, H, W)``.
    """
    raise NotImplementedError("Implement in this file.")
