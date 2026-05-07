"""Integrated Gradients — Sundararajan et al., ICML 2017.

Wraps Captum's ``IntegratedGradients`` with a baseline strategy
adapted to medical imaging.

For X-ray imagery, prefer ``baseline=blurred_original`` — black is informative.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger


def make_baseline(images: torch.Tensor, baseline_type: str, **kwargs: Any) -> torch.Tensor:
    """Build a baseline tensor for IG.

    Args:
        images: Input batch ``(B, C, H, W)``.
        baseline_type: ``"black_image"``, ``"white_image"``, ``"random_noise"``,
            ``"blurred_original"``, ``"mean_image"``.

    Returns:
        Baseline tensor of same shape as ``images``.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images of shape (B, C, H, W), got {images.shape}")

    baseline_type = str(baseline_type)
    if baseline_type == "black_image":
        return torch.zeros_like(images)

    if baseline_type == "white_image":
        return torch.ones_like(images)

    if baseline_type == "random_noise":
        # Uniform noise in the same value range as the inputs
        return torch.rand_like(images)

    if baseline_type == "mean_image":
        # Mean per-sample, per-channel baseline
        return images.mean(dim=(-1, -2), keepdim=True).expand_as(images)

    if baseline_type == "blurred_original":
        # Fast blur approximation via avg pool (kernel size derived from sigma)
        sigma = float(kwargs.get("blur_sigma", 10.0))
        # Kernel size ~= 6*sigma + 1, force odd and cap for speed
        k = int(max(3, min(51, 2 * round(3 * sigma) + 1)))
        pad = k // 2
        return F.avg_pool2d(images, kernel_size=k, stride=1, padding=pad)

    raise ValueError(
        "Unknown baseline_type. Expected one of: "
        "black_image, white_image, random_noise, blurred_original, mean_image"
    )


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
    from captum.attr import IntegratedGradients

    if images.ndim != 4:
        raise ValueError(f"Expected images of shape (B, C, H, W), got {images.shape}")
    if len(targets) != images.shape[0]:
        raise ValueError(
            f"targets length ({len(targets)}) must match batch size ({images.shape[0]})"
        )

    model = model.to(images.device).eval()

    def _forward(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        # HuggingFace-style outputs
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, tuple):
            return out[0]
        return out

    ig = IntegratedGradients(_forward)

    baseline = make_baseline(
        images,
        baseline_type=cfg.baseline.type,
        blur_sigma=getattr(cfg.baseline, "blur_sigma", 10.0),
    )

    attr = ig.attribute(
        inputs=images,
        baselines=baseline,
        target=torch.tensor(targets, device=images.device),
        n_steps=int(cfg.n_steps),
        internal_batch_size=int(cfg.internal_batch_size),
        method=str(cfg.method),
    )  # (B, C, H, W)

    # ---- Post-process to (B, H, W) ----
    reduce = str(cfg.postprocess.reduce)
    if reduce == "sum":
        sal = attr.sum(dim=1)
    elif reduce == "mean":
        sal = attr.mean(dim=1)
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

    logger.debug(f"Integrated Gradients: {images.shape[0]} samples processed")
    return sal
