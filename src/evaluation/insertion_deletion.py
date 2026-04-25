"""Insertion / Deletion AUC — Petsiuk et al., 2018 ("RISE").

Faithfulness metrics:
    - **Insertion AUC** (↑ better): start from baseline, add pixels by
      decreasing importance, measure how confidence rises.
    - **Deletion AUC** (↓ better): start from image, remove pixels by
      decreasing importance, measure how confidence drops.

Combined: ``Faithfulness = Insertion AUC - Deletion AUC``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_baseline(image: torch.Tensor, kind: str) -> torch.Tensor:
    """Build a baseline image used as the 'empty' state for Insertion."""
    if kind == "black":
        return torch.zeros_like(image)
    if kind == "blurred":
        # Gaussian blur via avg pool — fast, good enough as baseline
        return F.avg_pool2d(image.unsqueeze(0), 31, 1, 15).squeeze(0)
    if kind == "mean":
        return image.mean(dim=(-1, -2), keepdim=True).expand_as(image)
    raise ValueError(f"Unknown baseline kind: {kind}")


def _ranked_pixel_indices(saliency: torch.Tensor) -> torch.Tensor:
    """Return pixel indices sorted by importance (descending)."""
    flat = saliency.flatten()
    return torch.argsort(flat, descending=True)


@torch.no_grad()
def insertion_curve(
    model: nn.Module,
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    baseline: str = "black",
) -> tuple[list[float], float]:
    """Compute the insertion curve and its AUC.

    Args:
        model: Trained model.
        image: ``(C, H, W)``.
        saliency: ``(H, W)`` — saliency map for the target class.
        target_class: Class index.
        n_steps: Number of insertion steps.
        baseline: ``"black"`` | ``"blurred"`` | ``"mean"``.

    Returns:
        ``(confidences_per_step, auc)``.
    """
    device = image.device
    model.eval()

    baseline_img = _build_baseline(image, baseline)
    indices = _ranked_pixel_indices(saliency)

    n_pixels = saliency.numel()
    pixels_per_step = max(1, n_pixels // n_steps)

    H, W = saliency.shape  # noqa: N806
    confidences = []
    current = baseline_img.clone()

    for step in range(n_steps + 1):
        with torch.no_grad():
            logits = model(current.unsqueeze(0).to(device))
            prob = torch.softmax(logits, dim=-1)[0, target_class].item()
        confidences.append(prob)

        if step < n_steps:
            # Add the next batch of pixels from the original image
            start = step * pixels_per_step
            end = min((step + 1) * pixels_per_step, n_pixels)
            idx = indices[start:end]
            ys, xs = idx // W, idx % W
            current[:, ys, xs] = image[:, ys, xs]

    auc = float(np.trapz(confidences, dx=1.0 / n_steps))
    return confidences, auc


@torch.no_grad()
def deletion_curve(
    model: nn.Module,
    image: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    replacement: str = "black",
) -> tuple[list[float], float]:
    """Compute the deletion curve and its AUC.

    Returns:
        ``(confidences_per_step, auc)``.
    """
    device = image.device
    model.eval()

    replacement_img = _build_baseline(image, replacement)
    indices = _ranked_pixel_indices(saliency)

    n_pixels = saliency.numel()
    pixels_per_step = max(1, n_pixels // n_steps)

    H, W = saliency.shape  # noqa: N806
    confidences = []
    current = image.clone()

    for step in range(n_steps + 1):
        with torch.no_grad():
            logits = model(current.unsqueeze(0).to(device))
            prob = torch.softmax(logits, dim=-1)[0, target_class].item()
        confidences.append(prob)

        if step < n_steps:
            # Replace the next batch of important pixels with baseline
            start = step * pixels_per_step
            end = min((step + 1) * pixels_per_step, n_pixels)
            idx = indices[start:end]
            ys, xs = idx // W, idx % W
            current[:, ys, xs] = replacement_img[:, ys, xs]

    auc = float(np.trapz(confidences, dx=1.0 / n_steps))
    return confidences, auc


def faithfulness_score(insertion_auc: float, deletion_auc: float) -> float:
    """Combined faithfulness: insertion - deletion.

    Interpretation:
        - > 0.4 : excellent
        - 0.2 - 0.4 : good
        - < 0.1 : weak
        - < 0 : the method's saliency is anti-correlated with importance
    """
    return insertion_auc - deletion_auc


def evaluate_faithfulness(
    model: nn.Module,
    saliency_maps: torch.Tensor,
    images: torch.Tensor,
    targets: list[int],
    cfg: Any,
) -> dict[str, list[float]]:
    """Evaluate faithfulness over a batch.

    Returns:
        Dict with per-sample lists of ``insertion_auc``, ``deletion_auc``,
        ``faithfulness_score``.
    """
    insertion_aucs = []
    deletion_aucs = []
    fscores = []

    n_steps = cfg.faithfulness.insertion.n_steps
    base = cfg.faithfulness.insertion.baseline
    repl = cfg.faithfulness.deletion.replacement

    for i in range(images.shape[0]):
        _, ins = insertion_curve(
            model, images[i], saliency_maps[i],
            target_class=targets[i], n_steps=n_steps, baseline=base,
        )
        _, dl = deletion_curve(
            model, images[i], saliency_maps[i],
            target_class=targets[i], n_steps=n_steps, replacement=repl,
        )
        insertion_aucs.append(ins)
        deletion_aucs.append(dl)
        fscores.append(faithfulness_score(ins, dl))

    return {
        "insertion_auc": insertion_aucs,
        "deletion_auc": deletion_aucs,
        "faithfulness_score": fscores,
    }
