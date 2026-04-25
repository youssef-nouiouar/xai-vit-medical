"""Sanity Checks for XAI — Adebayo et al., NeurIPS 2018.

CRITICAL: every XAI method must pass these tests before publication.

Tests included:
    1. **Model Parameter Randomization**: cascading randomization of weights
       — saliency from trained vs random model should differ.
    2. **Label Randomization**: train on shuffled labels — saliency should differ.
    3. **Stability under noise**: small input perturbations should not drastically
       change explanations.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def model_randomization_test(
    model: nn.Module,
    xai_method: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    strategy: str = "cascading",
    n_seeds: int = 3,
) -> dict[str, float]:
    """Compare XAI on trained model vs progressively randomized model.

    Args:
        model: Trained model.
        xai_method: Callable ``(model, image) -> saliency``.
        images: Test batch.
        strategy: ``"cascading"`` (Adebayo 2018), ``"independent"``, ``"full"``.
        n_seeds: Number of random initializations.

    Returns:
        Dict with ``correlation_per_layer`` and ``passed`` (correlation < 0.3).
    """
    raise NotImplementedError("Implement in this file.")


def label_randomization_test(
    model_class: type,
    train_dataset: Any,
    xai_method: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    test_images: torch.Tensor,
    cfg: Any,
) -> dict[str, float]:
    """Train a model on randomly shuffled labels and compare XAI.

    Note:
        EXPENSIVE — full retraining required. Run only for the final paper.
    """
    raise NotImplementedError


def stability_under_noise(
    model: nn.Module,
    xai_method: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    sigma: float = 0.05,
    n_trials: int = 10,
) -> float:
    """Average SSIM between original and noised-input saliency maps.

    Returns:
        SSIM in [0, 1] — higher = more stable.
    """
    raise NotImplementedError


def run_all_sanity_checks(
    model: nn.Module,
    xai_method: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    test_loader: Any,
    cfg: Any,
) -> dict[str, dict[str, Any]]:
    """Run the full sanity-check suite and return a summary report.

    Returns:
        ``{test_name: {result_dict, passed: bool}}``.
    """
    raise NotImplementedError("Implement in this file.")
