"""Sparse Autoencoders for ViT — Contribution C1.

References:
    - Cunningham et al., ICLR 2024 — "SAEs Find Highly Interpretable Features"
    - Templeton et al., 2024 — "Scaling Monosemanticity" (Anthropic)
    - Gao et al., 2024 — "Scaling and Evaluating SAEs" (OpenAI; TopK)

Pipeline:
    1. Collect activations from a target layer of a trained ViT.
    2. Train a TopK-SAE on these activations.
    3. Analyze each feature: top-K activating images.
    4. Optionally validate clinically (with dermatologist).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Args:
        d_in: Input dimension (e.g., 768 for ViT-Base).
        d_sae: Hidden dim (typically ``d_in × expansion_factor``).
        k: Number of active features per input (sparsity).

    Architecture::

        encoder: Linear(d_in -> d_sae)
        topk:    Keep top-k activations, zero others
        decoder: Linear(d_sae -> d_in)
    """

    def __init__(self, d_in: int = 768, d_sae: int = 12288, k: int = 40) -> None:
        super().__init__()
        raise NotImplementedError("Implement in this file.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return top-k sparse codes ``(B, d_sae)``."""
        raise NotImplementedError

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse codes."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(reconstructed, codes)``."""
        raise NotImplementedError


def collect_activations(
    model: nn.Module,
    dataloader: Any,
    target_layer: str,
    cfg: Any,
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Collect activations from a target layer over a dataset.

    Args:
        model: Trained ViT.
        dataloader: DataLoader (typically train split).
        target_layer: e.g. ``"blocks.8.mlp"``.
        cfg: ``cfg.xai`` (sae config).
        cache_path: If provided, cache to disk and skip recompute.

    Returns:
        Activations tensor ``(N, d_in)`` where N = total samples × n_tokens.
    """
    raise NotImplementedError("Implement in this file.")


def train_sae(
    sae: TopKSAE,
    activations: torch.Tensor,
    cfg: Any,
) -> dict[str, list[float]]:
    """Train the SAE on collected activations.

    Returns:
        Training history (loss, L0, variance_explained, dead_features per epoch).
    """
    raise NotImplementedError("Implement in this file.")


def analyze_features(
    sae: TopKSAE,
    activations: torch.Tensor,
    images: list[Any],
    cfg: Any,
) -> dict[int, dict[str, Any]]:
    """For each feature, find top-K activating images.

    Returns:
        Dict ``feature_idx -> {"top_images": [...], "activation_strengths": [...]}``.
    """
    raise NotImplementedError("Implement in this file.")
