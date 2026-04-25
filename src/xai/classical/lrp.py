"""Layer-wise Relevance Propagation — Bach et al., 2015.

ViT adaptation: Chefer et al., 2021. Use ``zennit`` library for
flexible composite rules.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_lrp_composite(model: nn.Module, cfg: Any) -> Any:
    """Build a zennit Composite for the model based on the LRP rules in cfg.

    Args:
        model: Trained model.
        cfg: ``cfg.xai`` Hydra subtree (lrp config).

    Returns:
        A ``zennit.composites.Composite`` instance.
    """
    raise NotImplementedError("Implement in this file.")


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
    raise NotImplementedError("Implement in this file.")
