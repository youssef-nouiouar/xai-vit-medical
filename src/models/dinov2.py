"""DINOv2 wrapper — loads from ``torch.hub`` and adds a classification head.

Supports three usage modes (see ``config/model/dinov2.yaml``):
    1. ``frozen_linear_probe`` — freeze backbone + linear head (recommended)
    2. ``full_finetune`` — fine-tune everything (LLRD recommended)
    3. ``feature_extractor`` — return features for downstream use (SAE, MIL)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + classification head.

    Args:
        num_classes: Number of output classes (8 for ISIC 2019).
        usage_mode: One of ``"frozen_linear_probe"``, ``"full_finetune"``,
            ``"feature_extractor"``.
        head_dropout: Dropout before linear head.
        hub_model: ``torch.hub`` model name (default: ``"dinov2_vitb14"``).
    """

    def __init__(
        self,
        num_classes: int = 8,
        usage_mode: str = "frozen_linear_probe",
        head_dropout: float = 0.0,
        hub_model: str = "dinov2_vitb14",
    ) -> None:
        super().__init__()
        raise NotImplementedError("Implement in this file.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (pre-head).

        Used for SAE training and feature extraction tasks.

        Returns:
            Features of shape ``(B, embed_dim)`` (CLS token) or
            ``(B, num_patches, embed_dim)`` (all tokens — see ``return_all_tokens``).
        """
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; head remains trainable."""
        raise NotImplementedError


def build_dinov2(cfg: Any) -> DINOv2Classifier:
    """Build :class:`DINOv2Classifier` from a Hydra config.

    Args:
        cfg: ``cfg.model`` Hydra subtree.
    """
    raise NotImplementedError("Implement in this file.")
