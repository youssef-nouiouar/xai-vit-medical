"""DINOv2 wrapper — loads from ``torch.hub`` and adds a classification head.

Supports three usage modes (see ``config/model/dinov2.yaml``):
    1. ``frozen_linear_probe`` — freeze backbone, train only head (Phase 1)
    2. ``full_finetune``       — fine-tune all layers (low LR required)
    3. ``feature_extractor``   — return CLS/all-token features (SAE, XAI)

Two-phase training (recommended):
    Phase 1 — Linear probe  : freeze_backbone()          LR ~ 1e-3
    Phase 2 — Fine-tuning   : unfreeze_last_n_blocks(4)  LR ~ 5e-5
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    """DINOv2-ViT-B/14 backbone + classification head.

    Args:
        num_classes:  Number of output classes (9 for CRC histology).
        usage_mode:   One of ``"frozen_linear_probe"``, ``"full_finetune"``,
                      ``"feature_extractor"``.
        head_dropout: Dropout probability before the linear head.
        hub_model:    torch.hub model name (default: ``"dinov2_vitb14"``).

    Attributes:
        EMBED_DIM: CLS token dimension for ViT-B (768).
    """

    EMBED_DIM: int = 768

    def __init__(
        self,
        num_classes: int = 9,
        usage_mode: str = "frozen_linear_probe",
        head_dropout: float = 0.0,
        hub_model: str = "dinov2_vitb14",
    ) -> None:
        super().__init__()
        self.usage_mode = usage_mode

        self.backbone: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", hub_model, pretrained=True
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.EMBED_DIM, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

        if usage_mode == "frozen_linear_probe":
            self.freeze_backbone()
        elif usage_mode == "feature_extractor":
            self.freeze_backbone()
        # full_finetune: nothing frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Args:
            x: Image tensor ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        cls_token = self.backbone(x)   # [B, 768]
        return self.head(cls_token)

    def extract_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """Return backbone features (pre-head).

        Used for SAE training and XAI analysis.

        Args:
            x:                Image tensor ``(B, 3, H, W)``.
            return_all_tokens: If True, return all patch tokens
                               ``(B, num_patches+1, 768)`` instead of
                               just the CLS token ``(B, 768)``.

        Returns:
            Features tensor.
        """
        if return_all_tokens:
            return self.backbone.forward_features(x)["x_norm_patchtokens"]
        return self.backbone(x)   # CLS token [B, 768]

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; head remains trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        """Unfreeze the last n transformer blocks + final norm.

        Used for Phase 2 fine-tuning. Keeps early blocks frozen to
        preserve DINOv2 SSL representations.

        Args:
            n: Number of transformer blocks to unfreeze (from the end).
               DINOv2-ViT-B has 12 blocks total.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

        total = len(self.backbone.blocks)
        for block in self.backbone.blocks[total - n:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the entire model (full fine-tuning)."""
        for p in self.parameters():
            p.requires_grad = True

    def count_trainable(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def build_dinov2(cfg: Any) -> DINOv2Classifier:
    """Build :class:`DINOv2Classifier` from a Hydra model config.

    Args:
        cfg: ``cfg.model`` Hydra subtree (from ``config/model/dinov2.yaml``).

    Returns:
        Configured :class:`DINOv2Classifier`.

    Example::

        with initialize(config_path='../../config'):
            cfg = compose('config', overrides=['model=dinov2'])
        model = build_dinov2(cfg.model)
    """
    usage_mode = cfg.get("usage_mode", "frozen_linear_probe")

    head_dropout = 0.0
    if usage_mode == "frozen_linear_probe":
        head_dropout = cfg.frozen_linear_probe.get("head_dropout", 0.0)
    elif usage_mode == "full_finetune":
        head_dropout = cfg.full_finetune.get("head_dropout", 0.1)

    model = DINOv2Classifier(
        num_classes=cfg.architecture.num_classes,
        usage_mode=usage_mode,
        head_dropout=head_dropout,
        hub_model=cfg.hub_model,
    )
    return model
