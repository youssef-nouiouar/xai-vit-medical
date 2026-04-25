"""Grad-CAM and variants — Selvaraju et al., ICCV 2017.

Variants supported: ``GradCAM``, ``GradCAMPlusPlus``, ``EigenCAM``, ``XGradCAM``.

For ViT, ``reshape_transform`` is REQUIRED — see the helper below.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def vit_reshape_transform(
    tensor: torch.Tensor,
    height: int = 14,
    width: int = 14,
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """Reshape ViT tokens to a 2D feature-map for Grad-CAM.

    Args:
        tensor: Shape ``(B, N, D)`` where ``N = num_patches + num_extra_tokens``.
        height, width: Spatial dimensions of the patch grid (e.g., 14×14 for ViT-B/16).
        num_extra_tokens: 1 for ViT/DINOv2 (CLS only), 2 for DeiT (CLS+DIST).

    Returns:
        Tensor of shape ``(B, D, H, W)`` compatible with Grad-CAM.

    Notes:
        - For ViT-Base/16 with 224×224 input: ``num_patches = 196``, so
          tensor shape is ``(B, 197, 768)`` — drop the [CLS] then reshape
          ``(B, 196, 768) → (B, 14, 14, 768) → (B, 768, 14, 14)``.
        - For DeiT: drop both [CLS] and [DIST] tokens (``num_extra_tokens=2``).
        - For DINOv2 at 224: patch_size=14 → grid is 16×16, not 14×14.
    """
    # Drop the extra tokens (CLS, optionally DIST)
    patches = tensor[:, num_extra_tokens:, :]

    expected_n = height * width
    actual_n = patches.shape[1]
    if actual_n != expected_n:
        raise ValueError(
            f"Patch count mismatch: got {actual_n} patches, "
            f"expected {expected_n} (={height}×{width}). "
            f"Check height/width/num_extra_tokens."
        )

    # (B, N, D) -> (B, H, W, D) -> (B, D, H, W)
    B, N, D = patches.shape  # noqa: N806
    return patches.reshape(B, height, width, D).permute(0, 3, 1, 2)


def make_reshape_transform(cfg: Any):
    """Build a reshape_transform partial function from cfg.

    Returns a callable suitable for ``pytorch_grad_cam`` constructors,
    or None if not required (CNN models).
    """
    rt_cfg = cfg.reshape_transform
    if not rt_cfg.enabled:
        return None
    return partial(
        vit_reshape_transform,
        height=rt_cfg.height,
        width=rt_cfg.width,
        num_extra_tokens=rt_cfg.num_extra_tokens,
    )


def run_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    targets: list[int],
    target_layer: nn.Module,
    cfg: Any,
) -> torch.Tensor:
    """Run Grad-CAM on a batch.

    Args:
        model: Trained model.
        images: Input batch ``(B, C, H, W)``.
        targets: Class index per image.
        target_layer: The ``nn.Module`` to hook (resolved by factory).
        cfg: ``cfg.xai`` Hydra subtree (gradcam config).

    Returns:
        Saliency maps ``(B, H, W)``, normalized to [0, 1].
    """
    from pytorch_grad_cam import (
        EigenCAM,
        GradCAM,
        GradCAMPlusPlus,
        XGradCAM,
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam_classes = {
        "GradCAM": GradCAM,
        "GradCAMPlusPlus": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM,
    }

    if cfg.variant not in cam_classes:
        raise ValueError(
            f"Unknown Grad-CAM variant: {cfg.variant}. "
            f"Choose from {list(cam_classes)}."
        )

    cam_cls = cam_classes[cfg.variant]
    reshape_transform = make_reshape_transform(cfg)

    cam = cam_cls(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    cam_targets = [ClassifierOutputTarget(t) for t in targets]
    grayscale_cam = cam(
        input_tensor=images,
        targets=cam_targets,
        aug_smooth=cfg.aug_smooth,
        eigen_smooth=cfg.eigen_smooth,
    )

    logger.debug(f"Grad-CAM ({cfg.variant}) computed for {images.shape[0]} images")
    return torch.from_numpy(grayscale_cam)
