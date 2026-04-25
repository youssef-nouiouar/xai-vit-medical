"""Model factory — creates models from Hydra config.

Supports:
    - ResNet-50 (timm)
    - ViT-Base/16 (timm)        ⭐ SOTA anchor — MANDATORY
    - DeiT-Base (timm)
    - DINOv2-ViT-B/14 (torch.hub) — see :mod:`src.models.dinov2`
    - Swin-Base (timm)         optional

DO NOT add other models without updating ``CLAUDE.md`` first.
"""

from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig


# Allowed models — keep in sync with CLAUDE.md
ALLOWED_MODELS = {
    "resnet50",
    "vit_base",
    "deit_base",
    "dinov2_vitb14",
    "swin_base",
}


def create_model(cfg: DictConfig) -> nn.Module:
    """Build a model from a Hydra ``model`` config.

    Args:
        cfg: ``cfg.model`` from Hydra. Must contain ``name``, ``source``,
            ``timm_name`` (or ``hub_*``), ``architecture.num_classes``.

    Returns:
        A PyTorch ``nn.Module`` with the classification head adapted
        to ``num_classes``.

    Raises:
        ValueError: If ``cfg.name`` is not in the allowed list.
    """
    if cfg.name not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{cfg.name}' is not in the allowed list for Phase 1. "
            f"Allowed: {sorted(ALLOWED_MODELS)}. "
            f"To add a new model, update CLAUDE.md first."
        )

    logger.info(f"Building model: {cfg.name} (role: {cfg.role})")

    # ---- DINOv2 — separate path (torch.hub) ----
    if cfg.source == "torch_hub":
        from src.models.dinov2 import build_dinov2

        model = build_dinov2(cfg)
        logger.info(f"DINOv2 built — usage_mode={cfg.usage_mode}")
        return model

    # ---- timm models ----
    if cfg.source != "timm":
        raise ValueError(f"Unknown source: {cfg.source}")

    timm_kwargs = {
        "model_name": cfg.timm_name,
        "pretrained": cfg.pretrained,
        "num_classes": cfg.architecture.num_classes,
    }

    # Optional drop rates
    arch = cfg.architecture
    if "drop_rate" in arch:
        timm_kwargs["drop_rate"] = arch.drop_rate
    if "drop_path_rate" in arch:
        timm_kwargs["drop_path_rate"] = arch.drop_path_rate

    model = timm.create_model(**timm_kwargs)

    n_params = count_parameters(model, trainable_only=True)
    logger.info(f"Model built — {n_params / 1e6:.1f}M trainable params")

    return model


def get_xai_target_layer(model: nn.Module, cfg: DictConfig, method: str) -> nn.Module:
    """Resolve the target layer for an XAI method on a given model.

    Args:
        model: Built model.
        cfg: Hydra model config (with ``xai_hooks``).
        method: XAI method name (e.g., ``"gradcam"``, ``"sae"``).

    Returns:
        The target ``nn.Module`` (used by hook-based XAI).
    """
    hooks = cfg.xai_hooks

    layer_path_map = {
        "gradcam": hooks.gradcam_target_layer,
        "sae": hooks.get("sae_target_layer", None),
    }

    layer_path = layer_path_map.get(method)
    if layer_path is None:
        raise ValueError(
            f"No target layer defined for method '{method}' on model {cfg.name}. "
            f"Add it to config/model/{cfg.name}.yaml under xai_hooks."
        )

    return _get_module_by_path(model, layer_path)


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path like ``'blocks.11.norm1'`` to a submodule."""
    module = model
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
