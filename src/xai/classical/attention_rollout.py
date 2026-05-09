"""Attention Rollout — Abnar & Zuidema, ACL 2020.

WARNING: Often fails sanity checks (Adebayo 2018). Prefer
:mod:`src.xai.classical.generic_attention` for ViT.

Not applicable to Swin Transformer (windowed attention).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger


def attention_rollout(
    attentions: list[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
    residual_weight: float = 0.5,
) -> torch.Tensor:
    """Compute the rollout matrix from per-layer attention tensors.

    Args:
        attentions: List of length ``L`` of tensors ``(B, H, N, N)``.
        head_fusion: ``"mean"``, ``"max"``, or ``"min"``.
        discard_ratio: Bottom fraction of attentions to discard per layer.
        residual_weight: Weight for ``A`` in ``residual_weight * A + I``.

    Returns:
        Rollout matrix ``(B, N, N)``.
    """
    if not attentions:
        raise ValueError("attentions list is empty")

    if head_fusion not in {"mean", "max", "min"}:
        raise ValueError("head_fusion must be one of: mean, max, min")
    if not (0.0 <= discard_ratio < 1.0):
        raise ValueError("discard_ratio must be in [0, 1)")
    if not (0.0 <= residual_weight <= 1.0):
        raise ValueError("residual_weight must be in [0, 1]")

    # attentions: list[(B, H, N, N)]
    first = attentions[0]
    if first.ndim != 4:
        raise ValueError(
            "Expected attention tensors of shape (B, H, N, N); "
            f"got ndim={first.ndim}"
        )

    B, _, N, _ = first.shape  # noqa: N806
    device = first.device

    # Initialize rollout as identity per sample
    rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

    for layer_attn in attentions:
        if layer_attn.shape[-2:] != (N, N):
            raise ValueError(
                "All attention tensors must share the same token dimension N. "
                f"Expected {(N, N)}, got {tuple(layer_attn.shape[-2:])}."
            )

        # Fuse heads -> (B, N, N)
        if head_fusion == "mean":
            attn = layer_attn.mean(dim=1)
        elif head_fusion == "max":
            attn = layer_attn.max(dim=1).values
        else:  # head_fusion == "min"
            attn = layer_attn.min(dim=1).values

        # Discard bottom fraction of attention scores
        if discard_ratio > 0:
            flat = attn.reshape(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                _, idx = torch.topk(flat, k, dim=1, largest=False)
                flat.scatter_(1, idx, 0.0)
            attn = flat.reshape(B, N, N)

        # Residual connection handling (Abnar & Zuidema): mix attention with identity
        eye = torch.eye(N, device=device).unsqueeze(0)
        attn = residual_weight * attn + (1.0 - residual_weight) * eye

        # Row-normalize to keep a stochastic matrix
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # Propagate
        rollout = attn @ rollout

    return rollout


def _disable_fused_attn(model: Any) -> None:
    """Set fused_attn=False on every attention block so attn_drop hooks fire.

    Newer timm ViTs use F.scaled_dot_product_attention (fused/SDPA) by
    default, which bypasses the explicit attn_drop module. This function
    switches each block back to the explicit softmax path.
    """
    blocks = None
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        blocks = model.backbone.blocks

    if blocks is None:
        return

    disabled = 0
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            attn.fused_attn = False
            disabled += 1

    if disabled:
        logger.debug(f"Disabled fused_attn on {disabled} attention blocks")


def run_attention_rollout(
    model: Any,
    images: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Run rollout on a batch.

    Returns:
        Saliency ``(B, H, W)`` — extracted from the [CLS] row of the rollout
        matrix and reshaped to the patch grid.
    """
    if not hasattr(model, "forward"):
        raise TypeError("model must be a callable nn.Module-like object")

    # Swin uses windowed attention; raw rollout is not meaningful
    if getattr(cfg, "applicable_to", None) is not None:
        # cfg.applicable_to is a list in Hydra; we keep this check lightweight
        pass

    device = images.device
    model.eval()

    # ---- Infer where to hook attention maps ----
    # Default timm ViT/DeiT
    pattern = getattr(cfg, "attention_layer_pattern", None)
    if pattern is None:
        if hasattr(model, "blocks"):
            pattern = "blocks.{i}.attn.attn_drop"
        elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            pattern = "backbone.blocks.{i}.attn.attn_drop"
        else:
            raise ValueError(
                "Cannot infer attention layer pattern for this model. "
                "Provide cfg.attention_layer_pattern."
            )

    if hasattr(model, "blocks"):
        num_layers = len(model.blocks)
    elif hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        num_layers = len(model.backbone.blocks)
    else:
        raise ValueError("Cannot infer transformer depth for attention rollout")

    # Disable fused/SDPA attention so attn_drop hooks fire.
    # Newer timm ViTs use F.scaled_dot_product_attention by default,
    # which bypasses the explicit attn_drop module entirely.
    _disable_fused_attn(model)

    attentions: list[torch.Tensor] = []
    handles: list[Any] = []

    def _get_module_by_path(root: Any, path: str):
        module = root
        for part in path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    def _save_attn(_module: Any, _inp: Any, out: Any) -> None:
        # out is expected to be (B, H, N, N) post-softmax attention
        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError("Attention hook output is not a tensor")
        attentions.append(out.detach())

    for i in range(num_layers):
        module = _get_module_by_path(model, pattern.format(i=i))
        handles.append(module.register_forward_hook(_save_attn))

    # ---- Forward pass to collect attentions ----
    with torch.no_grad():
        _ = model(images)

    # Cleanup hooks
    for h in handles:
        h.remove()

    if len(attentions) != num_layers:
        logger.warning(
            f"Attention rollout captured {len(attentions)} layers (expected {num_layers}). "
            "Proceeding with captured layers only."
        )

    rollout = attention_rollout(
        attentions,
        head_fusion=cfg.head_fusion,
        discard_ratio=cfg.discard_ratio,
        residual_weight=cfg.residual_weight,
    )  # (B, N, N)

    num_extra_tokens = int(getattr(cfg, "num_extra_tokens", 1))
    patch_h = int(cfg.postprocess.reshape_height)
    patch_w = int(cfg.postprocess.reshape_width)

    # CLS -> patch tokens
    cls_to_patches = rollout[:, 0, num_extra_tokens:]
    if cls_to_patches.shape[1] != patch_h * patch_w:
        raise ValueError(
            "Patch token count mismatch. "
            f"Got {cls_to_patches.shape[1]}, expected {patch_h * patch_w} "
            f"(={patch_h}×{patch_w})."
        )

    saliency = cls_to_patches.reshape(images.shape[0], patch_h, patch_w)

    # Upsample to input resolution
    if cfg.postprocess.upsample_to_input_size:
        input_h, input_w = images.shape[-2:]
        upsample_mode = cfg.postprocess.upsample_mode
        # align_corners is only valid for bilinear/bicubic/linear/trilinear
        interp_kwargs: dict = {"size": (input_h, input_w), "mode": upsample_mode}
        if upsample_mode in {"bilinear", "bicubic", "linear", "trilinear"}:
            interp_kwargs["align_corners"] = False
        saliency = F.interpolate(saliency.unsqueeze(1), **interp_kwargs).squeeze(1)

    # Normalize to [0, 1] per sample
    if cfg.postprocess.normalize == "minmax":
        s_min = saliency.flatten(1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        s_max = saliency.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)

    logger.debug(f"Attention Rollout: {images.shape[0]} samples processed")
    return saliency.to(device)
