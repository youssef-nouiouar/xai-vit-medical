"""Generic Attention — Chefer et al., CVPR 2021.

Reference: "Transformer Interpretability Beyond Attention Visualization"
https://arxiv.org/abs/2012.09838

⭐ Recommended XAI method for ViT (passes sanity checks, better fidelity
than raw attention rollout).

Core idea: combine attention with its gradient ::

    R_layer = mean_heads( ReLU(A * dA) )

then rollout across layers with residual handling::

    R_layer_with_residual = R_layer + I
    R_layer_normalized    = R_layer_with_residual / sum_rows
    R_total               = R_L @ R_{L-1} @ ... @ R_1
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger


class GenericAttentionExplainer:
    """Explainer that registers hooks for attention + gradients.

    Args:
        model: A ViT (timm or HF style).
        attention_layer_pattern: Format string for attention modules,
            e.g. ``"blocks.{i}.attn.attn_drop"``. Used to register hooks.
        num_layers: Number of transformer blocks (auto-detected if None).

    Notes:
        Hooks must be registered ON ``attn_drop`` (post-softmax) for the
        attention values to be correctly captured. See Chefer 2021 §4.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_pattern: str = "blocks.{i}.attn.attn_drop",
        num_layers: int | None = None,
    ) -> None:
        self.model = model
        self.pattern = attention_layer_pattern
        self.num_layers = num_layers or self._detect_num_layers()

        self.attention_maps: list[torch.Tensor] = []
        self.attention_grads: list[torch.Tensor] = []
        self.handles: list[Any] = []

        self._register_hooks()

    def _detect_num_layers(self) -> int:
        """Auto-detect number of transformer blocks."""
        # Works for timm ViT/DeiT/Swin
        if hasattr(self.model, "blocks"):
            return len(self.model.blocks)
        # DINOv2 (torch.hub) backbone
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "blocks"):
            return len(self.model.backbone.blocks)
        raise ValueError(
            "Cannot auto-detect num_layers. Pass num_layers explicitly."
        )

    def _register_hooks(self) -> None:
        """Register forward + backward hooks on attention_drop layers."""

        def _save_attn(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            # Out shape: (B, H, N, N) — post-softmax attention
            self.attention_maps.append(out.detach())

        def _save_grad(module: nn.Module, grad_in: Any, grad_out: tuple) -> None:
            # grad_out[0] shape: (B, H, N, N)
            self.attention_grads.append(grad_out[0].detach())

        for i in range(self.num_layers):
            layer_path = self.pattern.format(i=i)
            module = self._get_module_by_path(layer_path)
            self.handles.append(module.register_forward_hook(_save_attn))
            self.handles.append(module.register_full_backward_hook(_save_grad))

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for part in path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    def explain(
        self,
        image: torch.Tensor,
        target_class: int,
        num_extra_tokens: int = 1,
        start_layer: int = 0,
    ) -> torch.Tensor:
        """Generate the saliency map for a single class.

        Args:
            image: Single image ``(1, C, H, W)``.
            target_class: Class index to explain.
            num_extra_tokens: 1 for ViT/DINOv2, 2 for DeiT (CLS+DIST).
            start_layer: Start propagation from this layer.

        Returns:
            Saliency ``(num_patches,)`` — caller reshapes to (H, W) grid.
        """
        # Reset buffers
        self.attention_maps = []
        self.attention_grads = []

        # Forward
        self.model.zero_grad()
        output = self.model(image)
        if isinstance(output, tuple):
            output = output[0]

        # Backward on target class
        score = output[0, target_class]
        score.backward(retain_graph=False)

        # Compute relevance via Chefer's rollout
        relevance = self._compute_relevance(num_extra_tokens, start_layer)

        # Extract CLS-row of relevance matrix and drop extra tokens
        # relevance shape: (N, N) where N = num_patches + num_extra_tokens
        cls_attribution = relevance[0, num_extra_tokens:]  # only CLS → patches
        return cls_attribution

    def _compute_relevance(self, num_extra_tokens: int, start_layer: int) -> torch.Tensor:
        """Compute attention × gradient rollout (Chefer 2021)."""
        # Reverse — backward hooks fire in reverse layer order
        attns = self.attention_maps  # (B, H, N, N) per layer, forward order
        grads = list(reversed(self.attention_grads))  # align with forward order

        # Get N from first layer
        B, H, N, _ = attns[0].shape  # noqa: N806
        device = attns[0].device

        # Identity matrix for residual
        rollout = torch.eye(N, device=device)

        for layer_idx in range(start_layer, len(attns)):
            attn = attns[layer_idx]  # (B, H, N, N)
            grad = grads[layer_idx]  # (B, H, N, N)

            # Element-wise product, ReLU, mean over heads, take batch[0]
            cam = (attn * grad).clamp(min=0).mean(dim=1)[0]  # (N, N)

            # Add residual identity
            cam = cam + torch.eye(N, device=device)

            # Normalize rows
            cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)

            # Propagate
            rollout = cam @ rollout

        return rollout

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.attention_maps.clear()
        self.attention_grads.clear()


def run_generic_attention(
    model: nn.Module,
    images: torch.Tensor,
    targets: list[int],
    cfg: Any,
) -> torch.Tensor:
    """Run Generic Attention on a batch.

    Args:
        model: Trained ViT model.
        images: Input batch ``(B, C, H, W)``.
        targets: Class index per image.
        cfg: ``cfg.xai`` Hydra subtree.

    Returns:
        Saliency maps ``(B, H, W)`` upsampled to input resolution.
    """
    import torch.nn.functional as F

    explainer = GenericAttentionExplainer(model)
    height = cfg.postprocess.reshape_height
    width = cfg.postprocess.reshape_width
    num_extra = cfg.num_extra_tokens

    saliencies = []
    for i in range(images.shape[0]):
        sal = explainer.explain(
            images[i : i + 1],
            target_class=targets[i],
            num_extra_tokens=num_extra,
            start_layer=cfg.start_layer,
        )
        # Reshape patch attribution to 2D grid
        sal_2d = sal.reshape(height, width)
        saliencies.append(sal_2d)

    explainer.cleanup()

    saliency_batch = torch.stack(saliencies)  # (B, h, w)

    # Upsample to input resolution
    input_h, input_w = images.shape[-2:]
    saliency_batch = F.interpolate(
        saliency_batch.unsqueeze(1),
        size=(input_h, input_w),
        mode=cfg.postprocess.upsample_mode,
        align_corners=False,
    ).squeeze(1)

    # Min-max normalize per image
    if cfg.postprocess.normalize == "minmax":
        b_min = saliency_batch.flatten(1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        b_max = saliency_batch.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        saliency_batch = (saliency_batch - b_min) / (b_max - b_min + 1e-8)

    logger.debug(f"Generic Attention: {images.shape[0]} samples processed")
    return saliency_batch
