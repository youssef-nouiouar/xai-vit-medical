"""Activation Patching for ViT — Contribution C2.

Reference: Heimersheim & Nanda, 2024 — "How to use and interpret
activation patching" (https://arxiv.org/abs/2404.15255).

Adapted for CRC histology ViT: identifies which transformer components
are causally responsible for the model's diagnostic behavior
(e.g. TUM vs NORM discrimination) and flags shortcut reliance.

Two patching modes
------------------
denoising : patch **clean** activation into **corrupt** forward pass.
            → identifies SUFFICIENT components (full IE ≈ 1 means that
              component alone accounts for the clean behavior).

noising   : patch **corrupt** activation into **clean** forward pass.
            → identifies NECESSARY components (full IE ≈ 1 means the
              model cannot function without that component).

Indirect Effect (IE)
--------------------
For denoising::

    IE(l,c) = (metric_patched - metric_corrupt) / (metric_clean - metric_corrupt)

For noising::

    IE(l,c) = (metric_clean - metric_patched) / (metric_clean - metric_corrupt)

Values close to 1.0 identify the most causally important components.

Attribution Patching (fast)
---------------------------
A first-order Taylor approximation that replaces the O(L × C) forward
passes of exact patching with a single forward + backward pass::

    attr(l,c) ≈ (act_clean[l,c] - act_corrupt[l,c]) · ∇_{act[l,c]} metric_clean

~100× faster than exact patching; use for initial screening.

Corruptions (CRC histology)
---------------------------
null_baseline   — replace image by black / dataset mean (uninformative input)
patch_shuffle   — randomly permute a fraction of 16×16 patches
gaussian_noise  — additive Gaussian noise
horizontal_flip — left-right flip (tests orientation bias)
color_jitter    — HSV perturbation (tests staining/color shortcuts)
lesion_masking  — fill a binary ROI mask with noise
artifact_stripe — inject a horizontal stripe artifact
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Corruption strategies
# ---------------------------------------------------------------------------

def corrupt_null_baseline(
    image: torch.Tensor,
    dataset_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace image by dataset mean (or black if mean is None).

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)``.
        dataset_mean: ``(C,)`` mean per channel. Uses zeros if None.

    Returns:
        Constant-valued image of same shape.
    """
    if dataset_mean is not None:
        mean = dataset_mean.to(image.device)
        while mean.ndim < image.ndim:
            mean = mean.unsqueeze(-1)
        return mean.expand_as(image)
    return torch.zeros_like(image)


def corrupt_patch_shuffle(
    image: torch.Tensor,
    ratio: float = 0.3,
    patch_size: int = 16,
) -> torch.Tensor:
    """Randomly permute a fraction of ``patch_size × patch_size`` patches.

    Args:
        image: ``(C, H, W)`` — single image.
        ratio: Fraction of patches to shuffle (0 = none, 1 = all).
        patch_size: Patch size in pixels.

    Returns:
        Corrupted image ``(C, H, W)``.
    """
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image size {H}×{W} must be divisible by patch_size={patch_size}"
    )
    gh, gw = H // patch_size, W // patch_size
    n_patches = gh * gw
    k = max(1, int(ratio * n_patches))

    # Fold into patches: (C, gh, ph, gw, pw) → (n_patches, C, ph, pw)
    img = image.clone()
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # patches: (C, gh, gw, ph, pw)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(n_patches, C, patch_size, patch_size)

    # Select k patches to shuffle
    selected = torch.randperm(n_patches)[:k]
    shuffled = selected[torch.randperm(k)]
    patches[selected] = patches[shuffled]

    # Unfold back
    patches = patches.reshape(gh, gw, C, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4)  # (C, gh, ph, gw, pw)
    return patches.reshape(C, H, W).clone()


def corrupt_gaussian_noise(
    image: torch.Tensor,
    std: float = 0.5,
) -> torch.Tensor:
    """Add Gaussian noise to the image.

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)``.
        std: Noise standard deviation in normalized pixel space.

    Returns:
        Noisy image (clipped to [0, 1] approximate range).
    """
    noise = torch.randn_like(image) * std
    return (image + noise).clamp(image.min(), image.max())


def corrupt_horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip image left-right (tests orientation / laterality bias).

    Args:
        image: ``(C, H, W)`` or ``(B, C, H, W)``.
    """
    return image.flip(-1)


def corrupt_color_jitter(
    image: torch.Tensor,
    brightness: float = 0.5,
    contrast: float = 0.5,
    saturation: float = 0.5,
) -> torch.Tensor:
    """Random color perturbation — tests staining / color shortcuts in histology.

    Operates in normalized image space (ImageNet stats assumed).
    Applies brightness, contrast, and saturation jitter.

    Args:
        image: ``(C, H, W)`` single image, normalized.

    Returns:
        Color-jittered image, clipped.
    """
    import random

    # brightness
    factor_b = 1.0 + random.uniform(-brightness, brightness)
    img = image * factor_b

    # contrast
    factor_c = 1.0 + random.uniform(-contrast, contrast)
    mean = img.mean(dim=(-2, -1), keepdim=True)
    img = (img - mean) * factor_c + mean

    # saturation (convert to grayscale proxy: mean of channels)
    factor_s = 1.0 + random.uniform(-saturation, saturation)
    gray = img.mean(dim=0, keepdim=True)
    img = img * factor_s + gray * (1 - factor_s)

    return img.clamp(image.min() - 1.0, image.max() + 1.0)


def corrupt_lesion_masking(
    image: torch.Tensor,
    mask: torch.Tensor,
    fill: str = "gaussian_noise",
    noise_std: float = 0.5,
) -> torch.Tensor:
    """Replace the masked ROI with noise / black / mean.

    Useful for testing if the model relies on a specific tissue region.

    Args:
        image: ``(C, H, W)``.
        mask: Binary mask ``(H, W)`` — pixels to corrupt are 1.
        fill: ``"gaussian_noise"`` | ``"black"`` | ``"mean"``.
        noise_std: Std for Gaussian noise (only when fill="gaussian_noise").

    Returns:
        Corrupted image ``(C, H, W)``.
    """
    img = image.clone()
    m = mask.bool().to(image.device)
    if fill == "black":
        fill_val = torch.zeros(image.shape[0], 1, device=image.device)
        img[:, m] = fill_val
    elif fill == "mean":
        fill_val = image[:, ~m].mean(dim=-1, keepdim=True) if (~m).any() else image.mean()
        img[:, m] = fill_val
    else:  # gaussian_noise
        noise = torch.randn(image.shape[0], m.sum(), device=image.device) * noise_std
        img[:, m] = image[:, m] + noise
        img = img.clamp(image.min(), image.max())
    return img


def corrupt_artifact_stripe(
    image: torch.Tensor,
    stripe_height: int = 8,
    position: str = "random",
) -> torch.Tensor:
    """Inject a bright horizontal stripe (simulates a scanning artifact).

    Tests whether the model is spuriously sensitive to such artifacts.

    Args:
        image: ``(C, H, W)``.
        stripe_height: Height of the stripe in pixels.
        position: ``"random"`` | ``"top"`` | ``"center"`` | ``"bottom"``.

    Returns:
        Image with injected stripe.
    """
    img = image.clone()
    H = image.shape[-2]
    if position == "top":
        start = 0
    elif position == "bottom":
        start = H - stripe_height
    elif position == "center":
        start = (H - stripe_height) // 2
    else:
        start = torch.randint(0, max(1, H - stripe_height), (1,)).item()
    end = min(start + stripe_height, H)
    # Fill stripe with maximum value (bright artifact)
    img[:, start:end, :] = image.max()
    return img


CORRUPTION_REGISTRY: dict[str, Callable] = {
    "null_baseline": corrupt_null_baseline,
    "patch_shuffle": corrupt_patch_shuffle,
    "gaussian_noise": corrupt_gaussian_noise,
    "horizontal_flip": corrupt_horizontal_flip,
    "color_jitter": corrupt_color_jitter,
    "artifact_stripe": corrupt_artifact_stripe,
}


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def logit_diff_metric(
    positive_class: int,
    negative_class: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a metric function computing logit[pos] - logit[neg].

    The returned function accepts logits ``(1, num_classes)`` or ``(num_classes,)``
    and returns a scalar tensor.
    """
    def _metric(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1:
            logits = logits[0]
        return logits[positive_class] - logits[negative_class]
    return _metric


def correct_logit_metric(target_class: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a metric function computing logit[target_class]."""
    def _metric(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1:
            logits = logits[0]
        return logits[target_class]
    return _metric


def kl_divergence_metric(
    reference_logits: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """KL(reference || patched) — measures distributional shift."""
    ref_log_probs = F.log_softmax(reference_logits.flatten(), dim=0)

    def _metric(logits: torch.Tensor) -> torch.Tensor:
        log_p = ref_log_probs
        log_q = F.log_softmax(logits.flatten(), dim=0)
        kl = (log_p.exp() * (log_p - log_q)).sum()
        return -kl  # negate so higher = better (closer to reference)
    return _metric


# ---------------------------------------------------------------------------
# Module path utilities
# ---------------------------------------------------------------------------

def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    module: Any = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _detect_backbone_prefix(model: nn.Module) -> str:
    """Return 'backbone.' for DINOv2 wrapper, '' for plain ViT/DeiT."""
    if not hasattr(model, "blocks") and hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return "backbone."
    return ""


def _component_path(
    layer_idx: int,
    component: str,
    backbone_prefix: str = "",
) -> str:
    """Dotted module path for a given (layer, component).

    Args:
        component: ``"resid_post"`` | ``"attn_output"`` | ``"mlp_output"``.
    """
    base = f"{backbone_prefix}blocks.{layer_idx}"
    if component == "resid_post":
        return base
    elif component == "attn_output":
        return f"{base}.attn"
    elif component == "mlp_output":
        return f"{base}.mlp"
    else:
        raise ValueError(
            f"Unknown component '{component}'. "
            "Choose from: resid_post, attn_output, mlp_output"
        )


def _detect_num_layers(model: nn.Module) -> int:
    if hasattr(model, "blocks"):
        return len(model.blocks)
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return len(model.backbone.blocks)
    raise ValueError("Cannot auto-detect number of transformer layers.")


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run model and unwrap HuggingFace / tuple outputs to logits."""
    out = model(x)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Activation caching
# ---------------------------------------------------------------------------

def _cache_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    path_dict: dict[str, str],
    no_grad: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Forward pass that caches activations at multiple module paths.

    Args:
        model: ViT model.
        inputs: ``(B, C, H, W)``.
        path_dict: ``{name: dotted_module_path}`` — which activations to cache.
        no_grad: If True, wrap in ``torch.no_grad()``.

    Returns:
        ``(cached_acts, logits)`` where cached_acts maps each name to
        its activation tensor (same device as inputs).
    """
    modules = {name: _get_module_by_path(model, path) for name, path in path_dict.items()}
    cached: dict[str, torch.Tensor] = {}
    handles = []

    for name, module in modules.items():
        def _make_hook(n: str) -> Callable:
            def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
                cached[n] = out
            return _hook
        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        if no_grad:
            with torch.no_grad():
                logits = _forward_logits(model, inputs)
        else:
            logits = _forward_logits(model, inputs)
    finally:
        for h in handles:
            h.remove()

    return cached, logits


# ---------------------------------------------------------------------------
# Single-component exact activation patching
# ---------------------------------------------------------------------------

def activation_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    target_layer: int,
    target_component: str = "resid_post",
    direction: str = "denoising",
    metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
    backbone_prefix: str | None = None,
) -> dict[str, Any]:
    """Single-component activation patching.

    Args:
        model: Trained ViT (eval mode).
        clean_input: Clean image ``(1, C, H, W)``.
        corrupt_input: Corrupted image ``(1, C, H, W)``.
        target_layer: Transformer block index.
        target_component: ``"resid_post"`` | ``"attn_output"`` | ``"mlp_output"``.
        direction: ``"denoising"`` | ``"noising"``.
        metric: Scalar metric ``logits → scalar_tensor``.
            Defaults to the mean of all logits (useful for testing).
        backbone_prefix: ``"backbone."`` for DINOv2, ``""`` for timm ViT.
            Auto-detected if None.

    Returns:
        Dict with ``indirect_effect``, ``metric_clean``, ``metric_corrupt``,
        ``metric_patched``.
    """
    model.eval()
    if backbone_prefix is None:
        backbone_prefix = _detect_backbone_prefix(model)

    if metric is None:
        metric = lambda logits: logits.mean()  # noqa: E731

    path = _component_path(target_layer, target_component, backbone_prefix)
    module = _get_module_by_path(model, path)

    # ---- Baseline passes (no grad) ----
    with torch.no_grad():
        logits_clean   = _forward_logits(model, clean_input)
        logits_corrupt = _forward_logits(model, corrupt_input)
    m_clean   = metric(logits_clean).item()
    m_corrupt = metric(logits_corrupt).item()
    denom = m_clean - m_corrupt

    # ---- Save the "source" activation (clean for denoising, corrupt for noising) ----
    source_input  = clean_input   if direction == "denoising" else corrupt_input
    target_input  = corrupt_input if direction == "denoising" else clean_input

    source_cached: dict[str, torch.Tensor] = {}

    def _save_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        source_cached["act"] = out.detach()

    h_save = module.register_forward_hook(_save_hook)
    with torch.no_grad():
        _forward_logits(model, source_input)
    h_save.remove()

    saved_act = source_cached["act"]

    # ---- Patched forward ----
    def _patch_hook(_mod: nn.Module, _inp: Any, _out: torch.Tensor) -> torch.Tensor:
        return saved_act

    h_patch = module.register_forward_hook(_patch_hook)
    with torch.no_grad():
        logits_patched = _forward_logits(model, target_input)
    h_patch.remove()

    m_patched = metric(logits_patched).item()

    # ---- Indirect Effect ----
    if abs(denom) < 1e-8:
        ie = 0.0
    elif direction == "denoising":
        ie = (m_patched - m_corrupt) / denom
    else:  # noising
        ie = (m_clean - m_patched) / denom

    return {
        "indirect_effect": ie,
        "metric_clean"   : m_clean,
        "metric_corrupt" : m_corrupt,
        "metric_patched" : m_patched,
        "layer"          : target_layer,
        "component"      : target_component,
        "direction"      : direction,
    }


# ---------------------------------------------------------------------------
# Full scan: all (layer, component) pairs
# ---------------------------------------------------------------------------

def patching_scan(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    layers: list[int] | None = None,
    components: list[str] | None = None,
    direction: str = "denoising",
    metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
    backbone_prefix: str | None = None,
) -> dict[tuple[int, str], dict[str, float]]:
    """Scan all (layer, component) pairs via exact activation patching.

    Args:
        layers: Block indices to scan. Defaults to all layers.
        components: Components to scan. Defaults to
            ``["resid_post", "attn_output", "mlp_output"]``.

    Returns:
        ``{(layer_idx, component): patching_result}``
    """
    model.eval()
    if backbone_prefix is None:
        backbone_prefix = _detect_backbone_prefix(model)
    if layers is None:
        layers = list(range(_detect_num_layers(model)))
    if components is None:
        components = ["resid_post", "attn_output", "mlp_output"]
    if metric is None:
        metric = lambda logits: logits.mean()  # noqa: E731

    results: dict[tuple[int, str], dict[str, float]] = {}
    total = len(layers) * len(components)
    done = 0

    for layer_idx in layers:
        for component in components:
            result = activation_patching(
                model=model,
                clean_input=clean_input,
                corrupt_input=corrupt_input,
                target_layer=layer_idx,
                target_component=component,
                direction=direction,
                metric=metric,
                backbone_prefix=backbone_prefix,
            )
            results[(layer_idx, component)] = result
            done += 1
            if done % 6 == 0 or done == total:
                logger.debug(f"patching_scan: {done}/{total}")

    return results


# ---------------------------------------------------------------------------
# Attribution Patching (fast approximation)
# ---------------------------------------------------------------------------

def attribution_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    layers: list[int] | None = None,
    components: list[str] | None = None,
    metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
    backbone_prefix: str | None = None,
) -> dict[tuple[int, str], float]:
    """Fast attribution patching via first-order Taylor approximation.

    Computes::

        attr(l, c) = (act_clean[l,c] - act_corrupt[l,c]) · ∇_{act[l,c]} metric_clean

    Requires one clean forward + backward and one corrupt forward (no grad).
    ~100× faster than exact patching for large scans.

    Args:
        model: Trained ViT (eval mode). Parameters must allow grad computation
            even if frozen (the graph is built through activations, not params).
        clean_input: ``(1, C, H, W)`` — requires_grad not needed.
        corrupt_input: ``(1, C, H, W)``.
        layers: Block indices. Defaults to all.
        components: Component names. Defaults to all three.
        metric: ``logits → scalar_tensor`` (must be differentiable).
        backbone_prefix: Auto-detected if None.

    Returns:
        ``{(layer_idx, component): attribution_score}``
    """
    model.eval()
    if backbone_prefix is None:
        backbone_prefix = _detect_backbone_prefix(model)
    if layers is None:
        layers = list(range(_detect_num_layers(model)))
    if components is None:
        components = ["resid_post", "attn_output", "mlp_output"]
    if metric is None:
        metric = lambda logits: logits.mean()  # noqa: E731

    # Build path dict for all (layer, component) pairs
    path_dict: dict[tuple[int, str], str] = {}
    for li in layers:
        for comp in components:
            path_dict[(li, comp)] = _component_path(li, comp, backbone_prefix)

    # ---- Clean forward (with grad, to get gradients) ----
    clean_acts: dict[tuple[int, str], torch.Tensor] = {}
    clean_act_refs: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    for key, path in path_dict.items():
        module = _get_module_by_path(model, path)
        def _make_hook(k: tuple) -> Callable:
            def _hook(_mod, _inp, out: torch.Tensor) -> torch.Tensor:
                retained = out.requires_grad_(True)
                clean_act_refs[k] = retained
                return retained
            return _hook
        handles.append(module.register_forward_hook(_make_hook(key)))

    # Enable gradients for clean pass (activations retained in graph)
    logits_clean = _forward_logits(model, clean_input)
    m_clean = metric(logits_clean)
    m_clean.backward()

    for h in handles:
        h.remove()

    # Collect clean activations + their gradients
    clean_grads: dict[tuple[int, str], torch.Tensor] = {}
    for key, act in clean_act_refs.items():
        clean_acts[key] = act.detach()
        if act.grad is not None:
            clean_grads[key] = act.grad.detach()
        else:
            clean_grads[key] = torch.zeros_like(act)

    # ---- Corrupt forward (no grad) ----
    corrupt_act_vals: dict[tuple[int, str], torch.Tensor] = {}
    handles2 = []

    for key, path in path_dict.items():
        module = _get_module_by_path(model, path)
        def _make_hook2(k: tuple) -> Callable:
            def _hook(_mod, _inp, out: torch.Tensor) -> None:
                corrupt_act_vals[k] = out.detach()
            return _hook
        handles2.append(module.register_forward_hook(_make_hook2(key)))

    with torch.no_grad():
        _forward_logits(model, corrupt_input)

    for h in handles2:
        h.remove()

    # ---- Attribution score ----
    attr_scores: dict[tuple[int, str], float] = {}
    for key in path_dict:
        diff = clean_acts[key] - corrupt_act_vals[key]      # (1, N_tok, D) or (1, D)
        grad = clean_grads[key]
        score = (diff * grad).sum().item()
        attr_scores[key] = score

    return attr_scores


# ---------------------------------------------------------------------------
# Per-head attribution patching
# ---------------------------------------------------------------------------

def attribution_patching_heads(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    layers: list[int] | None = None,
    num_heads: int = 12,
    head_dim: int = 64,
    metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
    backbone_prefix: str | None = None,
) -> dict[tuple[int, int], float]:
    """Attribution patching at the individual attention-head level.

    Hooks the pre-projection concatenated head output of each attention block
    (using a forward pre-hook on ``attn.proj``).

    Args:
        num_heads: Number of attention heads (12 for ViT-Base).
        head_dim: Dimension per head (64 for ViT-Base with embed_dim=768).

    Returns:
        ``{(layer_idx, head_idx): attribution_score}``
    """
    model.eval()
    if backbone_prefix is None:
        backbone_prefix = _detect_backbone_prefix(model)
    if layers is None:
        layers = list(range(_detect_num_layers(model)))
    if metric is None:
        metric = lambda logits: logits.mean()  # noqa: E731

    # Build path dict: (layer, head) → proj module path
    proj_paths: dict[int, str] = {
        li: _component_path(li, "attn_output", backbone_prefix) + ".proj"
        for li in layers
    }

    # ---- Clean forward: hook on attn.proj inputs ----
    clean_head_acts: dict[tuple[int, int], torch.Tensor] = {}
    clean_head_grads: dict[tuple[int, int], torch.Tensor] = {}
    handles = []

    for li, proj_path in proj_paths.items():
        proj_mod = _get_module_by_path(model, proj_path)

        def _make_pre_hook(layer_i: int) -> Callable:
            def _hook(_mod: nn.Module, inp: tuple) -> tuple:
                # inp[0]: (B, N_tok, n_heads * head_dim)
                x = inp[0]
                # Split into per-head slices and retain grad
                heads = []
                for h in range(num_heads):
                    head_slice = x[..., h * head_dim : (h + 1) * head_dim]
                    retained = head_slice.requires_grad_(True)
                    clean_head_acts[(layer_i, h)] = retained
                    heads.append(retained)
                new_inp = torch.cat(heads, dim=-1)
                return (new_inp,)
            return _hook

        handles.append(proj_mod.register_forward_pre_hook(_make_pre_hook(li)))

    logits_clean = _forward_logits(model, clean_input)
    m_clean = metric(logits_clean)
    m_clean.backward()

    for h in handles:
        h.remove()

    for key, act in clean_head_acts.items():
        clean_head_grads[key] = act.grad.detach() if act.grad is not None else torch.zeros_like(act)
        clean_head_acts[key] = act.detach()

    # ---- Corrupt forward ----
    corrupt_head_acts: dict[tuple[int, int], torch.Tensor] = {}
    handles2 = []

    for li, proj_path in proj_paths.items():
        proj_mod = _get_module_by_path(model, proj_path)

        def _make_pre_hook2(layer_i: int) -> Callable:
            def _hook(_mod: nn.Module, inp: tuple) -> None:
                x = inp[0].detach()
                for h in range(num_heads):
                    corrupt_head_acts[(layer_i, h)] = x[..., h * head_dim : (h + 1) * head_dim]
            return _hook

        handles2.append(proj_mod.register_forward_pre_hook(_make_pre_hook2(li)))

    with torch.no_grad():
        _forward_logits(model, corrupt_input)

    for h in handles2:
        h.remove()

    # ---- Attribution scores ----
    head_scores: dict[tuple[int, int], float] = {}
    for key in clean_head_acts:
        diff  = clean_head_acts[key] - corrupt_head_acts.get(key, torch.zeros_like(clean_head_acts[key]))
        grad  = clean_head_grads[key]
        head_scores[key] = (diff * grad).sum().item()

    return head_scores


# ---------------------------------------------------------------------------
# Shortcut detection
# ---------------------------------------------------------------------------

def detect_shortcuts(
    model: nn.Module,
    dataloader: Any,
    layers: list[int] | None = None,
    components: list[str] | None = None,
    corruptions: list[str] | None = None,
    corruption_kwargs: dict[str, dict] | None = None,
    metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_samples: int = 100,
    ie_threshold: float = 0.3,
    backbone_prefix: str | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Scan for shortcut-reliant components across corruption types.

    For each corruption and each (layer, component) pair, computes the
    mean attribution-patching score (IE approximation) over ``max_samples``
    clean/corrupt image pairs.

    A component with ``mean_IE > ie_threshold`` on a semantic corruption
    (e.g., ``color_jitter``) indicates the model relies on that feature as
    a shortcut.

    Args:
        model: Trained ViT.
        dataloader: Yields ``(images, labels)`` batches.
        layers: Block indices to scan.
        components: Component names.
        corruptions: Corruption names from ``CORRUPTION_REGISTRY``.
            Defaults to ``["null_baseline", "patch_shuffle", "gaussian_noise",
            "horizontal_flip", "color_jitter"]``.
        corruption_kwargs: Optional per-corruption keyword arguments.
        metric: Logit-space metric. Defaults to mean logit.
        max_samples: Maximum number of images to process.
        ie_threshold: IE above this → potential shortcut.
        backbone_prefix: Auto-detected if None.
        device: Inference device.

    Returns:
        ``{corruption_name: {
            "mean_ie_matrix": Tensor(n_layers, n_components),
            "std_ie_matrix" : Tensor(n_layers, n_components),
            "shortcuts"     : list of {"layer", "component", "mean_ie"},
        }}``
    """
    model = model.to(device).eval()
    if backbone_prefix is None:
        backbone_prefix = _detect_backbone_prefix(model)
    if layers is None:
        layers = list(range(_detect_num_layers(model)))
    if components is None:
        components = ["resid_post", "attn_output", "mlp_output"]
    if corruptions is None:
        corruptions = ["null_baseline", "patch_shuffle", "gaussian_noise",
                       "horizontal_flip", "color_jitter"]
    if corruption_kwargs is None:
        corruption_kwargs = {}
    if metric is None:
        metric = lambda logits: logits.mean()  # noqa: E731

    n_layers = len(layers)
    n_comp   = len(components)

    # Accumulate per-sample attribution scores for each corruption
    # shape: (n_corruptions, n_samples, n_layers, n_comp)
    scores_by_corr: dict[str, list[dict]] = {c: [] for c in corruptions}

    n_processed = 0
    for batch in dataloader:
        if n_processed >= max_samples:
            break
        images = batch[0].to(device)
        for i in range(images.shape[0]):
            if n_processed >= max_samples:
                break
            clean_img = images[i : i + 1]  # (1, C, H, W)

            for corr_name in corruptions:
                corr_fn = CORRUPTION_REGISTRY.get(corr_name)
                if corr_fn is None:
                    logger.warning(f"Unknown corruption '{corr_name}'; skipping")
                    continue
                kwargs = corruption_kwargs.get(corr_name, {})
                try:
                    corrupted = corr_fn(clean_img[0], **kwargs).unsqueeze(0).to(device)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Corruption '{corr_name}' failed: {e}")
                    continue

                try:
                    attr_scores = attribution_patching(
                        model=model,
                        clean_input=clean_img,
                        corrupt_input=corrupted,
                        layers=layers,
                        components=components,
                        metric=metric,
                        backbone_prefix=backbone_prefix,
                    )
                    scores_by_corr[corr_name].append(attr_scores)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Attribution patching failed ({corr_name}): {e}")

            n_processed += 1

        if n_processed % 10 == 0:
            logger.debug(f"detect_shortcuts: {n_processed}/{max_samples} samples")

    # Aggregate
    results: dict[str, Any] = {}
    for corr_name, sample_list in scores_by_corr.items():
        if not sample_list:
            logger.warning(f"No valid samples for corruption '{corr_name}'")
            continue

        # Build (n_samples, n_layers, n_comp) tensor
        ie_arr = torch.zeros(len(sample_list), n_layers, n_comp)
        for si, sample_dict in enumerate(sample_list):
            for li, layer_idx in enumerate(layers):
                for ci, comp in enumerate(components):
                    ie_arr[si, li, ci] = sample_dict.get((layer_idx, comp), 0.0)

        mean_ie = ie_arr.mean(0)  # (n_layers, n_comp)
        std_ie  = ie_arr.std(0)

        shortcuts = []
        for li, layer_idx in enumerate(layers):
            for ci, comp in enumerate(components):
                m = mean_ie[li, ci].item()
                if abs(m) > ie_threshold:
                    shortcuts.append({
                        "layer"    : layer_idx,
                        "component": comp,
                        "mean_ie"  : m,
                        "std_ie"   : std_ie[li, ci].item(),
                    })

        if shortcuts:
            logger.warning(
                f"[detect_shortcuts] '{corr_name}': {len(shortcuts)} potential shortcuts "
                f"(IE > {ie_threshold}): " +
                ", ".join(f"L{s['layer']}/{s['component']} IE={s['mean_ie']:.3f}" for s in shortcuts[:5])
            )

        results[corr_name] = {
            "mean_ie_matrix" : mean_ie,
            "std_ie_matrix"  : std_ie,
            "layers"         : layers,
            "components"     : components,
            "shortcuts"      : shortcuts,
            "n_samples"      : len(sample_list),
        }

    logger.info(
        f"detect_shortcuts complete — {n_processed} samples, "
        f"{len(corruptions)} corruptions, "
        f"{sum(len(v['shortcuts']) for v in results.values() if 'shortcuts' in v)} total flags"
    )
    return results
