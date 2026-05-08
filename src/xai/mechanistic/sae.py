"""Sparse Autoencoders for ViT — Contribution C1.

References:
    - Cunningham et al., ICLR 2024 — "SAEs Find Highly Interpretable Features"
    - Templeton et al., 2024 — "Scaling Monosemanticity" (Anthropic)
    - Gao et al., 2024 — "Scaling and Evaluating SAEs" (OpenAI; TopK)

Pipeline:
    1. Collect activations from a target layer of a trained ViT.
    2. Train a TopK-SAE on these activations.
    3. Analyze each feature: top-K activating tokens/images.
    4. Optionally validate clinically (with domain expert).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder (Gao et al., 2024).

    Architecture::

        x_centered = x - b_pre
        pre_acts   = ReLU(x_centered @ W_enc + b_enc)   # (B, d_sae)
        acts       = topk(pre_acts, k)                   # sparse: (B, d_sae)
        x_hat      = acts @ W_dec + b_pre                # (B, d_in)

    Decoder columns are kept unit-norm throughout training.
    b_pre is shared between encoder and decoder — acts as a data-centring bias
    (Templeton et al. 2024 "Scaling Monosemanticity").

    Args:
        d_in: Input dimension (768 for ViT-Base / DeiT-Base / DINOv2-ViT-B).
        d_sae: Hidden (feature) dimension. Typically ``d_in × expansion_factor``.
        k: Number of active features per token (sparsity).
        aux_k: Features used in the auxiliary dead-feature loss.
            Defaults to ``d_sae // 2``.
        aux_loss_scale: Scalar weight on the auxiliary loss (Gao et al. use 1/32).
    """

    def __init__(
        self,
        d_in: int = 768,
        d_sae: int = 4096,
        k: int = 40,
        aux_k: int | None = None,
        aux_loss_scale: float = 1.0 / 32,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.aux_k = aux_k if aux_k is not None else max(k, d_sae // 2)
        self.aux_loss_scale = aux_loss_scale

        # Shared pre-encoder / decoder bias — data-centring (init = 0, set later from mean)
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            # Decoder init: transpose of encoder, then normalize
            self.W_dec.data = self.W_enc.data.T.clone()
        self._normalize_decoder()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Project decoder row vectors to unit norm (in-place)."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def set_b_pre_to_mean(self, mean: torch.Tensor) -> None:
        """Data-dependent initialization of b_pre from the dataset mean."""
        with torch.no_grad():
            self.b_pre.data.copy_(mean.to(self.b_pre.device))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse TopK encoding.

        Args:
            x: ``(B, d_in)``

        Returns:
            Sparse activation codes ``(B, d_sae)``.
        """
        pre_acts = F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sparse codes.

        Args:
            codes: ``(B, d_sae)``

        Returns:
            ``(B, d_in)``
        """
        return codes @ self.W_dec + self.b_pre

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, d_in)``

        Returns:
            ``(x_hat, acts, aux_loss)``
            - x_hat: reconstructed input ``(B, d_in)``
            - acts: sparse codes ``(B, d_sae)``
            - aux_loss: scalar auxiliary loss for dead-feature prevention
        """
        pre_acts = F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

        # TopK sparse codes
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)

        x_hat = acts @ self.W_dec + self.b_pre

        # Auxiliary loss: encourage dead features to reconstruct the residual
        residual = (x - x_hat).detach()
        aux_k = min(self.aux_k, self.d_sae)
        aux_vals, aux_idx = pre_acts.topk(aux_k, dim=-1)
        aux_acts = torch.zeros_like(pre_acts)
        aux_acts.scatter_(-1, aux_idx, aux_vals)
        x_hat_aux = aux_acts @ self.W_dec.detach() + self.b_pre.detach()
        aux_loss = F.mse_loss(x_hat_aux, residual + x_hat.detach()) * self.aux_loss_scale

        return x_hat, acts, aux_loss

    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor) -> dict[str, float]:
        """Quality metrics on a batch (no grad).

        Returns:
            Dict with ``reconstruction_loss``, ``L0``, ``variance_explained``.
        """
        x_hat, acts, _ = self(x)
        recon = F.mse_loss(x_hat, x).item()
        L0 = (acts > 0).float().sum(-1).mean().item()
        x_var = F.mse_loss(x, x.mean(0, keepdim=True)).item()
        var_exp = max(0.0, 1.0 - recon / (x_var + 1e-8))
        return {"reconstruction_loss": recon, "L0": L0, "variance_explained": var_exp}


# ---------------------------------------------------------------------------
# Module path resolver (shared utility)
# ---------------------------------------------------------------------------

def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    module: Any = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_activations(
    model: nn.Module,
    dataloader: Any,
    target_layer: str,
    device: torch.device | str = "cuda",
    max_batches: int | None = None,
    cache_path: Path | None = None,
    token_selection: str = "patch",
    num_extra_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect per-token activations from a ViT MLP layer.

    Registers a forward hook on ``target_layer``, runs inference over
    ``dataloader`` (no grad), and returns the stacked activation vectors.

    Args:
        model: Trained ViT (should be in eval mode on ``device``).
        dataloader: Yields ``(images, labels)`` batches.
        target_layer: Dotted module path, e.g. ``"blocks.8.mlp"`` or
            ``"backbone.blocks.8.mlp"`` for DINOv2.
        device: Inference device.
        max_batches: Number of batches to process (``None`` = all).
        cache_path: If provided, save/load the result to/from disk.
        token_selection: Which tokens to keep.
            ``"patch"``  — patch tokens only (skip first ``num_extra_tokens``).
            ``"cls"``    — CLS token only (index 0).
            ``"all"``    — all tokens.
        num_extra_tokens: Number of non-patch tokens (1 for ViT/DINOv2, 2 for DeiT).

    Returns:
        ``(activations, image_indices)`` where

        - ``activations``: ``(N, d_in)`` float tensor on CPU.
        - ``image_indices``: ``(N,)`` long tensor — global image index for each token.
    """
    if cache_path is not None and Path(cache_path).exists():
        idx_path = Path(str(cache_path) + ".idx.pt")
        logger.info(f"Loading cached activations from {cache_path}")
        acts = torch.load(cache_path, map_location="cpu")
        img_idx = torch.load(idx_path, map_location="cpu") if idx_path.exists() else None
        if img_idx is not None:
            return acts, img_idx

    model = model.to(device).eval()
    module = _get_module_by_path(model, target_layer)

    collected_acts: list[torch.Tensor] = []
    collected_img_idx: list[torch.Tensor] = []
    global_img_counter = 0

    def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        nonlocal global_img_counter
        if out.ndim == 3:
            B, N_tok, D = out.shape
            if token_selection == "cls":
                vecs = out[:, 0:1, :].reshape(-1, D)  # (B, D)
                n_per_img = 1
            elif token_selection == "patch":
                vecs = out[:, num_extra_tokens:, :].reshape(-1, D)
                n_per_img = N_tok - num_extra_tokens
            else:
                vecs = out.reshape(-1, D)
                n_per_img = N_tok
        elif out.ndim == 2:
            vecs = out  # (B, D) — already flat
            n_per_img = 1
        else:
            logger.warning(f"Unexpected activation ndim={out.ndim}; skipping")
            return

        collected_acts.append(vecs.detach().cpu())
        # Build image_indices for this batch
        img_ids = torch.arange(global_img_counter, global_img_counter + B)
        img_ids = img_ids.unsqueeze(1).expand(-1, n_per_img).reshape(-1)
        collected_img_idx.append(img_ids)
        global_img_counter += B

    handle = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = batch[0].to(device)
                _ = model(images)
                if batch_idx % 20 == 0:
                    n_so_far = sum(a.shape[0] for a in collected_acts)
                    logger.debug(
                        f"collect_activations: batch {batch_idx} | "
                        f"{n_so_far:,} token vectors so far"
                    )
    finally:
        handle.remove()

    activations = torch.cat(collected_acts, dim=0)
    image_indices = torch.cat(collected_img_idx, dim=0)
    logger.info(
        f"Collected {activations.shape[0]:,} activation vectors "
        f"({global_img_counter:,} images, d_in={activations.shape[1]}) "
        f"from '{target_layer}'"
    )

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activations, p)
        torch.save(image_indices, Path(str(p) + ".idx.pt"))
        logger.info(f"Cached activations → {p}")

    return activations, image_indices


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------

def train_sae(
    sae: TopKSAE,
    activations: torch.Tensor,
    cfg: Any,
    device: torch.device | str = "cuda",
) -> dict[str, list[float]]:
    """Train TopK-SAE on collected activation vectors.

    Args:
        sae: Initialized ``TopKSAE`` (moved to ``device`` internally).
        activations: Float tensor ``(N, d_in)`` — all collected token vectors.
        cfg: Config object with attributes accessible as ``cfg.training.optimizer.lr``,
            ``cfg.training.batch_size``, ``cfg.training.num_epochs``,
            ``cfg.training.scheduler.warmup_steps``.
        device: Training device.

    Returns:
        Training history dict with per-epoch lists for
        ``reconstruction_loss``, ``aux_loss``, ``L0``, ``variance_explained``.
    """
    sae = sae.to(device)

    def _cfg(path: str, default: Any) -> Any:
        obj = cfg
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        try:
            return type(default)(obj)
        except Exception:
            return default

    lr = _cfg("training.optimizer.lr", 5e-5)
    batch_size = _cfg("training.batch_size", 4096)
    num_epochs = _cfg("training.num_epochs", 10)
    warmup_steps = _cfg("training.scheduler.warmup_steps", 1000)

    # Data-dependent init of b_pre
    n_init = min(10_000, activations.shape[0])
    sae.set_b_pre_to_mean(activations[:n_init].mean(0))
    logger.info(f"b_pre ← dataset mean (from {n_init:,} samples)")

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    N = activations.shape[0]
    steps_per_epoch = max(1, N // batch_size)
    total_steps = num_epochs * steps_per_epoch

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * prog)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    history: dict[str, list[float]] = {
        "reconstruction_loss": [],
        "aux_loss": [],
        "L0": [],
        "variance_explained": [],
    }

    activations_dev = activations.to(device)

    for epoch in range(num_epochs):
        sae.train()
        perm = torch.randperm(N, device=device)
        e_recon, e_aux, e_L0, steps = 0.0, 0.0, 0.0, 0

        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            x = activations_dev[idx]

            x_hat, acts, aux_loss = sae(x)
            recon_loss = F.mse_loss(x_hat, x)
            loss = recon_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sae._normalize_decoder()

            e_recon += recon_loss.item()
            e_aux += aux_loss.item()
            e_L0 += (acts > 0).float().sum(-1).mean().item()
            steps += 1

        if steps == 0:
            continue

        # Variance explained on a held-out sample
        sae.eval()
        with torch.no_grad():
            val_idx = torch.randperm(N, device=device)[:min(4096, N)]
            val_x = activations_dev[val_idx]
            metrics = sae.compute_metrics(val_x)

        history["reconstruction_loss"].append(e_recon / steps)
        history["aux_loss"].append(e_aux / steps)
        history["L0"].append(e_L0 / steps)
        history["variance_explained"].append(metrics["variance_explained"])

        logger.info(
            f"SAE epoch {epoch + 1}/{num_epochs} | "
            f"recon={e_recon/steps:.4f} | aux={e_aux/steps:.4f} | "
            f"L0={e_L0/steps:.1f}/{sae.k} | R²={metrics['variance_explained']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

def analyze_features(
    sae: TopKSAE,
    activations: torch.Tensor,
    image_indices: torch.Tensor,
    top_k_images: int = 10,
    batch_size: int = 2048,
    device: torch.device | str = "cuda",
    num_features: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Find top-K activating images for each SAE feature.

    Uses a streaming approach — processes ``activations`` in batches and
    maintains a running top-K per feature without materializing the full
    ``(d_sae, N)`` activation matrix.

    Args:
        sae: Trained ``TopKSAE``.
        activations: ``(N, d_in)`` — collected token activation vectors.
        image_indices: ``(N,)`` — source image index for each token.
        top_k_images: Number of top-activating images to keep per feature.
        batch_size: Tokens per encoding batch.
        device: Device for SAE forward pass.
        num_features: Analyse only the first ``num_features`` features
            (``None`` = all). Useful for quick exploration.

    Returns:
        ``feature_idx -> dict`` with keys:

        - ``top_token_indices``: ``(top_k,)`` global token indices.
        - ``top_image_indices``: ``(top_k,)`` source image indices.
        - ``activation_strengths``: ``(top_k,)`` activation values.
        - ``mean_activation``: mean activation across all tokens.
        - ``activation_frequency``: fraction of tokens where feature fires.
    """
    sae = sae.to(device).eval()
    N = activations.shape[0]
    n_feats = num_features if num_features is not None else sae.d_sae
    K = min(top_k_images, N)

    # Running top-K per feature (on CPU to save GPU VRAM)
    topk_vals = torch.full((n_feats, K), float("-inf"))
    topk_idx = torch.zeros(n_feats, K, dtype=torch.long)

    # Running statistics
    feat_sum = torch.zeros(n_feats)
    feat_nnz = torch.zeros(n_feats)

    logger.info(f"Analyzing {n_feats} SAE features over {N:,} activation vectors…")

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = activations[start:end].to(device)
            codes = sae.encode(x)[:, :n_feats].cpu()  # (B, n_feats)
            B = codes.shape[0]

            global_positions = torch.arange(start, end)  # (B,)

            # Per-batch stats
            feat_sum += codes.sum(0)
            feat_nnz += (codes > 0).float().sum(0)

            # Merge current batch into running top-K
            # combined shape: (n_feats, K + B)
            combined_vals = torch.cat([topk_vals, codes.T], dim=1)
            combined_idx_batch = global_positions.unsqueeze(0).expand(n_feats, -1)
            combined_idx = torch.cat([topk_idx, combined_idx_batch], dim=1)

            new_vals, sel = combined_vals.topk(K, dim=1)
            topk_vals = new_vals
            topk_idx = combined_idx.gather(1, sel)

            if start % (batch_size * 20) == 0:
                logger.debug(f"  feature analysis: {end:,}/{N:,} tokens processed")

    mean_act = feat_sum / max(N, 1)
    act_freq = feat_nnz / max(N, 1)

    result: dict[int, dict[str, Any]] = {}
    img_idx_tensor = image_indices  # (N,)

    for fi in range(n_feats):
        tok_idx = topk_idx[fi]  # (K,)
        result[fi] = {
            "top_token_indices": tok_idx,
            "top_image_indices": img_idx_tensor[tok_idx],
            "activation_strengths": topk_vals[fi],
            "mean_activation": mean_act[fi].item(),
            "activation_frequency": act_freq[fi].item(),
        }

    logger.info(
        f"Feature analysis complete — "
        f"{(act_freq > 0).sum().item()} / {n_feats} features are active "
        f"(dead: {(act_freq == 0).sum().item()})"
    )
    return result
