"""
SAHAR (training-free adaptation) for timm Vision Transformers (ViT-B/16, CLS readout).

The source method --- Scale-Aware Hierarchical Attention Refinement, Imran-style
"SAHAR" --- is a *learnable* post-hoc module (scale projections, cross-attention
linear layers, a 7x7 gating conv, and fusion weights) trained with a
binary-cross-entropy faithfulness loss against radiologist lesion masks on a
Swin-V2 backbone. The CRC-histology benchmark has **no lesion masks** (each patch
is a single tissue class), so that supervised objective cannot be instantiated,
and the paper's mask-based metrics (Pointing Game, IoU, EBPG) are not computable.

This module therefore implements the *training-free* core of SAHAR: the four
mechanisms with their learnable parts replaced by parameter-free, deterministic
operators. It isolates the method's **inductive bias**, which is exactly what an
insertion/deletion test can measure without masks.

    MRTP  multi-resolution average pooling            (learnable proj -> identity)
    SACA  sigmoid-cosine anatomical-prior recalibration (parameter-free by design)
    RGAG  resolution-guided gating                    (learned 7x7 conv -> fixed low-pass)
    LWA   scale fusion                                 (learned weights -> uniform mean;
                                                        the paper inits them to 0 == uniform)

Everything here is post-hoc and non-differentiable; it only reads captured
attention and token embeddings and returns a 14x14 saliency map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Capture attention + token embeddings from one timm ViT block
# --------------------------------------------------------------------------

class SAHARCapture:
    """Capture the post-softmax attention and output token embeddings of one block.

    Unlike a rollout catcher this stores a single tensor per forward (it
    overwrites, never appends), so repeated forward passes --- e.g. the masked
    sweeps of an insertion/deletion metric --- cannot leak memory. Read
    ``.attn`` / ``.tokens`` immediately after the forward you care about.
    """

    def __init__(self, model, layer: int):
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("Model has no `.blocks` — is this a timm ViT?")
        if not (0 <= layer < len(blocks)):
            raise ValueError(f"layer {layer} out of range [0, {len(blocks)})")

        self.attn: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None
        self._attn_mod = blocks[layer].attn
        self._saved_fused = None
        if hasattr(self._attn_mod, "fused_attn"):
            self._saved_fused = self._attn_mod.fused_attn
            self._attn_mod.fused_attn = False   # force materialisation of the (B,H,N,N) map

        self._h_attn = self._attn_mod.attn_drop.register_forward_hook(self._attn_hook)
        self._h_tok = blocks[layer].register_forward_hook(self._tok_hook)

    def _attn_hook(self, module, inp, out):
        self.attn = out.detach()

    def _tok_hook(self, module, inp, out):
        self.tokens = (out[0] if isinstance(out, tuple) else out).detach()

    def remove(self):
        self._h_attn.remove()
        self._h_tok.remove()
        if self._saved_fused is not None:
            self._attn_mod.fused_attn = self._saved_fused

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# --------------------------------------------------------------------------
# 2. Config
# --------------------------------------------------------------------------

@dataclass
class SAHARConfig:
    layer: int = 11                          # block index to read attention/tokens from
    scales: tuple[int, ...] = (2, 4, 0)      # avg-pool kernels: micro=2, meso=4, macro=0 (global)
    use_mrtp: bool = True                    # multi-scale pooling (False -> single full-grid scale)
    use_saca: bool = True                    # sigmoid-cosine anatomical-prior recalibration
    use_rgag: bool = True                    # fixed low-pass gating
    gate_kernel: int = 7                     # RGAG smoothing kernel (odd); paper's conv is 7x7
    gate_gain: float = 10.0                  # RGAG sigmoid slope (fixed; replaces the learned conv)
    fusion: str = "uniform"                  # LWA training-free fusion (uniform mean)
    num_extra_tokens: int = 1                # [CLS]
    eps: float = 1e-8


# --------------------------------------------------------------------------
# 3. Building blocks
# --------------------------------------------------------------------------

def _gauss_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Separable reflect-padded Gaussian blur on (B,1,g,g)."""
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * (k / 3.0) ** 2))
    g = g / g.sum()
    gx = g.view(1, 1, 1, k)
    gy = g.view(1, 1, k, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), gx)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), gy)
    return x


def saca_coherence(Xp: torch.Tensor, k: int, grid: int, cfg: SAHARConfig) -> torch.Tensor:
    """SACA anatomical prior (parameter-free): per patch, the strongest
    sigmoid-cosine similarity to the pooled tokens at scale `k`.

    Xp   : (B, P, d) patch-token embeddings.
    k    : pooling kernel (0 -> global/macro).
    grid : spatial side (sqrt(P)).
    Returns (B, P) coherence in (0,1); high == aligned with a coherent region.
    """
    B, P, d = Xp.shape
    spatial = Xp.transpose(1, 2).reshape(B, d, grid, grid)      # (B,d,g,g)
    if k == 0:                                                  # macro: single global token
        C = spatial.mean(dim=(2, 3)).reshape(B, 1, d)          # (B,1,d)
    else:
        pooled = F.avg_pool2d(spatial, kernel_size=k, stride=k, ceil_mode=True)
        C = pooled.flatten(2).transpose(1, 2)                  # (B, M, d)
    Xn = F.normalize(Xp, dim=-1, eps=cfg.eps)
    Cn = F.normalize(C, dim=-1, eps=cfg.eps)
    sim = torch.bmm(Xn, Cn.transpose(1, 2))                    # (B, P, M) cosine similarity
    return torch.sigmoid(sim).amax(dim=-1)                     # (B, P) max over pooled tokens


def rgag_gate(S_hw: torch.Tensor, cfg: SAHARConfig) -> torch.Tensor:
    """RGAG (parameter-free): suppress high-frequency / below-average activations.

    Replaces the paper's learned 7x7 conv with a fixed low-pass filter: gate on
    the *smoothed* map so isolated speckle is attenuated and spatially coherent,
    above-average regions are preserved.
    """
    smooth = _gauss_blur(S_hw, cfg.gate_kernel)
    mu = smooth.mean(dim=(2, 3), keepdim=True)
    gate = torch.sigmoid(cfg.gate_gain * (smooth - mu))
    return S_hw * gate


# --------------------------------------------------------------------------
# 4. Full pipeline
# --------------------------------------------------------------------------

def sahar_saliency(attn: torch.Tensor, tokens: torch.Tensor, cfg: SAHARConfig) -> torch.Tensor:
    """Training-free SAHAR saliency.

    attn   : (B, H, N, N) post-softmax attention at the selected block.
    tokens : (B, N, d)    token embeddings at the selected block.
    Returns (B, grid, grid) min-max-normalised saliency over the patch grid.
    """
    B, H, N, _ = attn.shape
    ne = cfg.num_extra_tokens

    A = attn.mean(dim=1)                       # head-average (B, N, N)
    base = A[:, 0, ne:]                        # CLS -> patch attention (B, P)
    P = base.shape[1]
    grid = int(round(P ** 0.5))
    assert grid * grid == P, f"non-square patch grid: {P}"

    Xp = tokens[:, ne:, :]                     # (B, P, d) patch embeddings

    # single-scale ablation uses one genuine pooled scale (the micro kernel), not
    # kernel 1: with no pooling, SACA's max-cosine is dominated by each token's
    # self-similarity and collapses to a near-constant, which is uninformative.
    scales = cfg.scales if cfg.use_mrtp else (cfg.scales[0],)
    maps = []
    for k in scales:
        coh = saca_coherence(Xp, k, grid, cfg) if cfg.use_saca else torch.ones_like(base)
        S = base * coh                         # recalibrate the base attention (B, P)
        S_hw = S.reshape(B, 1, grid, grid)
        if cfg.use_rgag:
            S_hw = rgag_gate(S_hw, cfg)
        maps.append(S_hw)

    S = torch.stack(maps, dim=0).mean(dim=0)   # LWA uniform fusion (B,1,g,g)
    S = S.reshape(B, grid, grid)

    flat = S.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values.view(B, 1, 1)
    mx = flat.max(dim=1, keepdim=True).values.view(B, 1, 1)
    return (S - mn) / (mx - mn).clamp_min(cfg.eps)


# Named ablation configs -> mirror the paper's ablation study (Table 4),
# adapted to the training-free setting.
def variant_config(variant: str, base: SAHARConfig) -> SAHARConfig:
    """Return a SAHARConfig for a named ablation variant."""
    kw = dict(
        layer=base.layer, scales=base.scales, gate_kernel=base.gate_kernel,
        gate_gain=base.gate_gain, fusion=base.fusion,
        num_extra_tokens=base.num_extra_tokens, eps=base.eps,
    )
    if variant == "raw_attention":
        return SAHARConfig(use_mrtp=False, use_saca=False, use_rgag=False, **kw)
    if variant == "sahar_full":
        return SAHARConfig(use_mrtp=True, use_saca=True, use_rgag=True, **kw)
    if variant == "sahar_no_saca":
        return SAHARConfig(use_mrtp=True, use_saca=False, use_rgag=True, **kw)
    if variant == "sahar_no_rgag":
        return SAHARConfig(use_mrtp=True, use_saca=True, use_rgag=False, **kw)
    if variant == "sahar_single_scale":
        return SAHARConfig(use_mrtp=False, use_saca=True, use_rgag=True, **kw)
    raise ValueError(f"unknown SAHAR variant: {variant}")
