"""
SLR-AR: Spectral-Laplacian Regularized Attention Rollout
Implementation for timm Vision Transformers (ViT-B/16, CLS-token readout).

Paper hyperparameters (Sec 5.1):
    tau (spectral norm threshold) = 0.95
    K   (power iteration steps)   = 5
    M   (Taylor expansion order)  = 8

Variants exposed for the ablation grid:
    'rollout' : standard Abnar & Zuidema rollout          (baseline)
    'snb'     : + spectral norm bounding only
    'gts'     : + graph-theoretic Laplacian smoothing only
    'slrar'   : full SLR-AR (SNB + GTS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Attention capture from timm
# --------------------------------------------------------------------------

class AttentionCatcher:
    """Capture per-layer softmax attention maps from a timm VisionTransformer.

    timm >=0.9 uses F.scaled_dot_product_attention when `fused_attn=True`,
    which never materialises the attention matrix. We disable fusion and hook
    `attn_drop`, whose *output* is exactly the post-softmax attention tensor
    of shape (B, heads, N, N).
    """

    def __init__(self, model):
        self.attns: list[torch.Tensor] = []
        self.handles = []
        self._saved_fused = []

        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("Model has no `.blocks` — is this a timm ViT?")

        for blk in blocks:
            attn = blk.attn
            if hasattr(attn, "fused_attn"):
                self._saved_fused.append((attn, attn.fused_attn))
                attn.fused_attn = False
            self.handles.append(attn.attn_drop.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        self.attns.append(out.detach())

    def clear(self):
        self.attns = []

    def remove(self):
        for h in self.handles:
            h.remove()
        for attn, flag in self._saved_fused:
            attn.fused_attn = flag
        self.handles = []
        self._saved_fused = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def get_attentions(model, x, catcher: AttentionCatcher):
    """Forward pass returning (logits, list of L tensors (B, heads, N, N))."""
    catcher.clear()
    logits = model(x)
    return logits, list(catcher.attns)


# --------------------------------------------------------------------------
# 2. SLR-AR building blocks
# --------------------------------------------------------------------------

@dataclass
class SLRARConfig:
    tau: float = 0.95          # spectral norm threshold      (paper: 0.95)
    K: int = 5                 # power iteration steps        (paper: 5)
    M: int = 8                 # Taylor expansion order       (paper: 8)
    residual_alpha: float = 0.5  # P = a*A + (1-a)*I, then row-normalise
    smoothing: str = "applied"   # 'paper'  -> Q = exp(-beta L)
                                 # 'applied'-> Q = exp(-beta L) @ P_tilde
    beta_mode: str = "adaptive"  # 'adaptive' -> beta = 1/lambda2 ; 'fixed'
    beta_fixed: float = 1.0
    beta_clamp: tuple[float, float] = (0.1, 20.0)  # guard for lambda2 -> 0
    exact_exp: bool = False      # True -> torch.linalg.matrix_exp (reference)
    renorm_rows: bool = False    # re-row-normalise Q (breaks the tau bound)
    eps: float = 1e-8


def head_average_and_residual(A: torch.Tensor, cfg: SLRARConfig) -> torch.Tensor:
    """(B, heads, N, N) -> (B, N, N) row-stochastic propagation matrix P.

    Standard practice (Abnar & Zuidema): average heads, add the residual
    connection as an identity mixture, then row-normalise.
    The paper (p.8, Fig. 2) prescribes this step but never writes the formula.
    """
    P = A.mean(dim=1)                                    # (B, N, N)
    N = P.shape[-1]
    I = torch.eye(N, device=P.device, dtype=P.dtype).expand_as(P)
    a = cfg.residual_alpha
    P = a * P + (1.0 - a) * I
    P = P / P.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
    return P


def power_iteration_sigma(P: torch.Tensor, K: int, eps: float = 1e-8):
    """Differentiable estimate of the dominant singular value (Eq. 4).

    Returns sigma (B,), u (B,N), v (B,N).
    """
    B, N, _ = P.shape
    v = torch.randn(B, N, 1, device=P.device, dtype=P.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)
    Pt = P.transpose(-2, -1)

    u = None
    for _ in range(K):
        u = P @ v
        u = u / u.norm(dim=1, keepdim=True).clamp_min(eps)
        v = Pt @ u
        v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)

    sigma = (u.transpose(-2, -1) @ P @ v).reshape(B)      # Eq. 4, sigma_K
    return sigma, u.squeeze(-1), v.squeeze(-1)


def spectral_projection(P: torch.Tensor, cfg: SLRARConfig) -> torch.Tensor:
    """Eq. 5: scale P by tau/sigma_K when sigma_K > tau.

    NOTE: this makes P *sub*-stochastic (every row sum is multiplied by
    tau/sigma < 1). The paper claims the result stays row-stochastic AND has
    spectral norm <= tau < 1; those two cannot hold simultaneously. We keep
    the tau-scaling (it is what actually bounds the recursion) and treat the
    output as a relevance score rather than a probability.
    """
    sigma, _, _ = power_iteration_sigma(P, cfg.K, cfg.eps)
    scale = torch.where(
        sigma > cfg.tau,
        cfg.tau / sigma.clamp_min(cfg.eps),
        torch.ones_like(sigma),
    )
    return P * scale.view(-1, 1, 1)


def normalized_laplacian(P_tilde: torch.Tensor, cfg: SLRARConfig):
    """Eq. 7. Returns L (B, N, N)."""
    W = 0.5 * (P_tilde + P_tilde.transpose(-2, -1))       # symmetrise
    W = W.clamp_min(0.0)                                  # affinities >= 0
    d = W.sum(dim=-1)                                     # (B, N)
    d_inv_sqrt = d.clamp_min(cfg.eps).pow(-0.5)
    N = W.shape[-1]
    I = torch.eye(N, device=W.device, dtype=W.dtype).expand_as(W)
    Wn = d_inv_sqrt.unsqueeze(-1) * W * d_inv_sqrt.unsqueeze(-2)
    return I - Wn


def algebraic_connectivity(L: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """lambda_2: second-smallest eigenvalue of the normalized Laplacian.

    N=197 -> a dense symmetric eigensolve costs ~ms; fine at this scale.
    """
    Lsym = 0.5 * (L + L.transpose(-2, -1))                # enforce symmetry
    evals = torch.linalg.eigvalsh(Lsym)                   # ascending, (B, N)
    return evals[:, 1].clamp_min(eps)


def laplacian_exponential(L: torch.Tensor, beta: torch.Tensor, cfg: SLRARConfig):
    """Eq. 9: truncated Taylor series for exp(-beta L)."""
    if cfg.exact_exp:
        return torch.linalg.matrix_exp(-beta.view(-1, 1, 1) * L)

    B, N, _ = L.shape
    I = torch.eye(N, device=L.device, dtype=L.dtype).expand(B, N, N)
    G = I.clone()
    term = I.clone()
    b = beta.view(-1, 1, 1)
    for m in range(1, cfg.M + 1):
        term = torch.bmm(term, L) * (-b) / m              # (-beta)^m/m! * L^m
        G = G + term
    return G


def graph_smoothing(P_tilde: torch.Tensor, cfg: SLRARConfig):
    """Eq. 8/10. Returns Q and diagnostics."""
    L = normalized_laplacian(P_tilde, cfg)

    if cfg.beta_mode == "adaptive":
        lam2 = algebraic_connectivity(L, cfg.eps)          # (B,)
        beta = (1.0 / lam2).clamp(*cfg.beta_clamp)         # beta = 1/lambda_2
    else:
        lam2 = torch.full((P_tilde.shape[0],), float("nan"), device=P_tilde.device)
        beta = torch.full((P_tilde.shape[0],), cfg.beta_fixed, device=P_tilde.device)

    G = laplacian_exponential(L, beta, cfg)

    if cfg.smoothing == "paper":
        Q = G                       # Eq. 10 verbatim: attention-free, symmetric
    elif cfg.smoothing == "applied":
        Q = torch.bmm(G, P_tilde)   # diffusion kernel applied to the signal
    else:
        raise ValueError(f"unknown smoothing mode: {cfg.smoothing}")

    if cfg.renorm_rows:
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

    return Q, {"lambda2": lam2.detach(), "beta": beta.detach()}


# --------------------------------------------------------------------------
# 3. Recursive aggregation (the four variants)
# --------------------------------------------------------------------------

def build_rollout(attns: list[torch.Tensor], variant: str, cfg: SLRARConfig):
    """attns: list of L tensors (B, heads, N, N) in shallow->deep order.

    Returns R (B, N, N) and a diagnostics dict.
    """
    assert variant in {"rollout", "snb", "gts", "slrar"}
    use_snb = variant in {"snb", "slrar"}
    use_gts = variant in {"gts", "slrar"}

    B, _, N, _ = attns[0].shape
    device, dtype = attns[0].device, attns[0].dtype
    R = torch.eye(N, device=device, dtype=dtype).expand(B, N, N).clone()

    diags = {"sigma": [], "lambda2": [], "beta": [], "q_norm": []}

    for A in attns:
        P = head_average_and_residual(A, cfg)

        sigma, _, _ = power_iteration_sigma(P, cfg.K, cfg.eps)
        diags["sigma"].append(sigma.detach().cpu())

        Pt = spectral_projection(P, cfg) if use_snb else P

        if use_gts:
            Q, d = graph_smoothing(Pt, cfg)
            diags["lambda2"].append(d["lambda2"].cpu())
            diags["beta"].append(d["beta"].cpu())
        else:
            Q = Pt

        qn, _, _ = power_iteration_sigma(Q, cfg.K, cfg.eps)
        diags["q_norm"].append(qn.detach().cpu())

        R = torch.bmm(R, Q)                               # Eq. 11

    return R, diags


def cls_attribution_map(R: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """Extract the CLS row, drop the CLS self-entry, reshape to (B, grid, grid)."""
    cls_row = R[:, 0, 1:]                                 # (B, N-1)
    B, P = cls_row.shape
    assert P == grid * grid, f"expected {grid*grid} patches, got {P}"
    m = cls_row.reshape(B, grid, grid)
    # min-max normalise per image for visualisation / ranking
    flat = m.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values.view(B, 1, 1)
    mx = flat.max(dim=1, keepdim=True).values.view(B, 1, 1)
    return (m - mn) / (mx - mn).clamp_min(1e-12)


# --------------------------------------------------------------------------
# 4. Spectral Stability Index (Sec 5.1)
# --------------------------------------------------------------------------

@torch.no_grad()
def spectral_stability_index(R: torch.Tensor) -> torch.Tensor:
    """SSI = |lambda_2| / |lambda_1| of the aggregated rollout matrix R.

    Paper reference values (ViT-B/16, ImageNet):
        standard rollout ~ 0.02      full SLR-AR ~ 0.24
    Label-free: this is your cheapest correctness check.
    """
    ev = torch.linalg.eigvals(R.double())                 # (B, N) complex
    mag = ev.abs()
    top2 = mag.topk(2, dim=-1).values                     # (B, 2)
    return (top2[:, 1] / top2[:, 0].clamp_min(1e-12)).float()


@torch.no_grad()
def eigen_spectrum(R: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Top-k eigenvalue magnitudes of R (for the Figure-4 style depth plot)."""
    ev = torch.linalg.eigvals(R.double()).abs()
    return ev.topk(min(k, ev.shape[-1]), dim=-1).values.float()


@torch.no_grad()
def ssi_by_depth(attns, variant: str, cfg: SLRARConfig):
    """SSI of the partial product R^(l) after each layer -> collapse curve."""
    B, _, N, _ = attns[0].shape
    device, dtype = attns[0].device, attns[0].dtype
    R = torch.eye(N, device=device, dtype=dtype).expand(B, N, N).clone()
    use_snb = variant in {"snb", "slrar"}
    use_gts = variant in {"gts", "slrar"}
    out = []
    for A in attns:
        P = head_average_and_residual(A, cfg)
        Pt = spectral_projection(P, cfg) if use_snb else P
        Q = graph_smoothing(Pt, cfg)[0] if use_gts else Pt
        R = torch.bmm(R, Q)
        out.append(spectral_stability_index(R).cpu())
    return torch.stack(out, dim=0)                        # (L, B)
