"""
SP-LRP: Semantic-Prior Guided Layer-Wise Relevance Propagation for ViT-B/16.

Adaptation of the "SP-LRP" pathology paper to the CRC-histology classification
setting. The method modifies Chefer-style Transformer-Attribution relevance
(``src/xai/classical/generic_attention.py``) by replacing the raw attention
A^(l) with a *morphology-weighted* attention

    A_hat^(l) = rownorm( A^(l) ⊙ M^(l) ),   M^(l)_{ij} = exp(-||s_i - s_j||^2 / tau_l)

where s_i is a per-patch semantic-prior vector and tau_l grows with depth. When
the modulation is disabled (M == 1), A_hat == A and the method reduces exactly to
Chefer Transformer-Attribution --- i.e. the paper's "w/o MSPM" baseline.

Deviations from the paper, and why:
  * The paper builds the semantic prior with a DenseNet-201; CLAUDE.md forbids
    DenseNet, so we substitute the authorised, CRC-trained **ResNet-50** applied
    densely (``DenseMorphologyPrior``) — still an independent CNN morphology
    extractor, per the paper's intent.
  * The paper targets gigapixel WSIs whose fields of view span multiple tissue
    types. NCT-CRC-HE-100K tiles are pre-cropped *single-class* 224px patches, so
    intra-tile morphological heterogeneity is limited and M is expected to be
    close to 1 (i.e. SP-LRP ~ Chefer TA). This is a property of the dataset, not
    a bug; the insertion/deletion test quantifies whatever effect remains.
  * Mask-based metrics (Pointing Game, MAS, DSC) require lesion annotations CRC
    lacks; only Insertion/Deletion AUC is applicable here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xai.classical.generic_attention import GenericAttentionExplainer


# --------------------------------------------------------------------------
# 1. Dense morphology prior (ResNet-50 substitute for DenseNet-201)
# --------------------------------------------------------------------------

class DenseMorphologyPrior:
    """Per-patch semantic-prior vectors from a CRC-trained ResNet-50.

    Applies the ResNet's classifier head convolutionally to its final feature
    map, giving spatially-resolved class probabilities, then upsamples to the
    ViT token grid. Output: (B, P, C) probability vectors, one per ViT patch.
    """

    def __init__(self, resnet: nn.Module, grid: int = 14):
        self.resnet = resnet.eval()
        self.grid = grid
        # ResNet head is a Linear (C, 2048); apply it as a 1x1 conv over the map.
        self.W = resnet.fc.weight.detach()[:, :, None, None]   # (C, 2048, 1, 1)
        self.b = resnet.fc.bias.detach()                       # (C,)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.resnet.forward_features(images)            # (B, 2048, h, w)
        logits = F.conv2d(feat, self.W, self.b)                # (B, C, h, w)
        logits = F.interpolate(logits, size=(self.grid, self.grid),
                               mode="bilinear", align_corners=False)
        probs = logits.softmax(dim=1)                          # (B, C, g, g)
        return probs.flatten(2).transpose(1, 2).contiguous()   # (B, P, C)


# --------------------------------------------------------------------------
# 2. Config
# --------------------------------------------------------------------------

@dataclass
class SPLRPConfig:
    tau0: float = 0.1                # base temperature (paper: 0.1)
    layer_dependent_tau: bool = True  # tau_l = tau0*(l+1); else fixed tau0
    num_extra_tokens: int = 1        # [CLS]
    start_layer: int = 0
    eps: float = 1e-8


def _pairwise_sqdist(s: torch.Tensor) -> torch.Tensor:
    """(P, C) semantic vectors -> (P, P) squared Euclidean distances."""
    sq = (s * s).sum(-1)                                        # (P,)
    d = sq[:, None] + sq[None, :] - 2.0 * (s @ s.t())
    return d.clamp_min(0.0)


# --------------------------------------------------------------------------
# 3. Explainer: Chefer relevance with morphology-weighted attention
# --------------------------------------------------------------------------

class SPLRPExplainer(GenericAttentionExplainer):
    """Chefer Transformer-Attribution with an optional morphology modulation.

    Capture (forward+backward) is done once per image; the relevance rollout can
    then be recomputed cheaply for several modulation settings (full / fixed-tau
    / random-prior / off), since they all reuse the same attention and gradients.
    """

    def capture(self, image: torch.Tensor, target_class: int) -> None:
        """Run forward+backward for `target_class` and stash attn maps + grads."""
        self.attention_maps = []
        self.attention_grads = []
        self.model.eval()
        self.model.zero_grad()
        out = self.model(image)
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, tuple):
            out = out[0]
        out[0, target_class].backward(retain_graph=False)
        self._cap_attns = list(self.attention_maps)             # forward order
        self._cap_grads = list(reversed(self.attention_grads))  # align to forward order
        if len(self._cap_grads) != len(self._cap_attns) or not self._cap_attns:
            raise RuntimeError(
                f"attention/grad capture mismatch: {len(self._cap_attns)} attns, "
                f"{len(self._cap_grads)} grads (SDPA not disabled?)"
            )

    def relevance(self, prior_vecs: torch.Tensor | None, cfg: SPLRPConfig,
                  modulate: bool) -> torch.Tensor:
        """Compute the CLS->patch attribution from the captured attn/grads.

        prior_vecs: (P, C) semantic vectors (ignored when modulate is False).
        Returns (P,) patch attribution.
        """
        attns, grads = self._cap_attns, self._cap_grads
        B, H, N, _ = attns[0].shape
        device = attns[0].device
        ne = cfg.num_extra_tokens
        eye = torch.eye(N, device=device)

        if modulate:
            D = _pairwise_sqdist(prior_vecs.to(device))         # (P, P)

        rollout = eye.clone()
        for l in range(cfg.start_layer, len(attns)):
            A = attns[l]                                        # (B, H, N, N)
            grad = grads[l]
            if modulate:
                tau = cfg.tau0 * (l + 1) if cfg.layer_dependent_tau else cfg.tau0
                M = torch.exp(-D / max(tau, cfg.eps))           # (P, P)
                Mfull = torch.ones(N, N, device=device)
                Mfull[ne:, ne:] = M                             # CLS row/col stay 1
                A = A * Mfull
                A = A / (A.sum(dim=-1, keepdim=True) + cfg.eps)  # re-normalise -> stochastic
            cam = (A * grad).clamp(min=0).mean(dim=1)[0]        # (N, N)
            cam = cam + eye
            cam = cam / (cam.sum(dim=-1, keepdim=True) + cfg.eps)
            rollout = cam @ rollout
        return rollout[0, ne:]                                  # CLS row -> patches (P,)


# --------------------------------------------------------------------------
# 4. Variant dispatch
# --------------------------------------------------------------------------

def variant_saliency(explainer: SPLRPExplainer, variant: str, prior_vec: torch.Tensor,
                     cfg: SPLRPConfig, grid: int) -> torch.Tensor:
    """Return a (grid, grid) min-max-normalised saliency for a named variant.

    Assumes `explainer.capture(...)` has already run for this image.
    Variants:
      chefer_ta          modulation off (== paper's Standard ViT-LRP baseline)
      splrp_full         real prior, layer-dependent tau
      splrp_fixed_tau    real prior, fixed tau0
      splrp_random_prior random prior, layer-dependent tau (paper's null control)
    """
    if variant == "chefer_ta":
        sal = explainer.relevance(None, cfg, modulate=False)
    elif variant == "splrp_full":
        sal = explainer.relevance(prior_vec, cfg, modulate=True)
    elif variant == "splrp_fixed_tau":
        cfg_fixed = SPLRPConfig(tau0=cfg.tau0, layer_dependent_tau=False,
                                num_extra_tokens=cfg.num_extra_tokens,
                                start_layer=cfg.start_layer, eps=cfg.eps)
        sal = explainer.relevance(prior_vec, cfg_fixed, modulate=True)
    elif variant == "splrp_random_prior":
        rand = torch.softmax(torch.randn_like(prior_vec), dim=-1)
        sal = explainer.relevance(rand, cfg, modulate=True)
    else:
        raise ValueError(f"unknown SP-LRP variant: {variant}")

    S = sal.reshape(grid, grid)
    S = (S - S.min()) / (S.max() - S.min() + cfg.eps)
    return S
