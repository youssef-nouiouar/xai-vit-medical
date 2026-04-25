"""Circuit Discovery (ACDC) — Contribution C3.

Reference: Conmy et al., NeurIPS 2023 — "Towards Automated Circuit
Discovery for Mechanistic Interpretability".

NOTE: This is **Phase 1 stretch goal** / Phase 2 work. The Phase 1
priority is C1 (SAE) and C2 (Activation Patching). Implement only
after C1 and C2 are complete and validated.

Goal: identify the minimal computational sub-graph in a ViT that
implements the diagnostic behavior (e.g., melanoma vs nevus on ISIC).
"""

from __future__ import annotations

from typing import Any


def acdc(
    model: Any,
    clean_dataset: Any,
    corrupt_dataset: Any,
    threshold_tau: float = 0.01,
    metric: str = "kl_divergence",
) -> dict[str, Any]:
    """Run ACDC algorithm to find the minimal circuit.

    Args:
        model: Trained ViT.
        clean_dataset: Clean inputs.
        corrupt_dataset: Corresponding corrupted inputs.
        threshold_tau: Edge importance threshold (paper default 0.01).
        metric: Behavior metric (``"kl_divergence"`` or ``"logit_diff"``).

    Returns:
        Dict with ``"nodes"``, ``"edges"``, ``"importance_scores"``.
    """
    raise NotImplementedError("Phase 1 stretch goal — implement after C1 & C2.")
