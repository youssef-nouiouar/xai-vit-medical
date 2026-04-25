"""Training entry point — Hydra app.

Usage:
    python -m src.training.trainer model=vit_base
    python -m src.training.trainer -m model=resnet50,vit_base,deit_base
    python -m src.training.trainer model=deit_base training.epochs=50
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: str,
    cfg: DictConfig,
) -> dict[str, float]:
    """Single training epoch.

    Returns:
        Dict with ``loss``, ``acc``, ``lr``, ``grad_norm`` keys.
    """
    raise NotImplementedError


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-driven training pipeline.

    Pipeline:
        1. Set seed
        2. Build dataloaders (patient-level split)
        3. Build model + optimizer + scheduler + EMA
        4. Train + validate (with early stopping)
        5. Save best checkpoint to ``outputs/models/{model.name}_best.pth``
        6. Final test evaluation
        7. Log all metrics to W&B
    """
    raise NotImplementedError("Implement in this file.")


if __name__ == "__main__":
    main()
