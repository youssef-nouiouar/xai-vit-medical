"""ISIC 2019 dataset loader.

Implements:
    - :class:`ISIC2019Dataset` — PyTorch ``Dataset``
    - :func:`build_dataloaders` — train/val/test ``DataLoader`` builders
    - :func:`make_patient_level_splits` — group K-fold by patient

Critical: splits MUST be patient-level (no leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset


class ISIC2019Dataset(Dataset):
    """ISIC 2019 dermoscopy dataset.

    Args:
        root: Root data directory containing ``images/`` and ``groundtruth.csv``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        transforms: Albumentations transform pipeline.
        return_metadata: If True, also return patient_id, lesion_id, etc.

    Returns:
        Dict with keys ``image`` (Tensor), ``label`` (int), and
        optionally ``metadata`` (dict).
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transforms: Any | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Implement in this file.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError


def make_patient_level_splits(
    metadata_csv: Path | str,
    n_folds: int = 5,
    test_ratio: float = 0.15,
    random_state: int = 42,
    output_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """Generate stratified group K-fold splits by patient.

    Args:
        metadata_csv: Path to ISIC metadata CSV.
        n_folds: Number of CV folds.
        test_ratio: Held-out test set proportion (patient-level).
        random_state: Seed.
        output_dir: If given, save splits as CSVs.

    Returns:
        Dict mapping ``"fold_{i}_train"``, ``"fold_{i}_val"``, ``"test"``
        to lists of image IDs.
    """
    raise NotImplementedError("Implement in this file.")


def build_dataloaders(
    cfg: Any,
    fold: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from a Hydra config.

    Args:
        cfg: Full Hydra config (uses ``cfg.dataset``).
        fold: Which CV fold to use as validation.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
    """
    raise NotImplementedError("Implement in this file.")
