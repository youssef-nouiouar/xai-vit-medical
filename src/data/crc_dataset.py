"""CRC Histology datasets (NCT-CRC-HE-100K + CRC-VAL-HE-7K).

This loader supports the folder-per-class structure of the colorectal
histology datasets:

Train/Val:
    data/NCT-CRC-HE-100K/{ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM}/

Test (external):
    data/CRC-VAL-HE-7K/{ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM}/

The dataset contains 9 classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


DEFAULT_CRC_CLASSES: list[str] = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
]


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class CRCSplits:
    """Container for dataset roots and split parameters."""

    trainval_root: Path
    test_root: Path
    classes: tuple[str, ...] = tuple(DEFAULT_CRC_CLASSES)
    val_ratio: float = 0.1
    random_state: int = 42


def build_crc_transforms(image_size: int = 224, split: str = "train") -> A.Compose:
    """Default transforms (ImageNet normalization).

    Notes:
        Albumentations' Normalize uses ``max_pixel_value=255`` by default, so it
        scales uint8 pixels by /255 before standardizing by mean/std.
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.5,
                ),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ]
        )

    # val / test
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )


class CRCHistologyDataset(Dataset):
    """Folder-per-class dataset for CRC histology.

    Args:
        split: ``"train"`` | ``"val"`` | ``"test"``.
        splits: Paths + split params.
        image_size: Used only for default transform construction.
        transform: Albumentations transform. If None, uses defaults.

    Returns:
        By default: ``(image_tensor, label_index)``.
        If ``return_id=True``: ``(image_tensor, label_index, image_id)``.
    """

    def __init__(
        self,
        *,
        split: str,
        splits: CRCSplits,
        image_size: int = 224,
        transform: A.Compose | None = None,
        return_id: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self.splits = splits
        self.classes = list(splits.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if transform is None:
            transform = build_crc_transforms(image_size=image_size, split=split)
        self.transform = transform
        self.return_id = return_id

        if split == "test":
            root = splits.test_root
            samples = _scan_folder_dataset(root, self.classes)
        else:
            root = splits.trainval_root
            all_samples = _scan_folder_dataset(root, self.classes)
            paths = np.array([p for (p, _) in all_samples], dtype=object)
            labels = np.array([y for (_, y) in all_samples], dtype=np.int64)

            train_idx, val_idx = train_test_split(
                np.arange(len(all_samples)),
                test_size=splits.val_ratio,
                random_state=splits.random_state,
                stratify=labels,
            )
            idx = train_idx if split == "train" else val_idx
            samples = [(Path(paths[i]), int(labels[i])) for i in idx]

        self._samples: list[tuple[Path, int]] = samples
        self.labels = np.array([y for _, y in self._samples], dtype=np.int64)

        logger.info(
            f"CRC dataset split={split}: {len(self._samples)} images "
            f"(root={root})"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        path, label = self._samples[idx]

        # PIL -> numpy RGB
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        image_t = self.transform(image=image_np)["image"]

        image_id = str(path.name)
        if self.return_id:
            return image_t, int(label), image_id
        return image_t, int(label)


def _scan_folder_dataset(root: Path, classes: Iterable[str]) -> list[tuple[Path, int]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: list[tuple[Path, int]] = []
    classes_list = list(classes)
    for label, class_name in enumerate(classes_list):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir} (under {root})"
            )

        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                samples.append((p, label))

    if not samples:
        raise RuntimeError(f"No images found under: {root}")

    return samples


def build_crc_dataloaders(
    *,
    trainval_root: str | Path,
    test_root: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
    random_state: int = 42,
    classes: list[str] | None = None,
) -> dict[str, DataLoader]:
    """Convenience to build train/val/test DataLoaders."""

    splits = CRCSplits(
        trainval_root=Path(trainval_root),
        test_root=Path(test_root),
        classes=tuple(classes or DEFAULT_CRC_CLASSES),
        val_ratio=val_ratio,
        random_state=random_state,
    )

    train_ds = CRCHistologyDataset(split="train", splits=splits, image_size=image_size)
    val_ds = CRCHistologyDataset(split="val", splits=splits, image_size=image_size)
    test_ds = CRCHistologyDataset(split="test", splits=splits, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
