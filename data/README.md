# Data Directory

> **Not versioned** — see ``.gitignore``.

## Phase 1 — CRC Histology

### Datasets

- Train/Val: **NCT-CRC-HE-100K** (folder-per-class)
- Test (external): **CRC-VAL-HE-7K** (folder-per-class)

Expected layout:

```
data/
└── NCT-CRC-HE-100K/
	├── ADI/
	├── BACK/
	├── DEB/
	├── LYM/
	├── MUC/
	├── MUS/
	├── NORM/
	├── STR/
	└── TUM/
└── CRC-VAL-HE-7K/
	├── ADI/
	├── BACK/
	└── ...
```

### Splits

- **Train/Val**: stratified split on NCT-CRC-HE-100K (see ``cfg.dataset.splits.val_ratio``)
- **Test**: all images from CRC-VAL-HE-7K

Note: there are no ground-truth segmentation masks in these datasets, so localization metrics
(Pointing Game, IoU) are disabled by default in ``config/xai/evaluation.yaml``.

## Phase 2 (DEFERRED — not for this repository)

BraTS 2023, CheXpert, CAMELYON17 are **out of scope** for Phase 1.
