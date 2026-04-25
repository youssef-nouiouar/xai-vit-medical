"""Clinical validation utilities — protocol for dermatologist review.

For Phase 1 final paper:
    - 2 dermatologists × 50 cases × N XAI methods (double-blind).
    - Rating scale 1–5 on three criteria:
        1. Clinical coherence (does saliency align with ABCDE criteria?)
        2. Usefulness for diagnosis
        3. Trust calibration (does the explanation help calibrate confidence?)
    - Inter-rater agreement: Fleiss' kappa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_evaluation_package(
    saliency_maps: dict[str, Any],
    images: list[Any],
    output_dir: Path,
    n_cases: int = 50,
    blind: bool = True,
) -> Path:
    """Prepare a blind evaluation package for dermatologists.

    Generates:
        - Anonymized image filenames (e.g., case_001.png, case_002.png, …).
        - A spreadsheet with empty rating columns.
        - A protocol PDF describing the rating criteria.

    Args:
        saliency_maps: ``{method_name: list of saliency arrays}``.
        images: Original images.
        output_dir: Where to put the package.
        n_cases: Number of cases per method.
        blind: If True, randomize method ordering and hide names.

    Returns:
        Path to the prepared package directory.
    """
    raise NotImplementedError("Implement in this file.")


def aggregate_ratings(ratings_csv: Path | str) -> dict[str, Any]:
    """Aggregate dermatologist ratings.

    Computes per-method:
        - mean ± std rating per criterion
        - Fleiss' kappa (inter-rater agreement)
        - Wilcoxon signed-rank test between methods

    Returns:
        Statistics dictionary suitable for paper tables.
    """
    raise NotImplementedError("Implement in this file.")
