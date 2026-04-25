"""Seed management for reproducibility.

Critical for medical imaging research — every experiment must be
reproducible bit-for-bit (when ``deterministic=True``).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix all random sources for reproducibility.

    Args:
        seed: Global seed value.
        deterministic: If True, set ``cudnn.deterministic = True``
            (slower but bit-exact reproducible). If False, allow
            ``cudnn.benchmark = True`` for speed.

    Notes:
        Sets seeds for: Python ``random``, ``numpy``, ``torch``,
        ``torch.cuda``, and the ``PYTHONHASHSEED`` env var.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Also set deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
        # CUBLAS workspace config required for full determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Set seed per DataLoader worker.

    Pass to ``DataLoader(worker_init_fn=worker_init_fn)``.

    Each worker gets a unique seed derived from ``torch.initial_seed()``
    plus the worker ID — required because multiprocessing forks before
    setting torch seeds.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
