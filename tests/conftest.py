"""Suite-wide settings.

The suite runs four worker processes by default (pytest-xdist), and CI runs one
per core (``-n auto``). Each worker keeps its numerical libraries to one thread,
so the workers share the machine instead of every worker's BLAS and torch pools
claiming all of it. Tests marked ``slow`` are left to CI by local runs that pass
``-m "not slow"``.
"""

from __future__ import annotations

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")


def pytest_configure(config) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        try:
            import torch
        except ImportError:
            return
        torch.set_num_threads(1)
