"""Shared utilities for the unified analysis runner."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    EVERYTHING,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
)


def standardize(X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Z-score standardize features. Constant features get std=1."""
    X = X.astype(np.float64)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X - X.mean(axis=0)) / std


def remove_constant(X: NDArray[np.float64], threshold: float = 1e-12) -> NDArray[np.float64]:
    """Remove features with near-zero variance."""
    variance = X.var(axis=0)
    mask = variance > threshold
    return X[:, mask]


def clean_for_json(obj: object) -> object:
    """Recursively make objects JSON-serializable.

    Handles: NaN/Inf floats (→ None), numpy scalars/arrays, dicts, lists,
    and pydantic BaseModel instances (dumped with ``exclude_none=True`` so
    Optional error fields vanish when unset).
    """
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, BaseModel):
        return clean_for_json(obj.model_dump(mode="json", exclude_none=True))
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


@contextmanager
def timer(label: str = "") -> Generator[dict[str, float], None, None]:
    """Context manager that tracks elapsed time."""
    result: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - t0
        if label:
            print(f"  [{label}] {result['elapsed']:.1f}s")


# Default block lists — used when analysis modules don't get dynamic lists.
# For v2 data, use get_available_blocks() to discover what's actually present.
ALL_BLOCKS = [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA, ATTENTION_AND_CACHE, ALL_CORE]
KEY_BLOCKS = [ATTENTION_AND_CACHE, ALL_CORE]


def get_available_blocks(data: object) -> tuple[list[str], list[str]]:
    """Discover which blocks and groups are available in loaded data.

    Parameters
    ----------
    data : AnalysisData or Run4Data
        Loaded data object with block_features and group_features.

    Returns
    -------
    all_blocks : list[str]
        All individual blocks + groups that are present.
    key_blocks : list[str]
        Key composite groups for expensive analyses.
    """
    run4 = getattr(data, "run4", data)
    individual = list(run4.block_features.keys())
    groups = list(run4.group_features.keys())

    all_blocks = individual + groups
    # Key blocks: composites that include multiple families
    key_blocks = [g for g in groups if g in {
        ATTENTION_AND_CACHE, ALL_CORE, ALL_FAMILIES, EVERYTHING,
        ATTENTION_AND_CACHE_WITH_FAMILIES,
    }]

    return all_blocks, key_blocks
