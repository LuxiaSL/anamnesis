"""Map-wide statistics: BH-FDR over the visibility-map grid, permutation helpers,
and the (n, M, law, floor-type) stamp every emitted number must carry (§6b).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.analysis.battery.manifest import FloorType


class ResultStamp(BaseModel):
    """Provenance stamp — no number ships without one (prereg §1 statistics block)."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(description="Samples/pairs behind the number")
    M: str = Field(description="Model the number is about (per-model claims only)")
    law: str = Field(description="Law reference, e.g. 'stage0-3b n_min=24 @alpha=0.0025 k=2'")
    floor_type: FloorType


class StampedValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float
    stamp: ResultStamp
    raw_artifact: Optional[str] = Field(
        default=None, description="Path to the raw artifact backing this number",
    )


def bh_fdr(pvals: Sequence[float], alpha: float = 0.05) -> tuple[NDArray, NDArray]:
    """Benjamini–Hochberg over the map grid. Returns (reject_mask, adjusted_p)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adjusted = np.empty(m)
    adjusted[order] = adj
    return adjusted <= alpha, adjusted


def permutation_pvalue(
    observed: float,
    null_samples: NDArray,
    alternative: str = "greater",
) -> float:
    """Exact-style permutation p with the +1 correction."""
    null_samples = np.asarray(null_samples, dtype=np.float64)
    n = len(null_samples)
    if n == 0:
        raise ValueError("empty null distribution")
    if alternative == "greater":
        hits = int((null_samples >= observed).sum())
    elif alternative == "less":
        hits = int((null_samples <= observed).sum())
    elif alternative == "two-sided":
        center = float(np.median(null_samples))
        hits = int((np.abs(null_samples - center) >= abs(observed - center)).sum())
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return (hits + 1) / (n + 1)
