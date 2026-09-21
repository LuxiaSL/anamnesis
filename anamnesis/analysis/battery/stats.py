"""Map-wide statistics: BH-FDR over the visibility-map grid, permutation helpers,
and the (n, M, law, floor-type) stamp every emitted number must carry (§6b).

This module is the one home for the two family-level statistics the instrument
reports, so that two analyses running the same test on the same input cannot
disagree about the number. ``anamnesis.analysis.gauntlet.classification`` reads
both from here rather than carrying its own copies: the battery is the metrology
layer and nothing in it imports the gauntlet, so the edge runs one way.

The conventions both callers therefore inherit:

* a permutation p-value carries the add-one correction, ``(hits + 1) / (n + 1)``,
  and is never zero;
* BH-FDR is the step-up adjustment, available over a sequence (``bh_fdr``) or
  over a family addressed by name (``bh_fdr_by_key``).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

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
    """Benjamini–Hochberg step-up over one family of tests.

    Parameters
    ----------
    pvals : Sequence[float]
        The family's p-values, in whatever order the caller holds them. The
        adjustment depends on the whole family, so a caller that tests a grid and
        reports one cell must still pass every cell it tested.
    alpha : float
        The false-discovery rate the reject mask is drawn at.

    Returns
    -------
    (reject_mask, adjusted_p) : tuple[NDArray, NDArray]
        Both in the input order, both length ``len(pvals)``. ``adjusted_p`` is the
        BH q-value: monotone in the sorted p-values, clipped into [0, 1], and never
        below the raw p-value. ``reject_mask`` is ``adjusted_p <= alpha``. An empty
        family gives two empty arrays rather than an error, because a grid with no
        testable cells is a legitimate outcome.
    """
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


def bh_fdr_by_key(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, float]:
    """BH q-values for a family whose members are addressed by name.

    The same adjustment as ``bh_fdr``, for the caller that holds its family as a
    mapping — a per-block permutation family, say — and wants the q-values back
    under the same keys. Implemented over ``bh_fdr`` so there is one step-up in
    the package rather than two that can drift.

    Parameters
    ----------
    pvals : Mapping[str, float]
        Test name → p-value. Every member of the family, for the reason stated on
        ``bh_fdr``.
    alpha : float
        Accepted so the two entry points take the same arguments; the reject
        decision is ``q <= alpha``, which the caller can make from the q-values.

    Returns
    -------
    dict[str, float]
        The input keys, each mapped to its q-value. An empty mapping in gives an
        empty dict out.
    """
    keys = list(pvals)
    _reject, adjusted = bh_fdr([pvals[key] for key in keys], alpha=alpha)
    return {key: float(q) for key, q in zip(keys, adjusted)}


def permutation_pvalue(
    observed: float,
    null_samples: NDArray,
    alternative: str = "greater",
) -> float:
    """Permutation p-value with the add-one correction.

    The statistic is ``(hits + 1) / (n + 1)``: the observed value is counted as one
    of the arrangements its own null could have produced. Two consequences the
    caller is relying on. The p-value is never zero — zero would assert that no
    relabelling whatever can match the observation, which ``n`` draws cannot
    establish — and ``n`` draws resolve nothing smaller than ``1 / (n + 1)``, so a
    claim below that resolution needs more permutations rather than a smaller
    number. Dividing the hit count by ``n`` instead, and taking the larger of that
    and ``1 / (n + 1)``, agrees only when no permutation reaches the observed value
    and is anti-conservative elsewhere: at one hit in 500 it gives 0.002 where this
    gives 0.003992.

    Parameters
    ----------
    observed : float
        The statistic computed on the real labels.
    null_samples : NDArray
        One statistic per permutation draw; ``n`` is its length.
    alternative : str
        ``"greater"`` (the default) counts null draws at or above ``observed``,
        ``"less"`` counts those at or below it, and ``"two-sided"`` centres the null
        on its median and counts draws whose absolute deviation from that centre is
        at least the observed one. A draw exactly equal to the observed value counts
        as a hit in all three, which is what keeps the test conservative when the
        statistic is discrete — a cross-validated accuracy over a fixed test set
        takes only multiples of one over its size, so exact ties are common.

    Returns
    -------
    float
        A p-value in ``(0, 1]``.

    Raises
    ------
    ValueError
        If the null distribution is empty, or ``alternative`` is not one of the
        three named above.
    """
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


def permutation_resolution(n_permutations: int) -> float:
    """The smallest p-value ``n_permutations`` draws can report: ``1 / (n + 1)``.

    A caller pre-registering a threshold needs this before it runs anything: a test
    planned at alpha = 0.001 is unfalsifiable on 500 permutations, whose finest
    resolution is 0.001996.
    """
    if n_permutations < 1:
        raise ValueError("a permutation test needs at least one draw")
    return 1.0 / (n_permutations + 1)
