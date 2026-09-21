"""Section 5 schemas: cross-condition generalization performance.

Train a separating hyperplane on some conditions, test it on held-out ones: the
per-dichotomy accuracies, and the summary whose ``all_perfect`` flag is what a
reader checks before believing the geometry is that clean.
"""

from __future__ import annotations

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class CCGPDichotomy(BaseModel):
    """Single binary dichotomy evaluation within a CCGP variant."""

    model_config = _FORBID

    group_a: list[str]
    group_b: list[str]
    mean_accuracy: float
    decodable: bool


class CCGPVariant(BaseModel):
    """One CCGP variant (classifier × seed × fold-count × optional block)."""

    model_config = _FORBID

    multiclass_mean: float
    multiclass_fold_accs: list[float]
    per_mode_recall: dict[str, float]
    n_decodable: int
    n_dichotomies: int
    ccgp_score: float
    dichotomies: list[CCGPDichotomy]


class CCGPSummary(BaseModel):
    """CCGP score summary across all variants."""

    model_config = _FORBID

    min_ccgp: float
    max_ccgp: float
    all_perfect: bool


class CCGPResult(BaseModel):
    """Section 5 result: CCGP across classifier/seed/fold variants.

    ``variants`` and ``summary`` are optional so a section that could not run —
    the union it reads is not in this corpus — round-trips as a stub carrying its
    reason. ``refused_variants`` names the variants the corpus could not support
    and why, keyed as ``variants`` is: a variant asking for more held-out folds
    than there are topics is a refusal, not a measurement, and it is reported
    beside the ones that ran rather than as a number that was never taken.
    """

    model_config = _FORBID

    variants: dict[str, CCGPVariant] | None = None
    summary: CCGPSummary | None = None
    refused_variants: dict[str, str] | None = None
    error: str | None = None
