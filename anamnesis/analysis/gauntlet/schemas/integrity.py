"""Section 1 schemas: data integrity and descriptive statistics.

What the gauntlet checked before it measured anything: sample balance, tier
dimensions, NaN/Inf counts, per-tier variance, generation lengths and feature
value ranges. ``all_clean`` is the field the orchestrator reads to shout.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class NanInfCount(BaseModel):
    """NaN/Inf counts for a single tier. ``error`` populated only when the
    tier could not be loaded (in which case ``nan``/``inf`` are sentinel -1).
    """

    model_config = _FORBID

    nan: int
    inf: int
    error: str | None = None


class TierVarianceReport(BaseModel):
    """Per-tier variance diagnostics: constant/near-constant feature counts."""

    model_config = _FORBID

    n_features: int
    n_constant: int
    n_near_constant: int


class LengthByModeStats(BaseModel):
    """Generation-length stats for a single mode (includes median)."""

    model_config = _FORBID

    mean: float
    std: float
    min: int
    max: int
    median: float


class LengthOverallStats(BaseModel):
    """Overall generation-length stats (no median — matches legacy shape)."""

    model_config = _FORBID

    mean: float
    std: float
    min: int
    max: int


class TierValueRange(BaseModel):
    """Per-tier feature-value range summary."""

    model_config = _FORBID

    global_mean: float
    global_std: float
    global_min: float
    global_max: float
    feature_mean_range: list[float] = Field(
        description="Two-element list [min, max] of per-feature means.",
    )


class IntegrityResult(BaseModel):
    """Section 1 result: data integrity + descriptive statistics."""

    model_config = _FORBID

    n_samples: int
    n_modes: int
    n_topics: int
    modes: list[str]
    topics: list[str]
    samples_per_mode: dict[str, int]
    samples_per_topic: dict[str, int]
    balanced: bool
    tier_dims: dict[str, int]
    total_features: int
    nan_inf: dict[str, NanInfCount]
    all_clean: bool
    variance_report: dict[str, TierVarianceReport]
    length_by_mode: dict[str, LengthByModeStats] | None = None
    length_overall: LengthOverallStats | None = None
    value_ranges: dict[str, TierValueRange]
