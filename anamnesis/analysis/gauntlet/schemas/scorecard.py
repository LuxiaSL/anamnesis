"""Section 10 schemas: the pre-registered predictions, scored.

Each prediction with the value that decided it and a confirmed/partial/wrong
outcome, plus the tally. This section consumes other sections rather than the
data, which is why the orchestrator reruns it every time.
"""

from __future__ import annotations

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ScorecardPrediction(BaseModel):
    """One row of the pre-registered prediction scorecard.

    Each of the 9 predictions carries different evidence fields, so
    the per-prediction extras (values, expected_order, mean_t1_t2,
    combined_accuracy, all_pairwise, etc.) are all Optional. Canonical
    runs populate the evidence fields relevant to each prediction; the
    serializer drops unset optionals.
    """

    model_config = _FORBID

    # Common fields (always set)
    prediction: str
    confidence: str
    importance: str
    outcome: str

    # Per-prediction evidence fields (at most a subset per row)
    metric: str | None = None
    surprise_threshold: str | None = None  # P1
    detail: str | None = None  # P2
    values: dict[str, float] | None = None  # P3
    expected_order: list[str] | None = None  # P4
    actual_order: list[str] | None = None  # P4
    mode_ids: dict[str, float] | None = None  # P4
    t3_id: float | None = None  # P5
    mean_t1_t2: float | None = None  # P5
    tier_inversion_holds: bool | None = None  # P6
    per_tier_accuracy: dict[str, float] | None = None  # P6
    removal_costs: dict[str, float | None] | None = None  # P6
    t2t25_accuracy: float | None = None  # P7
    combined_accuracy: float | None = None  # P7
    mean_hard_pair_accuracy: float | None = None  # P8
    mean_easy_pair_accuracy: float | None = None  # P8
    all_pairwise: dict[str, float] | None = None  # P8
    delta_rel_8b: float | None = None  # P9
    delta_rel_3b: float | None = None  # P9


class ScorecardSummary(BaseModel):
    """Counts of outcome categories across the 9 predictions."""

    model_config = _FORBID

    confirmed: int
    partial: int
    wrong: int
    noted: int
    insufficient: int
    total: int


class ScorecardResult(BaseModel):
    """Section 10 result: pre-registered 8B prediction evaluation."""

    model_config = _FORBID

    predictions: list[ScorecardPrediction]
    summary: ScorecardSummary
