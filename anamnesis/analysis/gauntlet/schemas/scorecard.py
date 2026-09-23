"""Section 10 schemas: the standing expectations, scored.

Each expectation with the value that decided it and a confirmed/partial/wrong
outcome, plus the tally. This section consumes other sections rather than the
data, which is why the orchestrator reruns it every time.

Two fields exist so that a verdict cannot travel without what bounds it.
``ScorecardPrediction.caveat`` carries the limit on one row's reading, and
``ScorecardResult.corpus_caveat`` carries what the corpus under the pass is and is
not. Both are written into the results file and both are printed beside the
verdict, because a limit kept in a docstring is a limit the reader of the output
never meets.
"""

from __future__ import annotations

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ScorecardPrediction(BaseModel):
    """One row of the prediction scorecard: an expectation, its threshold, its verdict.

    Each of the 9 predictions carries different evidence fields, so
    the per-prediction extras (values, expected_order, mean_norms_and_attention_id,
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

    # Why the row could not be scored, when ``outcome`` is INSUFFICIENT_DATA:
    # which upstream section did not run, and the reason it carried. A prediction
    # whose evidence is absent is unscored, never scored WRONG — a missing
    # measurement is not a failed one.
    unscorable_because: str | None = None

    # The limit on what this row's verdict supports, when the reading it scores
    # supports less than its own wording would suggest. It travels with the row so
    # that the outcome and the limit reach a reader together, whether the reader is
    # looking at the printed summary or at the results file.
    caveat: str | None = None

    # Per-prediction evidence fields (at most a subset per row)
    metric: str | None = None
    surprise_threshold: str | None = None  # P1
    detail: str | None = None  # P2
    values: dict[str, float] | None = None  # P3
    expected_order: list[str] | None = None  # P4
    actual_order: list[str] | None = None  # P4
    mode_ids: dict[str, float] | None = None  # P4
    residual_pca_id: float | None = None  # P5
    mean_norms_and_attention_id: float | None = None  # P5
    cache_beats_attention_beats_norms: bool | None = None  # P6
    per_block_accuracy: dict[str, float] | None = None  # P6
    removal_costs: dict[str, float | None] | None = None  # P6
    attention_and_cache_accuracy: float | None = None  # P7
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
    """Section 10 result: the nine standing expectations, scored on this corpus.

    Every row is always present, so the scorecard is a complete answer about all
    nine predictions whatever the upstream sections managed: a row whose evidence
    is absent reads INSUFFICIENT_DATA and says which section it was waiting on.
    ``error`` is set only when *no* row could be scored, which is the case where
    the section produced nothing and the pass is short by it.
    """

    model_config = _FORBID

    predictions: list[ScorecardPrediction]
    summary: ScorecardSummary
    error: str | None = None

    # What corpus the nine expectations were fixed against, and what the corpus
    # under this pass is. A verdict describes the agreement between the two, so a
    # pass over some other corpus states that here rather than letting a reader take
    # CONFIRMED for a finding about their own model.
    corpus_caveat: str | None = None
