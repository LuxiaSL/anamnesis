"""Section 3 schemas: the readout over the stored feature blocks.

Combinations and leave-one-out drops over the blocks, the resulting ranking, the
top feature importances, the static-versus-dynamic split, and per-topic effect
sizes. ``cache_beats_attention_beats_norms`` is a named prediction being scored,
not a summary of the table above it.

The field names here are the Python side; several banked files carry an older
spelling for the same field, and `anamnesis/analysis/gauntlet/schemas/compat.py`
is what maps one onto the other on the way in.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_serializer

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class PairwiseBlockCombo(BaseModel):
    """Accuracy of two baseline blocks concatenated."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    individual_max: float
    gain_over_best_individual: float


class TripleBlockCombo(BaseModel):
    """Accuracy of three baseline blocks concatenated."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    best_pairwise_subset: float
    gain_over_best_pair: float


class CrossGroupAblation(BaseModel):
    """Accuracy of a baseline composite + a single engineered family."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    baseline_accuracy: float
    engineered_alone: float
    gain_over_baseline: float


class LeaveOneOutEntry(BaseModel):
    """Per-block leave-one-out accuracy + cost of removal."""

    model_config = _FORBID

    accuracy_without: float
    cost_of_removal: float


class BlockRankingEntry(BaseModel):
    """One row of the block-ranking table (block name + its standalone accuracy)."""

    model_config = _FORBID

    block: str
    accuracy: float


class FeatureImportanceEntry(BaseModel):
    """One row of a feature-importance table (RF or LR top-N)."""

    model_config = _FORBID

    name: str
    importance: float


class StdVsMeanResult(BaseModel):
    """*_std vs *_mean feature RF accuracy split.

    Success path sets ``n_std_features``, ``n_mean_features`` plus the
    two accuracies + comparator. Error path sets ``error``, ``n_names``,
    ``n_features``. The two shapes have disjoint keys.
    """

    model_config = _FORBID

    n_std_features: int | None = None
    n_mean_features: int | None = None
    std_accuracy: float | None = None
    mean_accuracy: float | None = None
    std_beats_mean: bool | None = None
    n_names: int | None = None
    n_features: int | None = None
    error: str | None = None


class PerTopicEffectSize(BaseModel):
    """Per-topic Cohen's d (success) or an error stub (too few samples /
    insufficient pairs). Success and error keys do not overlap.
    """

    model_config = _FORBID

    cohens_d: float | None = None
    mean_within: float | None = None
    mean_between: float | None = None
    n_within_pairs: int | None = None
    n_between_pairs: int | None = None
    n_samples: int | None = None
    n: int | None = None
    error: str | None = None


class CohensDPerTopicResult(BaseModel):
    """Cohen's d summary across topics.

    When no topics produce a successful d, the summary fields are ``null``
    on the wire (not absent). The custom serializer preserves this shape.
    """

    model_config = _FORBID

    per_topic: dict[str, PerTopicEffectSize]
    mean_d: float | None
    median_d: float | None
    std_d: float | None
    min_d: float | None
    max_d: float | None
    all_positive: bool | None
    n_topics: int

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {
            "per_topic": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.per_topic.items()
            },
            "mean_d": self.mean_d,
            "median_d": self.median_d,
            "std_d": self.std_d,
            "min_d": self.min_d,
            "max_d": self.max_d,
            "all_positive": self.all_positive,
            "n_topics": self.n_topics,
        }


class LegacyBinReadoutResult(BaseModel):
    """Section 3 result: block ablation + feature importance.

    Several fields are present only for v2 runs (``cross_group_ablation``,
    ``top_features_rf``, etc.). ``top_features_rf_combined`` is a legacy
    key from pre-``feature_importance_composite`` snapshots and is
    preserved for round-trip of older baseline runs.
    """

    model_config = _FORBID

    per_block_accuracy: dict[str, float]
    pairwise_block_combinations: dict[str, PairwiseBlockCombo]
    triple_block_combinations: dict[str, TripleBlockCombo] | None = None
    cross_group_ablation: dict[str, CrossGroupAblation] | None = None
    cross_group_baseline: str | None = None
    leave_one_block_out: dict[str, LeaveOneOutEntry]
    leave_one_out_baseline_accuracy: float | None = None
    block_ranking: list[BlockRankingEntry]
    cache_beats_attention_beats_norms: bool
    top_features_rf: list[FeatureImportanceEntry] | None = None
    top_features_lr: list[FeatureImportanceEntry] | None = None
    feature_importance_composite: str | None = None
    top_features_rf_attention_and_cache: list[FeatureImportanceEntry]
    top_features_lr_attention_and_cache: list[FeatureImportanceEntry]
    top_features_rf_combined: list[FeatureImportanceEntry] | None = Field(
        default=None,
        description="Legacy top-features key from pre-v2 baseline snapshots.",
    )
    block_contribution_ratio: dict[str, float]
    std_vs_mean: StdVsMeanResult
    cohens_d_per_topic: CohensDPerTopicResult
