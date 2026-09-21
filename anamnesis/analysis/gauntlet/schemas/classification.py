"""Section 2 schemas: mode discrimination, per block.

The battery each block is put through — five-way random forest with its confusion
matrix, topic-held-out CV, a linear probe, every pairwise binary, and the
four-way set with analogical removed — plus the multi-seed stability and
label-permutation null that only the key composite blocks pay for, and the
length-only baseline that says how much of it generation length could explain.

The wire format stores block keys at the top level beside ``length_only``, so the
validator gathers them into ``by_block`` and the serializer flattens them back.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_serializer, model_validator

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ClassifierAccuracyResult(BaseModel):
    """Basic classifier CV output used by ``linear_probe`` and each pair in
    ``pairwise_binary``: accuracy + per-fold accuracies."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]


class ClassifierWithConfusionResult(BaseModel):
    """RF CV output with confusion matrix.

    Used by ``rf_5way`` (always succeeds) and ``rf_4way_no_analogical``
    (returns just ``{"error": ...}`` when analogical is absent).
    """

    model_config = _FORBID

    accuracy: float | None = None
    fold_accuracies: list[float] | None = None
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] | None = None
    error: str | None = None


class TopicHeldoutResult(BaseModel):
    """GroupKFold-by-topic CV output."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]
    n_groups: int


class CVStabilityResult(BaseModel):
    """Multi-seed RF stability distribution (key blocks only)."""

    model_config = _FORBID

    mean: float
    median: float
    std: float
    ci_lo: float
    ci_hi: float
    min: float
    max: float
    all_accuracies: list[float]
    n_seeds: int


class PermutationTestResult(BaseModel):
    """Label-permutation null distribution (key blocks only)."""

    model_config = _FORBID

    observed_accuracy: float
    p_value: float
    null_mean: float
    null_std: float
    null_max: float
    null_p95: float
    null_p99: float
    n_permutations: int
    # BH-FDR q-value across the per-group permutation family for this run
    # (2026-07-11 sweep). None when the family has a single member.
    q_value: float | None = None


class PerModeLengthStats(BaseModel):
    """Generation-length stats for the length-only confound baseline."""

    model_config = _FORBID

    mean: float
    std: float
    min: float
    max: float


class LengthOnlyResult(BaseModel):
    """Length-only confound baseline (Section 2).

    Two shapes exist on disk:
    - Error path (no length metadata): ``{"accuracy": null, "error": "..."}``.
    - Success path: adds ``fold_accuracies``, ``confusion_matrix``, ``labels``,
      ``per_mode_lengths``.

    The custom serializer keeps these wire shapes intact. ``accuracy`` is
    explicitly emitted even when ``None`` to preserve the error-stub shape.
    """

    model_config = _FORBID

    accuracy: float | None = None
    fold_accuracies: list[float] | None = None
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] | None = None
    per_mode_lengths: dict[str, PerModeLengthStats] | None = None
    error: str | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        if self.error is not None:
            return {"accuracy": self.accuracy, "error": self.error}
        out: dict[str, Any] = {"accuracy": self.accuracy}
        if self.fold_accuracies is not None:
            out["fold_accuracies"] = self.fold_accuracies
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix
        if self.labels is not None:
            out["labels"] = self.labels
        if self.per_mode_lengths is not None:
            out["per_mode_lengths"] = {
                k: v.model_dump(mode="json") for k, v in self.per_mode_lengths.items()
            }
        return out


class BlockClassificationResult(BaseModel):
    """Per-block classification battery.

    ``cv_stability`` / ``permutation_test`` are populated only for the union
    blocks worth the sweep; other blocks omit them on the wire.
    """

    model_config = _FORBID

    rf_5way: ClassifierWithConfusionResult
    topic_heldout: TopicHeldoutResult
    linear_probe: ClassifierAccuracyResult
    pairwise_binary: dict[str, ClassifierAccuracyResult]
    rf_4way_no_analogical: ClassifierWithConfusionResult
    cv_stability: CVStabilityResult | None = None
    permutation_test: PermutationTestResult | None = None


class ClassificationResult(BaseModel):
    """Section 2 result: per-block classification + length-only confound.

    The wire format stores one key per block at the top level, beside
    ``length_only``, and which blocks those are depends on the run. The
    validator gathers them into ``by_block`` so a consumer can use attribute
    access (``result.by_block[block].rf_5way.accuracy``), and the serializer
    spreads them back out. `anamnesis/analysis/gauntlet/signature_io.py` is
    where the block labels come from.
    """

    model_config = _FORBID

    by_block: dict[str, BlockClassificationResult]
    length_only: LengthOnlyResult | None = None

    @model_validator(mode="before")
    @classmethod
    def _reshape_in(cls, data: Any) -> Any:
        if isinstance(data, dict) and "by_block" not in data:
            flat = dict(data)
            length_only = flat.pop("length_only", None)
            return {"by_block": flat, "length_only": length_only}
        return data

    @model_serializer(mode="plain")
    def _reshape_out(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            block: block_result.model_dump(mode="json", exclude_none=True)
            for block, block_result in self.by_block.items()
        }
        if self.length_only is not None:
            out["length_only"] = self.length_only.model_dump(mode="json")
        return out
