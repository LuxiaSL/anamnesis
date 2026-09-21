"""Section 8 schemas: learned projections and their ablations.

A supervised contrastive encoder trained per tier, with the kNN readout on its
embedding, the tier ablation and pairwise grids, the super-additivity test, and
the capacity sweep and linear baselines that say whether a nonlinear encoder
was needed. ``_ON_DISK_RENAMES`` is the frozen wire vocabulary: banked JSON
carries the older key spelling and is read through the map.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, model_serializer, model_validator

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ContrastiveTierResult(BaseModel):
    """Top-level per-tier contrastive result (T2+T2.5 / combined)."""

    model_config = _FORBID

    knn_accuracy_mean: float
    knn_accuracy_std: float
    knn_fold_accs: list[float]
    silhouette_mean: float | None


class ContrastiveAblationEntry(BaseModel):
    """Single tier entry inside ``contrastive.tier_ablation.individual``
    or the combined/T2+T2.5 sub-keys of ``tier_ablation``."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_features: int


class ContrastivePairwiseEntry(BaseModel):
    """Pairwise contrastive ablation entry (adds best-individual baseline)."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_features: int
    best_individual_knn: float
    gain_over_best_individual: float


class ContrastiveSuperAdditivity(BaseModel):
    """T2+T2.5 super-additivity summary.

    Keys like ``T2.5_alone`` and ``T2+T2.5_pair`` aren't legal Python
    identifiers, so the model exposes them as ``T2_5_alone`` etc. A
    custom validator + serializer translates to/from the on-disk names.
    """

    model_config = _FORBID

    T2_alone: float
    T2_5_alone: float
    T2_T2_5_pair: float
    best_individual: float
    gain: float
    combined_knn: float
    T2_T2_5_beats_combined: bool

    _ON_DISK_RENAMES: ClassVar[dict[str, str]] = {
        "T2.5_alone": "T2_5_alone",
        "T2+T2.5_pair": "T2_T2_5_pair",
        "T2+T2.5_beats_combined": "T2_T2_5_beats_combined",
    }

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {cls._ON_DISK_RENAMES.get(k, k): v for k, v in data.items()}
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        return {
            "T2_alone": self.T2_alone,
            "T2.5_alone": self.T2_5_alone,
            "T2+T2.5_pair": self.T2_T2_5_pair,
            "best_individual": self.best_individual,
            "gain": self.gain,
            "combined_knn": self.combined_knn,
            "T2+T2.5_beats_combined": self.T2_T2_5_beats_combined,
        }


class ContrastiveTierAblation(BaseModel):
    """Contrastive MLP tier ablation bundle.

    ``T2+T2.5`` on the wire becomes ``T2_T2_5`` in Python; translated by
    the validator + serializer pair below.
    """

    model_config = _FORBID

    individual: dict[str, ContrastiveAblationEntry]
    pairwise: dict[str, ContrastivePairwiseEntry]
    T2_T2_5: ContrastiveAblationEntry
    combined: ContrastiveAblationEntry
    super_additivity: ContrastiveSuperAdditivity

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "T2+T2.5" in data:
            out = dict(data)
            out["T2_T2_5"] = out.pop("T2+T2.5")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        return {
            "individual": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.individual.items()
            },
            "pairwise": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.pairwise.items()
            },
            "T2+T2.5": self.T2_T2_5.model_dump(mode="json", exclude_none=True),
            "combined": self.combined.model_dump(mode="json", exclude_none=True),
            "super_additivity": self.super_additivity.model_dump(mode="json"),
        }


class CapacitySweepEntry(BaseModel):
    """One hidden-dim row in the capacity sweep."""

    model_config = _FORBID

    knn_accuracy: float
    silhouette: float | None


class LinearBaselineEntry(BaseModel):
    """LDA / NCA linear projection baseline."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_components: int


class ContrastiveResult(BaseModel):
    """Section 8 result: contrastive projection (MLP + triplet loss).

    Top-level fields are all ``Optional`` so the PyTorch-missing error
    stub (``{"error": "..."}``) round-trips cleanly. The ``T2+T2.5`` key
    is renamed to ``T2_T2_5`` via validator/serializer for the same
    reason as ContrastiveTierAblation.
    """

    model_config = _FORBID

    T2_T2_5: ContrastiveTierResult | None = None
    combined: ContrastiveTierResult | None = None
    capacity_sweep: dict[str, CapacitySweepEntry] | None = None
    tier_ablation: ContrastiveTierAblation | None = None
    linear_baselines: dict[str, LinearBaselineEntry] | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "T2+T2.5" in data:
            out = dict(data)
            out["T2_T2_5"] = out.pop("T2+T2.5")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.T2_T2_5 is not None:
            out["T2+T2.5"] = self.T2_T2_5.model_dump(mode="json", exclude_none=True)
        if self.combined is not None:
            out["combined"] = self.combined.model_dump(mode="json", exclude_none=True)
        if self.capacity_sweep is not None:
            out["capacity_sweep"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.capacity_sweep.items()
            }
        if self.tier_ablation is not None:
            out["tier_ablation"] = self.tier_ablation.model_dump(mode="json")
        if self.linear_baselines is not None:
            out["linear_baselines"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.linear_baselines.items()
            }
        if self.error is not None:
            out["error"] = self.error
        return out
