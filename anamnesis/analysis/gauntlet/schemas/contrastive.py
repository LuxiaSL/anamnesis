"""Section 8 schemas: learned projections and their ablations.

A supervised contrastive encoder trained per feature block, with the kNN readout
on its embedding, the per-block and pairwise ablation grids, the super-additivity
test, and the capacity sweep and linear baselines that say whether a nonlinear
encoder was needed.

The block labels on disk are a frozen wire vocabulary and several are not legal
Python identifiers, so each model that carries them translates in one place:
``_ON_DISK_RENAMES`` plus the validator/serializer pair beneath it. The Python
side says which substrates a block reads; the wire side stays byte-for-byte what
banked JSON holds.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, model_serializer, model_validator

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ContrastiveBlockResult(BaseModel):
    """One block's contrastive result: the kNN readout and silhouette on its embedding."""

    model_config = _FORBID

    knn_accuracy_mean: float
    knn_accuracy_std: float
    knn_fold_accs: list[float]
    silhouette_mean: float | None


class ContrastiveAblationEntry(BaseModel):
    """One block's entry in the ablation grid: ``individual``, or a union sub-key."""

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
    """Does the attention block plus the cache-and-keys block beat either alone.

    The comparison the numbers answer: each of the two blocks on its own, the two
    concatenated, and the concatenation against the whole vector.
    """

    model_config = _FORBID

    attention_alone: float
    cache_alone: float
    attention_and_cache_pair: float
    best_individual: float
    gain: float
    combined_knn: float
    attention_and_cache_beats_combined: bool

    _ON_DISK_RENAMES: ClassVar[dict[str, str]] = {
        "T2_alone": "attention_alone",
        "T2.5_alone": "cache_alone",
        "T2+T2.5_pair": "attention_and_cache_pair",
        "T2+T2.5_beats_combined": "attention_and_cache_beats_combined",
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
            "T2_alone": self.attention_alone,
            "T2.5_alone": self.cache_alone,
            "T2+T2.5_pair": self.attention_and_cache_pair,
            "best_individual": self.best_individual,
            "gain": self.gain,
            "combined_knn": self.combined_knn,
            "T2+T2.5_beats_combined": self.attention_and_cache_beats_combined,
        }


class ContrastiveBlockAblation(BaseModel):
    """The contrastive encoder's per-block ablation bundle.

    The union of the attention block and the cache-and-keys block is stored under
    a label that is not a legal Python identifier, so it is translated on the way
    in and out by the validator + serializer pair below.
    """

    model_config = _FORBID

    individual: dict[str, ContrastiveAblationEntry]
    pairwise: dict[str, ContrastivePairwiseEntry]
    attention_and_cache: ContrastiveAblationEntry
    combined: ContrastiveAblationEntry
    super_additivity: ContrastiveSuperAdditivity

    _ON_DISK_ATTENTION_AND_CACHE: ClassVar[str] = "T2+T2.5"

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls._ON_DISK_ATTENTION_AND_CACHE in data:
            out = dict(data)
            out["attention_and_cache"] = out.pop(cls._ON_DISK_ATTENTION_AND_CACHE)
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
            self._ON_DISK_ATTENTION_AND_CACHE: self.attention_and_cache.model_dump(
                mode="json", exclude_none=True
            ),
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
    stub (``{"error": "..."}``) round-trips cleanly. The union of the attention
    block and the cache-and-keys block is translated on the way in and out for
    the same reason as in ContrastiveBlockAblation.
    """

    model_config = _FORBID

    attention_and_cache: ContrastiveBlockResult | None = None
    combined: ContrastiveBlockResult | None = None
    capacity_sweep: dict[str, CapacitySweepEntry] | None = None
    block_ablation: ContrastiveBlockAblation | None = None
    linear_baselines: dict[str, LinearBaselineEntry] | None = None
    error: str | None = None

    _ON_DISK_ABLATION: ClassVar[str] = "tier_ablation"
    _ON_DISK_RENAMES: ClassVar[dict[str, str]] = {
        ContrastiveBlockAblation._ON_DISK_ATTENTION_AND_CACHE: "attention_and_cache",
        _ON_DISK_ABLATION: "block_ablation",
    }

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {cls._ON_DISK_RENAMES.get(k, k): v for k, v in data.items()}
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.attention_and_cache is not None:
            out[ContrastiveBlockAblation._ON_DISK_ATTENTION_AND_CACHE] = self.attention_and_cache.model_dump(
                mode="json", exclude_none=True
            )
        if self.combined is not None:
            out["combined"] = self.combined.model_dump(mode="json", exclude_none=True)
        if self.capacity_sweep is not None:
            out["capacity_sweep"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.capacity_sweep.items()
            }
        if self.block_ablation is not None:
            out[self._ON_DISK_ABLATION] = self.block_ablation.model_dump(mode="json")
        if self.linear_baselines is not None:
            out["linear_baselines"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.linear_baselines.items()
            }
        if self.error is not None:
            out["error"] = self.error
        return out
