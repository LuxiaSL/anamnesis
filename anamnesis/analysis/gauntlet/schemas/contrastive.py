"""Section 8 schemas: learned projections and their ablations.

A supervised contrastive encoder trained per feature block, with the kNN readout
on its embedding, the per-block and pairwise ablation grids, the super-additivity
test, and the capacity sweep and linear baselines that say whether a nonlinear
encoder was needed.

Every field here is named for what it holds, and a written result carries those
names. A banked file whose keys predate them is read forward by
``anamnesis/analysis/gauntlet/schemas/compat.py``, which is where the retired
spellings live.
"""

from __future__ import annotations

from pydantic import BaseModel

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


class ContrastiveBlockAblation(BaseModel):
    """The contrastive encoder's per-block ablation bundle.

    ``individual`` is keyed by block label and ``pairwise`` by a pair of them
    joined with ``+``; the two named entries are the unions those keys cannot
    spell as a field name.
    """

    model_config = _FORBID

    individual: dict[str, ContrastiveAblationEntry]
    pairwise: dict[str, ContrastivePairwiseEntry]
    attention_and_cache: ContrastiveAblationEntry
    combined: ContrastiveAblationEntry
    super_additivity: ContrastiveSuperAdditivity


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

    Every field is ``Optional`` so that a section that could not run — PyTorch
    absent, or the union it reads not in this corpus — round-trips as an error
    stub carrying only its reason.
    """

    model_config = _FORBID

    attention_and_cache: ContrastiveBlockResult | None = None
    combined: ContrastiveBlockResult | None = None
    capacity_sweep: dict[str, CapacitySweepEntry] | None = None
    block_ablation: ContrastiveBlockAblation | None = None
    linear_baselines: dict[str, LinearBaselineEntry] | None = None
    error: str | None = None
