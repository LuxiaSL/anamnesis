"""Section 9 schemas: is the signature reading content or computation?

The orthogonality battery. A TF-IDF surface baseline says how well the generated
text alone predicts the mode; Mantel correlations and text-to-compute R^2 say how
much of the signature geometry the text explains; the per-mode surface-versus-
compute split counts the modes that separate in computation while staying
inseparable in text. Shuffle controls and the prompt-swap confound are here
because a claim of orthogonality is only as good as its nulls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class ClassificationScore(BaseModel):
    """Per-classifier CV output inside a SemanticClassifierBundle."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]


class SemanticClassifierBundle(BaseModel):
    """Bundle of rf + knn accuracies for a given feature set.

    ``dims`` is present for tfidf / sbert / combined / semantic_noise
    (which record the feature-set dimensionality). Newer v2 runs omit
    ``dims`` for compute_classification (it pulls from
    per_block_semantic.classification, which never set it).
    """

    model_config = _FORBID

    rf: ClassificationScore | None = None
    knn: ClassificationScore | None = None
    dims: int | None = None
    error: str | None = None


class MantelResult(BaseModel):
    """Mantel test distance-matrix correlation."""

    model_config = _FORBID

    r: float
    p_value: float
    null_mean: float
    null_std: float


class TextToComputeR2(BaseModel):
    """Ridge R² summary for text → compute-feature prediction."""

    model_config = _FORBID

    median_r2: float
    mean_r2: float
    n_features_r2_above_01: int
    n_features_r2_above_0: int
    best_r2: float
    worst_r2: float


class PerModeSurfaceVsCompute(BaseModel):
    """One mode's surface-vs-compute recall gap."""

    model_config = _FORBID

    surface_recall: float
    compute_recall: float
    gap_compute_minus_surface: float
    sub_semantic: bool
    surface_n: int
    compute_n: int


class PerModeSurfaceVsComputeResult(BaseModel):
    """Per-mode surface-vs-compute decomposition summary."""

    model_config = _FORBID

    per_mode: dict[str, PerModeSurfaceVsCompute]
    n_sub_semantic: int
    sub_semantic_modes: list[str]
    n_surface_dominant: int
    surface_dominant_modes: list[str]


class ShuffleControlsResult(BaseModel):
    """Shuffle-control classifications for one compute feature set."""

    model_config = _FORBID

    within_topic_shuffle: ClassificationScore
    global_shuffle: ClassificationScore


class PerBlockSemanticResult(BaseModel):
    """Semantic orthogonality battery for a single block."""

    model_config = _FORBID

    n_features: int | None = None
    classification: SemanticClassifierBundle | None = None
    mantel_tfidf_cosine: MantelResult | None = None
    mantel_sbert_cosine: MantelResult | None = None
    text_to_compute_r2: TextToComputeR2 | None = None
    per_mode_surface_vs_compute: PerModeSurfaceVsComputeResult | None = None
    shuffle_controls: ShuffleControlsResult | None = None
    error: str | None = None


class RetrievalFeatureSet(BaseModel):
    """kNN mode-match retrieval output for one feature set."""

    model_config = _FORBID

    mean_mode_match: float
    std_mode_match: float
    per_mode: dict[str, float]
    k: int


class JaccardStats(BaseModel):
    """Jaccard overlap statistics between two kNN neighbour sets."""

    model_config = _FORBID

    mean: float
    std: float


class RetrievalResult(BaseModel):
    """kNN retrieval diagnostics across feature sets."""

    model_config = _FORBID

    compute_attention_and_cache: RetrievalFeatureSet | None = None
    tfidf: RetrievalFeatureSet | None = None
    sbert: RetrievalFeatureSet | None = None
    combined_compute_sbert: RetrievalFeatureSet | None = None
    jaccard_compute_tfidf: JaccardStats | None = None
    jaccard_compute_sbert: JaccardStats | None = None


class ContrastiveProjectionComparisonEntry(BaseModel):
    """One feature-set entry in the contrastive projection comparison."""

    model_config = _FORBID

    test_knn_accuracy: float
    train_knn_accuracy: float
    train_test_gap: float
    silhouette: float | None
    fold_accs: list[float]


class ContrastiveProjectionComparisonResult(BaseModel):
    """Contrastive projection comparison across feature sets.

    The section either returns an error stub or populates one or more
    feature-set entries (tfidf, compute_attention_and_cache, sbert, combined_compute_sbert).
    """

    model_config = _FORBID

    compute_attention_and_cache: ContrastiveProjectionComparisonEntry | None = None
    tfidf: ContrastiveProjectionComparisonEntry | None = None
    sbert: ContrastiveProjectionComparisonEntry | None = None
    combined_compute_sbert: ContrastiveProjectionComparisonEntry | None = None
    error: str | None = None


class PromptSwapBlockResult(BaseModel):
    """Per-block prompt-swap confound outcome."""

    model_config = _FORBID

    n_features: int
    follows_system_prompt: int
    follows_execution: int
    follows_neither: int
    pct_system: float
    pct_execution: float
    signal_type: str
    per_swap_type: dict[str, Any]


class PromptSwapConfoundResult(BaseModel):
    """Prompt-swap confound diagnostics.

    All fields Optional because the canonical runs hit the error path
    (``{"error": "No prompt-swap samples found"}``).
    """

    model_config = _FORBID

    n_swap_samples: int | None = None
    swap_types: list[str] | None = None
    per_block: dict[str, PromptSwapBlockResult] | None = None
    error: str | None = None


class SemanticResult(BaseModel):
    """Section 9 result.

    Top-level error ``{"error": "No generated text available"}`` round-trips
    via the Optional fields. ``per_block_semantic`` is only populated by
    v2 runs; older baseline snapshots omit it.
    """

    model_config = _FORBID

    tfidf_classification: SemanticClassifierBundle | None = None
    sbert_classification: SemanticClassifierBundle | None = None
    per_block_semantic: dict[str, PerBlockSemanticResult] | None = None
    compute_classification: SemanticClassifierBundle | None = None
    combined_classification: SemanticClassifierBundle | None = None
    semantic_noise_classification: SemanticClassifierBundle | None = None
    mantel_tfidf: MantelResult | None = None
    mantel_sbert: MantelResult | None = None
    mantel_tfidf_cosine: MantelResult | None = None
    mantel_sbert_cosine: MantelResult | None = None
    mantel_tfidf_euclidean: MantelResult | None = None
    mantel_sbert_euclidean: MantelResult | None = None
    text_to_compute_r2: TextToComputeR2 | None = None
    per_mode_surface_vs_compute: PerModeSurfaceVsComputeResult | None = None
    shuffle_controls: ShuffleControlsResult | None = None
    contrastive_projection_comparison: ContrastiveProjectionComparisonResult | None = None
    retrieval: RetrievalResult | None = None
    prompt_swap_confound: PromptSwapConfoundResult | None = None
    error: str | None = None
