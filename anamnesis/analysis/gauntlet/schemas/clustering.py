"""Section 7 schemas: silhouette and k-means agreement.

Unsupervised structure read against the mode labels: per-tier silhouette with
its per-mode breakdown, k-means adjusted Rand, and the 2-D embeddings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class TierSilhouette(BaseModel):
    """Per-tier silhouette scores.

    ``mode_silhouette_cosine`` / ``_euclidean`` / ``mode_silhouette`` are
    floats on success; they fall back to ``"ERROR: ..."`` strings when
    silhouette_score raises (e.g. degenerate cluster set).
    ``topic_silhouette`` is float or None.
    """

    model_config = _FORBID

    mode_silhouette_cosine: float | str
    mode_silhouette_euclidean: float | str
    mode_silhouette: float | str
    topic_silhouette: float | None


class PerModeSilhouetteStats(BaseModel):
    """Per-mode silhouette distribution summary."""

    model_config = _FORBID

    mean: float
    std: float
    min: float
    max: float
    n_negative: int


class EmbeddingResult(BaseModel):
    """2-D embedding (t-SNE or UMAP) for plotting.

    Success path stores ``coords`` (n×2 float), ``modes``, ``topics``.
    Error path stores only ``error`` (e.g. UMAP not installed).
    """

    model_config = _FORBID

    coords: list[list[float]] | None = None
    modes: list[str] | None = None
    topics: list[str] | None = None
    error: str | None = None


class ClusteringResult(BaseModel):
    """Section 7 result: silhouette + K-Means ARI + 2-D embeddings.

    Both ``per_mode_silhouette`` (top-level backward-compat key that
    aliases the cosine variant) and the explicit cosine/euclidean
    variants are preserved. When silhouette_samples raises, the
    per-mode dict contains a top-level ``"error": "..."`` key alongside
    any mode entries — the dict type here is ``dict[str, Any]`` so
    those heterogeneous entries round-trip cleanly.
    """

    model_config = _FORBID

    silhouette_by_tier: dict[str, TierSilhouette]
    per_mode_silhouette: dict[str, Any]
    per_mode_silhouette_cosine: dict[str, Any]
    per_mode_silhouette_euclidean: dict[str, Any]
    kmeans_ari: dict[str, float | str]
    embeddings: dict[str, EmbeddingResult]
