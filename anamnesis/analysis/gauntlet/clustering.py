"""Section 7: Silhouette scores and clustering visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from .signature_io import ALL_CORE, ATTENTION_AND_CACHE, AnalysisData
from .schemas import (
    ClusteringResult,
    EmbeddingResult,
    PerModeSilhouetteStats,
    BlockSilhouette,
)
from .utils import absence_reason, get_available_blocks


def run_clustering(data: AnalysisData) -> ClusteringResult:
    """Silhouette analysis, K-Means ARI, and dimensionality reduction."""
    silhouette_by_block: dict[str, BlockSilhouette] = {}

    available_blocks, _ = get_available_blocks(data)
    # Silhouette per block (mode labels) — compute both cosine and euclidean
    for block in available_blocks:
        X = data.get_block(block)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        scores: dict[str, float | str] = {}
        for metric in ["cosine", "euclidean"]:
            try:
                scores[metric] = float(silhouette_score(X_std, data.modes, metric=metric))
            except Exception as e:
                scores[metric] = f"ERROR: {e}"

        # Topic silhouette (should be low — mode signal shouldn't correlate with topic)
        topic_score: float | None
        try:
            topic_score = float(silhouette_score(X_std, data.topics, metric="cosine"))
        except Exception:
            topic_score = None

        silhouette_by_block[block] = BlockSilhouette(
            mode_silhouette_cosine=scores["cosine"],
            mode_silhouette_euclidean=scores["euclidean"],
            mode_silhouette=scores["cosine"],  # backward compat alias
            topic_silhouette=topic_score,
        )

    # Per-mode silhouette decomposition (attention-and-cache union). The per-block
    # silhouettes above stand without that union, so its absence costs the readings
    # below and is recorded in each of them.
    union_absent = absence_reason(data, ATTENTION_AND_CACHE)

    per_mode_by_metric: dict[str, dict[str, Any]] = {"cosine": {}, "euclidean": {}}
    if union_absent is not None:
        for metric in ["cosine", "euclidean"]:
            per_mode_by_metric[metric]["error"] = union_absent
    else:
        X_key = data.get_block(ATTENTION_AND_CACHE)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_key)
        for metric in ["cosine", "euclidean"]:
            try:
                sample_sils = silhouette_samples(X_std, data.modes, metric=metric)
                for mode in data.unique_modes:
                    mask = data.mode_mask(mode)
                    mode_sils = sample_sils[mask]
                    per_mode_by_metric[metric][mode] = PerModeSilhouetteStats(
                        mean=float(np.mean(mode_sils)),
                        std=float(np.std(mode_sils)),
                        min=float(np.min(mode_sils)),
                        max=float(np.max(mode_sils)),
                        n_negative=int(np.sum(mode_sils < 0)),
                    ).model_dump(mode="json")
            except Exception as e:
                per_mode_by_metric[metric]["error"] = str(e)

    per_mode_silhouette_cosine = per_mode_by_metric["cosine"]
    per_mode_silhouette_euclidean = per_mode_by_metric["euclidean"]
    # Backward-compat alias pointing to cosine
    per_mode_silhouette = per_mode_silhouette_cosine

    # K-Means ARI
    kmeans_ari: dict[str, float | str] = {}
    for block in [ATTENTION_AND_CACHE, ALL_CORE]:
        block_absent = absence_reason(data, block)
        if block_absent is not None:
            kmeans_ari[block] = f"ABSENT: {block_absent}"
            continue
        X = data.get_block(block)
        X_std = StandardScaler().fit_transform(X)

        mode_to_int = {m: i for i, m in enumerate(data.unique_modes)}
        y_true = np.array([mode_to_int[m] for m in data.modes])

        try:
            km = KMeans(n_clusters=len(data.unique_modes), random_state=42, n_init=10)
            y_pred = km.fit_predict(X_std)
            kmeans_ari[block] = float(adjusted_rand_score(y_true, y_pred))
        except Exception as e:
            kmeans_ari[block] = f"ERROR: {e}"

    # UMAP / t-SNE embeddings (save coordinates for plotting)
    embeddings: dict[str, EmbeddingResult] = {}

    if union_absent is not None:
        for name in ("tsne_attention_and_cache", "umap_attention_and_cache"):
            embeddings[name] = EmbeddingResult(error=union_absent)
        return ClusteringResult(
            silhouette_by_block=silhouette_by_block,
            per_mode_silhouette=per_mode_silhouette,
            per_mode_silhouette_cosine=per_mode_silhouette_cosine,
            per_mode_silhouette_euclidean=per_mode_silhouette_euclidean,
            kmeans_ari=kmeans_ari,
            embeddings=embeddings,
        )

    # t-SNE (always available via sklearn)
    from sklearn.manifold import TSNE

    X_key_std = StandardScaler().fit_transform(data.get_block(ATTENTION_AND_CACHE))
    try:
        tsne = TSNE(
            n_components=2, random_state=42,
            perplexity=min(30, data.n_samples - 1),
        )
        coords = tsne.fit_transform(X_key_std)
        embeddings["tsne_attention_and_cache"] = EmbeddingResult(
            coords=coords.tolist(),
            modes=data.modes.tolist(),
            topics=data.topics.tolist(),
        )
    except Exception as e:
        embeddings["tsne_attention_and_cache"] = EmbeddingResult(error=str(e))

    # UMAP (optional)
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, random_state=42,
            n_neighbors=min(15, data.n_samples - 1),
        )
        coords = reducer.fit_transform(X_key_std)
        embeddings["umap_attention_and_cache"] = EmbeddingResult(
            coords=coords.tolist(),
            modes=data.modes.tolist(),
            topics=data.topics.tolist(),
        )
    except ImportError:
        embeddings["umap_attention_and_cache"] = EmbeddingResult(error="umap not installed")
    except Exception as e:
        embeddings["umap_attention_and_cache"] = EmbeddingResult(error=str(e))

    return ClusteringResult(
        silhouette_by_block=silhouette_by_block,
        per_mode_silhouette=per_mode_silhouette,
        per_mode_silhouette_cosine=per_mode_silhouette_cosine,
        per_mode_silhouette_euclidean=per_mode_silhouette_euclidean,
        kmeans_ari=kmeans_ari,
        embeddings=embeddings,
    )
