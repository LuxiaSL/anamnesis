"""Mode clustering visualization and silhouette analysis.

UMAP/t-SNE of computational signatures, colored by topic and shaped by mode.
Quantitative: silhouette scores for mode vs topic clustering.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig, MODE_INDEX

logger = logging.getLogger(__name__)

# Mode markers for visualization
MODE_MARKERS = {m: marker for m, marker in zip(MODE_INDEX.keys(), ["o", "s", "^", "D", "P"])}


def analyze_clustering(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run clustering analysis on Sets A + B.

    Returns silhouette scores, ARI, UMAP/t-SNE embeddings.
    """
    # Filter to Sets A + B
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]

    gen_ids = []
    features_list = []
    mode_labels = []
    topic_strings = []

    for m in ab_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        gen_ids.append(gid)
        features_list.append(signatures[gid]["features"])
        mode_labels.append(m["mode"])
        # Use actual topic string to avoid conflating different topics
        # that share the same topic_idx across Sets A and B
        topic_strings.append(m["topic"])

    n = len(gen_ids)
    if n < 10:
        return {"error": f"Too few generations for clustering: {n}"}

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mode_indices = np.array([MODE_INDEX.get(m, 0) for m in mode_labels])
    # Map unique topic strings to integer indices for silhouette computation
    unique_topic_strings = sorted(set(topic_strings))
    topic_str_to_idx = {t: i for i, t in enumerate(unique_topic_strings)}
    topic_indices = np.array([topic_str_to_idx[t] for t in topic_strings])

    results: dict[str, Any] = {"n_generations": n}

    # ── Silhouette scores ──
    try:
        if len(set(mode_indices)) > 1:
            mode_sil = silhouette_score(X_scaled, mode_indices, metric="cosine")
            results["mode_silhouette"] = float(mode_sil)
            logger.info(f"Mode silhouette: {mode_sil:.4f}")
        else:
            results["mode_silhouette"] = None

        if len(set(topic_indices)) > 1:
            topic_sil = silhouette_score(X_scaled, topic_indices, metric="cosine")
            results["topic_silhouette"] = float(topic_sil)
            logger.info(f"Topic silhouette: {topic_sil:.4f}")
        else:
            results["topic_silhouette"] = None
    except Exception as e:
        logger.warning(f"Silhouette computation failed: {e}")
        results["mode_silhouette"] = None
        results["topic_silhouette"] = None

    # ── K-Means clustering and ARI ──
    try:
        n_modes = len(set(mode_indices))
        kmeans = KMeans(n_clusters=n_modes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        ari = adjusted_rand_score(mode_indices, cluster_labels)
        results["ari_mode_vs_kmeans"] = float(ari)
        logger.info(f"ARI (mode vs K-Means): {ari:.4f}")
    except Exception as e:
        logger.warning(f"K-Means ARI failed: {e}")

    # ── UMAP ──
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        X_umap = reducer.fit_transform(X_scaled)
        results["umap_computed"] = True

        _plot_embedding(
            X_umap, mode_labels, topic_indices, mode_indices,
            title="UMAP of Computational Signatures",
            filename=config.figures_dir / "mode_clustering_umap.png",
        )

        # Also plot topic-focused
        _plot_topic_embedding(
            X_umap, topic_indices, mode_labels,
            title="UMAP Colored by Topic",
            filename=config.figures_dir / "topic_clustering_umap.png",
        )

    except ImportError:
        logger.warning("umap-learn not installed, falling back to t-SNE")
        results["umap_computed"] = False
    except Exception as e:
        logger.warning(f"UMAP failed: {e}")
        results["umap_computed"] = False

    # ── t-SNE ──
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
        X_tsne = tsne.fit_transform(X_scaled)

        _plot_embedding(
            X_tsne, mode_labels, topic_indices, mode_indices,
            title="t-SNE of Computational Signatures",
            filename=config.figures_dir / "mode_clustering_tsne.png",
        )
        results["tsne_computed"] = True
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")
        results["tsne_computed"] = False

    # ── kNN-LM baseline UMAP ──
    try:
        knnlm_vecs = []
        for gid in gen_ids:
            knnlm = signatures[gid].get("knnlm_baseline")
            if knnlm is not None:
                knnlm_vecs.append(knnlm)

        if len(knnlm_vecs) == n:
            import umap
            knnlm_mat = np.stack(knnlm_vecs)
            knnlm_mat = np.nan_to_num(knnlm_mat, nan=0.0)
            knnlm_scaled = StandardScaler().fit_transform(knnlm_mat)
            knnlm_umap = umap.UMAP(n_components=2, random_state=42, metric="cosine").fit_transform(knnlm_scaled)

            _plot_embedding(
                knnlm_umap, mode_labels, topic_indices, mode_indices,
                title="UMAP of kNN-LM Baseline Signatures",
                filename=config.figures_dir / "knnlm_baseline_umap.png",
            )

            # Silhouette for kNN-LM
            if len(set(mode_indices)) > 1:
                knnlm_sil = silhouette_score(knnlm_scaled, mode_indices, metric="cosine")
                results["knnlm_mode_silhouette"] = float(knnlm_sil)
                logger.info(f"kNN-LM mode silhouette: {knnlm_sil:.4f}")
    except Exception as e:
        logger.warning(f"kNN-LM UMAP failed: {e}")

    # ── Spectral features UMAP ──
    try:
        spectral_vecs = []
        for gid in gen_ids:
            sig = signatures[gid]
            feat_names = sig.get("feature_names", [])
            feats = sig["features"]
            spectral_mask = np.array(["spectral_" in n for n in feat_names])
            if spectral_mask.any():
                spectral_vecs.append(feats[spectral_mask])

        if len(spectral_vecs) == n and len(spectral_vecs[0]) > 0:
            import umap
            spec_mat = np.stack(spectral_vecs)
            spec_mat = np.nan_to_num(spec_mat, nan=0.0)
            spec_scaled = StandardScaler().fit_transform(spec_mat)
            spec_umap = umap.UMAP(n_components=2, random_state=42, metric="cosine").fit_transform(spec_scaled)

            _plot_embedding(
                spec_umap, mode_labels, topic_indices, mode_indices,
                title="UMAP of Spectral Features Only",
                filename=config.figures_dir / "spectral_features_umap.png",
            )

            if len(set(mode_indices)) > 1:
                spec_sil = silhouette_score(spec_scaled, mode_indices, metric="cosine")
                results["spectral_mode_silhouette"] = float(spec_sil)
                logger.info(f"Spectral mode silhouette: {spec_sil:.4f}")
    except Exception as e:
        logger.warning(f"Spectral UMAP failed: {e}")

    return results


def _plot_embedding(
    X_2d: np.ndarray,
    mode_labels: list[str],
    topic_indices: np.ndarray,
    mode_indices: np.ndarray,
    title: str,
    filename: Any,
) -> None:
    """Plot 2D embedding colored by topic, shaped by mode."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    unique_topics = sorted(set(topic_indices))
    n_topics = len(unique_topics)
    cmap = plt.cm.get_cmap("tab20", n_topics)

    for i, (x, mode, topic_idx) in enumerate(zip(X_2d, mode_labels, topic_indices)):
        color_idx = unique_topics.index(topic_idx)
        marker = MODE_MARKERS.get(mode, "o")
        ax.scatter(x[0], x[1], c=[cmap(color_idx)], marker=marker,
                   s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Legend for modes
    for mode, marker in MODE_MARKERS.items():
        ax.scatter([], [], marker=marker, c="gray", s=60, label=mode, edgecolors="black")
    ax.legend(title="Processing Mode", loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {filename}")


def _plot_topic_embedding(
    X_2d: np.ndarray,
    topic_indices: np.ndarray,
    mode_labels: list[str],
    title: str,
    filename: Any,
) -> None:
    """Plot 2D embedding colored by topic with annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    unique_topics = sorted(set(topic_indices))
    n_topics = len(unique_topics)
    cmap = plt.cm.get_cmap("tab20", n_topics)

    for topic_idx in unique_topics:
        mask = topic_indices == topic_idx
        color_idx = unique_topics.index(topic_idx)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap(color_idx)],
                   s=60, alpha=0.7, edgecolors="black", linewidth=0.5,
                   label=f"Topic {topic_idx}")

    ax.legend(title="Topic", loc="upper right", fontsize=7, ncol=2)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {filename}")
