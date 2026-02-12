"""Deep-dive analyses for Phase 0 experiment interpretation.

New analyses suggested by research review:
  1. Distance matrix heatmaps (mode-sorted vs topic-sorted)
  2. Noise floor vs between-condition overlay
  3. Per-mode silhouette decomposition
  4. Pairwise mode discriminability (10 binary classifiers)
  5. Topic-controlled mode distance (effect size per topic)
  6. Feature importance x noise floor CV cross-reference
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig, MODE_INDEX

logger = logging.getLogger(__name__)


def run_deep_dive(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
    noise_floor_results: dict[str, Any] | None = None,
    feature_importance_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all deep-dive analyses. Returns combined results dict."""
    results: dict[str, Any] = {}

    # Collect Set A+B data (the main analysis sets)
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]
    set_c_meta = [m for m in metadata_list if m["prompt_set"] == "C"]

    gen_ids: list[int] = []
    features_list: list[np.ndarray] = []
    mode_labels: list[str] = []
    topic_strings: list[str] = []

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
        return {"error": f"Too few Set A+B generations: {n}"}

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modes_arr = np.array(mode_labels)
    # Map unique topic strings to integer indices
    unique_topic_strings = sorted(set(topic_strings))
    topic_str_to_idx = {t: i for i, t in enumerate(unique_topic_strings)}
    topics_arr = np.array([topic_str_to_idx[t] for t in topic_strings])
    mode_indices = np.array([MODE_INDEX.get(m, 0) for m in mode_labels])

    # 1. Distance matrix heatmaps
    logger.info("=== Distance Matrix Heatmaps ===")
    try:
        heatmap_results = _distance_matrix_heatmaps(
            X_scaled, modes_arr, topics_arr, mode_indices, config,
        )
        results["distance_heatmaps"] = heatmap_results
    except Exception as e:
        logger.error(f"Distance heatmap analysis failed: {e}", exc_info=True)
        results["distance_heatmaps"] = {"error": str(e)}

    # 2. Noise floor vs between-condition overlay
    logger.info("=== Noise Floor vs Between-Condition ===")
    try:
        overlay_results = _noise_floor_overlay(
            signatures, metadata_list, X_scaled, modes_arr, topics_arr, config,
        )
        results["noise_overlay"] = overlay_results
    except Exception as e:
        logger.error(f"Noise overlay analysis failed: {e}", exc_info=True)
        results["noise_overlay"] = {"error": str(e)}

    # 3. Per-mode silhouette decomposition
    logger.info("=== Per-Mode Silhouette ===")
    try:
        per_mode_sil = _per_mode_silhouette(X_scaled, mode_indices, mode_labels, config)
        results["per_mode_silhouette"] = per_mode_sil
    except Exception as e:
        logger.error(f"Per-mode silhouette failed: {e}", exc_info=True)
        results["per_mode_silhouette"] = {"error": str(e)}

    # 4. Pairwise mode discriminability
    logger.info("=== Pairwise Mode Discriminability ===")
    try:
        pairwise_results = _pairwise_mode_discriminability(
            X_scaled, modes_arr, config,
        )
        results["pairwise_discriminability"] = pairwise_results
    except Exception as e:
        logger.error(f"Pairwise discriminability failed: {e}", exc_info=True)
        results["pairwise_discriminability"] = {"error": str(e)}

    # 5. Topic-controlled mode distance
    logger.info("=== Topic-Controlled Mode Distance ===")
    try:
        topic_ctrl = _topic_controlled_mode_distance(
            X_scaled, modes_arr, topics_arr, config,
        )
        results["topic_controlled_distance"] = topic_ctrl
    except Exception as e:
        logger.error(f"Topic-controlled distance failed: {e}", exc_info=True)
        results["topic_controlled_distance"] = {"error": str(e)}

    # 6. Feature importance x noise floor CV cross-reference
    logger.info("=== Feature Importance x Noise Floor CV ===")
    try:
        if noise_floor_results and feature_importance_results:
            cv_xref = _feature_cv_cross_reference(
                noise_floor_results, feature_importance_results, config,
            )
            results["feature_cv_crossref"] = cv_xref
        else:
            results["feature_cv_crossref"] = {"skipped": "Missing noise floor or feature importance results"}
    except Exception as e:
        logger.error(f"Feature CV cross-reference failed: {e}", exc_info=True)
        results["feature_cv_crossref"] = {"error": str(e)}

    return results


def _distance_matrix_heatmaps(
    X_scaled: np.ndarray,
    modes: np.ndarray,
    topics: np.ndarray,
    mode_indices: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Plot pairwise distance heatmaps sorted by mode and by topic.

    The visual "does it work" test: if mode-sorted shows block structure
    (within-mode distances smaller) but topic-sorted doesn't, we have signal.
    """
    n = len(X_scaled)
    dist_mat = squareform(pdist(X_scaled, metric="cosine"))

    # Sort by mode
    mode_sort = np.argsort(mode_indices, kind="stable")
    dist_by_mode = dist_mat[np.ix_(mode_sort, mode_sort)]
    modes_sorted = modes[mode_sort]

    # Sort by topic
    topic_sort = np.argsort(topics, kind="stable")
    dist_by_topic = dist_mat[np.ix_(topic_sort, topic_sort)]
    topics_sorted = topics[topic_sort]

    # Compute block-diagonal vs off-diagonal averages for quantification
    unique_modes = sorted(set(modes))
    mode_block_within = []
    mode_block_between = []
    for m in unique_modes:
        mask = modes == m
        within = dist_mat[np.ix_(mask, mask)]
        np.fill_diagonal(within, np.nan)
        mode_block_within.extend(within[~np.isnan(within)].tolist())
        between = dist_mat[mask][:, ~mask]
        mode_block_between.extend(between.ravel().tolist())

    unique_topics = sorted(set(topics))
    topic_block_within = []
    topic_block_between = []
    for t in unique_topics:
        mask = topics == t
        within = dist_mat[np.ix_(mask, mask)]
        np.fill_diagonal(within, np.nan)
        topic_block_within.extend(within[~np.isnan(within)].tolist())
        between = dist_mat[mask][:, ~mask]
        topic_block_between.extend(between.ravel().tolist())

    mode_effect = float(np.mean(mode_block_between) - np.mean(mode_block_within))
    topic_effect = float(np.mean(topic_block_between) - np.mean(topic_block_within))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Mode-sorted
    im0 = axes[0].imshow(dist_by_mode, cmap="viridis", aspect="equal")
    axes[0].set_title(f"Sorted by Mode (effect={mode_effect:.4f})")
    # Add mode boundary lines
    idx = 0
    for m in unique_modes:
        count = int((modes_sorted == m).sum())
        if idx > 0:
            axes[0].axhline(idx - 0.5, color="red", linewidth=0.8, alpha=0.7)
            axes[0].axvline(idx - 0.5, color="red", linewidth=0.8, alpha=0.7)
        axes[0].text(-2, idx + count / 2, m[:4], fontsize=7, ha="right", va="center")
        idx += count
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Cosine Distance")

    # Topic-sorted
    im1 = axes[1].imshow(dist_by_topic, cmap="viridis", aspect="equal")
    axes[1].set_title(f"Sorted by Topic (effect={topic_effect:.4f})")
    idx = 0
    for t in unique_topics:
        count = int((topics_sorted == t).sum())
        if idx > 0:
            axes[1].axhline(idx - 0.5, color="red", linewidth=0.8, alpha=0.7)
            axes[1].axvline(idx - 0.5, color="red", linewidth=0.8, alpha=0.7)
        idx += count
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Cosine Distance")

    fig.suptitle("Pairwise Computational Distance Matrices", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "distance_heatmaps.png", dpi=150)
    plt.close(fig)
    logger.info("Saved distance_heatmaps.png")

    return {
        "mode_within_mean": float(np.mean(mode_block_within)),
        "mode_between_mean": float(np.mean(mode_block_between)),
        "mode_effect": mode_effect,
        "topic_within_mean": float(np.mean(topic_block_within)),
        "topic_between_mean": float(np.mean(topic_block_between)),
        "topic_effect": topic_effect,
    }


def _noise_floor_overlay(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    X_scaled_ab: np.ndarray,
    modes_ab: np.ndarray,
    topics_ab: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Overlay noise floor (Set C within-pair distances) on between-mode distances.

    Critical question: what fraction of between-mode distances exceed the
    95th percentile of the noise floor?
    """
    # Collect Set C within-pair distances
    set_c = [m for m in metadata_list if m["prompt_set"] == "C"]
    pairs: dict[tuple[int, str], list[int]] = {}
    for m in set_c:
        key = (m["topic_idx"], m["mode"])
        pairs.setdefault(key, []).append(m["generation_id"])

    # Collect ALL vectors (Set C + Set A) for a global scaler
    # Previous bug: z-scoring within pairs of n=2 produces degenerate distances
    all_vecs_for_scaler: list[np.ndarray] = []
    for m in metadata_list:
        gid = m["generation_id"]
        if gid in signatures and m["prompt_set"] in ("A", "B", "C"):
            vec = np.nan_to_num(signatures[gid]["features"], nan=0.0, posinf=0.0, neginf=0.0)
            all_vecs_for_scaler.append(vec)

    if len(all_vecs_for_scaler) < 2:
        return {"error": "Too few vectors for noise floor overlay"}

    all_mat = np.stack(all_vecs_for_scaler)
    global_mean = all_mat.mean(axis=0)
    global_std = all_mat.std(axis=0)
    global_std[global_std < 1e-10] = 1.0

    within_dists: list[float] = []
    per_mode_within: dict[str, list[float]] = {}
    for (topic_idx, mode), gen_ids in pairs.items():
        vecs = []
        for gid in gen_ids:
            if gid in signatures:
                vec = np.nan_to_num(signatures[gid]["features"], nan=0.0, posinf=0.0, neginf=0.0)
                vecs.append(vec)
        if len(vecs) < 2:
            continue
        mat = np.stack(vecs)
        # Use global scaler for comparability (fixes n=2 z-scoring bug)
        mat_scaled = (mat - global_mean) / global_std
        dists = pdist(mat_scaled, metric="cosine").tolist()
        within_dists.extend(dists)
        per_mode_within.setdefault(mode, []).extend(dists)

    if not within_dists:
        return {"error": "No Set C within-pair distances available"}

    noise_arr = np.array(within_dists)
    noise_p95 = float(np.percentile(noise_arr, 95))

    # Compute between-mode-same-topic distances from Set A
    set_a_meta = [m for m in metadata_list if m["prompt_set"] == "A"]
    topic_mode_map: dict[int, dict[str, np.ndarray]] = {}
    for m in set_a_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        tid = m["topic_idx"]
        mode = m["mode"]
        vec = signatures[gid]["features"]
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        topic_mode_map.setdefault(tid, {})[mode] = vec

    between_mode_dists: list[float] = []
    for tid, mode_vecs in topic_mode_map.items():
        modes_list = list(mode_vecs.keys())
        for i, m1 in enumerate(modes_list):
            for m2 in modes_list[i + 1:]:
                v1 = mode_vecs[m1].reshape(1, -1)
                v2 = mode_vecs[m2].reshape(1, -1)
                # Use same global scaler as within-pair distances
                v1_scaled = (v1 - global_mean) / global_std
                v2_scaled = (v2 - global_mean) / global_std
                d = float(cdist(v1_scaled, v2_scaled, metric="cosine")[0, 0])
                between_mode_dists.append(d)

    between_arr = np.array(between_mode_dists) if between_mode_dists else np.array([0.0])

    # Key metric: what fraction of between-mode distances exceed noise p95?
    frac_above_noise = float((between_arr > noise_p95).mean()) if len(between_arr) > 0 else 0.0

    # Per-mode noise floor variance
    per_mode_noise_stats: dict[str, dict[str, float]] = {}
    for mode, dists in per_mode_within.items():
        arr = np.array(dists)
        per_mode_noise_stats[mode] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n_pairs": len(dists),
        }

    # Plot: overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(noise_arr, bins=30, alpha=0.6, color="steelblue", edgecolor="black",
            label=f"Noise floor (Set C, n={len(noise_arr)})", density=True)
    if len(between_mode_dists) > 0:
        ax.hist(between_arr, bins=30, alpha=0.6, color="coral", edgecolor="black",
                label=f"Between-mode same-topic (Set A, n={len(between_arr)})", density=True)
    ax.axvline(noise_p95, color="orange", linestyle="--", linewidth=2,
               label=f"Noise 95th %ile: {noise_p95:.4f}")
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Noise Floor vs Between-Mode Distances ({frac_above_noise:.0%} above noise p95)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.figures_dir / "noise_overlay.png", dpi=150)
    plt.close(fig)
    logger.info("Saved noise_overlay.png")

    # Per-mode noise floor bar chart
    if per_mode_noise_stats:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        modes_sorted = sorted(per_mode_noise_stats.keys())
        means = [per_mode_noise_stats[m]["mean"] for m in modes_sorted]
        stds = [per_mode_noise_stats[m]["std"] for m in modes_sorted]
        ax.bar(range(len(modes_sorted)), means, yerr=stds, color="steelblue",
               edgecolor="black", capsize=5)
        ax.set_xticks(range(len(modes_sorted)))
        ax.set_xticklabels(modes_sorted)
        ax.set_ylabel("Mean Within-Pair Distance (cosine)")
        ax.set_title("Noise Floor by Mode (Set C)")
        fig.tight_layout()
        fig.savefig(config.figures_dir / "noise_per_mode.png", dpi=150)
        plt.close(fig)
        logger.info("Saved noise_per_mode.png")

    return {
        "noise_p95": noise_p95,
        "noise_mean": float(noise_arr.mean()),
        "between_mode_mean": float(between_arr.mean()),
        "between_mode_n": len(between_mode_dists),
        "fraction_above_noise_p95": frac_above_noise,
        "per_mode_noise": per_mode_noise_stats,
    }


def _per_mode_silhouette(
    X_scaled: np.ndarray,
    mode_indices: np.ndarray,
    mode_labels: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Compute per-mode silhouette scores.

    Which modes cluster tightest? Prediction: analytical/confident highest,
    uncertain lowest.
    """
    sample_sils = silhouette_samples(X_scaled, mode_indices, metric="cosine")

    unique_modes = sorted(set(mode_labels))
    per_mode: dict[str, dict[str, float]] = {}
    for mode in unique_modes:
        mask = np.array(mode_labels) == mode
        mode_sils = sample_sils[mask]
        per_mode[mode] = {
            "mean_silhouette": float(mode_sils.mean()),
            "std_silhouette": float(mode_sils.std()),
            "min_silhouette": float(mode_sils.min()),
            "max_silhouette": float(mode_sils.max()),
            "n_samples": int(mask.sum()),
        }
        logger.info(f"  {mode}: silhouette = {mode_sils.mean():.4f} +/- {mode_sils.std():.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    means = [per_mode[m]["mean_silhouette"] for m in unique_modes]
    stds = [per_mode[m]["std_silhouette"] for m in unique_modes]
    colors = ["steelblue" if m >= 0 else "coral" for m in means]
    ax.bar(range(len(unique_modes)), means, yerr=stds, color=colors,
           edgecolor="black", capsize=5)
    ax.set_xticks(range(len(unique_modes)))
    ax.set_xticklabels(unique_modes)
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_title("Per-Mode Silhouette Decomposition")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "per_mode_silhouette.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per_mode_silhouette.png")

    return per_mode


def _pairwise_mode_discriminability(
    X_scaled: np.ndarray,
    modes: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Train 10 binary classifiers (one per mode pair) and report accuracy.

    Gives a 5x5 matrix: "how distinguishable is mode X from mode Y."
    """
    unique_modes = sorted(set(modes))
    n_modes = len(unique_modes)
    discrim_matrix = np.full((n_modes, n_modes), np.nan)

    pair_results: dict[str, float] = {}
    for i, m1 in enumerate(unique_modes):
        discrim_matrix[i, i] = 1.0  # self
        for j, m2 in enumerate(unique_modes):
            if j <= i:
                continue
            mask = (modes == m1) | (modes == m2)
            X_pair = X_scaled[mask]
            y_pair = (modes[mask] == m1).astype(int)

            if len(set(y_pair)) < 2 or len(y_pair) < 6:
                continue

            cv_folds = min(5, min(sum(y_pair == 0), sum(y_pair == 1)))
            cv_folds = max(cv_folds, 2)

            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, X_pair, y_pair, cv=cv_folds, scoring="accuracy")
            acc = float(scores.mean())

            discrim_matrix[i, j] = acc
            discrim_matrix[j, i] = acc
            pair_key = f"{m1}_vs_{m2}"
            pair_results[pair_key] = acc
            logger.info(f"  {m1} vs {m2}: accuracy = {acc:.3f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(discrim_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(n_modes))
    ax.set_yticks(range(n_modes))
    ax.set_xticklabels(unique_modes, rotation=45, ha="right")
    ax.set_yticklabels(unique_modes)
    ax.set_title("Pairwise Mode Discriminability (RF Binary Accuracy)")

    for i in range(n_modes):
        for j in range(n_modes):
            val = discrim_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.8 or val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Binary Classification Accuracy")
    ax.axhline(-0.5, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(config.figures_dir / "pairwise_discriminability.png", dpi=150)
    plt.close(fig)
    logger.info("Saved pairwise_discriminability.png")

    return {
        "matrix": discrim_matrix.tolist(),
        "labels": unique_modes,
        "pairs": pair_results,
    }


def _topic_controlled_mode_distance(
    X_scaled: np.ndarray,
    modes: np.ndarray,
    topics: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """For each topic, compute within-mode vs between-mode distances.

    Gives per-topic effect sizes. If consistent across topics, signal is robust.
    """
    unique_topics = sorted(set(topics))
    per_topic: dict[int, dict[str, float]] = {}

    for tid in unique_topics:
        topic_mask = topics == tid
        topic_X = X_scaled[topic_mask]
        topic_modes = modes[topic_mask]

        if len(topic_X) < 3:
            continue

        unique_modes_here = sorted(set(topic_modes))
        if len(unique_modes_here) < 2:
            continue

        within_dists: list[float] = []
        between_dists: list[float] = []

        for i in range(len(topic_X)):
            for j in range(i + 1, len(topic_X)):
                d = float(cdist(topic_X[i:i+1], topic_X[j:j+1], metric="cosine")[0, 0])
                if topic_modes[i] == topic_modes[j]:
                    within_dists.append(d)
                else:
                    between_dists.append(d)

        if not between_dists:
            continue

        within_mean = float(np.mean(within_dists)) if within_dists else 0.0
        between_mean = float(np.mean(between_dists))
        # Cohen's d effect size: (between - within) / pooled std
        if within_dists:
            pooled = np.sqrt(
                (np.var(within_dists) + np.var(between_dists)) / 2.0
            )
            effect_size = (between_mean - within_mean) / max(float(pooled), 1e-10)
        else:
            effect_size = 0.0

        per_topic[int(tid)] = {
            "within_mean": within_mean,
            "between_mean": between_mean,
            "effect_size": float(effect_size),
            "n_within_pairs": len(within_dists),
            "n_between_pairs": len(between_dists),
        }
        logger.info(f"  Topic {tid}: within={within_mean:.4f}, between={between_mean:.4f}, "
                     f"effect={effect_size:.3f}")

    if not per_topic:
        return {"error": "No topics with enough data for comparison"}

    effect_sizes = [v["effect_size"] for v in per_topic.values()]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effect sizes per topic
    topic_ids = sorted(per_topic.keys())
    effects = [per_topic[t]["effect_size"] for t in topic_ids]
    colors = ["steelblue" if e > 0 else "coral" for e in effects]
    axes[0].bar(range(len(topic_ids)), effects, color=colors, edgecolor="black")
    axes[0].set_xticks(range(len(topic_ids)))
    axes[0].set_xticklabels([str(t) for t in topic_ids], fontsize=8)
    axes[0].set_xlabel("Topic Index")
    axes[0].set_ylabel("Effect Size (Cohen's d)")
    axes[0].set_title("Mode Effect Size per Topic")
    axes[0].axhline(0, color="gray", linestyle="-", linewidth=0.5)
    axes[0].axhline(np.mean(effects), color="red", linestyle="--", alpha=0.7,
                     label=f"Mean: {np.mean(effects):.3f}")
    axes[0].legend()

    # Within vs between distances per topic
    within_means = [per_topic[t]["within_mean"] for t in topic_ids]
    between_means = [per_topic[t]["between_mean"] for t in topic_ids]
    x = np.arange(len(topic_ids))
    width = 0.35
    axes[1].bar(x - width / 2, within_means, width, label="Within-mode", color="steelblue",
                edgecolor="black")
    axes[1].bar(x + width / 2, between_means, width, label="Between-mode", color="coral",
                edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(t) for t in topic_ids], fontsize=8)
    axes[1].set_xlabel("Topic Index")
    axes[1].set_ylabel("Mean Cosine Distance")
    axes[1].set_title("Within vs Between-Mode Distance per Topic")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.figures_dir / "topic_controlled_distance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved topic_controlled_distance.png")

    return {
        "per_topic": per_topic,
        "mean_effect_size": float(np.mean(effect_sizes)),
        "std_effect_size": float(np.std(effect_sizes)),
        "min_effect_size": float(np.min(effect_sizes)),
        "max_effect_size": float(np.max(effect_sizes)),
        "n_positive_topics": int(sum(1 for e in effect_sizes if e > 0)),
        "n_total_topics": len(effect_sizes),
    }


def _feature_cv_cross_reference(
    noise_floor_results: dict[str, Any],
    feature_importance_results: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Cross-reference top RF features with their noise floor CV.

    Features that are top-5 important AND have low CV (< 0.3) are robust signals.
    Features that are important but have high CV might be important because
    of mode-specific variance (which is valid but different in interpretation).
    """
    feature_cvs = noise_floor_results.get("feature_cvs", [])
    top_features = feature_importance_results.get("top_20_features", [])

    if not feature_cvs or not top_features:
        return {"error": "Missing feature CVs or feature importance data"}

    cross_ref: list[dict[str, Any]] = []
    for feat in top_features:
        idx = feat["index"]
        cv = feature_cvs[idx] if idx < len(feature_cvs) else None
        entry: dict[str, Any] = {
            "name": feat["name"],
            "importance": feat["importance"],
            "index": idx,
            "cv": float(cv) if cv is not None else None,
        }
        if cv is not None:
            if cv < 0.3:
                entry["signal_type"] = "robust"
            elif cv < 0.8:
                entry["signal_type"] = "moderate"
            else:
                entry["signal_type"] = "variance_driven"
        cross_ref.append(entry)

    n_robust = sum(1 for c in cross_ref if c.get("signal_type") == "robust")
    n_variance = sum(1 for c in cross_ref if c.get("signal_type") == "variance_driven")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    names = [c["name"] for c in cross_ref]
    importances = [c["importance"] for c in cross_ref]
    cvs = [c["cv"] if c["cv"] is not None else 0 for c in cross_ref]

    colors = []
    for c in cross_ref:
        st = c.get("signal_type", "unknown")
        if st == "robust":
            colors.append("seagreen")
        elif st == "variance_driven":
            colors.append("coral")
        else:
            colors.append("steelblue")

    y_pos = range(len(names))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (CV={cv:.2f})" for n, cv in zip(names, cvs)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RF Feature Importance")
    ax.set_title("Top 20 Features: Importance vs Noise Floor Stability\n"
                  "(Green=robust CV<0.3, Blue=moderate, Red=variance-driven CV>0.8)")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "feature_cv_crossref.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature_cv_crossref.png")

    return {
        "features": cross_ref,
        "n_robust": n_robust,
        "n_variance_driven": n_variance,
        "summary": (
            f"{n_robust} of top 20 features are robust (CV<0.3), "
            f"{n_variance} are variance-driven (CV>0.8)"
        ),
    }
