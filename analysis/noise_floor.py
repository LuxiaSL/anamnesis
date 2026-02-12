"""Noise floor estimation (Set C) and positive control validation (Set D).

Set C: 5 topic-mode pairs × 10 reps → within-condition distance distributions.
Set D: 5 "knows" + 5 "doesn't know" → LDA separation as pipeline sanity check.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def analyze_noise_floor(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Analyze Set C to establish the noise floor.

    Returns dict with:
        mean_within_distance, std_within_distance, p95_within_distance,
        per_pair distances, per_feature CVs, unreliable_features.
    """
    # Group Set C generations by topic-mode pair
    set_c = [m for m in metadata_list if m["prompt_set"] == "C"]
    if not set_c:
        return {"error": "No Set C generations found"}

    # Group by (topic_idx, mode)
    pairs: dict[tuple[int, str], list[int]] = {}
    for m in set_c:
        key = (m["topic_idx"], m["mode"])
        pairs.setdefault(key, []).append(m["generation_id"])

    all_within_dists: list[float] = []
    per_pair_results: dict[str, dict] = {}
    all_feature_values: list[np.ndarray] = []  # for CV computation

    for (topic_idx, mode), gen_ids in pairs.items():
        # Collect feature vectors
        vecs = []
        for gid in gen_ids:
            if gid in signatures:
                vecs.append(signatures[gid]["features"])

        if len(vecs) < 2:
            continue

        mat = np.stack(vecs)
        all_feature_values.append(mat)

        # Pairwise cosine distances
        dists = pdist(mat, metric="cosine")
        all_within_dists.extend(dists.tolist())

        pair_key = f"topic{topic_idx}_{mode}"
        per_pair_results[pair_key] = {
            "n_reps": len(vecs),
            "mean_dist": float(np.mean(dists)),
            "std_dist": float(np.std(dists)),
            "min_dist": float(np.min(dists)),
            "max_dist": float(np.max(dists)),
        }

    if not all_within_dists:
        return {"error": "No valid Set C pairs found"}

    within_arr = np.array(all_within_dists)

    # Per-feature coefficient of variation across repeats
    # Compute CV WITHIN each pair separately, then average across pairs
    # (avoids conflating within-pair and between-pair variance)
    n_features = all_feature_values[0].shape[1] if all_feature_values else 0
    feature_cvs: list[float] = []

    if all_feature_values and n_features > 0:
        per_pair_cvs = np.zeros((len(all_feature_values), n_features), dtype=np.float64)
        for pair_idx, mat in enumerate(all_feature_values):
            for i in range(n_features):
                col = mat[:, i].astype(np.float64)
                mean = np.abs(col.mean())
                std = col.std()
                per_pair_cvs[pair_idx, i] = std / max(mean, 1e-10)
        # Average CV across pairs for each feature
        feature_cvs = per_pair_cvs.mean(axis=0).tolist()

    results = {
        "mean_within_distance": float(within_arr.mean()),
        "std_within_distance": float(within_arr.std()),
        "p95_within_distance": float(np.percentile(within_arr, 95)),
        "median_within_distance": float(np.median(within_arr)),
        "n_pairs": len(per_pair_results),
        "total_comparisons": len(all_within_dists),
        "per_pair": per_pair_results,
        "n_unreliable_features": len([cv for cv in feature_cvs if cv > 1.0]),
        "feature_cvs": feature_cvs,  # expose for downstream use
    }

    # Plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(within_arr, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(results["mean_within_distance"], color="red", linestyle="--",
                    label=f"Mean: {results['mean_within_distance']:.4f}")
        ax.axvline(results["p95_within_distance"], color="orange", linestyle="--",
                    label=f"95th %ile: {results['p95_within_distance']:.4f}")
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("Count")
        ax.set_title("Noise Floor: Within-Condition Pairwise Distances (Set C)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.figures_dir / "noise_floor.png", dpi=150)
        plt.close(fig)
        logger.info("Saved noise_floor.png")
    except Exception as e:
        logger.warning(f"Failed to save noise floor plot: {e}")

    return results


def validate_positive_control(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Validate extraction pipeline with Set D (knows vs doesn't know).

    Uses LDA to test if features separate the two conditions.
    """
    set_d = [m for m in metadata_list if m["prompt_set"] == "D"]
    if not set_d:
        return {"error": "No Set D generations found"}

    features_list = []
    labels = []
    knows_lengths: list[int] = []
    doesnt_know_lengths: list[int] = []

    for m in set_d:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        features_list.append(signatures[gid]["features"])
        # Support both labeling schemes:
        # Legacy: "knows_*" vs "doesnt_know_*"
        # Context-prefix: "bare_*" vs "context_*"
        topic = m["topic"]
        is_condition_a = topic.startswith("knows_") or topic.startswith("bare_")
        label = 1 if is_condition_a else 0
        labels.append(label)
        n_tokens = m.get("num_generated_tokens", 0)
        if is_condition_a:
            knows_lengths.append(n_tokens)
        else:
            doesnt_know_lengths.append(n_tokens)

    if len(features_list) < 4:
        return {"error": f"Too few Set D generations: {len(features_list)}"}

    X = np.stack(features_list)
    y = np.array(labels)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalize
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    X = (X - X.mean(axis=0)) / std

    # LDA
    n_samples = len(y)
    n_per_class = min(sum(y == 0), sum(y == 1))

    if n_per_class < 2:
        return {"error": "Need at least 2 samples per class"}

    lda = LinearDiscriminantAnalysis(n_components=1)

    # Use leave-one-out CV for small samples (most robust with n=10)
    from sklearn.model_selection import LeaveOneOut
    try:
        loo = LeaveOneOut()
        scores = cross_val_score(lda, X, y, cv=loo, scoring="accuracy")
        accuracy = float(scores.mean())
        cv_method = "leave-one-out"
    except Exception as e:
        logger.warning(f"LOO cross-validation failed: {e}")
        # Fallback: report as unreliable rather than using train-set accuracy
        accuracy = 0.0
        cv_method = "failed"
        logger.warning("Positive control accuracy set to 0 (CV failed — cannot evaluate)")

    # Fit final LDA for projection visualization
    lda.fit(X, y)
    X_proj = lda.transform(X)

    # Length diagnostic: flag if knows vs doesn't-know have wildly different lengths
    length_diag: dict[str, Any] = {}
    if knows_lengths and doesnt_know_lengths:
        knows_mean = float(np.mean(knows_lengths))
        doesnt_know_mean = float(np.mean(doesnt_know_lengths))
        length_ratio = doesnt_know_mean / max(knows_mean, 1.0)
        length_diag = {
            "knows_mean_tokens": knows_mean,
            "doesnt_know_mean_tokens": doesnt_know_mean,
            "length_ratio": length_ratio,
        }
        if length_ratio > 3.0 or length_ratio < 0.33:
            length_diag["warning"] = (
                f"Large length disparity (ratio={length_ratio:.1f}x) — "
                "LDA may be learning generation length rather than epistemic state. "
                "Check which features drive classification."
            )
            logger.warning(f"Set D length confound: knows mean={knows_mean:.0f}, "
                           f"doesn't_know mean={doesnt_know_mean:.0f}, ratio={length_ratio:.1f}x")
        else:
            logger.info(f"Set D lengths: knows={knows_mean:.0f}, doesn't_know={doesnt_know_mean:.0f} "
                         f"(ratio={length_ratio:.1f}x, OK)")

    results = {
        "lda_accuracy": accuracy,
        "n_knows": int(sum(y == 1)),
        "n_doesnt_know": int(sum(y == 0)),
        "cv_method": cv_method,
        "passed": accuracy > 0.7,
        "length_diagnostic": length_diag,
    }

    # Plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for label, name, color in [(1, "Knows", "green"), (0, "Doesn't Know", "red")]:
            mask = y == label
            ax.scatter(X_proj[mask, 0], np.zeros(mask.sum()), c=color, s=100,
                       label=name, alpha=0.7, edgecolors="black")
        ax.set_xlabel("LDA Projection")
        ax.set_title(f"Positive Control: Knows vs Doesn't Know (accuracy={accuracy:.2f})")
        ax.legend()
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(config.figures_dir / "positive_control.png", dpi=150)
        plt.close(fig)
        logger.info("Saved positive_control.png")
    except Exception as e:
        logger.warning(f"Failed to save positive control plot: {e}")

    return results
