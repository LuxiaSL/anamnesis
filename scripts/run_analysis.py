#!/usr/bin/env python3
"""Run the full analysis pipeline on extracted signatures.

Steps:
  1. Load all signatures and metadata
  2. Noise floor estimation (Set C)
  3. Positive control validation (Set D)
  4. Semantic vs computational distance correlation (Mantel test)
  5. Mode clustering visualization (UMAP/t-SNE)
  6. Retrieval comparison
  7. Feature importance analysis
  8. Generate all figures
  9. Write results.json with GO/NO-GO recommendation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_all_signatures(config: ExperimentConfig) -> tuple[dict, list[dict]]:
    """Load all .npz signatures and their metadata.

    Returns:
        (signatures_by_id, metadata_list)
        signatures_by_id: {gen_id: {features, feature_names, knnlm_baseline, tier slices}}
        metadata_list: list of metadata dicts
    """
    metadata_path = config.metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata found at {metadata_path}. Run the experiment first.")

    with open(metadata_path) as f:
        master = json.load(f)

    signatures: dict[int, dict] = {}
    metadata_list: list[dict] = master["generations"]

    for meta in metadata_list:
        gen_id = meta["generation_id"]
        npz_path = config.signatures_dir / f"gen_{gen_id:03d}.npz"
        if not npz_path.exists():
            logger.warning(f"Missing signature file: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        signatures[gen_id] = {
            "features": data["features"],
            "feature_names": data["feature_names"].tolist() if "feature_names" in data else [],
            "knnlm_baseline": data["knnlm_baseline"] if "knnlm_baseline" in data else None,
        }
        # Load per-tier features
        for key in data.files:
            if key.startswith("features_tier"):
                signatures[gen_id][key] = data[key]

    logger.info(f"Loaded {len(signatures)} signatures, {len(metadata_list)} metadata entries")
    return signatures, metadata_list


def main() -> None:
    config = ExperimentConfig()

    logger.info("Loading signatures...")
    signatures, metadata_list = load_all_signatures(config)

    if len(signatures) == 0:
        logger.error("No signatures found. Run the experiment first.")
        return

    # Import analysis modules
    from analysis.noise_floor import analyze_noise_floor
    from analysis.distance_matrices import analyze_distances
    from analysis.clustering import analyze_clustering
    from analysis.retrieval_test import analyze_retrieval
    from analysis.feature_importance import analyze_feature_importance
    from analysis.deep_dive import run_deep_dive

    results: dict = {"experiment": {"total_generations": len(signatures)}}

    # ── Step 0a: Verify calibration ──
    logger.info("=== Calibration Verification ===")
    try:
        from analysis.positional_decomp import verify_calibration
        calib_results = verify_calibration(config)
        results["calibration"] = calib_results
        if calib_results.get("error"):
            logger.warning(f"Calibration issue: {calib_results['error']}")
        else:
            logger.info(
                f"Calibration OK: shape={calib_results['shape']}, "
                f"nonzero_fraction={calib_results.get('nonzero_fraction', 0):.3f}"
            )
    except Exception as e:
        logger.warning(f"Calibration verification failed: {e}")
        results["calibration"] = {"error": str(e)}

    # ── Step 0b: Per-mode generation length diagnostic ──
    logger.info("=== Generation Length by Mode ===")
    try:
        length_results = _analyze_generation_lengths(metadata_list, config)
        results["generation_lengths"] = length_results
    except Exception as e:
        logger.warning(f"Generation length analysis failed: {e}")
        results["generation_lengths"] = {"error": str(e)}

    # ── Step 1: Noise Floor (Set C) ──
    logger.info("=== Noise Floor Analysis (Set C) ===")
    try:
        noise_results = analyze_noise_floor(signatures, metadata_list, config)
        results["noise_floor"] = noise_results
        logger.info(f"Noise floor: mean within-condition distance = {noise_results.get('mean_within_distance', 'N/A')}")
    except Exception as e:
        logger.error(f"Noise floor analysis failed: {e}", exc_info=True)
        results["noise_floor"] = {"error": str(e)}

    # ── Step 2: Positive Control (Set D) ──
    logger.info("=== Positive Control (Set D) ===")
    try:
        from analysis.noise_floor import validate_positive_control
        pc_results = validate_positive_control(signatures, metadata_list, config)
        results["positive_control"] = pc_results
        logger.info(f"Positive control: separation = {pc_results.get('lda_accuracy', 'N/A')}")
    except Exception as e:
        logger.error(f"Positive control failed: {e}", exc_info=True)
        results["positive_control"] = {"error": str(e)}

    # ── Step 3: Distance Correlation (Mantel test) ──
    logger.info("=== Distance Matrix Analysis ===")
    try:
        dist_results = analyze_distances(signatures, metadata_list, config)
        results["distances"] = dist_results
        logger.info(f"Mantel r = {dist_results.get('mantel_r', 'N/A')}, p = {dist_results.get('mantel_p', 'N/A')}")
    except Exception as e:
        logger.error(f"Distance analysis failed: {e}", exc_info=True)
        results["distances"] = {"error": str(e)}

    # ── Step 4: Clustering ──
    logger.info("=== Clustering Analysis ===")
    try:
        cluster_results = analyze_clustering(signatures, metadata_list, config)
        results["clustering"] = cluster_results
        logger.info(f"Mode silhouette = {cluster_results.get('mode_silhouette', 'N/A')}")
        logger.info(f"Topic silhouette = {cluster_results.get('topic_silhouette', 'N/A')}")
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}", exc_info=True)
        results["clustering"] = {"error": str(e)}

    # ── Step 5: Retrieval ──
    logger.info("=== Retrieval Analysis ===")
    try:
        retrieval_results = analyze_retrieval(signatures, metadata_list, config)
        results["retrieval"] = retrieval_results
    except Exception as e:
        logger.error(f"Retrieval analysis failed: {e}", exc_info=True)
        results["retrieval"] = {"error": str(e)}

    # ── Step 6: Feature Importance ──
    logger.info("=== Feature Importance Analysis ===")
    try:
        importance_results = analyze_feature_importance(signatures, metadata_list, config)
        results["feature_importance"] = importance_results
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}", exc_info=True)
        results["feature_importance"] = {"error": str(e)}

    # ── Step 7: Deep Dive Analyses ──
    logger.info("=== Deep Dive Analyses ===")
    try:
        deep_dive_results = run_deep_dive(
            signatures, metadata_list, config,
            noise_floor_results=results.get("noise_floor"),
            feature_importance_results=results.get("feature_importance"),
        )
        results["deep_dive"] = deep_dive_results
    except Exception as e:
        logger.error(f"Deep dive analysis failed: {e}", exc_info=True)
        results["deep_dive"] = {"error": str(e)}

    # ── GO/NO-GO Decision ──
    results["decision"] = make_decision(results)

    # Save results
    results_path = config.results_path
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info(f"Results saved to {results_path}")

    # Print decision
    decision = results["decision"]
    logger.info("=" * 60)
    logger.info(f"DECISION: {decision['verdict']}")
    logger.info(f"Justification: {decision['justification']}")
    logger.info("=" * 60)


def _analyze_generation_lengths(
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Report token count distribution by mode as a pre-analysis diagnostic.

    Flags if any mode's mean length differs from the overall mean by >2x,
    since temporal features scale with generation length.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only use Sets A + B (mode-labeled, not noise floor or positive control)
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]
    if not ab_meta:
        return {"error": "No Set A+B generations found"}

    mode_lengths: dict[str, list[int]] = {}
    for m in ab_meta:
        mode = m["mode"]
        n_tokens = m.get("num_generated_tokens", 0)
        mode_lengths.setdefault(mode, []).append(n_tokens)

    per_mode: dict[str, dict[str, float]] = {}
    all_lengths: list[int] = []
    for mode, lengths in sorted(mode_lengths.items()):
        arr = np.array(lengths, dtype=np.float64)
        all_lengths.extend(lengths)
        per_mode[mode] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "median": float(np.median(arr)),
            "n": len(lengths),
        }
        logger.info(
            f"  {mode}: mean={arr.mean():.0f}, std={arr.std():.0f}, "
            f"range=[{int(arr.min())}, {int(arr.max())}]"
        )

    overall_mean = float(np.mean(all_lengths))
    overall_std = float(np.std(all_lengths))

    # Flag modes with mean length >2x or <0.5x the overall mean
    length_warnings: list[str] = []
    for mode, stats in per_mode.items():
        ratio = stats["mean"] / max(overall_mean, 1.0)
        if ratio > 2.0 or ratio < 0.5:
            warning = (
                f"Mode '{mode}' mean length ({stats['mean']:.0f}) is "
                f"{ratio:.1f}x the overall mean ({overall_mean:.0f}) "
                "— temporal features may be length-confounded"
            )
            length_warnings.append(warning)
            logger.warning(warning)

    # Plot histogram by mode
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        modes_sorted = sorted(mode_lengths.keys())
        for mode in modes_sorted:
            ax.hist(
                mode_lengths[mode], bins=20, alpha=0.5,
                label=f"{mode} (mean={per_mode[mode]['mean']:.0f})",
            )
        ax.set_xlabel("Number of Generated Tokens")
        ax.set_ylabel("Count")
        ax.set_title("Generation Length Distribution by Mode")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.figures_dir / "generation_lengths.png", dpi=150)
        plt.close(fig)
        logger.info("Saved generation_lengths.png")
    except Exception as e:
        logger.warning(f"Failed to save generation length plot: {e}")

    return {
        "per_mode": per_mode,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "length_warnings": length_warnings,
    }


def make_decision(results: dict) -> dict:
    """Apply the decision framework from the spec."""
    verdict = "UNKNOWN"
    justification_parts: list[str] = []

    clustering = results.get("clustering", {})
    distances = results.get("distances", {})
    noise = results.get("noise_floor", {})
    positive = results.get("positive_control", {})

    mode_sil = clustering.get("mode_silhouette")
    topic_sil = clustering.get("topic_silhouette")
    mantel_r = distances.get("mantel_r")

    # Check positive control first — if it fails, no verdict is trustworthy
    pc_passed = True
    if positive.get("error"):
        justification_parts.append(f"Positive control failed: {positive['error']}")
        pc_passed = False
    elif positive.get("lda_accuracy", 0) < 0.7:
        justification_parts.append(
            f"WARNING: Positive control weak (accuracy={positive.get('lda_accuracy', 0):.2f}) "
            "— extraction pipeline may have issues"
        )
        pc_passed = False

    # Apply decision framework
    if not pc_passed:
        verdict = "BLOCKED"
        justification_parts.append(
            "Positive control did not pass — cannot trust GO/NO-GO metrics. "
            "Debug extraction pipeline before interpreting results."
        )
    elif mode_sil is not None and mantel_r is not None:
        if mode_sil > 0.3 and (topic_sil is None or mode_sil > topic_sil) and mantel_r < 0.7:
            verdict = "STRONG GO"
            justification_parts.append(
                f"Mode silhouette ({mode_sil:.3f}) > 0.3 and > topic silhouette ({topic_sil}), "
                f"Mantel r ({mantel_r:.3f}) < 0.7"
            )
        elif (mode_sil > 0.1 and mantel_r < 0.85
              and (topic_sil is None or mode_sil >= topic_sil * 0.5)):
            # WEAK GO requires BOTH moderate silhouette AND non-redundant distance,
            # and mode clustering shouldn't be dwarfed by topic clustering
            verdict = "WEAK GO"
            justification_parts.append(
                f"Mode silhouette ({mode_sil:.3f}) > 0.1, Mantel r ({mantel_r:.3f}) < 0.85"
            )
        elif mode_sil < 0.1 and mantel_r > 0.85:
            verdict = "NO-GO"
            justification_parts.append(
                f"Mode silhouette ({mode_sil:.3f}) < 0.1, Mantel r ({mantel_r:.3f}) > 0.85"
            )
        else:
            verdict = "INCONCLUSIVE"
            justification_parts.append(
                f"Mode silhouette ({mode_sil:.3f}), Mantel r ({mantel_r:.3f}) — mixed signals"
            )
    else:
        verdict = "INCOMPLETE"
        justification_parts.append("Missing clustering or distance metrics")

    return {
        "verdict": verdict,
        "justification": "; ".join(justification_parts),
        "mode_silhouette": mode_sil,
        "topic_silhouette": topic_sil,
        "mantel_r": mantel_r,
        "positive_control_passed": pc_passed,
    }


def _json_default(obj: object) -> object:
    """JSON serialization fallback."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
