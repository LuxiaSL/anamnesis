#!/usr/bin/env python3
"""Analyze supplementary generation data: temperature control + prompt-swap control.

Part 1 — Temperature positive control:
  Binary RF: temp_low (0.3) vs temp_high (0.9) — does the pipeline detect
  genuine generation-time computational differences from temperature alone?

Part 2 — Prompt-swap control:
  Socratic system prompt + linear user directive. Compare to pure linear and
  pure socratic to disambiguate Explanation A (mode execution creates signal)
  vs Explanation B (prompt presence creates signal).

Requires supplementary generations from run_supplementary_gens.py.

Usage:
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_supplementary_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUTS_DIR, ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_supplementary_signatures(
    supp_dir: Path,
    supp_metadata: list[dict],
) -> dict[int, dict]:
    """Load supplementary .npz files."""
    signatures: dict[int, dict] = {}
    for m in supp_metadata:
        gid = m["generation_id"]
        sig_path = supp_dir / f"gen_{gid:04d}.npz"
        if not sig_path.exists():
            logger.warning(f"Missing supplementary signature: {sig_path}")
            continue
        data = np.load(sig_path, allow_pickle=True)
        signatures[gid] = {
            "features": data["features"],
            "feature_names": (
                data["feature_names"].tolist() if "feature_names" in data else []
            ),
        }
        for key in data.files:
            if key.startswith("features_tier"):
                signatures[gid][key] = data[key]
    return signatures


def load_main_signatures(
    config: ExperimentConfig,
    main_metadata: list[dict],
    gen_ids: list[int],
) -> dict[int, dict]:
    """Load main run .npz files for specific generation IDs."""
    signatures: dict[int, dict] = {}
    for m in main_metadata:
        gid = m["generation_id"]
        if gid not in gen_ids:
            continue
        sig_path = config.signatures_dir / f"gen_{gid:03d}.npz"
        if not sig_path.exists():
            logger.warning(f"Missing main signature: {sig_path}")
            continue
        data = np.load(sig_path, allow_pickle=True)
        signatures[gid] = {
            "features": data["features"],
            "feature_names": (
                data["feature_names"].tolist() if "feature_names" in data else []
            ),
        }
        for key in data.files:
            if key.startswith("features_tier"):
                signatures[gid][key] = data[key]
    return signatures


def binary_rf_analysis(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    description: str,
) -> dict[str, Any]:
    """Run binary RF + LDA with 5-fold CV and LOO cross-check.

    Returns dict with accuracy, confidence info, per-fold scores.
    """
    results: dict[str, Any] = {"description": description, "n_samples": len(y)}

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    n_per_class = [int(np.sum(y == c)) for c in sorted(set(y))]
    results["n_per_class"] = n_per_class
    results["label_names"] = label_names

    # 5-fold stratified CV (or fewer folds if samples are very limited)
    min_class_size = min(n_per_class)
    n_folds = min(5, min_class_size)
    if n_folds < 2:
        logger.warning(f"  {description}: too few samples for CV (min class = {min_class_size})")
        results["error"] = "Too few samples for cross-validation"
        return results

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    results["rf_accuracy_mean"] = float(rf_scores.mean())
    results["rf_accuracy_std"] = float(rf_scores.std())
    results["rf_fold_scores"] = rf_scores.tolist()
    results["rf_n_folds"] = n_folds
    logger.info(f"  {description} RF: {rf_scores.mean():.1%} +/- {rf_scores.std():.1%} ({n_folds}-fold)")

    # LDA
    try:
        lda = LinearDiscriminantAnalysis()
        lda_scores = cross_val_score(lda, X_scaled, y, cv=cv, scoring="accuracy")
        results["lda_accuracy_mean"] = float(lda_scores.mean())
        results["lda_accuracy_std"] = float(lda_scores.std())
        logger.info(f"  {description} LDA: {lda_scores.mean():.1%} +/- {lda_scores.std():.1%}")
    except Exception as e:
        logger.warning(f"  {description} LDA failed: {e}")
        results["lda_accuracy_mean"] = None

    # LOO-CV cross-check (more stable for small samples)
    try:
        loo = LeaveOneOut()
        rf_loo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        loo_scores = cross_val_score(rf_loo, X_scaled, y, cv=loo, scoring="accuracy")
        results["loo_accuracy"] = float(loo_scores.mean())
        results["loo_n_correct"] = int(loo_scores.sum())
        results["loo_n_total"] = len(loo_scores)
        logger.info(
            f"  {description} LOO: {loo_scores.mean():.1%} "
            f"({int(loo_scores.sum())}/{len(loo_scores)})"
        )
    except Exception as e:
        logger.warning(f"  {description} LOO failed: {e}")
        results["loo_accuracy"] = None

    return results


def tier_ablation(
    signatures_by_condition: dict[str, list[dict]],
    y: np.ndarray,
    description: str,
) -> dict[str, Any]:
    """Per-tier RF accuracy for a binary comparison."""
    tier_keys = ["features_tier1", "features_tier2", "features_tier2_5"]
    results: dict[str, Any] = {}

    n_per_class = [int(np.sum(y == c)) for c in sorted(set(y))]
    min_class_size = min(n_per_class)
    n_folds = min(5, min_class_size)
    if n_folds < 2:
        return {"error": "Too few samples"}

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for tier_key in tier_keys:
        tier_vecs: list[np.ndarray] = []
        has_tier = True
        for condition_name, sig_list in signatures_by_condition.items():
            for sig in sig_list:
                if tier_key in sig:
                    tier_vecs.append(sig[tier_key])
                else:
                    has_tier = False
                    break
            if not has_tier:
                break

        if not has_tier or len(tier_vecs) != len(y):
            results[tier_key] = {"error": "Missing tier data"}
            continue

        tier_mat = np.nan_to_num(np.stack(tier_vecs), nan=0.0)
        tier_scaled = StandardScaler().fit_transform(tier_mat)

        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, tier_scaled, y, cv=cv, scoring="accuracy")
            results[tier_key] = {
                "n_features": tier_mat.shape[1],
                "rf_accuracy": float(scores.mean()),
                "rf_std": float(scores.std()),
            }
            logger.info(f"    {tier_key}: {scores.mean():.1%}")
        except Exception as e:
            results[tier_key] = {"error": str(e)}

    return results


def analyze_temperature_control(
    supp_signatures: dict[int, dict],
    supp_metadata: list[dict],
    main_signatures: dict[int, dict],
    main_metadata: list[dict],
) -> dict[str, Any]:
    """Part 1: Temperature positive control analysis."""
    logger.info("=== Temperature Positive Control ===")
    results: dict[str, Any] = {}

    # Split supplementary into temp_low and temp_high
    temp_low_meta = [m for m in supp_metadata if m.get("condition") == "temp_low"]
    temp_high_meta = [m for m in supp_metadata if m.get("condition") == "temp_high"]

    temp_low_features = [
        supp_signatures[m["generation_id"]]["features"]
        for m in temp_low_meta
        if m["generation_id"] in supp_signatures
    ]
    temp_high_features = [
        supp_signatures[m["generation_id"]]["features"]
        for m in temp_high_meta
        if m["generation_id"] in supp_signatures
    ]

    logger.info(f"temp_low: {len(temp_low_features)} samples, temp_high: {len(temp_high_features)} samples")

    if not temp_low_features or not temp_high_features:
        return {"error": "Missing temperature condition data"}

    # Binary: temp_low vs temp_high
    X_temp = np.stack(temp_low_features + temp_high_features)
    y_temp = np.array([0] * len(temp_low_features) + [1] * len(temp_high_features))

    results["binary_low_vs_high"] = binary_rf_analysis(
        X_temp, y_temp,
        label_names=["temp_0.3", "temp_0.9"],
        description="temp_low vs temp_high",
    )

    # Per-tier ablation
    low_sigs = [supp_signatures[m["generation_id"]] for m in temp_low_meta if m["generation_id"] in supp_signatures]
    high_sigs = [supp_signatures[m["generation_id"]] for m in temp_high_meta if m["generation_id"] in supp_signatures]
    results["tier_ablation_low_vs_high"] = tier_ablation(
        {"temp_low": low_sigs, "temp_high": high_sigs},
        y_temp,
        "temp binary tier ablation",
    )

    # 3-way: temp_low + temp_high + existing linear at temp=0.7
    set_a_linear = [
        m for m in main_metadata
        if m["prompt_set"] == "A" and m["mode"] == "linear"
    ]
    linear_07_features = [
        main_signatures[m["generation_id"]]["features"]
        for m in set_a_linear
        if m["generation_id"] in main_signatures
    ]
    logger.info(f"Existing linear (temp=0.7): {len(linear_07_features)} samples")

    if linear_07_features:
        X_3way = np.stack(temp_low_features + linear_07_features + temp_high_features)
        y_3way = np.array(
            [0] * len(temp_low_features)
            + [1] * len(linear_07_features)
            + [2] * len(temp_high_features)
        )
        results["3way_temp"] = binary_rf_analysis(
            X_3way, y_3way,
            label_names=["temp_0.3", "temp_0.7", "temp_0.9"],
            description="3-way temperature",
        )

    # Length comparison (diagnostic)
    results["length_diagnostic"] = {
        "temp_low_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in temp_low_meta
        ])),
        "temp_high_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in temp_high_meta
        ])),
        "linear_07_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in set_a_linear
        ])) if set_a_linear else None,
    }

    return results


def analyze_prompt_swap(
    supp_signatures: dict[int, dict],
    supp_metadata: list[dict],
    main_signatures: dict[int, dict],
    main_metadata: list[dict],
) -> dict[str, Any]:
    """Part 2: Prompt-swap control analysis."""
    logger.info("=== Prompt-Swap Control ===")
    results: dict[str, Any] = {}

    # Prompt-swap signatures
    swap_meta = [m for m in supp_metadata if m.get("condition") == "prompt_swap"]
    swap_features = [
        supp_signatures[m["generation_id"]]["features"]
        for m in swap_meta
        if m["generation_id"] in supp_signatures
    ]
    logger.info(f"prompt_swap: {len(swap_features)} samples")

    if not swap_features:
        return {"error": "Missing prompt_swap data"}

    # Existing Set A linear
    set_a_linear = [
        m for m in main_metadata
        if m["prompt_set"] == "A" and m["mode"] == "linear"
    ]
    linear_features = [
        main_signatures[m["generation_id"]]["features"]
        for m in set_a_linear
        if m["generation_id"] in main_signatures
    ]

    # Existing Set A socratic
    set_a_socratic = [
        m for m in main_metadata
        if m["prompt_set"] == "A" and m["mode"] == "socratic"
    ]
    socratic_features = [
        main_signatures[m["generation_id"]]["features"]
        for m in set_a_socratic
        if m["generation_id"] in main_signatures
    ]

    logger.info(f"Set A linear: {len(linear_features)}, Set A socratic: {len(socratic_features)}")

    if not linear_features or not socratic_features:
        return {"error": "Missing main run Set A linear/socratic data"}

    # Binary 1: prompt_swap vs pure_linear
    # ~50% → Explanation A (mode execution): swap looks like linear
    # ~65%+ → Explanation B (prompt detection): swap is distinct from linear
    X_swap_vs_lin = np.stack(swap_features + linear_features)
    y_swap_vs_lin = np.array([0] * len(swap_features) + [1] * len(linear_features))
    results["swap_vs_linear"] = binary_rf_analysis(
        X_swap_vs_lin, y_swap_vs_lin,
        label_names=["prompt_swap", "pure_linear"],
        description="prompt_swap vs linear",
    )

    # Binary 2: prompt_swap vs pure_socratic
    X_swap_vs_soc = np.stack(swap_features + socratic_features)
    y_swap_vs_soc = np.array([0] * len(swap_features) + [1] * len(socratic_features))
    results["swap_vs_socratic"] = binary_rf_analysis(
        X_swap_vs_soc, y_swap_vs_soc,
        label_names=["prompt_swap", "pure_socratic"],
        description="prompt_swap vs socratic",
    )

    # 3-way: socratic vs prompt_swap vs linear
    X_3way = np.stack(socratic_features + swap_features + linear_features)
    y_3way = np.array(
        [0] * len(socratic_features)
        + [1] * len(swap_features)
        + [2] * len(linear_features)
    )
    results["3way_swap"] = binary_rf_analysis(
        X_3way, y_3way,
        label_names=["socratic", "prompt_swap", "linear"],
        description="3-way socratic/swap/linear",
    )

    # Per-tier ablation for swap vs linear (the key comparison)
    swap_sigs = [supp_signatures[m["generation_id"]] for m in swap_meta if m["generation_id"] in supp_signatures]
    linear_sigs = [main_signatures[m["generation_id"]] for m in set_a_linear if m["generation_id"] in main_signatures]
    results["tier_ablation_swap_vs_linear"] = tier_ablation(
        {"prompt_swap": swap_sigs, "linear": linear_sigs},
        y_swap_vs_lin,
        "swap vs linear tier ablation",
    )

    # Interpretation helper
    swap_vs_lin_acc = results["swap_vs_linear"].get("rf_accuracy_mean", 0.5)
    swap_vs_soc_acc = results["swap_vs_socratic"].get("rf_accuracy_mean", 0.5)
    if swap_vs_lin_acc < 0.55:
        interpretation = (
            "EXPLANATION A (mode execution): prompt_swap is computationally "
            "identical to linear. The system prompt's mode is overridden by "
            "the user directive. Signal comes from actual execution, not "
            "prompt presence."
        )
    elif swap_vs_lin_acc >= 0.65:
        interpretation = (
            "EXPLANATION B (prompt detection): prompt_swap is distinct from "
            "linear. The system prompt's presence creates signal even when "
            "the user overrides the mode. Some signal may come from prompt "
            "routing, not just execution."
        )
    else:
        interpretation = (
            "AMBIGUOUS: prompt_swap vs linear accuracy is in the 55-65% zone. "
            "Cannot clearly distinguish between explanations A and B."
        )

    results["interpretation"] = interpretation
    results["swap_vs_linear_summary"] = {
        "accuracy": swap_vs_lin_acc,
        "explanation": "A" if swap_vs_lin_acc < 0.55 else "B" if swap_vs_lin_acc >= 0.65 else "ambiguous",
    }

    # Length diagnostic
    results["length_diagnostic"] = {
        "swap_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in swap_meta
        ])),
        "linear_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in set_a_linear
        ])),
        "socratic_mean_tokens": float(np.mean([
            m.get("num_generated_tokens", 0) for m in set_a_socratic
        ])),
    }

    return results


def main() -> None:
    config = ExperimentConfig()

    # Load supplementary data
    supp_dir = OUTPUTS_DIR / "supplementary"
    supp_meta_path = OUTPUTS_DIR / "supplementary_metadata.json"

    if not supp_meta_path.exists():
        logger.error(f"Supplementary metadata not found at {supp_meta_path}")
        logger.error("Run run_supplementary_gens.py first.")
        sys.exit(1)

    with open(supp_meta_path) as f:
        supp_raw = json.load(f)
    supp_metadata: list[dict] = supp_raw["generations"]

    logger.info(f"Loaded {len(supp_metadata)} supplementary metadata entries")
    supp_signatures = load_supplementary_signatures(supp_dir, supp_metadata)
    logger.info(f"Loaded {len(supp_signatures)} supplementary signatures")

    # Load main run metadata
    with open(config.metadata_path) as f:
        main_raw = json.load(f)
    main_metadata: list[dict] = main_raw["generations"]

    # Figure out which main gen IDs we need (Set A linear + socratic)
    needed_ids = [
        m["generation_id"]
        for m in main_metadata
        if m["prompt_set"] == "A" and m["mode"] in ("linear", "socratic")
    ]
    main_signatures = load_main_signatures(config, main_metadata, needed_ids)
    logger.info(f"Loaded {len(main_signatures)} main run signatures")

    all_results: dict[str, Any] = {}

    # Part 1: Temperature control
    all_results["temperature_control"] = analyze_temperature_control(
        supp_signatures, supp_metadata, main_signatures, main_metadata,
    )

    # Part 2: Prompt-swap control
    all_results["prompt_swap_control"] = analyze_prompt_swap(
        supp_signatures, supp_metadata, main_signatures, main_metadata,
    )

    # Save results
    output_path = OUTPUTS_DIR / "supplementary_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    logger.info(f"Saved results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY ANALYSIS RESULTS")
    print("=" * 60)

    print("\n--- Temperature Positive Control ---")
    tc = all_results["temperature_control"]
    if "error" not in tc:
        blh = tc.get("binary_low_vs_high", {})
        print(f"temp_low vs temp_high RF: {blh.get('rf_accuracy_mean', 'N/A'):.1%}" if isinstance(blh.get('rf_accuracy_mean'), float) else f"temp_low vs temp_high: {blh.get('error', 'N/A')}")
        if blh.get("loo_accuracy") is not None:
            print(f"  LOO cross-check: {blh['loo_accuracy']:.1%} ({blh['loo_n_correct']}/{blh['loo_n_total']})")
        if blh.get("lda_accuracy_mean") is not None:
            print(f"  LDA: {blh['lda_accuracy_mean']:.1%}")

        if "tier_ablation_low_vs_high" in tc:
            print("  Tier ablation:")
            for tier, vals in tc["tier_ablation_low_vs_high"].items():
                if isinstance(vals, dict) and "rf_accuracy" in vals:
                    print(f"    {tier}: {vals['rf_accuracy']:.1%}")

        if "3way_temp" in tc:
            t3 = tc["3way_temp"]
            print(f"  3-way (0.3/0.7/0.9) RF: {t3.get('rf_accuracy_mean', 'N/A'):.1%}" if isinstance(t3.get('rf_accuracy_mean'), float) else "")

        if "length_diagnostic" in tc:
            ld = tc["length_diagnostic"]
            print(f"  Lengths: low={ld['temp_low_mean_tokens']:.0f}, high={ld['temp_high_mean_tokens']:.0f}, linear_0.7={ld.get('linear_07_mean_tokens', 'N/A')}")
    else:
        print(f"  Error: {tc['error']}")

    print("\n--- Prompt-Swap Control ---")
    ps = all_results["prompt_swap_control"]
    if "error" not in ps:
        svl = ps.get("swap_vs_linear", {})
        svs = ps.get("swap_vs_socratic", {})
        s3w = ps.get("3way_swap", {})

        if isinstance(svl.get("rf_accuracy_mean"), float):
            print(f"prompt_swap vs linear RF: {svl['rf_accuracy_mean']:.1%}")
        if svl.get("loo_accuracy") is not None:
            print(f"  LOO cross-check: {svl['loo_accuracy']:.1%}")
        if isinstance(svs.get("rf_accuracy_mean"), float):
            print(f"prompt_swap vs socratic RF: {svs['rf_accuracy_mean']:.1%}")
        if isinstance(s3w.get("rf_accuracy_mean"), float):
            print(f"3-way (socratic/swap/linear) RF: {s3w['rf_accuracy_mean']:.1%}")

        if "tier_ablation_swap_vs_linear" in ps:
            print("  Tier ablation (swap vs linear):")
            for tier, vals in ps["tier_ablation_swap_vs_linear"].items():
                if isinstance(vals, dict) and "rf_accuracy" in vals:
                    print(f"    {tier}: {vals['rf_accuracy']:.1%}")

        if "interpretation" in ps:
            print(f"\nInterpretation: {ps['interpretation']}")
        if "swap_vs_linear_summary" in ps:
            summary = ps["swap_vs_linear_summary"]
            print(f"Explanation: {summary['explanation']} (accuracy: {summary['accuracy']:.1%})")
    else:
        print(f"  Error: {ps['error']}")


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
