#!/usr/bin/env python3
"""Build a consolidated review pack from existing experiment artifacts.

Creates:
  review_pack/samples.parquet  — all samples with prompts, texts, token counts, labels
  review_pack/features.npz     — stacked feature matrix + labels + feature names + tier indices
  review_pack/surface_baseline.json — TF-IDF surface text baseline (confound check)

Usage:
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/build_review_pack.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ExperimentConfig, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REVIEW_DIR = Path(__file__).resolve().parent.parent / "review_pack"


def load_metadata(config: ExperimentConfig) -> list[dict]:
    """Load and return the generations list from metadata.json."""
    with open(config.metadata_path) as f:
        raw = json.load(f)
    return raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw


def build_samples_parquet(metadata: list[dict], output_path: Path) -> None:
    """Consolidate all sample metadata into a single parquet file."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for parquet output. Install with: pip install pandas pyarrow")
        raise

    rows: list[dict[str, Any]] = []
    for m in metadata:
        rows.append({
            "generation_id": m["generation_id"],
            "prompt_set": m["prompt_set"],
            "topic": m["topic"],
            "topic_idx": m["topic_idx"],
            "mode": m["mode"],
            "mode_idx": m["mode_idx"],
            "system_prompt": m["system_prompt"],
            "user_prompt": m["user_prompt"],
            "generated_text": m.get("generated_text", ""),
            "num_generated_tokens": m.get("num_generated_tokens", 0),
            "prompt_length": m.get("prompt_length", 0),
            "seed": m.get("seed", -1),
            "repetition": m.get("repetition", 0),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df)} samples to {output_path}")
    logger.info(f"  Sets: {df['prompt_set'].value_counts().to_dict()}")
    logger.info(f"  Modes: {df['mode'].value_counts().to_dict()}")


def build_features_npz(
    metadata: list[dict],
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    """Stack all per-sample .npz files into a single feature matrix."""
    ab_meta = [m for m in metadata if m["prompt_set"] in ("A", "B")]

    features_list: list[np.ndarray] = []
    labels: list[str] = []
    topics: list[str] = []
    gen_ids: list[int] = []
    feature_names: np.ndarray | None = None

    # Tier boundaries (populated from first sample)
    tier1_list: list[np.ndarray] = []
    tier2_list: list[np.ndarray] = []
    tier2_5_list: list[np.ndarray] = []
    tier3_list: list[np.ndarray] = []

    for m in ab_meta:
        gid = m["generation_id"]
        npz_path = config.signatures_dir / f"gen_{gid:03d}.npz"
        if not npz_path.exists():
            logger.warning(f"Missing: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        if "features" not in data:
            continue

        features_list.append(data["features"].flatten())
        labels.append(m["mode"])
        topics.append(m["topic"])
        gen_ids.append(gid)

        if feature_names is None and "feature_names" in data:
            feature_names = data["feature_names"]

        # Collect tier slices
        for tier_key, tier_list in [
            ("features_tier1", tier1_list),
            ("features_tier2", tier2_list),
            ("features_tier2_5", tier2_5_list),
            ("features_tier3", tier3_list),
        ]:
            if tier_key in data:
                tier_list.append(data[tier_key].flatten())

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    save_kwargs: dict[str, Any] = {
        "features": X,
        "labels": np.array(labels),
        "topics": np.array(topics),
        "generation_ids": np.array(gen_ids),
    }

    if feature_names is not None:
        save_kwargs["feature_names"] = feature_names

    # Stack tier arrays if we collected them for all samples
    n = len(features_list)
    if len(tier1_list) == n:
        save_kwargs["tier1"] = np.stack(tier1_list)
    if len(tier2_list) == n:
        save_kwargs["tier2"] = np.stack(tier2_list)
    if len(tier2_5_list) == n:
        save_kwargs["tier2_5"] = np.stack(tier2_5_list)
    if len(tier3_list) == n:
        save_kwargs["tier3"] = np.stack(tier3_list)

    np.savez_compressed(output_path, **save_kwargs)
    logger.info(f"Saved features.npz: {X.shape[0]} samples x {X.shape[1]} features")
    if "tier1" in save_kwargs:
        logger.info(
            f"  Tier shapes: T1={save_kwargs['tier1'].shape[1]}, "
            f"T2={save_kwargs['tier2'].shape[1]}, "
            f"T2.5={save_kwargs['tier2_5'].shape[1]}, "
            f"T3={save_kwargs.get('tier3', np.empty((0, 0))).shape[1]}"
        )


def run_surface_text_baseline(
    metadata: list[dict],
    output_path: Path,
) -> dict[str, Any]:
    """TF-IDF on output text -> mode classification.

    This is the "dumb baseline" that should fail — if the internal-state
    features are detecting something beyond what's in the surface text,
    the surface baseline must score substantially lower.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder

    ab_meta = [m for m in metadata if m["prompt_set"] in ("A", "B")]

    texts: list[str] = []
    labels: list[str] = []
    for m in ab_meta:
        text = m.get("generated_text", "")
        if not text:
            continue
        texts.append(text)
        labels.append(m["mode"])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    mode_names = le.classes_.tolist()

    n_samples = len(texts)
    n_classes = len(mode_names)
    logger.info(f"Surface baseline: {n_samples} samples, {n_classes} classes")

    results: dict[str, Any] = {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "mode_names": mode_names,
        "description": (
            "TF-IDF (max 5000 features, 1-2 grams) on generated text, "
            "classified with RandomForest (100 trees, 5-fold stratified CV). "
            "This tests whether surface text statistics can explain the mode signal. "
            "If the surface baseline matches or exceeds internal-state RF, "
            "the internal-state signal is redundant with output text."
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Unigram + bigram TF-IDF
    for ngram_label, ngram_range in [("unigram", (1, 1)), ("bigram", (1, 2))]:
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        X_tfidf = tfidf.fit_transform(texts)

        # RF on TF-IDF
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X_tfidf, y, cv=cv, scoring="accuracy")

        results[f"tfidf_{ngram_label}_rf_mean"] = float(rf_scores.mean())
        results[f"tfidf_{ngram_label}_rf_std"] = float(rf_scores.std())
        results[f"tfidf_{ngram_label}_rf_folds"] = [float(s) for s in rf_scores]
        results[f"tfidf_{ngram_label}_n_features"] = X_tfidf.shape[1]

        logger.info(
            f"  TF-IDF ({ngram_label}) + RF: "
            f"{rf_scores.mean():.1%} +/- {rf_scores.std():.1%} "
            f"({X_tfidf.shape[1]} features)"
        )

    # Character n-gram TF-IDF (catches tokenization-level patterns)
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        sublinear_tf=True,
    )
    X_char = tfidf_char.fit_transform(texts)
    rf_char = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    char_scores = cross_val_score(rf_char, X_char, y, cv=cv, scoring="accuracy")

    results["tfidf_char_ngram_rf_mean"] = float(char_scores.mean())
    results["tfidf_char_ngram_rf_std"] = float(char_scores.std())
    results["tfidf_char_ngram_rf_folds"] = [float(s) for s in char_scores]
    results["tfidf_char_ngram_n_features"] = X_char.shape[1]

    logger.info(
        f"  TF-IDF (char 3-5) + RF: "
        f"{char_scores.mean():.1%} +/- {char_scores.std():.1%}"
    )

    # ── 4-way excluding analogical (the hard case) ──
    logger.info("  --- 4-way surface baseline (excluding analogical) ---")
    ab_no_anal = [m for m in ab_meta if m["mode"] != "analogical"]
    texts_4way = [m.get("generated_text", "") for m in ab_no_anal if m.get("generated_text")]
    labels_4way = [m["mode"] for m in ab_no_anal if m.get("generated_text")]

    le_4way = LabelEncoder()
    y_4way = le_4way.fit_transform(labels_4way)
    cv_4way = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ngram_label, ngram_range in [("unigram", (1, 1)), ("bigram", (1, 2))]:
        tfidf_4 = TfidfVectorizer(
            max_features=5000, ngram_range=ngram_range,
            stop_words="english", sublinear_tf=True,
        )
        X_4 = tfidf_4.fit_transform(texts_4way)
        rf_4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores_4 = cross_val_score(rf_4, X_4, y_4way, cv=cv_4way, scoring="accuracy")
        results[f"4way_tfidf_{ngram_label}_rf_mean"] = float(scores_4.mean())
        results[f"4way_tfidf_{ngram_label}_rf_std"] = float(scores_4.std())
        logger.info(
            f"  4-way TF-IDF ({ngram_label}) + RF: "
            f"{scores_4.mean():.1%} +/- {scores_4.std():.1%}"
        )

    results["4way_n_samples"] = len(texts_4way)
    results["4way_modes"] = le_4way.classes_.tolist()

    # ── Per-mode confusion (which modes does TF-IDF identify?) ──
    logger.info("  --- Per-mode TF-IDF confusion ---")
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix as sklearn_cm

    tfidf_full = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 1),
        stop_words="english", sublinear_tf=True,
    )
    X_full = tfidf_full.fit_transform(texts)
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    y_pred_surface = cross_val_predict(rf_full, X_full, y, cv=cv)
    cm = sklearn_cm(y, y_pred_surface)
    per_mode_surface: dict[str, dict[str, Any]] = {}
    for i, mode in enumerate(mode_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_mode_surface[mode] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 3) if total > 0 else 0.0,
        }
        logger.info(f"    {mode}: {correct}/{total} ({correct/total:.0%})")
    results["per_mode_tfidf_accuracy"] = per_mode_surface
    results["tfidf_confusion_matrix"] = cm.tolist()
    results["tfidf_confusion_labels"] = mode_names

    # Comparison to internal-state RF (load from results.json if available)
    results_json_path = OUTPUTS_DIR / "results.json"
    if results_json_path.exists():
        try:
            with open(results_json_path) as f:
                run_results = json.load(f)
            fi = run_results.get("feature_importance", {})
            internal_rf = fi.get("rf_accuracy_mean", fi.get("rf_cv_accuracy"))
            if internal_rf is not None:
                results["internal_state_rf_accuracy"] = float(internal_rf)
                best_surface = max(
                    results.get("tfidf_unigram_rf_mean", 0),
                    results.get("tfidf_bigram_rf_mean", 0),
                    results.get("tfidf_char_ngram_rf_mean", 0),
                )
                results["gap_internal_minus_surface"] = float(internal_rf) - best_surface
                logger.info(
                    f"  Internal-state RF: {internal_rf:.1%}, "
                    f"best surface: {best_surface:.1%}, "
                    f"gap: {float(internal_rf) - best_surface:+.1%}"
                )

            # Also load 4-way internal for comparison
            fourway_path = OUTPUTS_DIR / "4way_no_analogical.json"
            if fourway_path.exists():
                with open(fourway_path) as f:
                    fourway_data = json.load(f)
                results["4way_internal_state_rf"] = fourway_data.get("rf_accuracy_mean")
                best_surface_4 = max(
                    results.get("4way_tfidf_unigram_rf_mean", 0),
                    results.get("4way_tfidf_bigram_rf_mean", 0),
                )
                results["4way_gap_internal_minus_surface"] = (
                    fourway_data.get("rf_accuracy_mean", 0) - best_surface_4
                )
        except Exception as e:
            logger.warning(f"Could not load results.json for comparison: {e}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved surface baseline to {output_path}")

    return results


def main() -> None:
    config = ExperimentConfig()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building review pack from {config.outputs_dir}")
    logger.info(f"Output directory: {REVIEW_DIR}")

    # Load metadata
    metadata = load_metadata(config)
    logger.info(f"Loaded {len(metadata)} generation metadata entries")

    # 1. Consolidated samples
    logger.info("\n=== Building samples.parquet ===")
    try:
        build_samples_parquet(metadata, REVIEW_DIR / "samples.parquet")
    except Exception as e:
        logger.error(f"Failed to build samples.parquet: {e}")

    # 2. Consolidated features
    logger.info("\n=== Building features.npz ===")
    try:
        build_features_npz(metadata, config, REVIEW_DIR / "features.npz")
    except Exception as e:
        logger.error(f"Failed to build features.npz: {e}")

    # 3. Surface text baseline
    logger.info("\n=== Running Surface Text Baseline ===")
    try:
        baseline_results = run_surface_text_baseline(
            metadata, REVIEW_DIR / "surface_baseline.json"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("SURFACE TEXT BASELINE")
        print("=" * 60)
        for key in ["tfidf_unigram_rf_mean", "tfidf_bigram_rf_mean", "tfidf_char_ngram_rf_mean"]:
            if key in baseline_results:
                label = key.replace("tfidf_", "").replace("_rf_mean", "")
                print(f"  TF-IDF ({label}): {baseline_results[key]:.1%}")
        if "internal_state_rf_accuracy" in baseline_results:
            print(f"  Internal-state RF: {baseline_results['internal_state_rf_accuracy']:.1%}")
            print(f"  Gap (internal - best surface): {baseline_results['gap_internal_minus_surface']:+.1%}")
        print()

        best = max(
            baseline_results.get("tfidf_unigram_rf_mean", 0),
            baseline_results.get("tfidf_bigram_rf_mean", 0),
            baseline_results.get("tfidf_char_ngram_rf_mean", 0),
        )
        chance = 1.0 / baseline_results.get("n_classes", 5)
        if best <= chance + 0.10:
            print("CONCLUSION: Surface text features are near chance — mode signal")
            print("            is NOT explained by output text content.")
        elif best < baseline_results.get("internal_state_rf_accuracy", 1.0):
            print("CONCLUSION: Surface text captures some mode signal, but less than")
            print("            internal-state features. Partial confound possible.")
        else:
            print("WARNING: Surface text matches or exceeds internal-state features.")
            print("         Internal-state signal may be redundant with output text.")

    except Exception as e:
        logger.error(f"Surface baseline failed: {e}")

    # 4. Copy protocol.md into review pack for self-contained bundle
    protocol_src = Path(__file__).resolve().parent.parent / "protocol.md"
    if protocol_src.exists():
        protocol_dst = REVIEW_DIR / "protocol.md"
        protocol_dst.write_text(protocol_src.read_text())
        logger.info(f"Copied protocol.md to {protocol_dst}")

    logger.info(f"\nReview pack complete: {REVIEW_DIR}")
    logger.info("Contents:")
    for p in sorted(REVIEW_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
