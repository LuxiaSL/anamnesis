#!/usr/bin/env python3
"""4-way analysis excluding a specified mode (default: analogical).

Runs RF classification, silhouette, Mantel, retrieval, tier ablation,
and pairwise discriminability on the remaining 4 modes. Designed for
Run 4 to assess how much analogical's 100% accuracy inflates the 5-way
average.

Usage:
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_4way_no_analogical.py
    ANAMNESIS_RUN_NAME=run4_format_controlled python scripts/run_4way_no_analogical.py --exclude-mode contrastive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUTS_DIR, ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[dict, list[dict]]:
    """Load signatures and metadata for the current run."""
    config = ExperimentConfig()
    metadata_path = config.metadata_path
    signatures_dir = config.signatures_dir

    with open(metadata_path) as f:
        raw = json.load(f)
    all_metadata: list[dict] = (
        raw["generations"] if isinstance(raw, dict) and "generations" in raw else raw
    )

    signatures: dict[int, dict] = {}
    for m in all_metadata:
        gid = m["generation_id"]
        sig_path = signatures_dir / f"gen_{gid:03d}.npz"
        if sig_path.exists():
            data = np.load(sig_path, allow_pickle=True)
            signatures[gid] = {
                "features": data["features"],
                "feature_names": (
                    data.get("feature_names", np.array([])).tolist()
                    if "feature_names" in data
                    else []
                ),
            }
            for tier_key in ["features_tier1", "features_tier2", "features_tier2_5"]:
                if tier_key in data:
                    signatures[gid][tier_key] = data[tier_key]
            if "knnlm_baseline" in data:
                signatures[gid]["knnlm_baseline"] = data["knnlm_baseline"]

    return signatures, all_metadata


def load_5way_results() -> dict | None:
    """Load the full 5-way results.json for comparison."""
    results_path = OUTPUTS_DIR / "results.json"
    if not results_path.exists():
        logger.warning(f"No 5-way results found at {results_path}")
        return None
    try:
        with open(results_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load 5-way results: {e}")
        return None


def pairwise_discriminability(
    X_scaled: np.ndarray,
    modes_arr: np.ndarray,
) -> dict:
    """Train binary classifiers for each mode pair and report accuracy."""
    unique_modes = sorted(set(modes_arr))
    n_modes = len(unique_modes)
    discrim_matrix = np.full((n_modes, n_modes), np.nan)
    pair_results: dict[str, float] = {}

    for i, m1 in enumerate(unique_modes):
        discrim_matrix[i, i] = 1.0
        for j, m2 in enumerate(unique_modes):
            if j <= i:
                continue
            mask = (modes_arr == m1) | (modes_arr == m2)
            X_pair = X_scaled[mask]
            y_pair = (modes_arr[mask] == m1).astype(int)

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

    return {
        "matrix": discrim_matrix.tolist(),
        "labels": unique_modes,
        "pairs": pair_results,
    }


def run_4way_analysis(exclude_mode: str) -> dict:
    """Run the full analysis excluding the specified mode."""
    signatures, all_metadata = load_data()

    # Filter to Sets A+B, excluding the specified mode
    ab_meta = [
        m
        for m in all_metadata
        if m["prompt_set"] in ("A", "B") and m["mode"] != exclude_mode
    ]

    gen_ids: list[int] = []
    features_list: list[np.ndarray] = []
    mode_labels: list[str] = []
    topic_strings: list[str] = []
    feature_names: list[str] | None = None

    for m in ab_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        gen_ids.append(gid)
        features_list.append(signatures[gid]["features"])
        mode_labels.append(m["mode"])
        topic_strings.append(m["topic"])
        if feature_names is None:
            feature_names = signatures[gid].get("feature_names", [])

    n = len(gen_ids)
    logger.info(f"4-way analysis (excluding {exclude_mode}): {n} samples, modes: {sorted(set(mode_labels))}")

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(mode_labels)
    mode_names = le.classes_.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: dict = {
        "exclude_mode": exclude_mode,
        "n_samples": n,
        "n_features": X.shape[1],
        "modes": mode_names,
    }

    # -- RF classification (5-fold stratified CV) --
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    results["rf_accuracy_mean"] = float(rf_scores.mean())
    results["rf_accuracy_std"] = float(rf_scores.std())
    logger.info(f"RF accuracy (4-way): {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

    # Confusion matrix
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    y_pred = cross_val_predict(rf_cv, X_scaled, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    results["confusion_matrix"] = cm.tolist()
    results["confusion_matrix_labels"] = mode_names
    logger.info(f"Confusion matrix:\n{cm}")

    # Per-mode accuracy
    per_mode_acc: dict[str, dict] = {}
    for i, mode in enumerate(mode_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_mode_acc[mode] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total > 0 else 0.0,
        }
    results["per_mode_accuracy"] = per_mode_acc

    # -- Silhouette scores --
    mode_sil = float(silhouette_score(X_scaled, y, metric="cosine"))
    results["mode_silhouette"] = mode_sil
    logger.info(f"Mode silhouette (4-way): {mode_sil:.4f}")

    # Topic silhouette
    unique_topic_strings = sorted(set(topic_strings))
    topic_str_to_idx = {t: i for i, t in enumerate(unique_topic_strings)}
    topics_arr = np.array([topic_str_to_idx[t] for t in topic_strings])
    try:
        topic_sil = float(silhouette_score(X_scaled, topics_arr, metric="cosine"))
        results["topic_silhouette"] = topic_sil
        logger.info(f"Topic silhouette (4-way): {topic_sil:.4f}")
    except Exception as e:
        logger.warning(f"Topic silhouette failed: {e}")
        results["topic_silhouette"] = None

    # -- Mantel test (computational vs semantic distance) --
    try:
        from scipy.stats import pearsonr
        from sentence_transformers import SentenceTransformer

        comp_dist = squareform(pdist(X_scaled, metric="cosine"))
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")

        prompts = [m["user_prompt"] for m in ab_meta if m["generation_id"] in set(gen_ids)]
        prompt_emb = sem_model.encode(prompts, normalize_embeddings=True)
        prompt_dist = squareform(pdist(prompt_emb, metric="cosine"))

        texts = [m["generated_text"] for m in ab_meta if m["generation_id"] in set(gen_ids)]
        text_emb = sem_model.encode(texts, normalize_embeddings=True)
        text_dist = squareform(pdist(text_emb, metric="cosine"))

        idx = np.triu_indices(n, k=1)
        r_prompt, _p_prompt = pearsonr(comp_dist[idx], prompt_dist[idx])
        r_text, _p_text = pearsonr(comp_dist[idx], text_dist[idx])
        results["mantel_r_prompt"] = float(r_prompt)
        results["mantel_r_text"] = float(r_text)
        logger.info(f"Mantel r (prompt): {r_prompt:.4f}, (text): {r_text:.4f}")
    except Exception as e:
        logger.warning(f"Mantel test failed (likely missing sentence-transformers): {e}")

    # -- Tier ablation --
    tier_results: dict[str, dict] = {}
    tier_keys = ["features_tier1", "features_tier2", "features_tier2_5"]

    for tier_key in tier_keys:
        tier_vecs: list[np.ndarray] = []
        has_tier = True
        for gid in gen_ids:
            sig = signatures[gid]
            if tier_key in sig:
                tier_vecs.append(sig[tier_key])
            else:
                has_tier = False
                break

        if not has_tier:
            continue

        tier_mat = np.nan_to_num(np.stack(tier_vecs), nan=0.0)
        tier_scaled = StandardScaler().fit_transform(tier_mat)

        try:
            tier_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            tier_scores = cross_val_score(tier_rf, tier_scaled, y, cv=cv, scoring="accuracy")
            tier_acc = float(tier_scores.mean())
        except Exception:
            tier_acc = None

        try:
            tier_sil = float(silhouette_score(tier_scaled, y, metric="cosine"))
        except Exception:
            tier_sil = None

        tier_results[tier_key] = {
            "n_features": tier_mat.shape[1],
            "rf_accuracy": tier_acc,
            "silhouette": tier_sil,
        }
        if tier_acc is not None:
            logger.info(f"{tier_key} alone (4-way): {tier_acc:.4f}")

    results["tier_ablation"] = tier_results

    # -- Engineered features only (T1+T2+T2.5, no PCA) --
    engineered_tiers = ["features_tier1", "features_tier2", "features_tier2_5"]
    eng_vecs: list[np.ndarray] = []
    for gid in gen_ids:
        parts = [signatures[gid][tk] for tk in engineered_tiers if tk in signatures[gid]]
        if parts:
            eng_vecs.append(np.concatenate(parts))

    if len(eng_vecs) == n:
        eng_mat = np.nan_to_num(np.stack(eng_vecs), nan=0.0)
        eng_scaled = StandardScaler().fit_transform(eng_mat)
        eng_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        eng_scores = cross_val_score(eng_rf, eng_scaled, y, cv=cv, scoring="accuracy")
        results["engineered_only_rf"] = float(eng_scores.mean())
        logger.info(f"Engineered only (T1+T2+T2.5, 4-way): {eng_scores.mean():.4f}")

    # -- Length-only baseline --
    lengths: list[int] = []
    for m in ab_meta:
        if m["generation_id"] in set(gen_ids):
            lengths.append(m.get("num_generated_tokens", len(m.get("generated_text", "").split())))
    if len(lengths) == n:
        X_len = np.array(lengths).reshape(-1, 1)
        len_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        len_scores = cross_val_score(len_rf, X_len, y, cv=cv, scoring="accuracy")
        results["length_only_rf"] = float(len_scores.mean())
        logger.info(f"Length-only baseline (4-way): {len_scores.mean():.4f}")

    # -- Retrieval comparison --
    try:
        from sentence_transformers import SentenceTransformer as ST2

        k = 5
        comp_dist_full = cdist(X_scaled, X_scaled, metric="cosine")
        modes_arr_np = np.array(mode_labels)

        mode_matches = 0
        total_ret = 0
        for i in range(n):
            dists = comp_dist_full[i].copy()
            dists[i] = np.inf
            nn_indices = np.argsort(dists)[:k]
            matches = (modes_arr_np[nn_indices] == modes_arr_np[i]).sum()
            mode_matches += matches
            total_ret += k
        results["retrieval_computational"] = float(mode_matches / total_ret)
        logger.info(f"Retrieval (computational, 4-way): {mode_matches / total_ret:.4f}")

        sem_model2 = ST2("all-MiniLM-L6-v2")
        texts_for_ret = [m["generated_text"] for m in ab_meta if m["generation_id"] in set(gen_ids)]
        sem_emb = sem_model2.encode(texts_for_ret, normalize_embeddings=True)
        sem_dist_full = cdist(sem_emb, sem_emb, metric="cosine")

        mode_matches_sem = 0
        for i in range(n):
            dists = sem_dist_full[i].copy()
            dists[i] = np.inf
            nn_indices = np.argsort(dists)[:k]
            matches = (modes_arr_np[nn_indices] == modes_arr_np[i]).sum()
            mode_matches_sem += matches
        results["retrieval_semantic"] = float(mode_matches_sem / total_ret)
        logger.info(f"Retrieval (semantic, 4-way): {mode_matches_sem / total_ret:.4f}")
    except Exception as e:
        logger.warning(f"Retrieval analysis failed: {e}")

    # -- Pairwise discriminability --
    logger.info("=== Pairwise Discriminability ===")
    modes_arr_np = np.array(mode_labels)
    pw_results = pairwise_discriminability(X_scaled, modes_arr_np)
    results["pairwise_discriminability"] = pw_results

    return results


def main() -> None:
    """Run 4-way analysis and save results."""
    parser = argparse.ArgumentParser(description="4-way analysis excluding a mode")
    parser.add_argument(
        "--exclude-mode",
        type=str,
        default="analogical",
        help="Mode to exclude from analysis (default: analogical)",
    )
    args = parser.parse_args()

    exclude_mode = args.exclude_mode
    results = run_4way_analysis(exclude_mode)

    # Save
    output_path = OUTPUTS_DIR / f"4way_no_{exclude_mode}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"4-WAY ANALYSIS (excluding {exclude_mode})")
    print("=" * 60)
    print(f"Samples: {results['n_samples']} ({results['modes']})")
    print(f"Features: {results['n_features']}")
    print()
    print(f"RF accuracy:      {results['rf_accuracy_mean']:.1%} +/- {results['rf_accuracy_std']:.1%}")
    print(f"Mode silhouette:  {results['mode_silhouette']:.4f}")
    if results.get("topic_silhouette") is not None:
        print(f"Topic silhouette: {results['topic_silhouette']:.4f}")
    if "mantel_r_prompt" in results:
        print(f"Mantel r (prompt): {results['mantel_r_prompt']:.4f}")
        print(f"Mantel r (text):   {results['mantel_r_text']:.4f}")
    print()

    # Confusion matrix
    print("Confusion matrix:")
    labels = results["confusion_matrix_labels"]
    cm = results["confusion_matrix"]
    header = "             " + "  ".join(f"{lbl[:5]:>5}" for lbl in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        correct = row[i]
        total = sum(row)
        print(f"{labels[i]:>12s}  {row_str}   {correct}/{total}")
    print()

    # Per-mode accuracy
    print("Per-mode accuracy:")
    for mode, stats in results["per_mode_accuracy"].items():
        print(f"  {mode}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    print()

    # Tier ablation
    if "tier_ablation" in results:
        print("Tier ablation (4-way):")
        for tier, vals in results["tier_ablation"].items():
            acc = vals.get("rf_accuracy")
            print(f"  {tier}: {acc:.1%}" if acc else f"  {tier}: N/A")
    if "engineered_only_rf" in results:
        print(f"  T1+T2+T2.5 combined: {results['engineered_only_rf']:.1%}")
    if "length_only_rf" in results:
        print(f"  Length only: {results['length_only_rf']:.1%}")
    print()

    # Pairwise discriminability
    if "pairwise_discriminability" in results:
        pw = results["pairwise_discriminability"]
        print("Pairwise discriminability:")
        for pair_key, acc in sorted(pw["pairs"].items()):
            print(f"  {pair_key}: {acc:.1%}")
    print()

    # Retrieval
    if "retrieval_computational" in results:
        print(f"Retrieval (computational): {results['retrieval_computational']:.1%}")
    if "retrieval_semantic" in results:
        print(f"Retrieval (semantic):      {results['retrieval_semantic']:.1%}")

    # Dynamic comparison to 5-way
    print()
    five_way = load_5way_results()
    if five_way:
        # Try to find RF accuracy from different possible locations
        five_rf = None
        five_sil = None
        if "feature_importance" in five_way:
            fi = five_way["feature_importance"]
            five_rf = fi.get("rf_accuracy_mean", fi.get("rf_cv_accuracy"))
        if "clustering" in five_way:
            five_sil = five_way["clustering"].get("mode_silhouette")
        # Also check deep_dive for pairwise results
        if "deep_dive" in five_way and "pairwise_discriminability" in five_way["deep_dive"]:
            five_pw = five_way["deep_dive"]["pairwise_discriminability"]
            print("5-way pairwise discriminability (from results.json):")
            if "pairs" in five_pw:
                for pair_key, acc in sorted(five_pw["pairs"].items()):
                    print(f"  {pair_key}: {acc:.1%}")
            print()

        print(f"Comparison to 5-way (all modes):")
        if five_rf is not None:
            print(f"  5-way RF:  {five_rf:.1%}")
        else:
            print(f"  5-way RF:  (not found in results.json)")
        print(f"  4-way RF:  {results['rf_accuracy_mean']:.1%}")
        if five_sil is not None:
            print(f"  5-way sil: {five_sil:.4f}")
        else:
            print(f"  5-way sil: (not found in results.json)")
        print(f"  4-way sil: {results['mode_silhouette']:.4f}")
    else:
        print("Comparison to 5-way: results.json not found — run run_analysis.py first")


if __name__ == "__main__":
    main()
