"""Feature importance analysis for mode classification.

Train simple classifiers to predict processing mode from features,
rank feature importance, and perform tier ablation.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import ExperimentConfig, MODE_INDEX

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run feature importance and tier ablation analysis.

    Uses Sets A + B for mode classification.
    """
    # Filter to Sets A + B
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]

    gen_ids = []
    features_list = []
    mode_labels = []
    feature_names = None

    for m in ab_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        gen_ids.append(gid)
        features_list.append(signatures[gid]["features"])
        mode_labels.append(m["mode"])
        if feature_names is None:
            feature_names = signatures[gid].get("feature_names", [])

    n = len(gen_ids)
    if n < 10:
        return {"error": f"Too few generations: {n}"}

    X = np.stack(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(mode_labels)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: dict[str, Any] = {"n_generations": n, "n_features": X.shape[1]}

    # Compute cv_folds once, outside try blocks
    cv_folds = min(5, n // max(len(set(y)), 1))
    cv_folds = max(cv_folds, 2)

    # ── Random Forest classification ──
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X_scaled, y, cv=cv_folds, scoring="accuracy")
        results["rf_accuracy_mean"] = float(rf_scores.mean())
        results["rf_accuracy_std"] = float(rf_scores.std())
        logger.info(f"Random Forest accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

        # Fit on full data for feature importances
        rf.fit(X_scaled, y)
        importances = rf.feature_importances_

        # Top 20 features
        top_indices = np.argsort(importances)[::-1][:20]
        top_features = []
        for idx in top_indices:
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
            top_features.append({
                "name": name,
                "importance": float(importances[idx]),
                "index": int(idx),
            })
        results["top_20_features"] = top_features
    except Exception as e:
        logger.warning(f"Random Forest failed: {e}")

    # ── Mode confusion matrix (RF cross-validated predictions) ──
    try:
        rf_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        y_pred = cross_val_predict(rf_cv, X_scaled, y, cv=cv_folds)
        cm = confusion_matrix(y, y_pred)
        mode_names = le.classes_.tolist()
        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_labels"] = mode_names

        # Log which mode pairs get most confused
        confused_pairs: list[dict[str, Any]] = []
        for i in range(len(mode_names)):
            for j in range(len(mode_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        "true": mode_names[i],
                        "predicted": mode_names[j],
                        "count": int(cm[i, j]),
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)
        results["top_confusions"] = confused_pairs[:10]
        if confused_pairs:
            logger.info(f"Top confusion: {confused_pairs[0]['true']} → "
                         f"{confused_pairs[0]['predicted']} ({confused_pairs[0]['count']} times)")
    except Exception as e:
        logger.warning(f"Confusion matrix failed: {e}")

    # ── Logistic Regression ──
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
        lr_scores = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring="accuracy")
        results["lr_accuracy_mean"] = float(lr_scores.mean())
        results["lr_accuracy_std"] = float(lr_scores.std())
        logger.info(f"Logistic Regression accuracy: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    except Exception as e:
        logger.warning(f"Logistic Regression failed: {e}")

    # ── Tier Ablation ──
    # Test each tier in isolation AND test removal of each tier (leave-one-tier-out)
    tier_ablation: dict[str, dict] = {}
    tier_keys = ["features_tier1", "features_tier2", "features_tier2_5"]

    # First, collect all tier data and build tier slice indices
    tier_data: dict[str, np.ndarray] = {}
    for tier_key in tier_keys:
        tier_vecs = []
        has_tier = True
        for gid in gen_ids:
            sig = signatures[gid]
            if tier_key in sig:
                tier_vecs.append(sig[tier_key])
            else:
                has_tier = False
                break
        if has_tier and tier_vecs:
            tier_data[tier_key] = np.nan_to_num(np.stack(tier_vecs), nan=0.0)

    for tier_key in tier_keys:
        if tier_key not in tier_data:
            continue

        tier_mat = tier_data[tier_key]
        tier_scaled = StandardScaler().fit_transform(tier_mat)

        # Test this tier ALONE
        try:
            tier_sil = float(silhouette_score(tier_scaled, y, metric="cosine"))
        except Exception:
            tier_sil = None

        try:
            tier_rf = RandomForestClassifier(n_estimators=50, random_state=42)
            tier_scores = cross_val_score(tier_rf, tier_scaled, y, cv=cv_folds, scoring="accuracy")
            tier_acc = float(tier_scores.mean())
        except Exception:
            tier_acc = None

        # Test with this tier REMOVED (leave-one-tier-out)
        other_tiers = [k for k in tier_keys if k != tier_key and k in tier_data]
        if other_tiers:
            other_mat = np.hstack([tier_data[k] for k in other_tiers])
            other_scaled = StandardScaler().fit_transform(other_mat)
            try:
                without_sil = float(silhouette_score(other_scaled, y, metric="cosine"))
            except Exception:
                without_sil = None
            try:
                other_rf = RandomForestClassifier(n_estimators=50, random_state=42)
                other_scores = cross_val_score(other_rf, other_scaled, y, cv=cv_folds, scoring="accuracy")
                without_acc = float(other_scores.mean())
            except Exception:
                without_acc = None
        else:
            without_sil = None
            without_acc = None

        tier_ablation[tier_key] = {
            "n_features": tier_mat.shape[1],
            "silhouette_alone": tier_sil,
            "rf_accuracy_alone": tier_acc,
            "silhouette_without": without_sil,
            "rf_accuracy_without": without_acc,
        }
        logger.info(f"{tier_key}: {tier_mat.shape[1]} features, "
                     f"alone: sil={tier_sil}, acc={tier_acc} | "
                     f"without: sil={without_sil}, acc={without_acc}")

    results["tier_ablation"] = tier_ablation

    # ── Feature importance plot ──
    try:
        if "top_20_features" in results:
            top = results["top_20_features"]
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            names = [f["name"] for f in top]
            imps = [f["importance"] for f in top]
            y_pos = range(len(names))
            ax.barh(y_pos, imps, color="steelblue", edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Random Forest Feature Importance")
            ax.set_title("Top 20 Most Important Features for Mode Classification")
            fig.tight_layout()
            fig.savefig(config.figures_dir / "feature_importance.png", dpi=150)
            plt.close(fig)
            logger.info("Saved feature_importance.png")
    except Exception as e:
        logger.warning(f"Failed to save feature importance plot: {e}")

    # ── Confusion matrix plot ──
    try:
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"])
            mode_names = results["confusion_matrix_labels"]
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))

            # Normalize by row (true label) for percentages
            cm_norm = cm.astype(np.float64)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_pct = cm_norm / row_sums

            im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(mode_names)))
            ax.set_yticks(range(len(mode_names)))
            ax.set_xticklabels(mode_names, rotation=45, ha="right")
            ax.set_yticklabels(mode_names)
            ax.set_xlabel("Predicted Mode")
            ax.set_ylabel("True Mode")
            ax.set_title("Mode Classification Confusion Matrix (RF, CV)")

            # Annotate cells with count and percentage
            for i in range(len(mode_names)):
                for j in range(len(mode_names)):
                    text_color = "white" if cm_pct[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0%})",
                            ha="center", va="center", color=text_color, fontsize=9)

            fig.colorbar(im, ax=ax, label="Proportion")
            fig.tight_layout()
            fig.savefig(config.figures_dir / "confusion_matrix.png", dpi=150)
            plt.close(fig)
            logger.info("Saved confusion_matrix.png")
    except Exception as e:
        logger.warning(f"Failed to save confusion matrix plot: {e}")

    # ── Tier ablation plot ──
    try:
        if tier_ablation:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            tier_names = list(tier_ablation.keys())
            sils = [tier_ablation[t].get("silhouette_alone", 0) or 0 for t in tier_names]
            accs = [tier_ablation[t].get("rf_accuracy_alone", 0) or 0 for t in tier_names]

            # Add full signature
            full_sil = results.get("mode_silhouette", 0)
            # Use the clustering module's result if available
            tier_names_display = [t.replace("features_", "") for t in tier_names]

            axes[0].bar(range(len(tier_names_display)), sils, color="steelblue", edgecolor="black")
            axes[0].set_xticks(range(len(tier_names_display)))
            axes[0].set_xticklabels(tier_names_display, rotation=45, ha="right")
            axes[0].set_ylabel("Silhouette Score (Mode)")
            axes[0].set_title("Clustering Quality per Tier")

            axes[1].bar(range(len(tier_names_display)), accs, color="coral", edgecolor="black")
            axes[1].set_xticks(range(len(tier_names_display)))
            axes[1].set_xticklabels(tier_names_display, rotation=45, ha="right")
            axes[1].set_ylabel("RF Classification Accuracy")
            axes[1].set_title("Mode Classification per Tier")
            # Chance line
            n_modes = len(set(y))
            axes[1].axhline(1.0 / n_modes, color="red", linestyle="--", alpha=0.5)

            fig.tight_layout()
            fig.savefig(config.figures_dir / "tier_ablation.png", dpi=150)
            plt.close(fig)
            logger.info("Saved tier_ablation.png")
    except Exception as e:
        logger.warning(f"Failed to save tier ablation plot: {e}")

    return results
