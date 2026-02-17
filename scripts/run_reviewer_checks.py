#!/usr/bin/env python3
"""Reviewer-requested validation checks for Phase 0.5 results.

Two analyses:
1. GroupKFold by topic (topic-heldout): Train on 16 topics, test on 4 unseen topics.
   Tests whether MLP learns "mode" vs "topic x mode" interaction artifacts.
2. Linear probe on learned embedding: After MLP projection, can a linear classifier
   separate modes? If yes, the MLP "unwraps" the nonlinear manifold into something
   linearly separable — mechanistic support for the "directions vs manifolds" framing.

Both run on existing data, no GPU needed. Expected runtime: ~5-10 minutes on CPU.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reuse ProjectionNet and training infrastructure from Exp 1
# ---------------------------------------------------------------------------


class ProjectionNet(nn.Module):
    """Contrastive projection network: input -> hidden -> bottleneck -> L2-norm."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return nn.functional.normalize(out, p=2, dim=1)


def mine_triplets(
    labels: np.ndarray, rng: np.random.RandomState, n_triplets: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully random triplet mining."""
    anchors, positives, negatives = [], [], []
    for _ in range(n_triplets):
        anchor_idx = rng.randint(len(labels))
        anchor_label = labels[anchor_idx]
        same_class = np.where(labels == anchor_label)[0]
        same_class = same_class[same_class != anchor_idx]
        if len(same_class) == 0:
            continue
        pos_idx = rng.choice(same_class)
        diff_class = np.where(labels != anchor_label)[0]
        neg_idx = rng.choice(diff_class)
        anchors.append(anchor_idx)
        positives.append(pos_idx)
        negatives.append(neg_idx)
    return np.array(anchors), np.array(positives), np.array(negatives)


def _train_projection_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
    torch_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, float, nn.Module]:
    """Train a projection MLP and return embeddings + model.

    Returns:
        (emb_train, emb_eval_or_None, final_loss, model)
    """
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32) if X_eval is not None else None

    model = ProjectionNet(X_train.shape[1], 256, bottleneck_dim, dropout)
    optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    model.train()
    for _epoch in range(n_epochs):
        a_idx, p_idx, n_idx = mine_triplets(y_train, rng, n_triplets=300)
        if len(a_idx) < 10:
            continue
        emb = model(X_train_t)
        loss = triplet_loss_fn(emb[a_idx], emb[p_idx], emb[n_idx])
        optimizer_obj.zero_grad()
        loss.backward()
        optimizer_obj.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        emb_train = model(X_train_t).numpy()
        emb_eval = model(X_eval_t).numpy() if X_eval_t is not None else None

    return emb_train, emb_eval, final_loss, model


# ---------------------------------------------------------------------------
# Analysis 1: GroupKFold by topic
# ---------------------------------------------------------------------------


def run_topic_heldout(
    X: np.ndarray,
    y: np.ndarray,
    topics: np.ndarray,
    mode_names: list[str],
    tier_label: str = "T2+T2.5",
) -> dict[str, Any]:
    """GroupKFold by topic: train on N-k topics, test on k unseen topics.

    With 20 topics, 5 groups of 4 topics each = each fold holds out 4 topics (20 samples).
    Training set: 16 topics (80 samples). Identical sample count to StratifiedKFold.
    """
    logger.info(f"=== Topic-heldout GroupKFold ({tier_label}) ===")

    # Encode topics as group IDs
    unique_topics = np.unique(topics)
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_groups = 5  # 20 topics / 5 = 4 topics per fold
    gkf = GroupKFold(n_splits=n_groups)

    fold_sils: list[float] = []
    fold_knns: list[float] = []
    fold_details: list[dict[str, Any]] = []
    all_y_test: list[int] = []
    all_y_pred: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        test_topics = np.unique(topics[test_idx])

        emb_train, emb_test, final_loss, _model = _train_projection_mlp(
            X_train, y_train, X_test,
            torch_seed=42 + fold_i,
        )

        sil_train = sil_test = float("nan")
        if len(np.unique(y_test)) > 1:
            sil_test = float(silhouette_score(emb_test, y_test, metric="cosine"))
            fold_sils.append(sil_test)
        if len(np.unique(y_train)) > 1:
            sil_train = float(silhouette_score(emb_train, y_train, metric="cosine"))

        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        knn_acc = float(knn.score(emb_test, y_test))
        fold_knns.append(knn_acc)

        preds = knn.predict(emb_test)
        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())

        fold_info = {
            "fold": fold_i,
            "heldout_topics": test_topics.tolist(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "sil_train": sil_train,
            "sil_test": sil_test,
            "knn_accuracy": knn_acc,
            "final_loss": final_loss,
        }
        fold_details.append(fold_info)
        logger.info(
            f"  Fold {fold_i}: topics={test_topics.tolist()}, "
            f"sil_test={sil_test:.4f}, kNN={knn_acc:.2%}, loss={final_loss:.6f}"
        )

    mean_sil = float(np.mean(fold_sils)) if fold_sils else 0.0
    mean_knn = float(np.mean(fold_knns))
    std_sil = float(np.std(fold_sils)) if fold_sils else 0.0
    std_knn = float(np.std(fold_knns))

    # Overall confusion matrix
    cm = sk_confusion_matrix(all_y_test, all_y_pred, labels=list(range(len(mode_names))))
    per_mode_recall = {}
    for i, name in enumerate(mode_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_mode_recall[name] = {
            "correct": correct,
            "total": total,
            "recall": correct / total if total > 0 else 0.0,
        }

    results = {
        "tier_label": tier_label,
        "n_folds": n_groups,
        "n_topics_per_fold": 4,
        "n_train_per_fold": 80,
        "n_test_per_fold": 20,
        "mean_sil_test": mean_sil,
        "std_sil_test": std_sil,
        "mean_knn": mean_knn,
        "std_knn": std_knn,
        "fold_details": fold_details,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": mode_names,
        "per_mode_recall": per_mode_recall,
    }

    logger.info(f"  Topic-heldout {tier_label}: sil={mean_sil:.4f} +/- {std_sil:.4f}, "
                f"kNN={mean_knn:.2%} +/- {std_knn:.2%}")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Linear probe on learned embedding
# ---------------------------------------------------------------------------


def run_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    mode_names: list[str],
    tier_label: str = "T2+T2.5",
) -> dict[str, Any]:
    """After MLP projection, test if a linear classifier works on the 32-d embedding.

    If the MLP genuinely "unwraps" a nonlinear manifold, the projected embeddings
    should be more linearly separable than the raw features. Compare:
    - LDA on raw features (Phase 0.5 Exp 1: sil = -0.045)
    - LogisticRegression on MLP embeddings (this test)
    - LDA on MLP embeddings (this test)
    """
    logger.info(f"=== Linear probe on MLP embedding ({tier_label}) ===")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Track results for three conditions
    conditions = {
        "logreg_on_embedding": {"sils": [], "accs": []},
        "lda_on_embedding": {"sils": [], "accs": []},
        "lda_on_raw": {"sils": [], "accs": []},
        "knn_on_embedding": {"sils": [], "accs": []},
    }

    all_test_info: list[dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # Train MLP, get embeddings
        emb_train, emb_test, final_loss, _model = _train_projection_mlp(
            X_train, y_train, X_test,
            torch_seed=42 + fold_i,
        )

        fold_info: dict[str, Any] = {"fold": fold_i}

        # --- Condition 1: LogisticRegression on MLP embedding ---
        lr_clf = LogisticRegression(max_iter=1000, random_state=42)
        lr_clf.fit(emb_train, y_train)
        lr_acc = float(lr_clf.score(emb_test, y_test))
        lr_preds = lr_clf.predict(emb_test)
        # Use predicted labels for "classification accuracy" but silhouette
        # on the embeddings with TRUE labels to measure geometric quality
        if len(np.unique(y_test)) > 1:
            lr_sil = float(silhouette_score(emb_test, y_test, metric="cosine"))
        else:
            lr_sil = float("nan")
        conditions["logreg_on_embedding"]["sils"].append(lr_sil)
        conditions["logreg_on_embedding"]["accs"].append(lr_acc)
        fold_info["logreg_acc"] = lr_acc
        fold_info["logreg_sil"] = lr_sil

        # --- Condition 2: LDA on MLP embedding ---
        n_components = min(emb_train.shape[1], len(np.unique(y_train)) - 1)
        try:
            lda_emb = LinearDiscriminantAnalysis(n_components=n_components)
            lda_emb_train = lda_emb.fit_transform(emb_train, y_train)
            lda_emb_test = lda_emb.transform(emb_test)
            lda_emb_acc = float(lda_emb.score(emb_test, y_test))
            if len(np.unique(y_test)) > 1:
                lda_emb_sil = float(silhouette_score(lda_emb_test, y_test, metric="cosine"))
            else:
                lda_emb_sil = float("nan")
        except Exception as e:
            logger.warning(f"  LDA on embedding failed fold {fold_i}: {e}")
            lda_emb_acc = float("nan")
            lda_emb_sil = float("nan")
        conditions["lda_on_embedding"]["sils"].append(lda_emb_sil)
        conditions["lda_on_embedding"]["accs"].append(lda_emb_acc)
        fold_info["lda_emb_acc"] = lda_emb_acc
        fold_info["lda_emb_sil"] = lda_emb_sil

        # --- Condition 3: LDA on raw features (baseline, for comparison) ---
        n_components_raw = min(X_train.shape[1], len(np.unique(y_train)) - 1)
        try:
            lda_raw = LinearDiscriminantAnalysis(n_components=n_components_raw)
            lda_raw_train = lda_raw.fit_transform(X_train, y_train)
            lda_raw_test = lda_raw.transform(X_test)
            lda_raw_acc = float(lda_raw.score(X_test, y_test))
            if len(np.unique(y_test)) > 1:
                lda_raw_sil = float(silhouette_score(lda_raw_test, y_test, metric="cosine"))
            else:
                lda_raw_sil = float("nan")
        except Exception as e:
            logger.warning(f"  LDA on raw failed fold {fold_i}: {e}")
            lda_raw_acc = float("nan")
            lda_raw_sil = float("nan")
        conditions["lda_on_raw"]["sils"].append(lda_raw_sil)
        conditions["lda_on_raw"]["accs"].append(lda_raw_acc)
        fold_info["lda_raw_acc"] = lda_raw_acc
        fold_info["lda_raw_sil"] = lda_raw_sil

        # --- Condition 4: kNN on embedding (the standard Exp 1 metric) ---
        knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        knn.fit(emb_train, y_train)
        knn_acc = float(knn.score(emb_test, y_test))
        conditions["knn_on_embedding"]["sils"].append(lr_sil)  # same embedding sil
        conditions["knn_on_embedding"]["accs"].append(knn_acc)
        fold_info["knn_acc"] = knn_acc

        all_test_info.append(fold_info)

        logger.info(
            f"  Fold {fold_i}: "
            f"LogReg={lr_acc:.2%}, LDA(emb)={lda_emb_acc:.2%}, "
            f"LDA(raw)={lda_raw_acc:.2%}, kNN={knn_acc:.2%}, "
            f"sil(emb)={lr_sil:.4f}, sil(LDA_emb)={lda_emb_sil:.4f}, "
            f"sil(LDA_raw)={lda_raw_sil:.4f}"
        )

    # Summarize
    summary: dict[str, Any] = {}
    for cond_name, cond_data in conditions.items():
        valid_sils = [s for s in cond_data["sils"] if not np.isnan(s)]
        valid_accs = [a for a in cond_data["accs"] if not np.isnan(a)]
        summary[cond_name] = {
            "mean_accuracy": float(np.mean(valid_accs)) if valid_accs else None,
            "std_accuracy": float(np.std(valid_accs)) if valid_accs else None,
            "mean_silhouette": float(np.mean(valid_sils)) if valid_sils else None,
            "std_silhouette": float(np.std(valid_sils)) if valid_sils else None,
        }

    results = {
        "tier_label": tier_label,
        "summary": summary,
        "fold_details": all_test_info,
        "interpretation": (
            "If LogReg/LDA on the MLP embedding substantially outperform LDA on raw features, "
            "the MLP performs a genuine geometric transformation ('unwrapping') that makes the "
            "nonlinear manifold linearly accessible. This is mechanistic evidence that the "
            "'directions vs manifolds' framing is correct."
        ),
    }

    logger.info("  === Linear probe summary ===")
    for cond_name, s in summary.items():
        if s["mean_accuracy"] is not None:
            logger.info(
                f"    {cond_name}: acc={s['mean_accuracy']:.2%} +/- {s['std_accuracy']:.2%}, "
                f"sil={s['mean_silhouette']:.4f} +/- {s['std_silhouette']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not TORCH_AVAILABLE:
        logger.error("PyTorch required. Install with: pip install torch")
        sys.exit(1)

    start_time = time.time()

    # Load data
    data = np.load(PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True)
    X_full = data["features"].astype(np.float64)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    labels_str = data["labels"]
    topics = data["topics"]

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    # Pre-split tiers
    X_t2 = data["tier2"].astype(np.float64)
    X_t25 = data["tier2_5"].astype(np.float64)
    X_t2_t25 = np.hstack([X_t2, X_t25])
    X_t2 = np.nan_to_num(X_t2, nan=0.0, posinf=0.0, neginf=0.0)
    X_t25 = np.nan_to_num(X_t25, nan=0.0, posinf=0.0, neginf=0.0)
    X_t2_t25 = np.nan_to_num(X_t2_t25, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Loaded: {X_full.shape[0]} samples, {X_full.shape[1]} features, "
                f"{len(mode_names)} modes, {len(np.unique(topics))} topics")
    logger.info(f"T2+T2.5: {X_t2_t25.shape[1]} features")

    output_dir = PROJECT_ROOT / "outputs" / "phase1" / "reviewer_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(X_full.shape[0]),
        "n_topics": int(len(np.unique(topics))),
        "n_modes": len(mode_names),
        "mode_names": mode_names,
    }

    # -----------------------------------------------------------------------
    # Analysis 1: Topic-heldout GroupKFold
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: Topic-heldout GroupKFold")
    logger.info("=" * 70)

    # Run on T2+T2.5 (the headline combination)
    topic_t2t25 = run_topic_heldout(X_t2_t25, y, topics, mode_names, tier_label="T2+T2.5")
    all_results["topic_heldout_T2_T25"] = topic_t2t25

    # Run on full features for comparison
    topic_full = run_topic_heldout(X_full, y, topics, mode_names, tier_label="combined")
    all_results["topic_heldout_combined"] = topic_full

    # Cross-reference summary
    logger.info("\n  --- Topic-heldout vs StratifiedKFold comparison ---")
    logger.info(f"  StratifiedKFold T2+T2.5: kNN=73%, sil=0.292 (from Exp 1.75)")
    logger.info(f"  Topic-heldout  T2+T2.5: kNN={topic_t2t25['mean_knn']:.2%}, "
                f"sil={topic_t2t25['mean_sil_test']:.4f}")
    logger.info(f"  StratifiedKFold combined: kNN=63%, sil=0.176 (from Exp 1.75)")
    logger.info(f"  Topic-heldout  combined: kNN={topic_full['mean_knn']:.2%}, "
                f"sil={topic_full['mean_sil_test']:.4f}")

    # -----------------------------------------------------------------------
    # Analysis 2: Linear probe on MLP embedding
    # -----------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: Linear probe on MLP embedding")
    logger.info("=" * 70)

    # Run on T2+T2.5
    probe_t2t25 = run_linear_probe(X_t2_t25, y, mode_names, tier_label="T2+T2.5")
    all_results["linear_probe_T2_T25"] = probe_t2t25

    # Run on full features for comparison
    probe_full = run_linear_probe(X_full, y, mode_names, tier_label="combined")
    all_results["linear_probe_combined"] = probe_full

    # Cross-reference summary
    logger.info("\n  --- Linear probe key comparison ---")
    logger.info(f"  Exp 1 LDA on raw 1837 features: sil=-0.045, acc=48%")
    lda_raw_s = probe_full["summary"]["lda_on_raw"]
    lda_emb_s = probe_full["summary"]["lda_on_embedding"]
    lr_emb_s = probe_full["summary"]["logreg_on_embedding"]
    logger.info(f"  LDA on raw (this run):           sil={lda_raw_s['mean_silhouette']:.4f}, "
                f"acc={lda_raw_s['mean_accuracy']:.2%}")
    logger.info(f"  LDA on MLP embedding:            sil={lda_emb_s['mean_silhouette']:.4f}, "
                f"acc={lda_emb_s['mean_accuracy']:.2%}")
    logger.info(f"  LogReg on MLP embedding:         acc={lr_emb_s['mean_accuracy']:.2%}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    elapsed = time.time() - start_time
    all_results["elapsed_seconds"] = elapsed

    output_path = output_dir / "reviewer_checks.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
