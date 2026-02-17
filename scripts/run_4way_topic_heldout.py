#!/usr/bin/env python3
"""4-way topic-heldout (no analogical) on T2+T2.5.

Filters out analogical, runs GroupKFold by topic on remaining 80 samples
(4 modes x 20 topics). Chance = 25%. Expected runtime: ~30 seconds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import silhouette_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ProjectionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), p=2, dim=1)


def mine_triplets(
    labels: np.ndarray, rng: np.random.RandomState, n_triplets: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    torch_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32) if X_eval is not None else None

    model = ProjectionNet(X_train.shape[1], 256, bottleneck_dim, drop=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    final_loss = 0.0
    model.train()
    for _ in range(n_epochs):
        a, p, n = mine_triplets(y_train, rng)
        if len(a) < 10:
            continue
        emb = model(X_train_t)
        loss = loss_fn(emb[a], emb[p], emb[n])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        emb_train = model(X_train_t).numpy()
        emb_eval = model(X_eval_t).numpy() if X_eval_t is not None else None

    return emb_train, emb_eval, final_loss


def main() -> None:
    start_time = time.time()

    # Load data
    data = np.load(PROJECT_ROOT / "review_pack" / "features.npz", allow_pickle=True)
    labels_str = data["labels"]
    topics = data["topics"]

    # Build T2+T2.5
    X_t2 = np.nan_to_num(data["tier2"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_t25 = np.nan_to_num(data["tier2_5"].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.hstack([X_t2, X_t25])

    # Filter out analogical
    keep_mask = labels_str != "analogical"
    X = X[keep_mask]
    labels_str = labels_str[keep_mask]
    topics = topics[keep_mask]

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    mode_names = le.classes_.tolist()

    unique_topics = np.unique(topics)
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    logger.info(f"4-way (no analogical): {X.shape[0]} samples, {X.shape[1]} features, "
                f"modes={mode_names}, {len(unique_topics)} topics, chance=25%")

    # GroupKFold: 5 folds, 4 topics held out per fold (16 test samples), 64 train
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

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

        emb_train, emb_test, final_loss = _train_projection_mlp(
            X_train, y_train, X_test, torch_seed=42 + fold_i,
        )

        sil_train = float(silhouette_score(emb_train, y_train, metric="cosine"))
        sil_test = float("nan")
        if len(np.unique(y_test)) > 1:
            sil_test = float(silhouette_score(emb_test, y_test, metric="cosine"))
            fold_sils.append(sil_test)

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

    # Confusion matrix
    cm = sk_confusion_matrix(all_y_test, all_y_pred, labels=list(range(len(mode_names))))
    per_mode_recall: dict[str, Any] = {}
    for i, name in enumerate(mode_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        per_mode_recall[name] = {
            "correct": correct,
            "total": total,
            "recall": correct / total if total > 0 else 0.0,
        }

    logger.info(f"\n  4-way topic-heldout T2+T2.5: sil={mean_sil:.4f} +/- {std_sil:.4f}, "
                f"kNN={mean_knn:.2%} +/- {std_knn:.2%}")
    logger.info(f"  (Stratified 4-way kNN was 55%, chance=25%)")
    logger.info(f"  Per-mode recall: {per_mode_recall}")
    logger.info(f"  Confusion matrix:\n{cm}")

    # Compare with stratified 5-way topic-heldout
    logger.info(f"\n  --- Cross-reference ---")
    logger.info(f"  5-way topic-heldout kNN: 78% (this removes analogical)")
    logger.info(f"  4-way stratified kNN:    55% (from Exp 1 follow-up)")
    logger.info(f"  4-way topic-heldout kNN: {mean_knn:.2%}")

    results: dict[str, Any] = {
        "experiment": "4way_topic_heldout_no_analogical",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_set": "T2+T2.5",
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "modes": mode_names,
        "n_folds": n_splits,
        "chance_level": 0.25,
        "mean_sil_test": mean_sil,
        "std_sil_test": std_sil,
        "mean_knn": mean_knn,
        "std_knn": std_knn,
        "fold_details": fold_details,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": mode_names,
        "per_mode_recall": per_mode_recall,
        "comparison": {
            "stratified_4way_knn": 0.55,
            "stratified_4way_sil": 0.047,
            "topic_heldout_5way_knn": 0.78,
            "topic_heldout_5way_sil": 0.338,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    output_dir = PROJECT_ROOT / "outputs" / "phase1" / "reviewer_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "4way_topic_heldout.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total elapsed: {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
