"""Retrieval comparison: semantic vs computational vs hybrid.

For each generation, find k=5 nearest neighbors by different distance metrics
and measure mode-match accuracy and retrieval overlap.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def analyze_retrieval(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
    k: int = 5,
) -> dict[str, Any]:
    """Run retrieval analysis on Sets A + B.

    Compares:
      (a) Semantic-only retrieval
      (b) Computational-only retrieval (full signature)
      (c) kNN-LM baseline retrieval
      (d) Hybrid retrieval at various alpha values
    """
    # Filter to Sets A + B
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]

    gen_ids = []
    comp_features = []
    knnlm_features = []
    texts = []
    modes = []
    topics = []

    for m in ab_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        gen_ids.append(gid)
        comp_features.append(signatures[gid]["features"])
        texts.append(m["generated_text"])
        modes.append(m["mode"])
        topics.append(m["topic_idx"])
        knnlm = signatures[gid].get("knnlm_baseline")
        if knnlm is not None:
            knnlm_features.append(knnlm)

    n = len(gen_ids)
    if n < k + 1:
        return {"error": f"Too few generations for k={k} retrieval: {n}"}

    modes_arr = np.array(modes)
    topics_arr = np.array(topics)

    # ── Build distance matrices ──

    # Computational
    comp_mat = np.stack(comp_features)
    comp_mat = np.nan_to_num(comp_mat, nan=0.0, posinf=0.0, neginf=0.0)
    comp_scaled = StandardScaler().fit_transform(comp_mat)
    comp_dist = cdist(comp_scaled, comp_scaled, metric="cosine")

    # Semantic
    try:
        from sentence_transformers import SentenceTransformer
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        sem_embeddings = sem_model.encode(texts, normalize_embeddings=True)
        sem_dist = cdist(sem_embeddings, sem_embeddings, metric="cosine")
        has_semantic = True
    except Exception as e:
        logger.warning(f"Semantic model failed: {e}")
        sem_dist = np.zeros((n, n))
        has_semantic = False

    # kNN-LM baseline
    knnlm_dist = None
    if len(knnlm_features) == n:
        knnlm_mat = np.stack(knnlm_features)
        knnlm_mat = np.nan_to_num(knnlm_mat, nan=0.0)
        knnlm_scaled = StandardScaler().fit_transform(knnlm_mat)
        knnlm_dist = cdist(knnlm_scaled, knnlm_scaled, metric="cosine")

    results: dict[str, Any] = {"n_generations": n, "k": k}

    # ── Retrieval analysis ──
    methods: dict[str, np.ndarray] = {"computational": comp_dist}
    if has_semantic:
        methods["semantic"] = sem_dist
    if knnlm_dist is not None:
        methods["knnlm"] = knnlm_dist

    # Hybrid methods at various alpha
    alphas = [0.3, 0.5, 0.7]
    if has_semantic:
        # Normalize distance matrices to [0, 1] range for hybrid
        comp_norm = (comp_dist - comp_dist.min()) / max(comp_dist.max() - comp_dist.min(), 1e-10)
        sem_norm = (sem_dist - sem_dist.min()) / max(sem_dist.max() - sem_dist.min(), 1e-10)
        for alpha in alphas:
            hybrid = alpha * sem_norm + (1 - alpha) * comp_norm
            methods[f"hybrid_alpha{alpha}"] = hybrid

    # For each method: compute mode-match accuracy
    for method_name, dist_matrix in methods.items():
        mode_matches = 0
        total = 0

        for i in range(n):
            # Get k nearest neighbors (excluding self)
            dists = dist_matrix[i].copy()
            dists[i] = np.inf  # exclude self
            nn_indices = np.argsort(dists)[:k]

            # Check mode match
            nn_modes = modes_arr[nn_indices]
            matches = (nn_modes == modes_arr[i]).sum()
            mode_matches += matches
            total += k

        mode_accuracy = mode_matches / max(total, 1)
        results[f"{method_name}_mode_match"] = float(mode_accuracy)
        logger.info(f"{method_name}: mode-match accuracy = {mode_accuracy:.4f}")

    # ── Retrieval overlap (Jaccard) between semantic and computational ──
    if has_semantic:
        jaccard_scores = []
        comp_unique_cases: list[dict] = []

        for i in range(n):
            sem_dists = sem_dist[i].copy()
            sem_dists[i] = np.inf
            sem_nn = set(np.argsort(sem_dists)[:k])

            comp_dists = comp_dist[i].copy()
            comp_dists[i] = np.inf
            comp_nn = set(np.argsort(comp_dists)[:k])

            intersection = sem_nn & comp_nn
            union = sem_nn | comp_nn
            jaccard = len(intersection) / max(len(union), 1)
            jaccard_scores.append(jaccard)

            # Track cases where computational finds things semantic doesn't
            comp_only = comp_nn - sem_nn
            if comp_only:
                comp_unique_cases.append({
                    "query_id": int(gen_ids[i]),
                    "query_mode": modes[i],
                    "query_topic": int(topics[i]),
                    "comp_unique_neighbors": [
                        {
                            "id": int(gen_ids[j]),
                            "mode": modes[j],
                            "topic": int(topics[j]),
                            "same_mode": modes[j] == modes[i],
                        }
                        for j in comp_only
                    ],
                })

        results["jaccard_semantic_vs_comp"] = float(np.mean(jaccard_scores))
        results["jaccard_std"] = float(np.std(jaccard_scores))
        results["n_comp_unique_cases"] = len(comp_unique_cases)

        # Save interesting cases (limit to 20)
        results["comp_unique_examples"] = comp_unique_cases[:20]

        logger.info(f"Semantic-Computational Jaccard overlap: {results['jaccard_semantic_vs_comp']:.4f}")

    # ── Plot: retrieval comparison ──
    try:
        method_names = []
        accuracies = []
        for key in sorted(results.keys()):
            if key.endswith("_mode_match"):
                method_names.append(key.replace("_mode_match", ""))
                accuracies.append(results[key])

        if method_names:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            bars = ax.bar(range(len(method_names)), accuracies, color="steelblue", edgecolor="black")
            ax.set_xticks(range(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha="right")
            ax.set_ylabel("Mode-Match Accuracy (k=5 NN)")
            ax.set_title("Retrieval Mode-Match by Method")
            # Add chance level line
            n_modes = len(set(modes))
            ax.axhline(1.0 / n_modes, color="red", linestyle="--", alpha=0.5, label=f"Chance ({1/n_modes:.2f})")
            ax.legend()
            # Value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.3f}", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(config.figures_dir / "retrieval_comparison.png", dpi=150)
            plt.close(fig)
            logger.info("Saved retrieval_comparison.png")
    except Exception as e:
        logger.warning(f"Failed to save retrieval plot: {e}")

    return results
