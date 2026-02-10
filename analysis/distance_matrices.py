"""Semantic vs computational distance correlation analysis.

Core test: Mantel test between computational distance and prompt-based
semantic distance (content-independence). Prompt-based Mantel uses the
user prompts — identical across modes for the same topic — to truly test
whether computational signatures carry information orthogonal to content.

Generated-text Mantel is also computed but is informational only, since
mode shapes what the model writes (inflating the correlation floor).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from config import ExperimentConfig

logger = logging.getLogger(__name__)


def _mantel_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_permutations: int = 9999,
) -> tuple[float, float]:
    """Mantel test: correlation between two distance matrices with permutation p-value.

    Args:
        dist_a, dist_b: square distance matrices (same size)
        n_permutations: number of permutations for p-value

    Returns:
        (r, p_value)
    """
    n = dist_a.shape[0]
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(n, k=1)
    vec_a = dist_a[idx]
    vec_b = dist_b[idx]

    observed_r, _ = pearsonr(vec_a, vec_b)

    # Permutation test
    count_ge = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        perm_b = dist_b[np.ix_(perm, perm)]
        perm_vec_b = perm_b[idx]
        r, _ = pearsonr(vec_a, perm_vec_b)
        if r >= observed_r:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return float(observed_r), float(p_value)


def analyze_distances(
    signatures: dict[int, dict],
    metadata_list: list[dict],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Compute and compare semantic vs computational distance matrices.

    Uses Sets A + B (75 mode-labeled generations).
    """
    # Filter to Sets A + B
    ab_meta = [m for m in metadata_list if m["prompt_set"] in ("A", "B")]
    if len(ab_meta) < 10:
        return {"error": f"Too few Set A+B generations: {len(ab_meta)}"}

    gen_ids = []
    comp_features = []
    knnlm_features = []
    texts = []
    prompts = []

    for m in ab_meta:
        gid = m["generation_id"]
        if gid not in signatures:
            continue
        gen_ids.append(gid)
        comp_features.append(signatures[gid]["features"])
        texts.append(m["generated_text"])
        prompts.append(m.get("user_prompt", ""))
        knnlm = signatures[gid].get("knnlm_baseline")
        if knnlm is not None:
            knnlm_features.append(knnlm)

    n = len(gen_ids)
    logger.info(f"Computing distances for {n} generations")

    # ── Computational distance matrix ──
    comp_mat = np.stack(comp_features)
    comp_mat = np.nan_to_num(comp_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score normalize
    std = comp_mat.std(axis=0)
    std[std < 1e-10] = 1.0
    comp_mat_norm = (comp_mat - comp_mat.mean(axis=0)) / std

    comp_dist = squareform(pdist(comp_mat_norm, metric="cosine"))

    # ── Semantic distance matrices ──
    # Two variants: prompt-based (true content distance) and text-based (informational)
    logger.info("Computing semantic embeddings...")
    has_semantic = False
    has_prompt_semantic = False
    sem_dist = np.zeros((n, n))
    prompt_sem_dist = np.zeros((n, n))

    try:
        from sentence_transformers import SentenceTransformer
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Prompt-based: embeds user prompts (identical across modes for same topic)
        # This is the primary content-independence test
        if prompts and all(p for p in prompts):
            prompt_embeddings = sem_model.encode(
                prompts, show_progress_bar=False, normalize_embeddings=True,
            )
            prompt_sem_dist = squareform(pdist(prompt_embeddings, metric="cosine"))
            has_prompt_semantic = True
            logger.info("Prompt-based semantic embeddings computed")

        # Text-based: embeds generated text (informational — mode shapes content)
        text_embeddings = sem_model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True,
        )
        sem_dist = squareform(pdist(text_embeddings, metric="cosine"))
        has_semantic = True
    except Exception as e:
        logger.warning(f"Semantic embedding failed: {e}. Using dummy distances.")

    # ── kNN-LM baseline distance matrix ──
    knnlm_dist = None
    if len(knnlm_features) == n:
        knnlm_mat = np.stack(knnlm_features)
        knnlm_mat = np.nan_to_num(knnlm_mat, nan=0.0, posinf=0.0, neginf=0.0)
        knnlm_dist = squareform(pdist(knnlm_mat, metric="cosine"))

    # ── Mantel tests ──
    results: dict[str, Any] = {"n_generations": n}

    # Primary gate metric: prompt-based Mantel (true content-independence test)
    if has_prompt_semantic:
        logger.info("Running Mantel test (computational vs prompt-semantic)...")
        mantel_r_prompt, mantel_p_prompt = _mantel_test(comp_dist, prompt_sem_dist)
        results["mantel_r_prompt"] = mantel_r_prompt
        results["mantel_p_prompt"] = mantel_p_prompt
        # Use prompt-based as the primary decision gate metric
        results["mantel_r"] = mantel_r_prompt
        results["mantel_p"] = mantel_p_prompt
        logger.info(f"Mantel test (prompt): r={mantel_r_prompt:.4f}, p={mantel_p_prompt:.4f}")

    # Informational: text-based Mantel (mode shapes content, so floor is expected)
    if has_semantic:
        logger.info("Running Mantel test (computational vs text-semantic)...")
        mantel_r_text, mantel_p_text = _mantel_test(comp_dist, sem_dist)
        results["mantel_r_text"] = mantel_r_text
        results["mantel_p_text"] = mantel_p_text
        logger.info(f"Mantel test (text): r={mantel_r_text:.4f}, p={mantel_p_text:.4f}")
        # If prompt-based wasn't available, fall back to text-based for gate
        if not has_prompt_semantic:
            results["mantel_r"] = mantel_r_text
            results["mantel_p"] = mantel_p_text

    if knnlm_dist is not None:
        # Comp vs kNN-LM
        r_comp_knnlm, p_comp_knnlm = _mantel_test(comp_dist, knnlm_dist)
        results["comp_vs_knnlm_r"] = r_comp_knnlm
        results["comp_vs_knnlm_p"] = p_comp_knnlm

        if has_semantic:
            # kNN-LM vs semantic
            r_knnlm_sem, p_knnlm_sem = _mantel_test(knnlm_dist, sem_dist)
            results["knnlm_vs_semantic_r"] = r_knnlm_sem
            results["knnlm_vs_semantic_p"] = p_knnlm_sem

    # ── Per-tier correlation with semantics ──
    if has_semantic:
        for tier_key in ["features_tier1", "features_tier2", "features_tier2_5"]:
            tier_vecs = []
            for gid in gen_ids:
                sig = signatures[gid]
                if tier_key in sig:
                    tier_vecs.append(sig[tier_key])
            if len(tier_vecs) == n:
                tier_mat = np.stack(tier_vecs)
                tier_mat = np.nan_to_num(tier_mat, nan=0.0, posinf=0.0, neginf=0.0)
                tier_std = tier_mat.std(axis=0)
                tier_std[tier_std < 1e-10] = 1.0
                tier_mat_norm = (tier_mat - tier_mat.mean(axis=0)) / tier_std
                tier_dist = squareform(pdist(tier_mat_norm, metric="cosine"))
                r, p = _mantel_test(tier_dist, sem_dist, n_permutations=999)
                results[f"{tier_key}_vs_semantic_r"] = r
                results[f"{tier_key}_vs_semantic_p"] = p
                logger.info(f"{tier_key} vs semantic: r={r:.4f}")

    # ── Plot: distance correlation ──
    try:
        idx = np.triu_indices(n, k=1)
        n_plots = int(has_prompt_semantic) + int(has_semantic)
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
            if n_plots == 1:
                axes = [axes]
            plot_idx = 0

            if has_prompt_semantic:
                ax = axes[plot_idx]
                ax.scatter(prompt_sem_dist[idx], comp_dist[idx], alpha=0.1, s=5)
                ax.set_xlabel("Prompt Semantic Distance (cosine)")
                ax.set_ylabel("Computational Distance (cosine)")
                r_val = results.get("mantel_r_prompt", 0)
                ax.set_title(f"Prompt-Semantic vs Computational\n(Mantel r={r_val:.3f}, primary gate)")
                z = np.polyfit(prompt_sem_dist[idx], comp_dist[idx], 1)
                x_line = np.linspace(prompt_sem_dist[idx].min(), prompt_sem_dist[idx].max(), 100)
                ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
                plot_idx += 1

            if has_semantic:
                ax = axes[plot_idx]
                ax.scatter(sem_dist[idx], comp_dist[idx], alpha=0.1, s=5)
                ax.set_xlabel("Text Semantic Distance (cosine)")
                ax.set_ylabel("Computational Distance (cosine)")
                r_val = results.get("mantel_r_text", 0)
                ax.set_title(f"Text-Semantic vs Computational\n(Mantel r={r_val:.3f}, informational)")
                z = np.polyfit(sem_dist[idx], comp_dist[idx], 1)
                x_line = np.linspace(sem_dist[idx].min(), sem_dist[idx].max(), 100)
                ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)

            fig.tight_layout()
            fig.savefig(config.figures_dir / "distance_correlation.png", dpi=150)
            plt.close(fig)
            logger.info("Saved distance_correlation.png")
    except Exception as e:
        logger.warning(f"Failed to save distance plot: {e}")

    return results
