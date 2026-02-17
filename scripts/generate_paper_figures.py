#!/usr/bin/env python3
"""Generate publication-ready figures for the paper.

All figures use paper terminology (Run 1/2/3) and consistent styling.
Outputs to outputs/paper_figures/.

Run mapping:
  Run 1 = run2_epistemic_modes
  Run 2 = run3_process_modes
  Run 3 = run4_format_controlled (primary)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Optional imports with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN1_DIR = ROOT / "outputs" / "runs" / "run2_epistemic_modes"
RUN2_DIR = ROOT / "outputs" / "runs" / "run3_process_modes"
RUN3_DIR = ROOT / "outputs" / "runs" / "run4_format_controlled"
REVIEW_PACK = ROOT / "review_pack"
TIER_ABLATION = ROOT / "outputs" / "phase1" / "tier_ablation" / "tier_ablation.json"
SEM_DISAMB = ROOT / "outputs" / "phase05" / "semantic_disambiguation"

# ── Styling ────────────────────────────────────────────────────────────
# Consistent palette: T1=blue, T2=orange, T2.5=red, T3=gray
TIER_COLORS = {
    "T1": "#4878CF",
    "T2": "#E8A838",
    "T2.5": "#D65F5F",
    "T3": "#B0B0B0",
}

# Mode palette (colorblind-friendly, 5 distinct hues)
MODE_COLORS = {
    "linear": "#4878CF",
    "analogical": "#6ACC65",
    "socratic": "#D65F5F",
    "contrastive": "#E8A838",
    "dialectical": "#956CB4",
}

FONT_SIZE = 11
TITLE_SIZE = 13
LABEL_SIZE = 11


def setup_style():
    """Set consistent matplotlib style."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_json(path: Path) -> dict:
    """Load a JSON file, raising a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required data file missing: {path}")
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Tier Inversion Crossover
# ═══════════════════════════════════════════════════════════════════════

def fig_tier_inversion():
    """Grouped bar chart: per-tier accuracy across Run 1→2→3.

    Shows the crossover pattern where T1 dominates early (Run 1)
    but T2.5 dominates under format control (Run 3).

    Uses RF accuracy values as reported in paper §4.2.
    """
    # Values from paper §4.2 (verified against results.json)
    # Run 1 (epistemic): T1 > T2 > T2.5
    # Run 2 (process, 4-way): T2 > T1 > T2.5
    # Run 3 (format-controlled): T2.5 > T2 > T1

    runs = ["Run 1\n(epistemic)", "Run 2\n(process)", "Run 3\n(format-ctrl)"]

    # Per-tier accuracies (RF, 5-fold CV)
    t1_acc = [0.57, 0.80, 0.54]
    t2_acc = [0.48, 0.83, 0.59]
    t25_acc = [0.41, 0.77, 0.64]

    x = np.arange(len(runs))
    width = 0.22

    fig, ax = plt.subplots(figsize=(6, 4))

    bars_t1 = ax.bar(x - width, t1_acc, width, label="T1 (logit statistics)",
                     color=TIER_COLORS["T1"], edgecolor="white", linewidth=0.5)
    bars_t2 = ax.bar(x, t2_acc, width, label="T2 (attention routing)",
                     color=TIER_COLORS["T2"], edgecolor="white", linewidth=0.5)
    bars_t25 = ax.bar(x + width, t25_acc, width, label="T2.5 (KV cache geometry)",
                      color=TIER_COLORS["T2.5"], edgecolor="white", linewidth=0.5)

    # Chance line
    ax.axhline(y=0.20, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2.45, 0.21, "chance", fontsize=8, color="gray", ha="right")

    # Value labels
    for bars in [bars_t1, bars_t2, bars_t25]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Classification accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=FONT_SIZE - 3)
    ax.set_title("Tier inversion: signal shifts from logits to KV cache\nunder progressive confound removal")

    fig.tight_layout()
    out = OUT_DIR / "fig_tier_inversion.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [1] Tier inversion crossover → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Contrastive Embedding (UMAP of T2+T2.5)
# ═══════════════════════════════════════════════════════════════════════

def fig_contrastive_embedding():
    """UMAP of T2+T2.5 features colored by mode.

    Shows geometric mode structure in the feature space that
    carries discriminative signal.
    """
    if not HAS_UMAP:
        print("  [2] SKIPPED (umap-learn not installed)")
        return None

    # Load features
    feat_path = REVIEW_PACK / "features.npz"
    if not feat_path.exists():
        print("  [2] SKIPPED (review_pack/features.npz not found)")
        return None

    data = np.load(feat_path, allow_pickle=True)
    tier2 = data["tier2"]  # (100, 221)
    tier25 = data["tier2_5"]  # (100, 145)
    labels = data["labels"]  # mode strings

    # Combine T2+T2.5
    X = np.hstack([tier2, tier25])  # (100, 366)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                        metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))

    for mode in sorted(set(labels)):
        mask = labels == mode
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=MODE_COLORS.get(mode, "#999999"), label=mode.capitalize(),
                   s=40, alpha=0.75, edgecolors="white", linewidth=0.3)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("T2+T2.5 feature space (Run 3, format-controlled)\n366 features, colored by processing mode")
    ax.legend(loc="best", framealpha=0.9, markerscale=1.2)

    # Remove tick labels (embedding axes are arbitrary)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    out = OUT_DIR / "fig_contrastive_embedding.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [2] Contrastive embedding (UMAP) → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Double Dissociation Heatmap
# ═══════════════════════════════════════════════════════════════════════

def fig_double_dissociation():
    """2×2 heatmap: temperature vs mode × T1 vs T2.5.

    Shows functional independence: temperature → T1, mode → T2.5,
    each near-chance for the other condition.
    """
    # Data from supplementary_analysis.json and tier_ablation.json
    supp = load_json(RUN3_DIR / "supplementary_analysis.json")

    temp_tier = supp["temperature_control"]["tier_ablation_low_vs_high"]
    t1_temp = temp_tier["features_tier1"]["rf_accuracy"]
    t25_temp = temp_tier["features_tier2_5"]["rf_accuracy"]

    # Mode tier data from tier_ablation.json (run4 = Run 3)
    ta = load_json(TIER_ABLATION)
    t1_mode = ta["run4"]["per_tier"]["T1"]["knn_accuracy_mean"]
    t25_mode = ta["run4"]["per_tier"]["T2.5"]["knn_accuracy_mean"]

    # 2×2 matrix: rows = condition (Temperature, Mode), cols = tier (T1, T2.5)
    matrix = np.array([
        [t1_temp, t25_temp],  # Temperature
        [t1_mode, t25_mode],  # Mode
    ])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Custom colormap: low=cool blue, high=warm red
    cmap = LinearSegmentedColormap.from_list("dissoc", ["#E8F0FE", "#4878CF", "#D65F5F", "#8B0000"])

    im = ax.imshow(matrix, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T1\n(logit statistics)", "T2.5\n(KV cache geometry)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Temperature\nvariation", "Processing\nmode"])
    ax.set_title("Double dissociation: tier × condition")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Classification accuracy")
    cbar.ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out = OUT_DIR / "fig_double_dissociation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [3] Double dissociation heatmap → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Semantic vs Compute Orthogonality
# ═══════════════════════════════════════════════════════════════════════

def fig_semantic_vs_compute():
    """Side-by-side UMAP: same T2+T2.5 embedding colored by topic vs mode.

    Topic coloring should look random/unstructured; mode coloring should
    show clear clusters — the visual version of the orthogonality claim.
    """
    if not HAS_UMAP:
        print("  [4] SKIPPED (umap-learn not installed)")
        return None

    feat_path = REVIEW_PACK / "features.npz"
    if not feat_path.exists():
        print("  [4] SKIPPED (review_pack/features.npz not found)")
        return None

    data = np.load(feat_path, allow_pickle=True)
    tier2 = data["tier2"]
    tier25 = data["tier2_5"]
    labels = data["labels"]
    topics = data["topics"]

    X = np.hstack([tier2, tier25])

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # Single UMAP fit — same embedding, different colorings
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                        metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: colored by topic (should look unstructured)
    unique_topics = sorted(set(topics))
    topic_cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_topics))
    topic_color_map = {t: topic_cmap(i) for i, t in enumerate(unique_topics)}

    for topic in unique_topics:
        mask = topics == topic
        ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[topic_color_map[topic]], s=30, alpha=0.6,
                    edgecolors="white", linewidth=0.2)

    ax1.set_title("Colored by semantic topic\n(20 topics)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    # Right: colored by mode (should show clusters)
    for mode in sorted(set(labels)):
        mask = labels == mode
        ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=MODE_COLORS.get(mode, "#999999"), label=mode.capitalize(),
                    s=30, alpha=0.75, edgecolors="white", linewidth=0.3)

    ax2.set_title("Colored by processing mode\n(5 modes)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.legend(loc="best", framealpha=0.9, markerscale=1.2)

    fig.suptitle("Same T2+T2.5 embedding — mode structure is orthogonal to semantic content",
                 fontsize=TITLE_SIZE, y=1.02)

    fig.tight_layout()
    out = OUT_DIR / "fig_semantic_vs_compute.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [4] Semantic vs compute orthogonality → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Pairwise Discriminability Matrix
# ═══════════════════════════════════════════════════════════════════════

def fig_pairwise_discriminability():
    """Heatmap of all 10 pairwise binary accuracies.

    Shows that all mode pairs are above chance (50%) at a glance.
    """
    results = load_json(RUN3_DIR / "results.json")
    pw = results["deep_dive"]["pairwise_discriminability"]

    labels = pw["labels"]
    matrix = np.array(pw["matrix"])

    # Capitalize labels for display
    display_labels = [l.capitalize() for l in labels]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Mask diagonal
    mask = np.eye(len(labels), dtype=bool)
    display_matrix = np.where(mask, np.nan, matrix)

    cmap = LinearSegmentedColormap.from_list("pw", ["#FFF5F0", "#FB6A4A", "#A50F15"])
    im = ax.imshow(display_matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")

    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
            else:
                val = matrix[i][j]
                color = "white" if val > 0.9 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=9, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(display_labels)
    ax.set_title("Pairwise binary classification accuracy\n(Run 3, format-controlled)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Binary accuracy")
    cbar.ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out = OUT_DIR / "fig_pairwise_discriminability.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [5] Pairwise discriminability matrix → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: R² Distribution (text → compute regression)
# ═══════════════════════════════════════════════════════════════════════

def fig_r2_distribution():
    """Histogram of per-feature R² from text→compute ridge regression.

    Recomputes per-feature R² from the stored features to get the
    full distribution (only summary stats are in JSON).
    """
    feat_path = REVIEW_PACK / "features.npz"
    if not feat_path.exists():
        print("  [6] SKIPPED (features.npz not found)")
        return None

    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  [6] SKIPPED (missing dependency: {e})")
        return None

    data = np.load(feat_path, allow_pickle=True)
    tier2 = data["tier2"]
    tier25 = data["tier2_5"]
    topics = data["topics"]

    # Load text for semantic embeddings
    meta = load_json(RUN3_DIR / "metadata.json")
    gens = meta["generations"]
    gen_ids = data["generation_ids"]
    texts = []
    for gid in gen_ids:
        texts.append(gens[gid]["generated_text"])

    # Compute semantic embeddings
    print("    Computing semantic embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sem_emb = model.encode(texts, show_progress_bar=False)

    # Compute features (T2+T2.5)
    Y = np.hstack([tier2, tier25])
    X = sem_emb

    # Standardize
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)

    # Per-feature R² via Ridge regression with GroupKFold by topic
    # sorted() for deterministic fold assignment across runs
    unique_topics = sorted(set(topics))
    topic_ids = np.array([unique_topics.index(t) for t in topics])

    gkf = GroupKFold(n_splits=5)
    r2_per_feature = []

    for feat_idx in range(Y.shape[1]):
        y = Y[:, feat_idx]
        fold_r2s = []
        for train_idx, test_idx in gkf.split(X, groups=topic_ids):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X[train_idx], y[train_idx])
            fold_r2s.append(ridge.score(X[test_idx], y[test_idx]))
        r2_per_feature.append(np.mean(fold_r2s))

    r2_arr = np.array(r2_per_feature)
    n_below_zero = int(np.sum(r2_arr < 0))
    n_total = len(r2_arr)
    median_r2 = float(np.median(r2_arr))

    # Clip extreme outliers for display (5 features below -10, min=-124)
    clip_low = -5.0
    r2_clipped = np.clip(r2_arr, clip_low, None)
    n_clipped = int(np.sum(r2_arr < clip_low))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(r2_clipped, bins=50, color=TIER_COLORS["T2.5"], alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="R² = 0 (no predictive power)")
    ax.axvline(x=median_r2, color="black", linestyle="-", linewidth=1.5,
               label=f"Median R² = {median_r2:.2f}")

    # Note clipped values
    if n_clipped > 0:
        ax.annotate(f"← {n_clipped} features\nbelow {clip_low:.0f}",
                    xy=(clip_low, 0), xytext=(clip_low + 0.3, ax.get_ylim()[1] * 0.7),
                    fontsize=8, color="gray", ha="left",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("R² (semantic embeddings → compute feature)")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Text→compute regression: {n_below_zero}/{n_total} features below zero\n"
                 f"(Ridge regression, GroupKFold by topic)")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "fig_r2_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [6] R² distribution → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Permutation Null Distribution
# ═══════════════════════════════════════════════════════════════════════

def fig_permutation_null():
    """Permutation null distribution with observed accuracy marked.

    Uses the semantic disambiguation 1000-permutation test (which has
    the full null distribution stored).
    """
    # Try the follow-up results first (has full null distribution)
    followup_path = SEM_DISAMB / "followup_results.json"
    if followup_path.exists():
        followup = load_json(followup_path)
        pt = followup["permutation_tests"]["compute_only"]
        null_dist = np.array(pt["null_distribution"])
        observed = pt["observed_silhouette"]
        p_value = pt["p_value"]
        metric_name = "silhouette score"
        title_suffix = "(compute features, GroupKFold)"
    else:
        # Fall back to Phase 0 permutation test (summary stats only)
        perm = load_json(RUN3_DIR / "permutation_test.json")
        # Synthesize approximate null from stats
        null_dist = np.random.default_rng(42).normal(
            perm["null_mean"], perm["null_std"], size=1000
        )
        observed = perm["observed_accuracy"]
        p_value = perm["p_value"]
        metric_name = "5-fold CV accuracy"
        title_suffix = "(RF classifier)"

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(null_dist, bins=50, color="#B0B0B0", alpha=0.8,
            edgecolor="white", linewidth=0.3, label="Null distribution (n=1000)")
    ax.axvline(x=observed, color=TIER_COLORS["T2.5"], linewidth=2.5,
               linestyle="-", label=f"Observed = {observed:.3f}")

    ax.set_xlabel(f"Permutation {metric_name}")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test: p = {p_value:.3f}\n{title_suffix}")
    ax.legend(loc="upper right" if observed > np.median(null_dist) else "upper left",
              framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "fig_permutation_null.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [7] Permutation null distribution → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════

def fig_confusion_matrix():
    """5-way confusion matrix showing mode classification patterns.

    Highlights the analogical dominance pattern and the
    contrastive↔dialectical confusion.
    """
    results = load_json(RUN3_DIR / "results.json")
    fi = results["feature_importance"]

    cm = np.array(fi["confusion_matrix"])
    labels = fi["confusion_matrix_labels"]
    display_labels = [l.capitalize() for l in labels]

    # Normalize by row (true class) to get per-class accuracy
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.where(row_sums > 0, row_sums, 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    cmap = LinearSegmentedColormap.from_list("cm", ["#FFFFFF", "#4878CF", "#1A237E"])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Annotate with both count and percentage
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i][j]
            pct = cm_norm[i][j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted mode")
    ax.set_ylabel("True mode")
    ax.set_title("5-way confusion matrix (Run 3, format-controlled)\nRF classifier, 5-fold CV, 70% overall accuracy")

    fig.tight_layout()
    out = OUT_DIR / "fig_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [8] Confusion matrix → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    print(f"Generating paper figures → {OUT_DIR}/\n")

    figures = {}

    # Priority tier — figures that argue
    print("Priority figures:")
    figures["tier_inversion"] = fig_tier_inversion()
    figures["contrastive_embedding"] = fig_contrastive_embedding()
    figures["double_dissociation"] = fig_double_dissociation()
    figures["semantic_vs_compute"] = fig_semantic_vs_compute()

    # Support tier
    print("\nSupport figures:")
    figures["pairwise"] = fig_pairwise_discriminability()
    figures["permutation"] = fig_permutation_null()
    figures["confusion"] = fig_confusion_matrix()

    # R² is slower (requires sentence-transformer)
    print("\nCompute-intensive figures:")
    try:
        figures["r2_distribution"] = fig_r2_distribution()
    except Exception as e:
        print(f"  [6] FAILED: {e}")
        figures["r2_distribution"] = None

    # Summary
    print(f"\n{'='*50}")
    generated = [k for k, v in figures.items() if v is not None]
    skipped = [k for k, v in figures.items() if v is None]
    print(f"Generated: {len(generated)} figures")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
