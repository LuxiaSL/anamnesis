"""Section 9: Semantic independence — compute vs text content signal."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .signature_io import AnalysisData
from .schemas import (
    ClassificationScore,
    ContrastiveProjectionComparisonEntry,
    ContrastiveProjectionComparisonResult,
    JaccardStats,
    MantelResult,
    PerModeSurfaceVsCompute,
    PerModeSurfaceVsComputeResult,
    PerTierSemanticResult,
    PromptSwapConfoundResult,
    PromptSwapTierResult,
    RetrievalFeatureSet,
    RetrievalResult,
    SemanticClassifierBundle,
    SemanticResult,
    ShuffleControlsResult,
    TextToComputeR2,
)

logger = logging.getLogger(__name__)


def _embed_cache_dir() -> Path:
    """Disk cache for corpus embeddings (recomputing SBERT/TF-IDF on an unchanged
    corpus dominated repeat semantic runs). Override via ANAMNESIS_EMBED_CACHE."""
    return Path(os.environ.get(
        "ANAMNESIS_EMBED_CACHE",
        str(Path.home() / ".cache" / "anamnesis" / "embeddings"),
    ))


def _corpus_cache_key(texts: list[str], tag: str) -> str:
    """Content hash of (method+params, corpus) — any text change invalidates."""
    h = hashlib.sha256(tag.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _cached_embedding_load(key: str) -> NDArray | None:
    """Best-effort cache read; any failure means recompute."""
    path = _embed_cache_dir() / f"{key}.npz"
    try:
        if path.exists():
            return np.load(path)["embedding"].astype(np.float32)
    except Exception as e:  # noqa: BLE001 — cache must never break the analysis
        logger.warning(f"Embedding cache read failed ({path.name}): {e}")
    return None


def _cached_embedding_save(key: str, embedding: NDArray) -> None:
    """Best-effort cache write (atomic rename; failures logged, not raised)."""
    cache_dir = _embed_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f".{key}.tmp.npz"
        np.savez_compressed(tmp, embedding=embedding)
        tmp.replace(cache_dir / f"{key}.npz")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Embedding cache write failed ({key[:12]}…): {e}")


def _embed_texts_tfidf(texts: list[str], n_components: int = 100) -> NDArray:
    """TF-IDF + SVD embedding of texts (disk-cached by corpus hash)."""
    key = _corpus_cache_key(texts, f"tfidf:max_features=5000:svd={n_components}:seed=42")
    cached = _cached_embedding_load(key)
    if cached is not None:
        return cached
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(n_components, X_tfidf.shape[1] - 1), random_state=42)
    out = svd.fit_transform(X_tfidf).astype(np.float32)
    _cached_embedding_save(key, out)
    return out


def _embed_texts_sbert(texts: list[str]) -> NDArray | None:
    """Sentence-BERT embedding (optional dependency; disk-cached by corpus hash)."""
    key = _corpus_cache_key(texts, "sbert:all-MiniLM-L6-v2")
    cached = _cached_embedding_load(key)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        out = np.array(embeddings, dtype=np.float32)
        _cached_embedding_save(key, out)
        return out
    except ImportError:
        return None
    except Exception:
        return None


def _classify_condition(
    X: NDArray, y: NDArray, topics: NDArray,
    clf_name: str = "rf", seed: int = 42,
) -> ClassificationScore:
    """Classify with GroupKFold by topic."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)
    fold_accs: list[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if clf_name == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        elif clf_name == "knn":
            clf = KNeighborsClassifier(n_neighbors=3)
        else:
            raise ValueError(clf_name)

        clf.fit(X_tr, y[train_idx])
        fold_accs.append(float(accuracy_score(y[test_idx], clf.predict(X_te))))

    return ClassificationScore(
        accuracy=float(np.mean(fold_accs)),
        fold_accuracies=fold_accs,
    )


def _classifier_bundle(
    X: NDArray, y: NDArray, topics: NDArray, *, dims: int | None = None,
) -> SemanticClassifierBundle:
    """Build the (rf, knn[, dims]) bundle used by many sub-results."""
    return SemanticClassifierBundle(
        rf=_classify_condition(X, y, topics, clf_name="rf"),
        knn=_classify_condition(X, y, topics, clf_name="knn"),
        dims=dims,
    )


def _mantel_test(
    D_compute: NDArray, D_semantic: NDArray, n_permutations: int = 1000, seed: int = 42,
) -> MantelResult:
    """Mantel test between two distance matrices."""
    n = D_compute.shape[0]
    idx = np.triu_indices(n, k=1)
    x = D_compute[idx]
    y = D_semantic[idx]

    observed_r = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(seed)
    null_rs: list[float] = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        D_perm = D_semantic[np.ix_(perm, perm)]
        y_perm = D_perm[idx]
        null_rs.append(float(np.corrcoef(x, y_perm)[0, 1]))

    null_arr = np.array(null_rs)
    p_value = float(np.mean(null_arr >= observed_r))

    return MantelResult(
        r=observed_r,
        p_value=max(p_value, 1.0 / (n_permutations + 1)),
        null_mean=float(np.mean(null_arr)),
        null_std=float(np.std(null_arr)),
    )


def _text_to_compute_r2(
    X_semantic: NDArray, X_compute: NDArray, topics: NDArray,
) -> TextToComputeR2:
    """Ridge regression: predict each compute feature from semantic embedding."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)

    n_compute = X_compute.shape[1]
    r2_per_feature = np.full(n_compute, np.nan)

    for feat_idx in range(n_compute):
        y_feat = X_compute[:, feat_idx]
        fold_r2s: list[float] = []

        for train_idx, test_idx in gkf.split(X_semantic, y_feat, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_semantic[train_idx])
            X_te = scaler.transform(X_semantic[test_idx])

            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            ridge.fit(X_tr, y_feat[train_idx])
            y_pred = ridge.predict(X_te)

            ss_res = np.sum((y_feat[test_idx] - y_pred) ** 2)
            ss_tot = np.sum((y_feat[test_idx] - np.mean(y_feat[test_idx])) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            fold_r2s.append(r2)

        r2_per_feature[feat_idx] = float(np.mean(fold_r2s))

    return TextToComputeR2(
        median_r2=float(np.median(r2_per_feature)),
        mean_r2=float(np.mean(r2_per_feature)),
        n_features_r2_above_01=int(np.sum(r2_per_feature > 0.1)),
        n_features_r2_above_0=int(np.sum(r2_per_feature > 0)),
        best_r2=float(np.max(r2_per_feature)),
        worst_r2=float(np.min(r2_per_feature)),
    )


def _contrastive_topic_heldout(
    X_tfidf: NDArray, X_compute: NDArray,
    X_sbert: NDArray | None,
    y: NDArray, topics: NDArray, seed: int = 42,
) -> ContrastiveProjectionComparisonResult:
    """Compare contrastive projection across feature types, topic-heldout."""
    try:
        from .contrastive import _train_contrastive_mlp, _embed, _build_topic_folds
    except ImportError:
        return ContrastiveProjectionComparisonResult(
            error="contrastive module not available (torch missing?)",
        )

    folds = _build_topic_folds(topics, n_folds=5, seed=seed)

    conditions: dict[str, NDArray] = {
        "tfidf": StandardScaler().fit_transform(X_tfidf),
        "compute_t2t25": StandardScaler().fit_transform(X_compute),
    }
    if X_sbert is not None:
        conditions["sbert"] = StandardScaler().fit_transform(X_sbert)
        conditions["combined_compute_sbert"] = StandardScaler().fit_transform(
            np.concatenate([X_compute, X_sbert], axis=1),
        )

    entries: dict[str, ContrastiveProjectionComparisonEntry] = {}
    for cond_name, X in conditions.items():
        fold_accs: list[float] = []
        fold_sils: list[float] = []
        train_accs: list[float] = []

        for train_mask, test_mask in folds:
            try:
                model = _train_contrastive_mlp(X[train_mask], y[train_mask])
                emb_train = _embed(model, X[train_mask])
                emb_test = _embed(model, X[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(emb_train, y[train_mask])

                y_pred_test = knn.predict(emb_test)
                fold_accs.append(float(np.mean(y_pred_test == y[test_mask])))

                y_pred_train = knn.predict(emb_train)
                train_accs.append(float(np.mean(y_pred_train == y[train_mask])))

                if len(set(y[test_mask])) > 1:
                    from sklearn.metrics import silhouette_score
                    fold_sils.append(float(silhouette_score(emb_test, y[test_mask])))
            except Exception:
                fold_accs.append(0.0)
                train_accs.append(0.0)

        test_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0

        entries[cond_name] = ContrastiveProjectionComparisonEntry(
            test_knn_accuracy=test_acc,
            train_knn_accuracy=train_acc,
            train_test_gap=train_acc - test_acc,
            silhouette=float(np.mean(fold_sils)) if fold_sils else None,
            fold_accs=fold_accs,
        )

    return ContrastiveProjectionComparisonResult(
        compute_t2t25=entries.get("compute_t2t25"),
        tfidf=entries.get("tfidf"),
        sbert=entries.get("sbert"),
        combined_compute_sbert=entries.get("combined_compute_sbert"),
    )


def _per_mode_surface_vs_compute(
    X_surface: NDArray, X_compute: NDArray,
    y: NDArray, topics: NDArray, seed: int = 42,
) -> PerModeSurfaceVsComputeResult:
    """Per-mode recall comparison: surface vs compute features."""
    unique_modes = sorted(set(y))
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)

    def _accumulate_per_mode(X: NDArray) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {
            m: {"correct": 0, "total": 0} for m in unique_modes
        }
        for train_idx, test_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            clf.fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)

            for mode in unique_modes:
                mask = y[test_idx] == mode
                counts[mode]["total"] += int(mask.sum())
                counts[mode]["correct"] += int((y_pred[mask] == mode).sum())
        return counts

    surface_counts = _accumulate_per_mode(X_surface)
    compute_counts = _accumulate_per_mode(X_compute)

    per_mode: dict[str, PerModeSurfaceVsCompute] = {}
    for mode in unique_modes:
        s_total = surface_counts[mode]["total"]
        s_correct = surface_counts[mode]["correct"]
        c_total = compute_counts[mode]["total"]
        c_correct = compute_counts[mode]["correct"]

        s_recall = s_correct / max(s_total, 1)
        c_recall = c_correct / max(c_total, 1)

        per_mode[mode] = PerModeSurfaceVsCompute(
            surface_recall=float(s_recall),
            compute_recall=float(c_recall),
            gap_compute_minus_surface=float(c_recall - s_recall),
            sub_semantic=c_recall > s_recall + 0.05,
            surface_n=s_total,
            compute_n=c_total,
        )

    sub_semantic_modes = [m for m, d in per_mode.items() if d.sub_semantic]
    surface_dominant_modes = [
        m for m, d in per_mode.items() if d.gap_compute_minus_surface < -0.05
    ]

    return PerModeSurfaceVsComputeResult(
        per_mode=per_mode,
        n_sub_semantic=len(sub_semantic_modes),
        sub_semantic_modes=sub_semantic_modes,
        n_surface_dominant=len(surface_dominant_modes),
        surface_dominant_modes=surface_dominant_modes,
    )


def _parse_swap_mode(mode_str: str) -> tuple[str, str] | None:
    """Parse 'swap_A→B' into (system_prompt_mode, execution_mode)."""
    match = re.match(r"swap_(\w+)→(\w+)", mode_str)
    if match:
        return match.group(1), match.group(2)
    return None


def _run_prompt_swap_confound(
    data: AnalysisData,
    signature_dir: Path,
    addon_dirs: list[Path] | None = None,
) -> PromptSwapConfoundResult:
    """Prompt-swap confound test: train on core set, predict on prompt-swap samples."""
    from .signature_io import TIER_KEYS, BASELINE_TIERS, ENGINEERED_TIERS

    all_npz_files = sorted(signature_dir.glob("gen_*.npz"))
    if not all_npz_files:
        return PromptSwapConfoundResult(error="No npz files in signature dir")

    swap_info: list[dict] = []
    swap_npz_paths: list[Path] = []
    for npz_path in all_npz_files:
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        parsed = _parse_swap_mode(meta.get("mode", ""))
        if parsed:
            swap_info.append({
                "file_stem": npz_path.stem,
                "system_prompt_mode": parsed[0],
                "execution_mode": parsed[1],
                "swap_name": meta["mode"],
                "topic": meta.get("topic", ""),
            })
            swap_npz_paths.append(npz_path)

    if not swap_info:
        return PromptSwapConfoundResult(error="No prompt-swap samples found")

    run4 = data.run4
    test_tiers: list[str] = []
    for tier in BASELINE_TIERS + ENGINEERED_TIERS:
        if tier in run4.tier_features:
            test_tiers.append(tier)
    for group in ["T2+T2.5", "combined_v2"]:
        if group in run4.group_features:
            test_tiers.append(group)

    individual_test_tiers = [t for t in test_tiers if t in TIER_KEYS]
    swap_tier_features: dict[str, list[NDArray | None]] = {}

    # Pass 1: load individual tier features from primary npz files
    for i, npz_path in enumerate(swap_npz_paths):
        npz_data = np.load(npz_path, allow_pickle=True)
        for tier_name in individual_test_tiers:
            npz_key = TIER_KEYS.get(tier_name, "")
            if npz_key and npz_key in npz_data.files:
                swap_tier_features.setdefault(tier_name, [None] * len(swap_npz_paths))
                swap_tier_features[tier_name][i] = npz_data[npz_key]

    # Pass 2: merge addon features
    if addon_dirs:
        for addon_dir in addon_dirs:
            addon_path = Path(addon_dir)
            if not addon_path.exists():
                continue
            for i in range(len(swap_npz_paths)):
                stem = swap_info[i]["file_stem"]
                addon_npz = addon_path / f"{stem}.npz"
                if addon_npz.exists():
                    addon_data = np.load(addon_npz, allow_pickle=True)
                    for tier_name in individual_test_tiers:
                        npz_key = TIER_KEYS.get(tier_name, "")
                        if npz_key and npz_key in addon_data.files:
                            swap_tier_features.setdefault(tier_name, [None] * len(swap_npz_paths))
                            if swap_tier_features[tier_name][i] is None:
                                swap_tier_features[tier_name][i] = addon_data[npz_key]

    complete_tiers = {
        t for t, arrays in swap_tier_features.items()
        if all(a is not None for a in arrays)
    }
    swap_tier_features = {
        t: arrays for t, arrays in swap_tier_features.items() if t in complete_tiers
    }

    for group in ["T2+T2.5", "combined_v2"]:
        if group in test_tiers and group not in swap_tier_features:
            from .signature_io import TIER_GROUPS
            members = TIER_GROUPS.get(group, [])
            available_members = [m for m in members if m in complete_tiers]
            if available_members:
                swap_tier_features[group] = [
                    np.concatenate([swap_tier_features[m][j] for m in available_members])
                    for j in range(len(swap_npz_paths))
                ]

    if not complete_tiers:
        return PromptSwapConfoundResult(
            error="Could not load any complete tier features for swap samples",
        )

    sys_modes = np.array([s["system_prompt_mode"] for s in swap_info])
    exec_modes = np.array([s["execution_mode"] for s in swap_info])

    per_tier: dict[str, PromptSwapTierResult] = {}

    for tier_name in test_tiers:
        if tier_name not in swap_tier_features or not swap_tier_features[tier_name]:
            continue
        if len(swap_tier_features[tier_name]) != len(swap_info):
            continue

        try:
            X_train = data.get_tier(tier_name)
        except KeyError:
            continue

        X_swap = np.stack(swap_tier_features[tier_name], axis=0)
        y_train = data.modes

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_swap_scaled = scaler.transform(X_swap)

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        clf.fit(X_train_scaled, y_train)

        predictions = clf.predict(X_swap_scaled)

        follows_system = int(np.sum(predictions == sys_modes))
        follows_execution = int(np.sum(predictions == exec_modes))
        follows_neither = int(len(predictions) - follows_system - follows_execution)

        per_swap_type: dict[str, Any] = {}
        for swap_name in set(s["swap_name"] for s in swap_info):
            mask = np.array([s["swap_name"] == swap_name for s in swap_info])
            n_this = int(mask.sum())
            preds_this = predictions[mask]
            sys_this = sys_modes[mask]
            exec_this = exec_modes[mask]

            per_swap_type[swap_name] = {
                "n": n_this,
                "follows_system_prompt": int(np.sum(preds_this == sys_this)),
                "follows_execution": int(np.sum(preds_this == exec_this)),
                "follows_neither": int(n_this - np.sum(preds_this == sys_this) - np.sum(preds_this == exec_this)),
                "predictions": predictions[mask].tolist(),
                "system_modes": sys_this.tolist(),
                "execution_modes": exec_this.tolist(),
            }

        per_tier[tier_name] = PromptSwapTierResult(
            n_features=int(X_train.shape[1]),
            follows_system_prompt=follows_system,
            follows_execution=follows_execution,
            follows_neither=follows_neither,
            pct_system=float(follows_system / len(predictions)),
            pct_execution=float(follows_execution / len(predictions)),
            signal_type=(
                "execution_based" if follows_execution > follows_system * 1.5
                else "system_prompt_based" if follows_system > follows_execution * 1.5
                else "ambiguous"
            ),
            per_swap_type=per_swap_type,
        )

    return PromptSwapConfoundResult(
        n_swap_samples=len(swap_info),
        swap_types=list(set(s["swap_name"] for s in swap_info)),
        per_tier=per_tier,
    )


def _get_semantic_test_tiers(data: AnalysisData) -> list[str]:
    """Discover which tiers/groups to test for semantic independence."""
    from .signature_io import BASELINE_TIERS, ENGINEERED_TIERS

    run4 = data.run4
    tiers: list[str] = []

    for tier in BASELINE_TIERS:
        if tier in run4.tier_features:
            tiers.append(tier)

    for tier in ENGINEERED_TIERS:
        if tier in run4.tier_features:
            tiers.append(tier)

    for group in ["T2+T2.5", "engineered", "combined_v2", "T2+T2.5+engineered"]:
        if group in run4.group_features:
            tiers.append(group)

    return tiers


def run_semantic(
    data: AnalysisData,
    signature_dir: Path | str | None = None,
    addon_dirs: list[Path | str] | None = None,
) -> SemanticResult:
    """Run semantic independence analyses."""
    if data.generated_texts is None or all(t == "" for t in data.generated_texts):
        return SemanticResult(error="No generated text available")

    y = data.modes
    topics = data.topics

    print("    TF-IDF surface baseline...")
    X_tfidf = _embed_texts_tfidf(data.generated_texts)
    tfidf_classification = _classifier_bundle(X_tfidf, y, topics, dims=int(X_tfidf.shape[1]))

    print("    Sentence-BERT embeddings...")
    X_sbert = _embed_texts_sbert(data.generated_texts)
    if X_sbert is not None:
        sbert_classification = _classifier_bundle(X_sbert, y, topics, dims=int(X_sbert.shape[1]))
    else:
        sbert_classification = SemanticClassifierBundle(error="sentence-transformers not available")

    # ── Per-tier semantic orthogonality battery ──
    test_tiers = _get_semantic_test_tiers(data)
    print(f"    Testing semantic orthogonality for {len(test_tiers)} tiers: {test_tiers}")

    from scipy.spatial.distance import pdist, squareform

    X_tfidf_std = StandardScaler().fit_transform(X_tfidf)
    X_sbert_std = StandardScaler().fit_transform(X_sbert) if X_sbert is not None else None
    semantic_emb = X_sbert if X_sbert is not None else X_tfidf

    per_tier_semantic: dict[str, PerTierSemanticResult] = {}

    for tier_name in test_tiers:
        print(f"      Tier: {tier_name}...")
        try:
            X_compute = data.get_tier(tier_name)
        except KeyError:
            per_tier_semantic[tier_name] = PerTierSemanticResult(
                error=f"tier {tier_name} not found",
            )
            continue

        classification = SemanticClassifierBundle(
            rf=_classify_condition(X_compute, y, topics, clf_name="rf"),
            knn=_classify_condition(X_compute, y, topics, clf_name="knn"),
        )

        X_comp_std = StandardScaler().fit_transform(X_compute)
        D_compute = squareform(pdist(X_comp_std, metric="cosine"))
        D_tfidf = squareform(pdist(X_tfidf_std, metric="cosine"))
        mantel_tfidf_cosine = _mantel_test(D_compute, D_tfidf)

        mantel_sbert_cosine: MantelResult | None = None
        if X_sbert_std is not None:
            D_sbert = squareform(pdist(X_sbert_std, metric="cosine"))
            mantel_sbert_cosine = _mantel_test(D_compute, D_sbert)

        text_to_compute_r2 = _text_to_compute_r2(semantic_emb, X_compute, topics)
        per_mode = _per_mode_surface_vs_compute(X_tfidf, X_compute, y, topics)
        shuffle_controls = _run_shuffle_controls(X_compute, y, topics)

        per_tier_semantic[tier_name] = PerTierSemanticResult(
            n_features=int(X_compute.shape[1]),
            classification=classification,
            mantel_tfidf_cosine=mantel_tfidf_cosine,
            mantel_sbert_cosine=mantel_sbert_cosine,
            text_to_compute_r2=text_to_compute_r2,
            per_mode_surface_vs_compute=per_mode,
            shuffle_controls=shuffle_controls,
        )

    # ── Legacy top-level keys (T2+T2.5, backward compat) ──
    t2t25_results = per_tier_semantic.get("T2+T2.5")
    X_compute_main = data.get_tier("T2+T2.5")
    compute_classification = t2t25_results.classification if t2t25_results is not None else None

    combined_classification: SemanticClassifierBundle | None = None
    semantic_noise_classification: SemanticClassifierBundle | None = None
    if X_sbert is not None:
        X_combined = np.concatenate([X_compute_main, X_sbert], axis=1)
        combined_classification = _classifier_bundle(
            X_combined, y, topics, dims=int(X_combined.shape[1]),
        )

        rng = np.random.default_rng(42)
        noise = rng.standard_normal((X_sbert.shape[0], X_compute_main.shape[1])).astype(np.float32)
        X_semantic_noise = np.concatenate([X_sbert, noise], axis=1)
        semantic_noise_classification = _classifier_bundle(
            X_semantic_noise, y, topics, dims=int(X_semantic_noise.shape[1]),
        )

    mantel_tfidf_cosine_top = (
        t2t25_results.mantel_tfidf_cosine if t2t25_results is not None else None
    )
    mantel_sbert_cosine_top = (
        t2t25_results.mantel_sbert_cosine if t2t25_results is not None else None
    )

    X_comp_main_std = StandardScaler().fit_transform(X_compute_main)
    D_compute_euc = squareform(pdist(X_comp_main_std, metric="euclidean"))
    D_tfidf_euc = squareform(pdist(X_tfidf_std, metric="euclidean"))
    mantel_tfidf_euclidean = _mantel_test(D_compute_euc, D_tfidf_euc)
    mantel_sbert_euclidean: MantelResult | None = None
    if X_sbert_std is not None:
        D_sbert_euc = squareform(pdist(X_sbert_std, metric="euclidean"))
        mantel_sbert_euclidean = _mantel_test(D_compute_euc, D_sbert_euc)

    # Backward-compat top-level copies
    text_to_compute_r2_top = (
        t2t25_results.text_to_compute_r2 if t2t25_results is not None else None
    )
    per_mode_surface_vs_compute_top = (
        t2t25_results.per_mode_surface_vs_compute if t2t25_results is not None else None
    )
    shuffle_controls_top = (
        t2t25_results.shuffle_controls if t2t25_results is not None else None
    )

    # ── Cross-tier analyses (run once) ──
    print("    Contrastive projection comparison (topic-heldout)...")
    cpc = _contrastive_topic_heldout(X_tfidf, X_compute_main, X_sbert, y, topics)

    print("    Retrieval analysis...")
    retrieval = _run_retrieval_analysis(X_compute_main, X_tfidf, X_sbert, y, topics)

    # ── Prompt-swap confound test ──
    prompt_swap: PromptSwapConfoundResult
    if signature_dir is not None:
        print("    Prompt-swap confound test...")
        try:
            addon_paths = [Path(d) for d in addon_dirs] if addon_dirs else None
            prompt_swap = _run_prompt_swap_confound(
                data, Path(signature_dir), addon_dirs=addon_paths,
            )
            if prompt_swap.per_tier is not None:
                n_swap = prompt_swap.n_swap_samples or 0
                print(f"      {n_swap} swap samples, {len(prompt_swap.per_tier)} tiers tested")
                for tier_name, tier_data in prompt_swap.per_tier.items():
                    print(f"        {tier_name}: {tier_data.signal_type} "
                          f"(exec={tier_data.pct_execution:.0%}, sys={tier_data.pct_system:.0%})")
            elif prompt_swap.error:
                print(f"      Skipped: {prompt_swap.error}")
        except Exception as e:
            logger.warning(f"Prompt-swap confound test failed: {e}")
            prompt_swap = PromptSwapConfoundResult(error=str(e))
    else:
        prompt_swap = PromptSwapConfoundResult(error="signature_dir not provided")

    return SemanticResult(
        tfidf_classification=tfidf_classification,
        sbert_classification=sbert_classification,
        per_tier_semantic=per_tier_semantic,
        compute_classification=compute_classification,
        combined_classification=combined_classification,
        semantic_noise_classification=semantic_noise_classification,
        mantel_tfidf=mantel_tfidf_cosine_top,
        mantel_sbert=mantel_sbert_cosine_top,
        mantel_tfidf_cosine=mantel_tfidf_cosine_top,
        mantel_sbert_cosine=mantel_sbert_cosine_top,
        mantel_tfidf_euclidean=mantel_tfidf_euclidean,
        mantel_sbert_euclidean=mantel_sbert_euclidean,
        text_to_compute_r2=text_to_compute_r2_top,
        per_mode_surface_vs_compute=per_mode_surface_vs_compute_top,
        shuffle_controls=shuffle_controls_top,
        contrastive_projection_comparison=cpc,
        retrieval=retrieval,
        prompt_swap_confound=prompt_swap,
    )


def _run_shuffle_controls(
    X_compute: NDArray, y: NDArray, topics: NDArray, seed: int = 42,
) -> ShuffleControlsResult:
    """Shuffle controls for compute features."""
    rng = np.random.default_rng(seed)

    X_within_shuffled = X_compute.copy()
    for topic in sorted(set(topics)):
        topic_mask = topics == topic
        topic_indices = np.where(topic_mask)[0]
        perm = rng.permutation(len(topic_indices))
        X_within_shuffled[topic_indices] = X_compute[topic_indices[perm]]

    within = _classify_condition(X_within_shuffled, y, topics, clf_name="rf", seed=seed)

    X_global_shuffled = X_compute[rng.permutation(len(X_compute))]
    global_ = _classify_condition(X_global_shuffled, y, topics, clf_name="rf", seed=seed)

    return ShuffleControlsResult(within_topic_shuffle=within, global_shuffle=global_)


def _run_retrieval_analysis(
    X_compute: NDArray, X_tfidf: NDArray, X_sbert: NDArray | None,
    y: NDArray, topics: NDArray, k: int = 5,
) -> RetrievalResult:
    """kNN mode-match retrieval accuracy in raw feature space."""
    from sklearn.neighbors import NearestNeighbors

    feature_sets: dict[str, NDArray] = {
        "compute_t2t25": StandardScaler().fit_transform(X_compute),
        "tfidf": StandardScaler().fit_transform(X_tfidf),
    }
    if X_sbert is not None:
        feature_sets["sbert"] = StandardScaler().fit_transform(X_sbert)
        feature_sets["combined_compute_sbert"] = StandardScaler().fit_transform(
            np.concatenate([X_compute, X_sbert], axis=1),
        )

    neighbor_indices: dict[str, NDArray] = {}
    entries: dict[str, RetrievalFeatureSet] = {}

    for name, X in feature_sets.items():
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        knn_indices = indices[:, 1:]
        neighbor_indices[name] = knn_indices

        mode_matches = np.array([
            np.mean(y[knn_indices[i]] == y[i]) for i in range(len(y))
        ])

        per_mode: dict[str, float] = {}
        for mode in sorted(set(y)):
            mask = y == mode
            per_mode[mode] = float(np.mean(mode_matches[mask]))

        entries[name] = RetrievalFeatureSet(
            mean_mode_match=float(np.mean(mode_matches)),
            std_mode_match=float(np.std(mode_matches)),
            per_mode=per_mode,
            k=k,
        )

    jaccard_compute_tfidf: JaccardStats | None = None
    jaccard_compute_sbert: JaccardStats | None = None
    if "compute_t2t25" in neighbor_indices and "tfidf" in neighbor_indices:
        compute_nn = neighbor_indices["compute_t2t25"]
        tfidf_nn = neighbor_indices["tfidf"]
        jaccards = []
        for i in range(len(y)):
            c_set = set(compute_nn[i])
            t_set = set(tfidf_nn[i])
            intersection = len(c_set & t_set)
            union = len(c_set | t_set)
            jaccards.append(intersection / max(union, 1))
        jaccard_compute_tfidf = JaccardStats(
            mean=float(np.mean(jaccards)),
            std=float(np.std(jaccards)),
        )

    if "compute_t2t25" in neighbor_indices and "sbert" in neighbor_indices:
        compute_nn = neighbor_indices["compute_t2t25"]
        sbert_nn = neighbor_indices["sbert"]
        jaccards = []
        for i in range(len(y)):
            c_set = set(compute_nn[i])
            s_set = set(sbert_nn[i])
            intersection = len(c_set & s_set)
            union = len(c_set | s_set)
            jaccards.append(intersection / max(union, 1))
        jaccard_compute_sbert = JaccardStats(
            mean=float(np.mean(jaccards)),
            std=float(np.std(jaccards)),
        )

    return RetrievalResult(
        compute_t2t25=entries.get("compute_t2t25"),
        tfidf=entries.get("tfidf"),
        sbert=entries.get("sbert"),
        combined_compute_sbert=entries.get("combined_compute_sbert"),
        jaccard_compute_tfidf=jaccard_compute_tfidf,
        jaccard_compute_sbert=jaccard_compute_sbert,
    )
