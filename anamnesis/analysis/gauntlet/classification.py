"""Section 2: Classification — 5-way mode discrimination.

Topic-grouped CV (StratifiedGroupKFold) is the DEFAULT for every number this module
reports — ungrouped StratifiedKFold lets same-topic samples straddle train/test, and
every repetition of a topic shares its prompt, so a classifier can recognise the
topic and be scored as if it had recognised the mode. That leak is what the grouped
default exists to prevent. The ungrouped path survives only as a fallback when no
topic labels exist, and is labeled as legacy in the printed output. Pairwise-binary
accuracies carry no p-values, and the ten-pair grid is one family read off one set
of signatures rather than ten findings: the spread between folds is of the same
order as the spread between pairs, so a pairwise gap can be fold noise and nothing
else. ``cv_stability`` on the key blocks measures that spread directly — its ``std``
and 95% interval are what a pairwise gap has to be weighed against, rather than
against zero.

One number here is not a finding but a ruler: the length-only baseline, a forest on
generated-token count alone. Every mode accuracy in this section is read against it,
because a corpus whose modes generate different amounts of text can be classified by
length while looking like it was classified by computation. It runs first, before the
stability sweeps and the permutation nulls, and it refuses rather than reporting a
stub — a length baseline that says nothing and a length baseline that could not be
measured look identical in a results file, and only one of them is honest.

The two family-level statistics this module reports — the permutation p-value and
the BH-FDR adjustment across the per-block permutation family (q_value on
PermutationTestResult) — come from ``anamnesis.analysis.battery.stats``, which is
their one home in the package. The convention that follows from that, and which
every number here is read under: a permutation p-value carries the add-one
correction, ``(hits + 1) / (n + 1)``, so it is never zero and never finer than
``1 / (n_permutations + 1)``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from anamnesis.analysis.battery.stats import bh_fdr_by_key, permutation_pvalue

from .signature_io import AnalysisData
from .schemas import (
    ClassificationResult,
    ClassifierAccuracyResult,
    ClassifierWithConfusionResult,
    CVStabilityResult,
    LengthOnlyResult,
    PerModeLengthStats,
    PermutationTestResult,
    BlockClassificationResult,
    TopicHeldoutResult,
)
from .utils import get_available_blocks

# n_jobs=1 on RF to avoid joblib thread pool deadlocks.
# We parallelize at the outer loop level instead.
_RF_KWARGS = dict(n_estimators=100, n_jobs=1)


class LengthDataUnavailable(ValueError):
    """No per-sample generated-token count is reachable on a loaded run.

    A refusal rather than a reported absence. The length baseline is the ruler the
    rest of this section is read against, and in a results file a stub saying it
    could not be measured sits in the same field as a number saying length carries
    nothing — so a pass that cannot measure it stops instead of banking the
    ambiguity.
    """


def _make_splits(
    X: NDArray, y: NDArray, groups: NDArray | None, n_splits: int, seed: int,
) -> list[tuple[NDArray, NDArray]]:
    """CV splits: topic-grouped + stratified when groups exist (leak-proof default),
    plain stratified otherwise (legacy fallback — pre-v3 behavior)."""
    if groups is not None:
        n_eff = min(n_splits, len(np.unique(groups)))
        sgkf = StratifiedGroupKFold(n_splits=n_eff, shuffle=True, random_state=seed)
        return list(sgkf.split(X, y, groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def _run_rf_cv(
    X: NDArray, y: NDArray, n_splits: int = 5, seed: int = 42,
    return_confusion: bool = True, groups: NDArray | None = None,
) -> ClassifierWithConfusionResult:
    """Single RF cross-validation run.

    Returns a typed result; callers that expect the lighter
    ``ClassifierAccuracyResult`` shape can narrow after the fact.
    """
    fold_accs: list[float] = []
    all_y_true: list = []
    all_y_pred: list = []

    for train_idx, test_idx in _make_splits(X, y, groups, n_splits, seed):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)

        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))
        if return_confusion:
            all_y_true.extend(y[test_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

    confusion = None
    labels = None
    if return_confusion and all_y_true:
        labels = sorted(set(all_y_true))
        confusion = confusion_matrix(
            all_y_true, all_y_pred, labels=labels,
        ).tolist()

    return ClassifierWithConfusionResult(
        accuracy=float(np.mean(fold_accs)),
        fold_accuracies=fold_accs,
        confusion_matrix=confusion,
        labels=labels,
    )


def _run_topic_heldout(
    X: NDArray, y: NDArray, topics: NDArray, n_splits: int = 5, seed: int = 42,
) -> TopicHeldoutResult:
    """GroupKFold by topic."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    actual_splits = min(n_splits, len(unique_topics))
    gkf = GroupKFold(n_splits=actual_splits)

    fold_accs: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))

    return TopicHeldoutResult(
        accuracy=float(np.mean(fold_accs)),
        fold_accuracies=fold_accs,
        n_groups=len(unique_topics),
    )


# ── Parallelized helpers for expensive loops ──

def _cv_single_seed(args: tuple) -> float:
    """Worker for CV stability — one seed, returns accuracy."""
    X, y, groups, seed = args
    return float(
        _run_rf_cv(X, y, seed=seed, return_confusion=False, groups=groups).accuracy
    )


def _perm_single(args: tuple) -> float:
    """Worker for permutation test — one permuted-label RF, returns accuracy."""
    X, y_perm, groups, seed = args
    return float(
        _run_rf_cv(X, y_perm, seed=seed, return_confusion=False, groups=groups).accuracy
    )


def _run_cv_stability(
    X: NDArray, y: NDArray, n_seeds: int = 50, groups: NDArray | None = None,
) -> CVStabilityResult:
    """RF accuracy across multiple CV seeds, parallelized."""
    n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(X, y, groups, seed) for seed in range(n_seeds)]

    accs: list[float] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_cv_single_seed, a) for a in args_list]
        for f in as_completed(futures):
            accs.append(f.result())

    return CVStabilityResult(
        mean=float(np.mean(accs)),
        median=float(np.median(accs)),
        std=float(np.std(accs)),
        ci_lo=float(np.percentile(accs, 2.5)),
        ci_hi=float(np.percentile(accs, 97.5)),
        min=float(np.min(accs)),
        max=float(np.max(accs)),
        all_accuracies=sorted(accs),
        n_seeds=n_seeds,
    )


def _run_permutation_test(
    X: NDArray, y: NDArray, n_permutations: int = 500, seed: int = 42,
    groups: NDArray | None = None,
) -> PermutationTestResult:
    """Permutation test for RF accuracy, parallelized.

    The labels are shuffled ``n_permutations`` times and the whole cross-validation
    rerun on each shuffle, so the null carries the same fold structure and the same
    grouping as the observed number. The reported ``p_value`` is
    ``permutation_pvalue`` from ``anamnesis.analysis.battery.stats``: add-one
    corrected, so it is never zero and never finer than
    ``1 / (n_permutations + 1)``.
    """
    observed = float(
        _run_rf_cv(X, y, seed=seed, return_confusion=False, groups=groups).accuracy
    )

    rng = np.random.default_rng(seed)
    perm_args = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_args.append((X, y_perm, groups, seed))

    n_workers = max(1, mp.cpu_count() - 1)
    null_accs: list[float] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_perm_single, a) for a in perm_args]
        for f in as_completed(futures):
            null_accs.append(f.result())

    null_arr = np.array(null_accs)

    return PermutationTestResult(
        observed_accuracy=observed,
        p_value=permutation_pvalue(observed, null_arr),
        null_mean=float(np.mean(null_arr)),
        null_std=float(np.std(null_arr)),
        null_max=float(np.max(null_arr)),
        null_p95=float(np.percentile(null_arr, 95)),
        null_p99=float(np.percentile(null_arr, 99)),
        n_permutations=n_permutations,
    )


def _run_pairwise_binary(
    X: NDArray, y: NDArray, seed: int = 42, groups: NDArray | None = None,
) -> dict[str, ClassifierAccuracyResult]:
    """Binary RF for all mode pairs.

    No per-pair p-values are computed. Read the 10-pair grid against the
    fold-to-fold spread ``cv_stability`` reports, not as ten independent findings.
    """
    modes = sorted(set(y))
    results: dict[str, ClassifierAccuracyResult] = {}
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            mask = np.isin(y, [m1, m2])
            X_sub = X[mask]
            y_sub = (y[mask] == m1).astype(int)
            g_sub = groups[mask] if groups is not None else None

            fold_accs: list[float] = []
            for train_idx, test_idx in _make_splits(X_sub, y_sub, g_sub, 5, seed):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_sub[train_idx])
                X_te = scaler.transform(X_sub[test_idx])

                clf = RandomForestClassifier(**_RF_KWARGS, random_state=seed)
                clf.fit(X_tr, y_sub[train_idx])
                y_pred = clf.predict(X_te)
                fold_accs.append(float(accuracy_score(y_sub[test_idx], y_pred)))

            results[f"{m1}_vs_{m2}"] = ClassifierAccuracyResult(
                accuracy=float(np.mean(fold_accs)),
                fold_accuracies=fold_accs,
            )

    return results


def _run_4way_no_analogical(
    X: NDArray, y: NDArray, seed: int = 42, groups: NDArray | None = None,
) -> ClassifierWithConfusionResult:
    """4-way classification excluding analogical."""
    mask = y != "analogical"
    if np.sum(mask) == 0:
        return ClassifierWithConfusionResult(error="no analogical mode found")
    g_sub = groups[mask] if groups is not None else None
    return _run_rf_cv(X[mask], y[mask], seed=seed, groups=g_sub)


def _run_linear_probe(
    X: NDArray, y: NDArray, seed: int = 42, groups: NDArray | None = None,
) -> ClassifierAccuracyResult:
    """Logistic regression classification."""
    fold_accs: list[float] = []

    for train_idx, test_idx in _make_splits(X, y, groups, 5, seed):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        fold_accs.append(float(accuracy_score(y[test_idx], y_pred)))

    return ClassifierAccuracyResult(
        accuracy=float(np.mean(fold_accs)),
        fold_accuracies=fold_accs,
    )


def run_classification(data: AnalysisData) -> ClassificationResult:
    """Run all classification analyses across feature groups."""
    y = data.modes
    topics = data.topics
    by_block: dict[str, BlockClassificationResult] = {}

    # Topic-grouped CV everywhere: the leak-proof default, because all
    # repetitions of a topic share its prompt.
    groups: NDArray | None = None
    if topics is not None and len(topics) == len(y):
        topic_to_id = {t: i for i, t in enumerate(sorted(set(topics)))}
        groups = np.array([topic_to_id[t] for t in topics])
        print(f"  CV: StratifiedGroupKFold by topic ({len(topic_to_id)} groups)")
    else:
        print("  CV: WARNING — no topic labels; falling back to ungrouped "
              "StratifiedKFold (legacy, topic-leak-prone)")

    # The confound check comes first. It is seconds of work, every accuracy below is
    # read against it, and a run that cannot supply lengths should refuse before the
    # stability sweeps and the thousand-permutation nulls rather than after them.
    print("  Length-only baseline...")
    length_only = _run_length_only_baseline(data, y, groups=groups)
    print(f"    Length-only RF: {length_only.accuracy:.1%} "
          f"against {1 / len(set(y)):.1%} chance")

    available_blocks, key_blocks = get_available_blocks(data)
    for block in available_blocks:
        print(f"  Classification: {block}")
        X = data.get_block(block)

        rf_5way = _run_rf_cv(X, y, groups=groups)
        topic_heldout = _run_topic_heldout(X, y, topics)
        linear_probe = _run_linear_probe(X, y, groups=groups)
        pairwise = _run_pairwise_binary(X, y, groups=groups)
        rf_4way = _run_4way_no_analogical(X, y, groups=groups)

        cv_stability = None
        permutation = None
        if block in key_blocks:
            print(f"    CV stability ({block})...")
            cv_stability = _run_cv_stability(X, y, n_seeds=100, groups=groups)

            print(f"    Permutation test ({block})...")
            permutation = _run_permutation_test(
                X, y, n_permutations=1000, groups=groups,
            )

        by_block[block] = BlockClassificationResult(
            rf_5way=rf_5way,
            topic_heldout=topic_heldout,
            linear_probe=linear_probe,
            pairwise_binary=pairwise,
            rf_4way_no_analogical=rf_4way,
            cv_stability=cv_stability,
            permutation_test=permutation,
        )
        print(f"    Done: {block} RF={rf_5way.accuracy:.1%}")

    # BH-FDR across the per-block permutation family: the blocks are nested reads of
    # one set of signatures, so their p-values are one family and not independent
    # findings. A single block has nothing to correct against and keeps q_value None.
    perm_ps = {
        t: r.permutation_test.p_value
        for t, r in by_block.items()
        if r.permutation_test is not None
    }
    if len(perm_ps) > 1:
        for t, q in bh_fdr_by_key(perm_ps).items():
            r = by_block[t]
            by_block[t] = r.model_copy(
                update={
                    "permutation_test": r.permutation_test.model_copy(
                        update={"q_value": q},
                    ),
                },
            )

    return ClassificationResult(by_block=by_block, length_only=length_only)


def _generation_lengths(data: AnalysisData) -> NDArray[np.float64]:
    """Generated tokens per sample, in the row order of the feature matrices.

    ``SampleMeta.num_generated_tokens`` is this number's one home: a required field
    of every loaded sample, filled from that signature's own sidecar, so it is there
    whether or not the pass also loaded generated text. It is read by name off the
    typed object rather than searched for among candidate spellings, because a
    search that comes up empty produces a baseline-shaped absence, while a name that
    has moved produces a failure somebody fixes.
    """
    try:
        samples = data.run4.samples
        lengths = np.array(
            [float(sample.num_generated_tokens) for sample in samples], dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise LengthDataUnavailable(
            "the length baseline reads run4.samples[i].num_generated_tokens and the "
            f"loaded run does not supply it ({exc})"
        ) from exc
    if lengths.size != data.n_samples:
        raise LengthDataUnavailable(
            f"{lengths.size} sample lengths against {data.n_samples} feature rows; "
            "a baseline aligned to the wrong rows measures nothing"
        )
    if not np.all(np.isfinite(lengths)):
        raise LengthDataUnavailable(
            "at least one sample carries a non-finite generated-token count"
        )
    return lengths


def _run_length_only_baseline(
    data: AnalysisData, y: NDArray, groups: NDArray | None = None,
) -> LengthOnlyResult:
    """RF on generated-token count alone, under the same folds as everything else.

    Chance is one over the number of modes. Near chance says the mode signal is not
    length in disguise. Well above chance says the modes generate different amounts
    of text, and then every accuracy in this section has to be read against this
    number rather than against chance — which is why the per-mode length
    distributions come back beside the accuracy: a mode whose generations end before
    the token cap while the rest are truncated at it is separable on length alone,
    and the distributions are what name it.

    Raises
    ------
    LengthDataUnavailable
        When no per-sample length is reachable.
    """
    lengths = _generation_lengths(data)
    X_length = lengths.reshape(-1, 1)
    base = _run_rf_cv(X_length, y, return_confusion=True, groups=groups)

    per_mode_lengths: dict[str, PerModeLengthStats] = {}
    for mode in sorted(set(y)):
        mask = y == mode
        mode_lens = lengths[mask]
        per_mode_lengths[mode] = PerModeLengthStats(
            mean=float(np.mean(mode_lens)),
            std=float(np.std(mode_lens)),
            min=float(np.min(mode_lens)),
            max=float(np.max(mode_lens)),
        )

    return LengthOnlyResult(
        accuracy=base.accuracy,
        fold_accuracies=base.fold_accuracies,
        confusion_matrix=base.confusion_matrix,
        labels=base.labels,
        per_mode_lengths=per_mode_lengths,
    )
