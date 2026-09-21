"""Section 2's machinery, at test scale.

``run_classification`` is not called here: on its key composite tiers it runs a
hundred-seed stability sweep and a thousand-permutation null, which is minutes of
CPU per tier and belongs to a real pass rather than to a suite. The parts it is
made of are all cheap at small parameters, and they are where the properties that
matter live — so they are tested directly:

  * the split helper is topic-grouped by default and falls back to plain
    stratified folds only when there are no groups, which is the leak-proof
    default the corpus of record was rerun under;
  * the permutation p-value cannot report zero, because with N permutations the
    smallest resolvable p is 1/(N+1);
  * BH-FDR over the per-tier permutation family is monotone in p;
  * the pairwise and four-way readouts name their own conditions, so a missing
    mode is an error stub rather than a silently smaller comparison.

One behaviour is pinned here as it stands rather than as it should be: the
length-only confound baseline probes attribute names (``token_counts``, ``texts``,
``run4.metadata``) that the loaded data object does not carry, so on a real run it
reports "no length data available" instead of measuring the confound. The port
carried it unchanged; the test records what it does so that fixing it is a
visible change rather than a surprise.

CPU only; no banked data, no model, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.classification import (
    _bh_fdr,
    _make_splits,
    _run_4way_no_analogical,
    _run_cv_stability,
    _run_length_only_baseline,
    _run_linear_probe,
    _run_pairwise_binary,
    _run_permutation_test,
    _run_rf_cv,
    _run_topic_heldout,
)
from anamnesis.analysis.gauntlet.schemas import (
    ClassifierAccuracyResult,
    ClassifierWithConfusionResult,
    LengthOnlyResult,
    TopicHeldoutResult,
)

MODES = ["linear", "socratic", "contrastive", "dialectical", "analogical"]
N_TOPICS = 6


@pytest.fixture(scope="module")
def separable() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five modes × six topics, each mode on its own offset: findable but noisy."""
    rng = np.random.default_rng(20260920)
    X, y, topics = [], [], []
    for mode_idx, mode in enumerate(MODES):
        for topic_idx in range(N_TOPICS):
            X.append(float(mode_idx) + 0.2 * rng.standard_normal(6))
            y.append(mode)
            topics.append(f"topic_{topic_idx}")
    return np.array(X), np.array(y), np.array(topics)


def groups_of(topics: np.ndarray) -> np.ndarray:
    order = {t: i for i, t in enumerate(sorted(set(topics)))}
    return np.array([order[t] for t in topics])


def test_splits_are_topic_grouped_by_default(separable) -> None:
    X, y, topics = separable
    groups = groups_of(topics)
    grouped = _make_splits(X, y, groups, n_splits=5, seed=42)
    assert len(grouped) == 5
    for train_idx, test_idx in grouped:
        assert not (set(groups[train_idx].tolist()) & set(groups[test_idx].tolist()))

    # No groups is the legacy fallback, and it does let a topic straddle.
    ungrouped = _make_splits(X, y, None, n_splits=5, seed=42)
    assert len(ungrouped) == 5
    assert sum(len(te) for _tr, te in ungrouped) == len(y)


def test_splits_never_ask_for_more_folds_than_groups(separable) -> None:
    X, y, topics = separable
    two_topics = np.where(groups_of(topics) < 2, groups_of(topics), 0)
    splits = _make_splits(X, y, two_topics, n_splits=5, seed=42)
    assert len(splits) == 2, "the fold count is clamped to the group count"


def test_rf_cv_reports_its_folds_and_its_confusion(separable) -> None:
    X, y, topics = separable
    result = _run_rf_cv(X, y, groups=groups_of(topics))
    assert isinstance(result, ClassifierWithConfusionResult)
    assert result.accuracy is not None and result.accuracy > 0.8
    assert result.fold_accuracies is not None and len(result.fold_accuracies) == 5
    assert result.labels == sorted(MODES)
    assert result.confusion_matrix is not None
    assert sum(sum(row) for row in result.confusion_matrix) == len(y)

    lean = _run_rf_cv(X, y, groups=groups_of(topics), return_confusion=False)
    assert lean.confusion_matrix is None and lean.labels is None


def test_topic_heldout_reports_the_group_count_it_actually_had(separable) -> None:
    X, y, topics = separable
    result = _run_topic_heldout(X, y, topics)
    assert isinstance(result, TopicHeldoutResult)
    assert result.n_groups == N_TOPICS
    assert len(result.fold_accuracies) == 5
    assert result.accuracy > 0.8


def test_linear_probe_and_pairwise_name_their_conditions(separable) -> None:
    X, y, topics = separable
    groups = groups_of(topics)
    probe = _run_linear_probe(X, y, groups=groups)
    assert isinstance(probe, ClassifierAccuracyResult)
    assert probe.accuracy > 0.8

    pairwise = _run_pairwise_binary(X, y, groups=groups)
    assert len(pairwise) == 10, "five modes give ten unordered pairs"
    for key, entry in pairwise.items():
        a, b = key.split("_vs_")
        assert a in MODES and b in MODES
        assert entry.accuracy > 0.8


def test_the_four_way_readout_says_when_analogical_is_absent(separable) -> None:
    X, y, topics = separable
    groups = groups_of(topics)
    present = _run_4way_no_analogical(X, y, groups=groups)
    assert present.error is None and present.accuracy is not None
    assert present.labels is not None and "analogical" not in present.labels

    only_analogical = y == "analogical"
    absent = _run_4way_no_analogical(
        X[only_analogical], y[only_analogical], groups=groups[only_analogical]
    )
    assert absent.error is not None, "a comparison with nothing left to compare says so"
    assert absent.accuracy is None


def test_a_permutation_p_value_cannot_be_reported_as_zero(separable) -> None:
    X, y, topics = separable
    result = _run_permutation_test(X, y, n_permutations=12, groups=groups_of(topics))
    assert result.n_permutations == 12
    assert result.p_value >= 1.0 / 13, "the resolution floor is 1/(N+1), not 0"
    assert result.observed_accuracy > result.null_mean
    assert result.null_p95 <= result.null_max
    assert result.q_value is None, "the q-value is attached by the section, not here"


def test_cv_stability_summarises_the_seeds_it_ran(separable) -> None:
    X, y, topics = separable
    result = _run_cv_stability(X, y, n_seeds=6, groups=groups_of(topics))
    assert result.n_seeds == 6
    assert len(result.all_accuracies) == 6
    assert result.min <= result.median <= result.max
    assert result.ci_lo <= result.median <= result.ci_hi


def test_bh_fdr_is_monotone_and_never_exceeds_one() -> None:
    q = _bh_fdr({"a": 0.001, "b": 0.02, "c": 0.5, "d": 0.9})
    assert set(q) == {"a", "b", "c", "d"}
    ordered = [q[k] for k in ["a", "b", "c", "d"]]
    assert ordered == sorted(ordered), "q-values follow the p-value order"
    assert all(0.0 <= v <= 1.0 for v in q.values())
    assert q["a"] >= 0.001, "adjustment can only raise a p-value"
    assert _bh_fdr({"only": 0.04})["only"] == pytest.approx(0.04)


def test_the_length_only_baseline_reports_when_it_has_no_lengths(separable) -> None:
    _X, y, _topics = separable

    class NoLengths:
        n_samples = len(y)
        run4 = object()

    result = _run_length_only_baseline(NoLengths(), y)
    assert isinstance(result, LengthOnlyResult)
    assert result.accuracy is None
    assert result.error == "no length data available"

    # Given lengths under one of the names it probes, it does measure the confound.
    class WithLengths:
        n_samples = len(y)
        token_counts = [100 + 10 * MODES.index(m) for m in y]
        run4 = object()

    measured = _run_length_only_baseline(WithLengths(), y)
    assert measured.error is None
    assert measured.accuracy is not None and measured.accuracy > 0.5
    assert measured.per_mode_lengths is not None
    assert set(measured.per_mode_lengths) == set(MODES)
