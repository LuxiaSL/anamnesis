"""Section 2's machinery, at test scale.

``run_classification`` is not called here: on its key composite blocks it runs a
hundred-seed stability sweep and a thousand-permutation null, which is minutes of
CPU per block and belongs to a real pass rather than to a suite. The parts it is
made of are all cheap at small parameters, and they are where the properties that
matter live — so they are tested directly:

  * the split helper is topic-grouped by default and falls back to plain
    stratified folds only when there are no groups, which is the leak-proof
    default the corpus of record was rerun under;
  * the grouped readouts actually refuse a topic-local rule. ``confounded`` is
    built so that the only way to score is to recognise the topic, and the
    grouped readouts are asserted near chance on it while the ungrouped fallback
    scores high. A splitter that stopped honouring its groups would pass every
    count-and-shape assertion in this file, so the confound is what pins it;
  * the permutation p-value is the add-one statistic from
    ``anamnesis.analysis.battery.stats``, recomputed here from a reproduced null
    so that the two call sites are checked against each other rather than each
    against itself;
  * BH-FDR over the per-block permutation family is monotone in p;
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

from anamnesis.analysis.battery.stats import (
    bh_fdr_by_key,
    permutation_pvalue,
    permutation_resolution,
)
from anamnesis.analysis.gauntlet.classification import (
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

#: The confound fixture's shape: ten topics, five repetitions of each mode in each.
CONFOUND_TOPICS = 10
CONFOUND_REPS = 5
#: Chance on five balanced modes.
CHANCE = 1.0 / len(MODES)


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


@pytest.fixture(scope="module")
def confounded() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deliberate topic confound: readable inside a topic, worthless across topics.

    Each row is a topic fingerprint (three dimensions, one well-separated vector
    per topic) beside a mode value (two dimensions), and the mode → value rule is an
    independent permutation in every topic. So a classifier that may train and test
    inside one topic reads the fingerprint, applies that topic's rule and scores
    near one; a classifier held out by topic meets a fingerprint it has never seen
    and a rule it cannot have learned, and scores near chance.

    The rows are shuffled, because plain ``StratifiedKFold`` assigns folds by
    position within a class: topic-major rows would land whole topics in single
    folds and hide the leak behind an accident of ordering.
    """
    rng = np.random.default_rng(20260921)
    X, y, topics = [], [], []
    for topic_idx in range(CONFOUND_TOPICS):
        fingerprint = 5.0 * rng.standard_normal(3)
        rule = rng.permutation(len(MODES)).astype(float)
        for mode_idx, mode in enumerate(MODES):
            for _ in range(CONFOUND_REPS):
                X.append(
                    np.concatenate([
                        fingerprint + 0.1 * rng.standard_normal(3),
                        np.full(2, rule[mode_idx]) + 0.1 * rng.standard_normal(2),
                    ])
                )
                y.append(mode)
                topics.append(f"topic_{topic_idx}")
    order = rng.permutation(len(y))
    return np.array(X)[order], np.array(y)[order], np.array(topics)[order]


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


def test_rf_cv_scores_a_topic_local_rule_at_chance_when_grouped(confounded) -> None:
    """The grouped path must not cash in a rule that holds only inside a topic.

    Both halves are asserted: grouped near chance, and the ungrouped fallback high
    on the same matrix. Without the second half the first could pass because the
    features are weak; with it, the gap is the confound being refused.
    """
    X, y, topics = confounded
    grouped = _run_rf_cv(X, y, groups=groups_of(topics))
    ungrouped = _run_rf_cv(X, y, groups=None)

    assert ungrouped.accuracy > 0.70, (
        "the confound is real: allowed to train and test inside a topic, the forest "
        "reads the fingerprint and applies that topic's rule"
    )
    assert grouped.accuracy < CHANCE + 0.20, (
        "held out by topic, the topic-local rule is worth nothing and the forest "
        f"sits near chance ({CHANCE:.0%}), not at the ungrouped "
        f"{ungrouped.accuracy:.0%}"
    )


def test_topic_heldout_reports_the_group_count_it_actually_had(separable) -> None:
    X, y, topics = separable
    result = _run_topic_heldout(X, y, topics)
    assert isinstance(result, TopicHeldoutResult)
    assert result.n_groups == N_TOPICS
    assert len(result.fold_accuracies) == 5
    assert result.accuracy > 0.8


def test_topic_heldout_holds_topics_out_and_lands_at_chance_on_a_confound(
    confounded,
) -> None:
    """The topic-held-out readout scores a topic-local rule at chance.

    This is the readout whose whole purpose is the leak, so the number is what has
    to be pinned, not the group count: ``n_groups`` is counted off the topic labels
    and reads the same whether or not the splitter honours them. Ungrouped folds on
    the same matrix are asserted beside it, so the assertion is a gap rather than a
    bare threshold.
    """
    X, y, topics = confounded
    result = _run_topic_heldout(X, y, topics)
    ungrouped = _run_rf_cv(X, y, groups=None)

    assert result.n_groups == CONFOUND_TOPICS
    assert len(result.fold_accuracies) == 5, "ten topics, five folds of two"
    assert ungrouped.accuracy > 0.70, "the confound is readable within a topic"
    assert result.accuracy < CHANCE + 0.20, (
        f"holding the topic out leaves chance ({CHANCE:.0%}); scoring like the "
        f"ungrouped {ungrouped.accuracy:.0%} would mean the folds share topics"
    )


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
    assert result.p_value >= permutation_resolution(12), "1/(N+1) is the resolution"
    assert result.observed_accuracy > result.null_mean
    assert result.null_p95 <= result.null_max
    assert result.q_value is None, "the q-value is attached by the section, not here"


def test_the_section_reports_the_p_value_the_battery_computes(separable) -> None:
    """The two call sites agree, checked by recomputing one against the other.

    The null is reproducible — one seed drives the label shuffles and every forest
    grown on them — so the section's number can be rebuilt here and handed to
    ``permutation_pvalue`` directly. Agreement is the receipt that the section owns
    no second copy of the arithmetic.
    """
    X, y, topics = separable
    groups = groups_of(topics)
    n_permutations = 8
    result = _run_permutation_test(
        X, y, n_permutations=n_permutations, groups=groups, seed=42,
    )

    rng = np.random.default_rng(42)
    null = np.array([
        _run_rf_cv(
            X, rng.permutation(y), seed=42, return_confusion=False, groups=groups,
        ).accuracy
        for _ in range(n_permutations)
    ])
    assert result.null_mean == pytest.approx(float(np.mean(null))), (
        "the reproduced null matches the one the section drew"
    )
    assert result.p_value == pytest.approx(
        permutation_pvalue(result.observed_accuracy, null)
    )


def test_the_permutation_p_value_carries_the_add_one_correction() -> None:
    """The reported p-value is (hits+1)/(N+1), not the hit rate with a floor under it.

    Features that are pure noise put the observation in the middle of its own null,
    so some permutations reach it and some do not — and that is the only case where
    the two conventions differ. ``(hits + 1) / (N + 1)`` lands on the 1/(N+1)
    lattice; ``hits / N`` clamped from below at 1/(N+1) lands off it and lower, by a
    factor approaching two at one hit. The fixture is built here rather than shared,
    so what this test depends on is the arithmetic and not any property of a
    splitter.
    """
    rng = np.random.default_rng(2)
    n_samples = 60
    X = rng.standard_normal((n_samples, 4))
    y = np.array([MODES[i % len(MODES)] for i in range(n_samples)])
    groups = np.array([i % N_TOPICS for i in range(n_samples)])

    n_permutations = 12
    result = _run_permutation_test(
        X, y, n_permutations=n_permutations, groups=groups, seed=42,
    )

    lattice_position = result.p_value * (n_permutations + 1)
    assert lattice_position == pytest.approx(round(lattice_position), abs=1e-9), (
        "an add-one p-value is a whole number of 1/(N+1) steps"
    )
    hits = round(lattice_position) - 1
    assert 1 <= hits <= n_permutations - 1, (
        "the observation sits inside its own null, which is the band the two "
        f"conventions disagree over; got {hits} of {n_permutations}"
    )
    assert result.p_value == pytest.approx((hits + 1) / (n_permutations + 1))
    assert result.p_value > max(hits / n_permutations, permutation_resolution(
        n_permutations
    )), "the hit rate with a floor under it is the anti-conservative reading"


def test_cv_stability_summarises_the_seeds_it_ran(separable) -> None:
    X, y, topics = separable
    result = _run_cv_stability(X, y, n_seeds=6, groups=groups_of(topics))
    assert result.n_seeds == 6
    assert len(result.all_accuracies) == 6
    assert result.min <= result.median <= result.max
    assert result.ci_lo <= result.median <= result.ci_hi


def test_bh_fdr_is_monotone_and_never_exceeds_one() -> None:
    """The per-block family is corrected by the shared step-up, keyed by block."""
    q = bh_fdr_by_key({"a": 0.001, "b": 0.02, "c": 0.5, "d": 0.9})
    assert set(q) == {"a", "b", "c", "d"}
    ordered = [q[k] for k in ["a", "b", "c", "d"]]
    assert ordered == sorted(ordered), "q-values follow the p-value order"
    assert all(0.0 <= v <= 1.0 for v in q.values())
    assert q["a"] >= 0.001, "adjustment can only raise a p-value"
    assert bh_fdr_by_key({"only": 0.04})["only"] == pytest.approx(0.04)
    assert bh_fdr_by_key({}) == {}, "a family with no members corrects to nothing"


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
