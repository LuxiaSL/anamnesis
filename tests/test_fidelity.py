"""Fidelity gates: an unexecuted check must never read as a pass.

This is the arithmetic of the verdict, not of the features. The module under test is
handed a reference batch, a candidate batch, an independent repeat of the candidate,
and a ruler; it renders the gates. Everything it can get wrong is a way of being
accidentally generous, so almost every test here asserts a refusal.

The gate order is load-bearing. Repeatability is evaluated first and, when it
fails, the agreement gate reports `None` rather than a boolean — a candidate whose own
repeat differs has not been measured, and `False` would misstate what is known. The
same distinction returns at the row level: a row whose bound is a declared lower bound
and whose distance exceeds it is `requires_full_floor`, meaning more work is owed, not
that the row failed.

The subtle case is a common shift. Two rows with identical inputs must cancel exactly;
a candidate that shifts both by the same amount preserves their difference, passes the
per-feature tolerance, and still fails the literal condition gate, because the gate is
scaled by a contrast distance that is zero. Exempting it is possible but must be
declared — a ruled zero-pair with a condition manifest digest, and the cancellation is
then verified byte-exact rather than trusted.

Three further refusals worth naming: a candidate and its "repeat" may not be the same
capture (shared replay ids), a null contrast needs an explicit stated reason rather than
an invented nonzero comparator, and a downstream statistic must reproduce its pinned
anchor before it can be used as a ruler at all.

Pure numpy; no lane, no model, no device.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anamnesis.extraction.equivalence.fidelity import (
    AGREEMENT,
    BANKED_GATE_KEYS,
    COVERED,
    DOWNSTREAM_PRESERVED,
    REPEATABLE,
    SINGLE_LANE,
    WITHIN_MEMORY,
    DownstreamAnchor,
    FidelityError,
    ReplayBatch,
    Ruler,
    verify_coverage,
    verify_sweep_coverage,
    verify_bank_scope_coverage,
    verify_downstream,
    verify_memory,
    verify_vectors,
    verify_within_tol,
    read_gate_receipt,
)


@pytest.fixture
def evidence():
    keys = tuple(("bank", f"d{i}", "F", 0, "h1") for i in range(6))
    names = ("a", "b", "c")
    cpu = ReplayBatch(
        np.arange(18, dtype=np.float32).reshape(6, 3),
        keys,
        names,
        ("cpu",) * 6,
        ("a" * 64,) * 6,
        tuple(f"cpu{i}" for i in range(6)),
        "b" * 64,
        "c" * 64,
        "d" * 64,
    )
    gpu = replace(
        cpu, lane_ids=("gpu",) * 6, replay_ids=tuple(f"gpu{i}" for i in range(6))
    )
    repeat = replace(gpu, replay_ids=tuple(f"repeat{i}" for i in range(6)))
    distances = {
        k: float(
            np.linalg.norm(cpu.features[i].astype(float) - cpu.features[(i + 1) % 6])
        )
        for i, k in enumerate(keys)
    }
    ruler = Ruler(
        np.ones(3),
        np.ones(3),
        "e" * 64,
        names,
        dict.fromkeys(keys, 0.05),
        distances,
        {k: (k, keys[(i + 1) % 6]) for i, k in enumerate(keys)},
        {k: i + 1 for i, k in enumerate(keys)},
        keys,
        "c" * 64,
        "d" * 64,
        "f" * 64,
    )
    return cpu, gpu, repeat, ruler


def test_identical_input_null_common_shift_still_fails_literal_condition_gate(evidence):
    cpu, gpu, repeat, ruler = evidence
    values = cpu.features.copy()
    values[1] = values[0]
    cpu = replace(cpu, features=values)
    shifted = values.copy()
    shifted[:, 0] += np.float32(0.00005)
    gpu = replace(gpu, features=shifted)
    repeat = replace(repeat, features=shifted.copy())
    pairs = dict(ruler.condition_pair)
    pairs[cpu.keys[0]] = (cpu.keys[0], cpu.keys[1])
    pairs[cpu.keys[1]] = (cpu.keys[1], cpu.keys[0])
    lookup = {k: i for i, k in enumerate(cpu.keys)}
    distances = {
        k: float(np.linalg.norm(values[lookup[a]].astype(float) - values[lookup[b]]))
        for k, (a, b) in pairs.items()
    }
    ruler = replace(ruler, condition_pair=pairs, condition_distance=distances)
    assert np.array_equal(
        cpu.features[0] - cpu.features[1], gpu.features[0] - gpu.features[1]
    )
    result = verify_vectors(cpu, gpu, repeat, ruler)
    assert result[REPEATABLE] is True
    assert result[AGREEMENT] is False
    assert result["rows"][0]["bound"] == 0
    assert result["rows"][0]["floor_c"] < ruler.floor_b[cpu.keys[0]]
    assert result["max_feature_sigma_error"] < 0.1

    ruled = replace(
        ruler,
        identical_input_zero_pairs=frozenset(cpu.keys[:2]),
        condition_manifest_sha256="a" * 64,
    )
    assert verify_vectors(cpu, gpu, repeat, ruled)[AGREEMENT] is True
    different_inputs = list(cpu.input_sha256)
    different_inputs[1] = "f" * 64
    with pytest.raises(FidelityError, match="inputs differ"):
        verify_vectors(
            *[
                replace(b, input_sha256=tuple(different_inputs))
                for b in (cpu, gpu, repeat)
            ],
            ruled,
        )
    bad = shifted.copy()
    bad[1, 0] += np.float32(0.00001)
    with pytest.raises(FidelityError, match="cancellation"):
        verify_vectors(
            cpu, replace(gpu, features=bad), replace(repeat, features=bad), ruled
        )


def test_cpu_self_passes_all_vector_gates(evidence):
    cpu, _, _, ruler = evidence
    other = replace(cpu, replay_ids=tuple(f"repeat{i}" for i in range(6)))
    result = verify_vectors(cpu, cpu, other, ruler)
    assert result[REPEATABLE] and result[AGREEMENT] and result[SINGLE_LANE]
    assert [t["n"] for t in result["terciles"]] == [2, 2, 2]


def test_ratified_sweep_composition_is_scoped(evidence):
    _, gpu, _, _ = evidence
    regimes = {
        name: gpu.keys
        for name in (
            "near-floor-B-E",
            "near-floor-F-E",
            "E",
            "null-fork-0",
            "null-fork-1",
            "null-fork-2",
        )
    }
    sweep = dict(
        lane_id="gpu",
        coverage_complete=True,
        independent_repeats_exact=True,
        per_feature_pass=True,
        lengths=[
            dict(cache_length=n, n=5, delta_self=0)
            for n in (64, 128, 299, 512, 1024, 2048)
        ],
    )
    result = verify_sweep_coverage(gpu, gpu.keys, regimes, sweep)
    bounds = dict.fromkeys(gpu.keys, (299, 427))
    bank = verify_bank_scope_coverage(gpu, gpu.keys, regimes, bounds, "a" * 64)
    assert bank[COVERED] and not bank["wider_scope_certified"]
    with pytest.raises(FidelityError, match="geometry"):
        verify_bank_scope_coverage(
            gpu, gpu.keys, regimes, dict.fromkeys(gpu.keys, (64, 192)), "a" * 64
        )
    assert result[COVERED] is True
    assert result["consumer_scope"] == "banked regime only"
    with pytest.raises(FidelityError, match="lane differs"):
        verify_sweep_coverage(gpu, gpu.keys, regimes, dict(sweep, lane_id="other"))
    assert not verify_sweep_coverage(
        gpu, gpu.keys, regimes, dict(sweep, independent_repeats_exact=False)
    )[COVERED]
    with pytest.raises(FidelityError, match="missing bank regime"):
        verify_sweep_coverage(gpu, gpu.keys, {"E": gpu.keys}, sweep)


def test_known_bad_path_distance_fails_despite_per_feature_pass(evidence):
    cpu, gpu, repeat, ruler = evidence
    changed = gpu.features.copy()
    changed[5, 0] += 0.06
    result = verify_vectors(
        cpu, replace(gpu, features=changed), replace(repeat, features=changed), ruler
    )
    assert result[REPEATABLE] and not result[AGREEMENT]
    assert result["max_feature_sigma_error"] < 0.1
    assert result["terciles"][2]["passed"] is False


def test_known_bad_coordinate_fails_even_with_large_path_bound(evidence):
    cpu, gpu, repeat, ruler = evidence
    changed = gpu.features.copy()
    changed[0, 0] += 0.11
    ruler = replace(ruler, floor_b=dict.fromkeys(ruler.reference_keys, 1.0))
    result = verify_vectors(
        cpu, replace(gpu, features=changed), replace(repeat, features=changed), ruler
    )
    assert result[REPEATABLE] and not result[AGREEMENT]
    assert result["rows"][0]["floor_c"] < result["rows"][0]["bound"]


def test_repeat_failure_stops_before_cross_gate_including_signed_zero(evidence):
    cpu, gpu, repeat, ruler = evidence
    changed = repeat.features.copy()
    changed[0, 0] = -0.0
    result = verify_vectors(cpu, gpu, replace(repeat, features=changed), ruler)
    assert result[REPEATABLE] is False and result[AGREEMENT] is None


@pytest.mark.parametrize(
    "change,match",
    [
        ({"feature_names": ("b", "a", "c")}, "feature order"),
        ({"lane_ids": ("gpu", "other", "gpu", "gpu", "gpu", "gpu")}, "mixed"),
        ({"calibration_sha256": "0" * 64}, "calibration"),
        ({"fresh_cache": False}, "fresh caches"),
        ({"channel": "excess"}, "not certified"),
    ],
)
def test_bad_metadata_refused(evidence, change, match):
    cpu, gpu, repeat, ruler = evidence
    with pytest.raises(FidelityError, match=match):
        verify_vectors(cpu, replace(gpu, **change), repeat, ruler)


def test_same_capture_cannot_satisfy_repeatability(evidence):
    cpu, gpu, _, ruler = evidence
    with pytest.raises(FidelityError, match="same replay"):
        verify_vectors(cpu, gpu, gpu, ruler)


def test_missing_row_and_nonfinite_refused(evidence):
    cpu, gpu, repeat, ruler = evidence
    with pytest.raises(FidelityError, match="missing/extra"):
        verify_vectors(cpu, gpu, repeat, replace(ruler, floor_b={}))
    bad = gpu.features.copy()
    bad[0, 0] = np.nan
    with pytest.raises(FidelityError, match="nonfinite"):
        verify_vectors(cpu, replace(gpu, features=bad), repeat, ruler)


def test_reordering_rows_joins_by_full_key(evidence):
    cpu, gpu, repeat, ruler = evidence
    fields = {
        name: getattr(gpu, name)[::-1]
        for name in ("features", "keys", "lane_ids", "input_sha256", "replay_ids")
    }
    assert verify_vectors(cpu, replace(gpu, **fields), repeat, ruler)[AGREEMENT]


def test_zero_condition_bound_is_not_silently_relaxed(evidence):
    cpu, gpu, repeat, ruler = evidence
    ruler = replace(
        ruler,
        condition_distance=dict.fromkeys(ruler.reference_keys, 0.0),
        condition_pair={k: (k, k) for k in ruler.reference_keys},
    )
    changed = gpu.features.copy()
    changed[0, 0] = 1e-8
    assert not verify_vectors(
        cpu, replace(gpu, features=changed), replace(repeat, features=changed), ruler
    )[AGREEMENT]


def test_anchor_null_by_rule_is_explicit_and_keeps_path_bound(evidence):
    cpu, gpu, repeat, ruler = evidence
    key = ruler.reference_keys[0]
    distances = dict(ruler.condition_distance)
    distances[key] = None
    pairs = dict(ruler.condition_pair)
    pairs[key] = None
    ruled = replace(ruler, condition_distance=distances, condition_pair=pairs)
    with pytest.raises(FidelityError, match="null-by-rule"):
        verify_vectors(cpu, gpu, repeat, ruled)
    ruled = replace(
        ruled,
        null_contrast_reasons={
            key: "Anchor has no nonzero self-contrast; separate null-fork checks apply."
        },
    )
    changed = gpu.features.copy()
    changed[0, 0] = 0.01
    result = verify_vectors(
        cpu, replace(gpu, features=changed), replace(repeat, features=changed), ruled
    )
    assert result[AGREEMENT]
    assert result["rows"][0]["contrast_status"] == "null-by-rule"
    assert result["rows"][0]["condition_distance"] is None
    assert result["rows"][0]["bound"] == ruler.floor_b[key]


def test_lower_bound_is_labeled_and_insufficiency_is_not_failure(evidence):
    cpu, gpu, repeat, ruler = evidence
    kinds = dict.fromkeys(ruler.reference_keys, "lower_bound")
    ruled = replace(
        ruler,
        floor_kinds=kinds,
        floor_proof_sha256=dict.fromkeys(ruler.reference_keys, "1" * 64),
    )
    result = verify_vectors(cpu, gpu, repeat, ruled)
    assert result[AGREEMENT]
    assert result["rows"][0]["floor_b"] is None
    assert result["rows"][0]["floor_b_lower_bound"] == 0.05
    changed = gpu.features.copy()
    changed[0, 0] = 0.06
    result = verify_vectors(
        cpu, replace(gpu, features=changed), replace(repeat, features=changed), ruled
    )
    assert result[AGREEMENT] is None
    assert result["rows"][0]["requires_full_floor"]


def test_null_se_warning_does_not_weaken_hard_gate():
    docs = tuple(f"d{i}" for i in range(383))

    def statistic(d, c):
        return SimpleNamespace(point=1.0, se=1.0 + float(d[0, 0]))

    anchor = DownstreamAnchor(1.0, 1.0, docs, docs, "a" * 64, "b" * 64)
    cpu = np.zeros((383, 1))
    candidate = cpu.copy()
    candidate[0, 0] = 0.04
    result = verify_downstream(
        cpu,
        candidate,
        docs,
        docs,
        anchor,
        statistic,
        "b" * 64,
        flag_se_sampling_scale=True,
    )
    assert result[DOWNSTREAM_PRESERVED] and result["se_change_warning"]
    assert 0.036 < result["se_sampling_scale_in_reference_se"] < 0.037
    candidate[0, 0] = 0.3
    assert not verify_downstream(
        cpu,
        candidate,
        docs,
        docs,
        anchor,
        statistic,
        "b" * 64,
        flag_se_sampling_scale=True,
    )[DOWNSTREAM_PRESERVED]


def test_coverage_and_memory_are_mandatory(evidence):
    cpu, gpu, repeat, ruler = evidence
    vec = verify_vectors(cpu, gpu, repeat, ruler)
    cover = verify_coverage(
        gpu, ruler.reference_keys, {"near-floor": [gpu.keys[0]], "short": []}
    )
    assert not cover[COVERED]
    mem = verify_memory(
        {"replay": [101]},
        {"replay": [100]},
        cap_bytes=100,
        metric="reserved",
        baseline_sha256="a" * 64,
    )
    assert not mem[WITHIN_MEMORY]
    assert not verify_within_tol(vec, {DOWNSTREAM_PRESERVED: True}, cover, mem)["passed"]
    assert not verify_within_tol(vec, {}, {COVERED: True}, {WITHIN_MEMORY: True})["passed"]


def test_downstream_pinned_anchor_and_matched_cohort():
    # Controlled statistic isolates gate behavior from the owner's tested estimator.
    def stat(d, c):
        return SimpleNamespace(point=float(d.mean()), se=float(d.std()))

    d = np.asarray([[1.0, 2.0], [4.0, 8.0], [3.0, 2.0]])
    docs = ("d0", "d1", "d2")
    clusters = ("c0", "c1", "c2")
    anchor = DownstreamAnchor(
        float(d.mean()), float(d.std()), docs, clusters, "a" * 64, "b" * 64
    )
    assert verify_downstream(d, d, docs, clusters, anchor, stat, "b" * 64)[DOWNSTREAM_PRESERVED]
    assert not verify_downstream(
        d, d + anchor.se, docs, clusters, anchor, stat, "b" * 64
    )[DOWNSTREAM_PRESERVED]
    assert not verify_downstream(d, d * 2, docs, clusters, anchor, stat, "b" * 64)[DOWNSTREAM_PRESERVED]
    with pytest.raises(FidelityError, match="cohort"):
        verify_downstream(d, d, docs[::-1], clusters, anchor, stat, "b" * 64)
    with pytest.raises(FidelityError, match="reproduce"):
        verify_downstream(
            d, d, docs, clusters, replace(anchor, point=99.0), stat, "b" * 64
        )


def test_banked_gate_keys_read_forward(evidence):
    """A receipt banked under the short gate keys combines like a fresh one."""
    cpu, gpu, repeat, ruler = evidence
    fresh = verify_vectors(cpu, gpu, repeat, ruler)
    banked = {
        {v: k for k, v in BANKED_GATE_KEYS.items()}.get(key, key): value
        for key, value in fresh.items()
    }
    assert "G0" in banked and REPEATABLE not in banked
    assert read_gate_receipt(banked) == fresh
    assert read_gate_receipt(fresh) == fresh, "current names pass through untouched"
    downstream = {"G2": True, "point_change_in_reference_se": 0.0}
    coverage, memory = {"G3": True}, {"G5": True}
    combined = verify_within_tol(banked, downstream, coverage, memory)
    assert combined["gates"] == verify_within_tol(
        fresh, {DOWNSTREAM_PRESERVED: True}, {COVERED: True}, {WITHIN_MEMORY: True}
    )["gates"]
    assert set(combined["gates"]) == set(BANKED_GATE_KEYS.values())
    assert combined["downstream"]["point_change_in_reference_se"] == 0.0


def test_a_verdict_under_both_spellings_must_agree():
    assert read_gate_receipt({"G3": True, COVERED: True}) == {COVERED: True}
    with pytest.raises(FidelityError, match="two different verdicts"):
        read_gate_receipt({"G3": True, COVERED: False})
