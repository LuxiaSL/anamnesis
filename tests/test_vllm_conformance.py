"""The install check's decision: identical, own-lane or refused.

:func:`anamnesis.extraction.vllm.conformance.decide` takes vectors a host has
already captured and never touches an engine, so every rule it applies is
exercised here on hand-built fixtures of five coordinates: the checkpoint
match, the lane's agreement with itself, finiteness, and the median ratio per
component against its recorded max, which decide; and the per-row ceilings, the
family maxima, the discrete coordinates, and the ordinary stratum's median and
tail readings, which are reported and never decide. The fixture set's round
trip, its digest and the receipt cache are here too.

What needs a device is the vectors themselves: which tier a real host earns is
the install check run on that host.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.extraction.vllm.conformance import (
    CapturedFixture,
    FixtureRow,
    FixtureSet,
    HostFingerprint,
    ReceiptCache,
    RowComponent,
    Tolerance,
    decide,
    own_lane_id,
)

NAMES = ("sub_a", "sub_b", "att_a", "att_b", "att_cov")
SHA = "a" * 64


def _fixtures(selected_by=("test", "test")):
    rows = tuple(FixtureRow(generation_id=g, population="native", input_ids=(1, 2, 3, 4),
                            prompt_length=1, end=4, floor_b=1.0, selected_by=rule)
                 for g, rule in zip((0, 5), selected_by, strict=True))
    vectors = {0: np.asarray([1, 2, 3, 4, 0.5], dtype=np.float32),
               5: np.asarray([0, 1, 0, 1, 0.25], dtype=np.float32)}
    return FixtureSet(model="8b", lane_id="lane-8b", checkpoint_sha256=SHA,
                      calibration_sha256=SHA, ruler_sha256=SHA, feature_names=NAMES,
                      sigma_cal=np.ones(5), weights=np.ones(5), rows=rows, vectors=vectors)


def _tolerance(family_max=0.5, max_ratio=0.8, median_stratum=None, p90=0.5, p99=None):
    return Tolerance(
        model="8b",
        components=(RowComponent(name="substrate", feature_names=("sub_a", "sub_b"),
                                 max_ratio=max_ratio, p90_ratio=p90, p99_ratio=p99,
                                 source="audit"),
                    RowComponent(name="attention", feature_names=("att_a", "att_b"),
                                 max_ratio=max_ratio, p90_ratio=p90, p99_ratio=p99,
                                 source="report")),
        families={"sub_a": "residual", "sub_b": "residual", "att_a": "flow", "att_b": "flow"},
        family_max_abs_sigma={"residual": family_max, "flow": family_max},
        discrete=("att_cov",), median_gate_stratum=median_stratum, sources={"audit": SHA})


def _fingerprint(fixtures, tolerance, checkpoint=SHA):
    return HostFingerprint(gpu_name="gpu", gpu_uuid="GPU-1", driver="580", cuda_runtime="12.8",
                           torch="2.9.1", vllm="0.16.0", anamnesis="x",
                           checkpoint_sha256=checkpoint, engine_settings_sha256="e" * 64,
                           fixture_digest=fixtures.digest, tolerance_digest=tolerance.digest)


def _captures(fixtures, shift=None, batched_shift=None):
    out = []
    for gid, vector in fixtures.vectors.items():
        first = vector.copy() if shift is None else (vector + shift).astype(np.float32)
        batched = first if batched_shift is None else (first + batched_shift).astype(np.float32)
        out.append(CapturedFixture(generation_id=gid, first=first, repeat=first.copy(),
                                   batched=batched))
    return out


def _one_row_shifted(fixtures, gid, shift):
    out = []
    for g, vector in fixtures.vectors.items():
        first = (vector + shift).astype(np.float32) if g == gid else vector.copy()
        out.append(CapturedFixture(generation_id=g, first=first, repeat=first.copy(),
                                   batched=first.copy()))
    return out


def test_byte_identical_vectors_keep_the_qualified_lane():
    f, t = _fixtures(), _tolerance()
    receipt = decide(f, t, _fingerprint(f, t), _captures(f))
    assert receipt.tier == "identical" and receipt.lane_id == "lane-8b"
    assert receipt.qualified_lane_id == "lane-8b" and not receipt.reasons
    assert all(r.byte_identical for r in receipt.rows)


def test_small_deviation_is_its_own_lane():
    f, t = _fixtures(), _tolerance()
    fp = _fingerprint(f, t)
    shift = np.asarray([0.1, 0, 0.2, 0, 0], dtype=np.float32)
    receipt = decide(f, t, fp, _captures(f, shift))
    assert receipt.tier == "own-lane" and not receipt.reasons and not receipt.readings
    assert receipt.lane_id == own_lane_id("lane-8b", fp)
    assert receipt.lane_id != "lane-8b" and receipt.lane_id.startswith("lane-8b+host-")
    assert receipt.rows[0].component_ratios["attention"] == pytest.approx(0.2, rel=1e-5)
    assert receipt.component_medians["attention"] == pytest.approx(0.2, rel=1e-5)


def test_an_own_lane_id_is_the_hosts_own():
    f, t = _fixtures(), _tolerance()
    fp = _fingerprint(f, t)
    other = fp.model_copy(update={"gpu_uuid": "GPU-2"})
    assert own_lane_id("lane-8b", fp) != own_lane_id("lane-8b", other)


def test_a_median_ratio_past_the_recorded_max_is_refused():
    f, t = _fixtures(), _tolerance(family_max=10)
    shift = np.asarray([0, 0, 0.9, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _captures(f, shift))
    assert receipt.tier == "refused" and receipt.lane_id is None
    assert list(receipt.reasons) == ["attention: median ratio 0.9 over the fixture rows "
                                     "exceeds the recorded max 0.8"]
    assert len(receipt.rows) == 2


def test_a_row_past_its_ceiling_is_reported_not_refused():
    f, t = _fixtures(), _tolerance(family_max=10)
    shift = np.asarray([0.7, 0.7, 0, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _one_row_shifted(f, 5, shift))
    assert receipt.tier == "own-lane" and not receipt.reasons
    assert not receipt.rows[1].components_within
    assert any(r.startswith("row 5:") and "ceiling" in r for r in receipt.readings)


def test_a_coordinate_past_its_family_maximum_is_reported_not_refused():
    f, t = _fixtures(), _tolerance(family_max=0.05)
    shift = np.asarray([0, 0, 0.1, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _captures(f, shift))
    assert receipt.tier == "own-lane"
    assert receipt.family_report["flow"] == pytest.approx(2.0, rel=1e-4)
    assert any("family flow" in r for r in receipt.readings)


def test_discrete_crossings_are_reported_never_gating():
    f, t = _fixtures(), _tolerance()
    shift = np.asarray([0, 0, 0, 0, 1.0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _captures(f, shift))
    assert receipt.tier == "own-lane"
    assert all(r.discrete_crossings == 1 for r in receipt.rows)


def test_a_non_finite_feature_is_refused():
    f, t = _fixtures(), _tolerance()
    shift = np.asarray([0, 0, 0, np.nan, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _one_row_shifted(f, 0, shift))
    assert receipt.tier == "refused"
    assert list(receipt.reasons) == ["row 0: 1 features are not finite"]


def test_a_lane_that_disagrees_with_itself_is_refused():
    f, t = _fixtures(), _tolerance()
    receipt = decide(f, t, _fingerprint(f, t),
                     _captures(f, batched_shift=np.full(5, 1e-3, dtype=np.float32)))
    assert receipt.tier == "refused"
    assert any("disagrees with itself" in r for r in receipt.reasons)


def test_a_capture_of_the_wrong_shape_is_refused():
    f, t = _fixtures(), _tolerance()
    captured = [CapturedFixture(generation_id=c.generation_id, first=c.first[:-1],
                                repeat=c.repeat[:-1], batched=c.batched[:-1])
                for c in _captures(f)]
    receipt = decide(f, t, _fingerprint(f, t), captured)
    assert receipt.tier == "refused"
    assert any("float32 [features]" in r for r in receipt.reasons)


def test_a_different_checkpoint_is_refused():
    f, t = _fixtures(), _tolerance()
    receipt = decide(f, t, _fingerprint(f, t, checkpoint="b" * 64), _captures(f))
    assert receipt.tier == "refused"
    assert any("checkpoint" in r for r in receipt.reasons)


def test_missing_captures_are_refused():
    f, t = _fixtures(), _tolerance()
    receipt = decide(f, t, _fingerprint(f, t), _captures(f)[:1])
    assert receipt.tier == "refused"
    assert any("exactly the fixture rows" in r for r in receipt.reasons)


def test_a_fingerprint_for_other_fixtures_is_an_error():
    f, t = _fixtures(), _tolerance()
    other = _tolerance(family_max=0.9)
    with pytest.raises(ValueError, match="other fixtures or tolerance"):
        decide(f, t, _fingerprint(f, other), _captures(f))


def test_a_tolerance_for_another_model_is_an_error():
    f = _fixtures()
    t = _tolerance().model_copy(update={"model": "3b"})
    with pytest.raises(ValueError, match="different models"):
        decide(f, t, _fingerprint(f, t), _captures(f))


def test_a_tolerance_naming_coordinates_the_fixtures_lack_is_an_error():
    f = _fixtures()
    t = _tolerance().model_copy(update={"discrete": ("att_cov", "absent")})
    with pytest.raises(ValueError, match="fixtures lack"):
        decide(f, t, _fingerprint(f, t), _captures(f))


def test_fixture_sets_round_trip_and_detect_tampering(tmp_path):
    f = _fixtures()
    f.save(tmp_path / "fx")
    assert FixtureSet.load(tmp_path / "fx").digest == f.digest
    with pytest.raises(FileExistsError):
        f.save(tmp_path / "fx")
    z = dict(np.load(tmp_path / "fx" / "vectors.npz"))
    z["vectors"][0, 0] += 1
    np.savez(tmp_path / "fx" / "vectors.npz", **z)
    with pytest.raises(ValueError, match="recorded digest"):
        FixtureSet.load(tmp_path / "fx")


def test_fixture_rows_must_partition_their_ids():
    with pytest.raises(ValueError, match="partition"):
        FixtureRow(generation_id=0, population="native", input_ids=(1, 2, 3), prompt_length=1,
                   end=4, floor_b=1.0, selected_by="x")


def test_a_fixture_set_needs_a_positive_standardizer_and_one_vector_per_row():
    f = _fixtures()
    fields = f.model_dump(exclude={"sigma_cal", "weights", "vectors"})
    arrays = dict(sigma_cal=f.sigma_cal, weights=f.weights, vectors=dict(f.vectors))
    with pytest.raises(ValueError, match="positive"):
        FixtureSet(**fields, **{**arrays, "sigma_cal": np.zeros(5)})
    with pytest.raises(ValueError, match="every feature name"):
        FixtureSet(**fields, **{**arrays, "weights": np.ones(4)})
    with pytest.raises(ValueError, match="exactly one vector"):
        FixtureSet(**fields, **{**arrays, "vectors": {0: f.vectors[0]}})
    with pytest.raises(ValueError, match="finite float32"):
        FixtureSet(**fields, **{**arrays, "vectors": {0: f.vectors[0].astype(np.float64),
                                                      5: f.vectors[5]}})


def test_tolerance_families_need_a_recorded_maximum():
    with pytest.raises(ValueError, match="recorded maximum"):
        Tolerance(model="8b", components=_tolerance().components, families={"sub_a": "x"},
                  family_max_abs_sigma={}, sources={})


def test_a_coordinate_cannot_be_continuous_and_discrete():
    with pytest.raises(ValueError, match="not both"):
        Tolerance(model="8b", components=_tolerance().components,
                  families={"att_cov": "flow"}, family_max_abs_sigma={"flow": 1.0},
                  discrete=("att_cov",), sources={})


def test_receipts_are_reused_only_for_the_same_fingerprint(tmp_path):
    f, t = _fixtures(), _tolerance()
    fp = _fingerprint(f, t)
    cache = ReceiptCache(tmp_path)
    assert cache.load(fp) is None
    cache.store(decide(f, t, fp, _captures(f)))
    assert cache.load(fp).tier == "identical"
    assert cache.load(fp.model_copy(update={"driver": "581"})) is None
    assert cache.load(fp.model_copy(update={"engine_settings_sha256": "f" * 64})) is None


def test_a_receipt_of_another_contract_is_decided_again(tmp_path):
    f, t = _fixtures(), _tolerance()
    fp = _fingerprint(f, t)
    cache = ReceiptCache(tmp_path)
    path = cache.store(decide(f, t, fp, _captures(f)))
    stored = path.read_text()
    path.write_text(stored.replace('"conformance-receipt/2"', '"conformance-receipt/1"'))
    assert cache.load(fp) is None
    path.write_text("not json")
    assert cache.load(fp) is None


def test_the_ordinary_median_past_its_p90_is_a_reading():
    f = _fixtures()
    t = _tolerance(max_ratio=0.8, median_stratum="test", p90=0.3)
    shift = np.asarray([0, 0, 0.7, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _captures(f, shift))
    assert receipt.tier == "own-lane" and not receipt.reasons
    assert any("median ratio" in r and "p90" in r for r in receipt.readings)
    assert all(r.components_within for r in receipt.rows)


def test_a_median_stratum_no_row_carries_is_a_reading():
    f = _fixtures()
    t = _tolerance(median_stratum="absent")
    shift = np.asarray([0, 0, 0.1, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _captures(f, shift))
    assert receipt.tier == "own-lane"
    assert any("stratum" in r for r in receipt.readings)


def test_an_ordinary_row_past_the_p99_is_a_reading():
    f = _fixtures(selected_by=("test", "tail"))
    t = _tolerance(family_max=1.0, max_ratio=0.8, median_stratum="test", p90=0.35, p99=0.5)
    shift = np.asarray([0, 0, 0.6, 0, 0], dtype=np.float32)
    receipt = decide(f, t, _fingerprint(f, t), _one_row_shifted(f, 5, shift))
    assert receipt.tier == "own-lane" and not receipt.readings
    receipt = decide(f, t, _fingerprint(f, t), _one_row_shifted(f, 0, shift))
    assert receipt.tier == "own-lane"
    assert any(r.startswith("row 0: attention ratio") and "p99" in r for r in receipt.readings)
