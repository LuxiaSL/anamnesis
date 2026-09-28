"""Extension lanes: declared by a file, admitted only on a matching passing receipt.

Each case writes a passing fine-tune of a synthetic ``8b`` (``synthetic_extension``),
names its entry in ``ANAMNESIS_VLLM_LANES`` and breaks one thing: the admitted lane
resolves like a shipped one; the guard and the file each refuse by name; the
calibration is verified against its pins; shipped lanes never read the file.
"""

from __future__ import annotations

import json

import pytest

from anamnesis.extraction.vllm import extensions, runtime
from anamnesis.extraction.vllm.envelope import (
    CONDITIONS,
    LANE_MODELS,
    REQUIRED_ENV,
    engine_settings,
    enforce_lane_envelope,
    extension_lane_id,
    lane_id,
    lane_model,
    lane_preset,
)
from anamnesis.extraction.vllm.extensions import LANES_ENV, LaneFileError, admit, lane_keys
from synthetic_extension import (
    BASE,
    CHECKPOINT,
    KEY,
    PRESET,
    base_tolerance,
    declare_extension,
    entry_row,
    registry,
    rewrite_entry,
    rewrite_receipt,
)


@pytest.fixture
def declared(tmp_path, monkeypatch):
    return declare_extension(tmp_path, monkeypatch)


def test_an_admitted_extension_resolves_like_a_shipped_lane(declared):
    fixtures, tolerance = runtime.load_fixtures(KEY)
    assert fixtures.digest == declared.result.fixtures.digest
    assert tolerance.digest == declared.result.tolerance.digest
    assert fixtures.lane_id == lane_id(KEY) == extension_lane_id(BASE, CHECKPOINT)
    assert lane_id(KEY) not in {lane_id(m) for m in LANE_MODELS}
    assert extension_lane_id(BASE, "d" * 64) != lane_id(KEY) != extension_lane_id("70b",
                                                                                 CHECKPOINT)
    assert lane_model(KEY) == LANE_MODELS[BASE] and lane_preset(KEY) == PRESET
    for condition in CONDITIONS:
        assert engine_settings(KEY, condition) == engine_settings(BASE, condition)
    settings = engine_settings(KEY, "full-b1-order0")
    record = enforce_lane_envelope(settings, dict(REQUIRED_ENV), lane=lane_id(KEY), model=KEY)
    assert record["dtype"] == LANE_MODELS[BASE]["dtype"]
    with pytest.raises(ValueError, match="dtype"):
        enforce_lane_envelope(dict(settings, dtype="float16"), dict(REQUIRED_ENV),
                              lane=lane_id(KEY), model=KEY)
    declared.receipt.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        enforce_lane_envelope(settings, dict(REQUIRED_ENV), lane=lane_id(KEY), model=KEY)


def test_both_commands_offer_the_declared_key(declared):
    from anamnesis.scripts import qualify_vllm, run_vllm_replay

    assert lane_keys() == (*sorted(LANE_MODELS), KEY)
    for command in (qualify_vllm, run_vllm_replay):
        action = next(a for a in command.parser()._actions if a.dest == "model")
        assert list(action.choices) == list(lane_keys())


def test_qualify_uses_the_verified_declared_calibration(declared, monkeypatch, tmp_path, capsys):
    from anamnesis.extraction.vllm import hub
    from anamnesis.scripts import qualify_vllm

    calls = []

    def check_install(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("stop")

    monkeypatch.setattr(runtime, "check_install", check_install)
    monkeypatch.setattr(hub, "fetch_calibration", lambda model: pytest.fail("fetched"))
    argv = ["--model", KEY, "--model-path", str(tmp_path / "m"), "--work-dir",
            str(tmp_path / "w")]
    assert qualify_vllm.main(argv) == 2
    assert calls[0][0] == KEY and calls[0][2] == declared.calib.resolve()
    (declared.calib / "pca_model.pkl").write_bytes(b"another basis")
    assert qualify_vllm.main(argv) == 2 and len(calls) == 1
    assert f"not extension lane {KEY!r}'s calibration" in capsys.readouterr().err


def test_a_broken_lane_file_stops_the_commands_but_never_a_shipped_lookup(tmp_path, monkeypatch,
                                                                        capsys):
    from anamnesis.scripts import qualify_vllm, run_vllm_replay

    (tmp_path / "lanes.json").write_text("{not json")
    monkeypatch.setenv(LANES_ENV, str(tmp_path / "lanes.json"))
    assert qualify_vllm.main(["--model", "8b"]) == run_vllm_replay.main(["--model", "8b"]) == 2
    assert "invalid JSON" in capsys.readouterr().err
    assert lane_model("8b") == LANE_MODELS["8b"]
    assert lane_id("70b") == "c34f17d8d1a4e42731d613a6bc6378828a8b19b4258c476753be3376451067d2"
    assert runtime.load_fixtures("3b")[0].lane_id == lane_id("3b")


def _base_tolerance_changed(declared, tmp_path, monkeypatch):
    (tmp_path / "shipped" / BASE / "tolerance.json").write_text(
        base_tolerance(residual_max=2.0).model_dump_json())


def _tolerance_altered(declared, tmp_path, monkeypatch):
    path = declared.out / "fixtures" / "tolerance.json"
    tolerance = json.loads(path.read_text())
    tolerance["family_max_abs_sigma"]["residual"] *= 2
    path.write_text(json.dumps(tolerance))


@pytest.mark.parametrize("breakage,match", [
    (lambda d, *_: d.receipt.unlink(), "does not exist"),
    (lambda d, *_: rewrite_entry(d, transfer_receipt_sha256="0" * 64), "declared digest"),
    (lambda d, *_: rewrite_receipt(d, verdict="refuse", reasons=["row 1: over"],
                                   fixture_digest=None, tolerance_digest=None),
     "refused it: row 1: over"),
    (lambda d, *_: rewrite_receipt(d, base_lane_id=lane_id("70b")), "receipt base_lane_id"),
    (lambda d, *_: rewrite_receipt(d, checkpoint_sha256="d" * 64), "receipt checkpoint"),
    (lambda d, *_: rewrite_entry(d, checkpoint_sha256="d" * 64), "receipt lane_id"),
    (_tolerance_altered, "tolerance digest"),
    (_base_tolerance_changed, "base tolerance digest"),
    (lambda d, tmp, mp: registry(tmp / "m3.json", mp, row={"extends": "3b", "model_id": "x"}),
     "does not extend '8b'"),
    (lambda d, tmp, mp: registry(tmp / "mp.json", mp, row={
        "extends": BASE, "model_id": "x", "sampled_layers": [0, 8, 16, 24, 31]}),
     "changes \\['sampled_layers'\\]"),
])
def test_the_guard_refuses_by_name(declared, tmp_path, monkeypatch, breakage, match):
    breakage(declared, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=f"extension lane '{KEY}' refused: .*{match}"):
        admit(KEY)


def test_an_undeclared_key_is_no_lane(declared):
    with pytest.raises(ValueError, match="not a declared extension lane"):
        admit("nothing")
    with pytest.raises(ValueError, match="has no vLLM lane"):
        lane_model("nothing")


@pytest.mark.parametrize("lanes,match", [
    (lambda d: {"8b": entry_row(d)}, "already a shipped or declared lane"),
    (lambda d: {KEY: entry_row(d), "alias": entry_row(d)}, "one checkpoint on one base"),
    (lambda d: {"x": entry_row(d, extends="405b")}, "not a shipped lane"),
    (lambda d: {"x": entry_row(d, extends=KEY)}, "not a shipped lane"),
    (lambda d: {KEY: entry_row(d, dtype="float16")}, "sets \\['dtype'\\]"),
    (lambda d: {KEY: entry_row(d, logprob_wrapper="none")}, "sets \\['logprob_wrapper'\\]"),
    (lambda d: {KEY: entry_row(d, max_model_len=4096)}, "sets \\['max_model_len'\\]"),
    (lambda d: {KEY: entry_row(d, calibration={})}, "calibration must pin exactly"),
])
def test_the_lane_file_refuses_what_an_entry_may_not_claim(declared, tmp_path, monkeypatch,
                                                            lanes, match):
    (tmp_path / "lanes.json").write_text(json.dumps({"lanes": lanes(declared)}))
    monkeypatch.setenv(LANES_ENV, str(tmp_path / "lanes.json"))
    with pytest.raises(LaneFileError, match=match):
        lane_keys()


def test_a_key_repeated_across_or_within_files_is_refused(declared, tmp_path, monkeypatch):
    row = json.dumps(entry_row(declared, checkpoint_sha256="d" * 64))
    (tmp_path / "second.json").write_text(f'{{"lanes": {{"{KEY}": {row}}}}}')
    (tmp_path / "repeat.json").write_text(f'{{"lanes": {{"{KEY}": {row}, "{KEY}": {row}}}}}')
    for value, match in ((f"{declared.entry}:{tmp_path / 'second.json'}", "already"),
                         (str(tmp_path / "repeat.json"), "more than once"),
                         (str(tmp_path / "absent.json"), "unreadable")):
        monkeypatch.setenv(LANES_ENV, value)
        with pytest.raises(LaneFileError, match=match):
            lane_keys()


def test_the_calibration_is_verified_against_its_pins(declared, tmp_path):
    entry = extensions.declared_lane(KEY)
    assert entry.fixtures_dir == declared.out / "fixtures"
    assert extensions.verify_calibration(KEY) == declared.calib.resolve()
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "positional_means.npz").write_bytes(b"other means")
    with pytest.raises(ValueError, match="positional_means.npz differs .*pca_model.pkl missing"):
        extensions.verify_calibration(KEY, copy)


def test_two_extensions_of_one_base_are_two_lanes(tmp_path, monkeypatch):
    first = declare_extension(tmp_path, monkeypatch, key="ft-a", checkpoint="a" * 64,
                              out_name="a")
    second = declare_extension(tmp_path, monkeypatch, key="ft-b", checkpoint="d" * 64,
                               out_name="b")
    monkeypatch.setenv(LANES_ENV, f"{first.entry}:{second.entry}")
    assert lane_keys() == (*sorted(LANE_MODELS), "ft-a", "ft-b")
    assert admit("ft-a").fixtures.lane_id != admit("ft-b").fixtures.lane_id
