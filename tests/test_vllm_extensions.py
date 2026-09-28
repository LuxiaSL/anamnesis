"""Extension lanes: declared by a file, admitted only on a matching passing receipt.

Every case builds a synthetic base lane and a passing fine-tune of it on disk
(``synthetic_extension``), names the written entry in ``ANAMNESIS_VLLM_LANES``,
and then breaks one thing. What is pinned:

* the admitted lane resolves everywhere a shipped one does — its facts and
  engine settings are its base's, its lane id is its own, its fixtures load, the
  startup guard admits it, and both commands offer it;
* each refusal the guard makes, by name: a missing or altered receipt, a receipt
  that refused or names another base or checkpoint, fixtures or a tolerance that
  are not the receipt's, a base tolerance other than the shipped one, and a
  preset that is not a structural copy of the base's;
* each refusal the file makes: a shipped key, a key declared twice, one
  checkpoint declared twice on one base, an unshipped base, and any inherited
  setting an entry tries to set;
* the calibration is verified against the declared pins before use;
* the shipped lanes do not read the file at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.extraction.vllm import extensions, runtime
from anamnesis.extraction.vllm.envelope import (
    CONDITIONS,
    LANE_MODELS,
    REQUIRED_ENV,
    engine_settings,
    enforce_lane_envelope,
    lane_base,
    lane_id,
    lane_identity,
    lane_model,
    lane_preset,
)
from anamnesis.extraction.vllm.extensions import LANES_ENV, LaneFileError, admit, lane_keys
from anamnesis.extraction.vllm.transfer import extension_lane_id
from synthetic_extension import (
    BASE,
    CHECKPOINT,
    KEY,
    PRESET,
    base_tolerance,
    declare_extension,
    registry,
    rewrite_entry,
    rewrite_receipt,
    ship_base,
)


@pytest.fixture
def declared(tmp_path, monkeypatch):
    return declare_extension(tmp_path, monkeypatch)


# --- an admitted lane resolves like a shipped one --------------------------------


def test_an_admitted_extension_carries_its_own_fixtures_and_lane_id(declared):
    admitted = admit(KEY)
    assert admitted.receipt.verdict == "pass"
    fixtures, tolerance = runtime.load_fixtures(KEY)
    assert fixtures.digest == declared.result.fixtures.digest
    assert tolerance.digest == declared.result.tolerance.digest
    assert runtime.fixtures_dir(KEY) == declared.out / "fixtures"
    assert fixtures.lane_id == lane_id(KEY) == extension_lane_id(BASE, CHECKPOINT)


def test_an_extension_inherits_its_base_s_facts_and_engine_settings(declared):
    assert lane_model(KEY) == LANE_MODELS[BASE]
    assert lane_base(KEY) == BASE and lane_preset(KEY) == PRESET
    for condition in CONDITIONS:
        assert engine_settings(KEY, condition) == engine_settings(BASE, condition)


def test_the_extension_lane_id_is_distinct_from_its_base_and_other_checkpoints(declared):
    ids = {lane_id(m) for m in LANE_MODELS}
    assert lane_id(KEY) not in ids
    assert lane_identity(KEY) == dict(base=lane_identity(BASE), extends=BASE,
                                      checkpoint_sha256=CHECKPOINT)
    assert extension_lane_id(BASE, "d" * 64) != lane_id(KEY)
    assert extension_lane_id("70b", CHECKPOINT) != lane_id(KEY)


def test_the_startup_guard_admits_an_extension_at_its_base_s_settings(declared):
    settings = engine_settings(KEY, "full-b1-order0")
    record = enforce_lane_envelope(settings, dict(REQUIRED_ENV), lane=lane_id(KEY), model=KEY)
    assert record["model"] == KEY and record["dtype"] == LANE_MODELS[BASE]["dtype"]
    with pytest.raises(ValueError, match="dtype"):
        enforce_lane_envelope(dict(settings, dtype="float16"), dict(REQUIRED_ENV),
                              lane=lane_id(KEY), model=KEY)


def test_the_startup_guard_refuses_an_extension_its_guard_refuses(declared):
    declared.receipt.unlink()
    settings = engine_settings(KEY, "full-b1-order0")
    with pytest.raises(ValueError, match="does not exist"):
        enforce_lane_envelope(settings, dict(REQUIRED_ENV), lane=lane_id(KEY), model=KEY)


def test_both_commands_offer_the_declared_key(declared):
    from anamnesis.scripts import qualify_vllm, run_vllm_replay

    assert lane_keys() == (*sorted(LANE_MODELS), KEY)
    for command in (qualify_vllm, run_vllm_replay):
        action = next(a for a in command.parser()._actions if a.dest == "model")
        assert list(action.choices) == list(lane_keys())


def test_qualify_checks_an_extension_against_its_declared_calibration(declared, monkeypatch,
                                                                      tmp_path, capsys):
    from anamnesis.extraction.vllm import hub
    from anamnesis.scripts import qualify_vllm

    calls = []
    monkeypatch.setattr(runtime, "check_install",
                        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(
                            RuntimeError("stop after the calibration")))
    monkeypatch.setattr(hub, "fetch_calibration", lambda model: pytest.fail("fetched"))
    argv = ["--model", KEY, "--model-path", str(tmp_path / "m"), "--work-dir",
            str(tmp_path / "w")]
    assert qualify_vllm.main(argv) == 2
    assert calls[0][0] == KEY and calls[0][2] == declared.calib
    (declared.calib / "pca_model.pkl").write_bytes(b"another basis")
    assert qualify_vllm.main(argv) == 2
    assert f"not extension lane {KEY!r}'s calibration" in capsys.readouterr().err
    assert len(calls) == 1


def test_a_broken_lane_file_stops_the_commands_with_exit_two(tmp_path, monkeypatch, capsys):
    from anamnesis.scripts import qualify_vllm, run_vllm_replay

    broken = tmp_path / "lanes.json"
    broken.write_text("{not json")
    monkeypatch.setenv(LANES_ENV, str(broken))
    assert qualify_vllm.main(["--model", "8b"]) == 2
    assert run_vllm_replay.main(["--model", "8b"]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_a_shipped_lane_never_reads_the_lane_file(tmp_path, monkeypatch):
    monkeypatch.setenv(LANES_ENV, str(tmp_path / "absent.json"))
    assert lane_model("8b") == LANE_MODELS["8b"]
    assert lane_id("70b") == "c34f17d8d1a4e42731d613a6bc6378828a8b19b4258c476753be3376451067d2"
    fixtures, _ = runtime.load_fixtures("3b")
    assert fixtures.lane_id == lane_id("3b")


# --- the guard's refusals -------------------------------------------------------------


def test_a_missing_receipt_is_refused_by_name(declared):
    declared.receipt.unlink()
    with pytest.raises(ValueError, match=f"extension lane '{KEY}' refused: .*does not exist"):
        admit(KEY)


def test_a_receipt_differing_from_its_declared_digest_is_refused(declared):
    rewrite_entry(declared, transfer_receipt_sha256="0" * 64)
    with pytest.raises(ValueError, match="differs from the declared digest"):
        admit(KEY)


def test_a_refusing_receipt_is_refused(declared):
    rewrite_receipt(declared, verdict="refuse", reasons=["row 1: over the ceiling"],
                    fixture_digest=None, tolerance_digest=None)
    with pytest.raises(ValueError, match="receipt refused it: row 1: over the ceiling"):
        admit(KEY)


def test_a_receipt_naming_another_base_lane_is_refused(declared):
    rewrite_receipt(declared, base_lane_id=lane_id("70b"))
    with pytest.raises(ValueError, match="base_lane_id"):
        admit(KEY)


def test_a_receipt_naming_another_checkpoint_is_refused(declared):
    rewrite_receipt(declared, checkpoint_sha256="d" * 64)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        admit(KEY)


def test_an_entry_naming_another_checkpoint_is_refused(declared):
    rewrite_entry(declared, checkpoint_sha256="d" * 64)
    with pytest.raises(ValueError, match="lane_id"):
        admit(KEY)


def test_a_tolerance_other_than_the_receipt_s_is_refused(declared):
    path = declared.out / "fixtures" / "tolerance.json"
    tolerance = json.loads(path.read_text())
    tolerance["family_max_abs_sigma"]["residual"] *= 2
    path.write_text(json.dumps(tolerance))
    with pytest.raises(ValueError, match="tolerance_digest"):
        admit(KEY)


def test_a_base_tolerance_changed_since_the_receipt_is_refused(declared, tmp_path, monkeypatch):
    (tmp_path / "shipped" / BASE / "tolerance.json").write_text(
        base_tolerance(median_gate_stratum=None).model_dump_json())
    with pytest.raises(ValueError, match="run the transfer check again"):
        admit(KEY)


def test_a_preset_that_does_not_extend_the_base_is_refused(declared, tmp_path, monkeypatch):
    registry(tmp_path / "models-3b.json", monkeypatch, row={"extends": "3b", "model_id": "x"})
    with pytest.raises(ValueError, match="does not extend '8b'"):
        admit(KEY)


def test_a_preset_that_changes_the_layer_plan_is_refused(declared, tmp_path, monkeypatch):
    registry(tmp_path / "models-plan.json", monkeypatch,
             row={"extends": BASE, "model_id": "x", "sampled_layers": [0, 8, 16, 24, 31]})
    with pytest.raises(ValueError, match="changes \\['sampled_layers'\\]"):
        admit(KEY)


def test_an_undeclared_key_is_not_an_extension(declared):
    with pytest.raises(ValueError, match="not a declared extension lane"):
        admit("nothing")
    with pytest.raises(ValueError, match="has no vLLM lane"):
        lane_model("nothing")


# --- the file's refusals -------------------------------------------------------------


def _lane_file(tmp_path: Path, monkeypatch, lanes: dict, name: str = "lanes.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"lanes": lanes}))
    monkeypatch.setenv(LANES_ENV, str(path))
    return path


def _row(declared) -> dict:
    (row,) = json.loads(declared.entry.read_text())["lanes"].values()
    return {k: (str(declared.out / v) if k in ("fixtures_dir", "transfer_receipt") else v)
            for k, v in row.items()}


def test_an_entry_may_not_redefine_a_shipped_lane(declared, tmp_path, monkeypatch):
    _lane_file(tmp_path, monkeypatch, {"8b": _row(declared)})
    with pytest.raises(LaneFileError, match="'8b' is a shipped lane"):
        lane_keys()


def test_a_key_declared_in_two_files_is_refused(declared, tmp_path, monkeypatch):
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"lanes": {KEY: dict(_row(declared),
                                                      checkpoint_sha256="d" * 64)}}))
    monkeypatch.setenv(LANES_ENV, f"{declared.entry}:{second}")
    with pytest.raises(LaneFileError, match="already declared"):
        lane_keys()


def test_a_key_repeated_in_one_file_is_refused(declared, tmp_path, monkeypatch):
    row = json.dumps(_row(declared))
    path = tmp_path / "repeated.json"
    path.write_text(f'{{"lanes": {{"{KEY}": {row}, "{KEY}": {row}}}}}')
    monkeypatch.setenv(LANES_ENV, str(path))
    with pytest.raises(LaneFileError, match="more than once"):
        lane_keys()


def test_one_checkpoint_on_one_base_is_one_lane(declared, tmp_path, monkeypatch):
    _lane_file(tmp_path, monkeypatch, {KEY: _row(declared), "alias": _row(declared)})
    with pytest.raises(LaneFileError, match="one checkpoint on one base is one lane"):
        lane_keys()


@pytest.mark.parametrize("extends", ["405b", KEY])
def test_an_entry_extends_a_shipped_lane_directly(declared, tmp_path, monkeypatch, extends):
    _lane_file(tmp_path, monkeypatch, {"other": dict(_row(declared), extends=extends)})
    with pytest.raises(LaneFileError, match="not a shipped lane"):
        lane_keys()


@pytest.mark.parametrize("field,value", [("dtype", "float16"),
                                         ("logprob_wrapper", "explicit-fp32-input"),
                                         ("max_model_len", 4096),
                                         ("enable_prefix_caching", True)])
def test_an_entry_may_not_set_what_it_inherits(declared, tmp_path, monkeypatch, field, value):
    _lane_file(tmp_path, monkeypatch, {KEY: dict(_row(declared), **{field: value})})
    with pytest.raises(LaneFileError, match=f"lane '{KEY}' sets \\['{field}'\\]"):
        lane_keys()


def test_an_entry_pins_exactly_the_calibration_files(declared, tmp_path, monkeypatch):
    row = _row(declared)
    row["calibration"].pop("pca_model.pkl")
    _lane_file(tmp_path, monkeypatch, {KEY: row})
    with pytest.raises(LaneFileError, match="calibration must pin exactly"):
        lane_keys()


def test_an_unreadable_lane_file_is_an_error_not_a_fall_through(tmp_path, monkeypatch):
    monkeypatch.setenv(LANES_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(LaneFileError, match="unreadable"):
        lane_keys()


def test_relative_paths_resolve_against_the_lane_file(declared):
    entry = extensions.declared_lane(KEY)
    assert entry.fixtures_dir == declared.out / "fixtures"
    assert entry.transfer_receipt == declared.out / "transfer_receipt.json"
    assert entry.calibration_dir == declared.calib.resolve()


# --- the calibration --------------------------------------------------------------


def test_the_declared_calibration_is_verified_before_use(declared, tmp_path):
    assert extensions.verify_calibration(KEY) == declared.calib.resolve()
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in runtime.CALIBRATION_FILES:
        (copy / name).write_bytes((declared.calib / name).read_bytes())
    assert extensions.verify_calibration(KEY, copy) == copy
    (copy / "positional_means.npz").write_bytes(b"other means")
    (copy / "pca_model.pkl").unlink()
    with pytest.raises(ValueError, match="positional_means.npz differs .*pca_model.pkl missing"):
        extensions.verify_calibration(KEY, copy)


def test_two_extensions_of_one_base_are_two_lanes(tmp_path, monkeypatch):
    first = declare_extension(tmp_path, monkeypatch, key="ft-a", checkpoint="a" * 64,
                              out_name="a")
    second = declare_extension(tmp_path, monkeypatch, key="ft-b", checkpoint="d" * 64,
                               out_name="b", ship=False)
    monkeypatch.setenv(LANES_ENV, f"{first.entry}:{second.entry}")
    assert lane_keys() == (*sorted(LANE_MODELS), "ft-a", "ft-b")
    assert admit("ft-a").fixtures.lane_id != admit("ft-b").fixtures.lane_id


def test_the_registry_records_what_a_row_extends(tmp_path, monkeypatch):
    from anamnesis.config.models import load_registry

    registry(tmp_path / "models.json", monkeypatch)
    ship_base(tmp_path / "shipped", monkeypatch)
    loaded = load_registry()
    assert loaded.extends_chain(PRESET) == (BASE,)
    assert loaded.extends_chain(BASE) == ()
