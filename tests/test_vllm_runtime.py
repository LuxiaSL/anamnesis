"""The vLLM lane's runtime: the two step processes, the fixtures, and the receipts.

:mod:`anamnesis.extraction.vllm.runtime` resolves everything a pass depends on
once — the environment each step starts in, the engine settings a receipt is
keyed by, the shipped fixtures and their calibration, the host's cached receipt
— and then launches the engine step and the readout step as separate
interpreters. Everything here is that resolution and its refusals, plus the
install check's flow with the two steps replaced by stand-ins that write the
vectors a step would, so the decision, the cache and the reuse of a cached
receipt are exercised end to end without an engine.

Every case that reads fixtures builds its own set and tolerance under
``tmp_path`` and points :data:`anamnesis.extraction.vllm.runtime.FIXTURES_ROOT`
at it, so no case depends on which models ship fixtures.

The last case holds the readout side of the package to its central property:
importing it never imports the engine.

What needs a device and the engine: :func:`anamnesis.extraction.vllm.runtime.capture_rows`
(it builds a vLLM engine), :func:`anamnesis.extraction.vllm.runtime.reduce_rows`
(it reduces on ``cuda:0``), :func:`anamnesis.extraction.vllm.runtime.host_fingerprint`
(it reads the first device's properties and the installed vLLM version), and so
a real install check or replay.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anamnesis.extraction.vllm import runtime, steps
from anamnesis.extraction.vllm.conformance import (
    ConformanceReceipt,
    FixtureRow,
    FixtureSet,
    HostFingerprint,
    ReceiptCache,
    RowComponent,
    Tolerance,
)
from anamnesis.extraction.vllm.envelope import (
    CONDITIONS,
    LANE_MODELS,
    READOUT_WORKSPACE,
    REQUIRED_ENV,
    SETTINGS,
    engine_settings,
    lane_id,
)
from anamnesis.provenance import digest_of_shas, file_sha

MODEL = "8b"
NAMES = ("a", "b")


def _calibration(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in runtime.CALIBRATION_FILES:
        (directory / name).write_bytes(name.encode())
    return directory


def _fixture_set(calib_dir: Path, *, model: str = MODEL, lane: str | None = None,
                 gids=(1, 2)) -> FixtureSet:
    rows = tuple(FixtureRow(generation_id=g, population="native", input_ids=(1, 2, 3, 4),
                            prompt_length=1, end=4, floor_b=1.0, selected_by="t") for g in gids)
    return FixtureSet(model=model, lane_id=lane_id(model) if lane is None else lane,
                      checkpoint_sha256="a" * 64,
                      calibration_sha256=runtime.calibration_digest(calib_dir),
                      ruler_sha256="c" * 64, feature_names=NAMES,
                      sigma_cal=np.ones(2), weights=np.ones(2), rows=rows,
                      vectors={g: np.asarray([g, g], dtype=np.float32) for g in gids})


def _tolerance(model: str = MODEL) -> Tolerance:
    return Tolerance(model=model, components=(RowComponent(
        name="all", feature_names=NAMES, max_ratio=1.0, p90_ratio=0.5, source="s"),),
        families={"a": "f", "b": "f"}, family_max_abs_sigma={"f": 1.0}, sources={})


def _fingerprint(fixtures: FixtureSet, tolerance: Tolerance) -> HostFingerprint:
    return HostFingerprint(gpu_name="g", gpu_uuid="u", driver="d", cuda_runtime="c",
                           torch="t", vllm="v", anamnesis="x",
                           checkpoint_sha256=fixtures.checkpoint_sha256,
                           engine_settings_sha256=runtime.settings_digest(fixtures.model),
                           fixture_digest=fixtures.digest, tolerance_digest=tolerance.digest)



PINNED = {"vllm": "0.16.0", "torch": "2.9.1", "triton": "3.5.1"}
"""What the stand-in installation reports: the engine packages the lane pins."""

def _ship(tmp_path: Path, monkeypatch, fixtures: FixtureSet, tolerance: Tolerance,
          directory: str | None = None) -> Path:
    """Write ``fixtures`` and ``tolerance`` where the runtime reads shipped fixtures."""
    root = tmp_path / "shipped"
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", root)
    target = root / (directory or fixtures.model)
    fixtures.save(target)
    (target / "tolerance.json").write_text(tolerance.model_dump_json())
    return target


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Shipped fixtures for ``MODEL``, their calibration, and a fixed host fingerprint."""
    calib = _calibration(tmp_path / "calib")
    fixtures, tolerance = _fixture_set(calib), _tolerance()
    _ship(tmp_path, monkeypatch, fixtures, tolerance)
    fingerprint = _fingerprint(fixtures, tolerance)
    monkeypatch.setattr(runtime, "host_fingerprint", lambda *a: fingerprint)
    monkeypatch.setattr(runtime, "require_pinned_packages", lambda: dict(PINNED))
    return SimpleNamespace(calib=calib, fixtures=fixtures, tolerance=tolerance,
                           fingerprint=fingerprint, cache=tmp_path / "cache",
                           model_path=tmp_path / "checkpoint")


# --- the environment each step starts in -------------------------------------------------


def test_the_capture_step_starts_in_the_required_environment():
    base = {"CUDA_VISIBLE_DEVICES": "3", "PATH": "/bin", "VLLM_PLUGINS": "some-plugin"}
    env = runtime.child_environment("capture", base)
    for key, value in REQUIRED_ENV.items():
        assert env[key] == value, key
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["PATH"] == "/bin"
    assert base["VLLM_PLUGINS"] == "some-plugin"


def test_the_reduce_step_strips_every_engine_variable_and_fixes_the_workspace():
    base = {"CUDA_VISIBLE_DEVICES": "0,1", "HOME": "/home/x", "VLLM_BATCH_INVARIANT": "1",
            "VLLM_PLUGINS": "", "VLLM_ANYTHING_ELSE": "y", "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    env = runtime.child_environment("reduce", base)
    assert not [key for key in env if key.startswith("VLLM_")]
    assert env["CUBLAS_WORKSPACE_CONFIG"] == READOUT_WORKSPACE
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["HOME"] == "/home/x"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert base["VLLM_BATCH_INVARIANT"] == "1"


def test_the_environment_defaults_to_this_process(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    env = runtime.child_environment("reduce")
    assert "VLLM_BATCH_INVARIANT" not in env and env["CUDA_VISIBLE_DEVICES"] == "5"


def test_an_unknown_step_has_no_environment():
    with pytest.raises(ValueError, match="unknown step"):
        runtime.child_environment("decode", {})


def test_run_step_writes_the_spec_and_launches_the_steps_module(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, env, check):
        seen.update(command=command, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec_path = tmp_path / "reduce.json"
    runtime.run_step("reduce", {"model": MODEL, "path": tmp_path}, spec_path)
    assert json.loads(spec_path.read_text()) == {"model": MODEL, "path": str(tmp_path)}
    assert seen["command"] == [sys.executable, "-m", runtime.STEP_MODULE, "reduce",
                               str(spec_path)]
    assert seen["env"]["CUBLAS_WORKSPACE_CONFIG"] == READOUT_WORKSPACE
    assert seen["check"] is False


def test_a_failed_step_is_an_error_naming_it(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3))
    with pytest.raises(RuntimeError, match="capture step exited 3"):
        runtime.run_step("capture", {}, tmp_path / "capture.json")


def test_the_steps_module_names_the_runtime_it_launches():
    assert runtime.STEP_MODULE == steps.__name__
    assert steps.STEPS == {"capture": runtime.capture_rows, "reduce": runtime.reduce_rows}


@pytest.mark.parametrize("argv", [[], ["capture"], ["decode", "spec.json"],
                                  ["reduce", "spec.json", "extra"]])
def test_the_steps_module_refuses_a_malformed_command_line(argv, capsys):
    assert steps.main(argv) == 2
    assert "usage" in capsys.readouterr().err


def test_the_steps_module_runs_the_named_step_on_its_spec(tmp_path, monkeypatch):
    received = []
    monkeypatch.setitem(steps.STEPS, "reduce", received.append)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": MODEL}))
    assert steps.main(["reduce", str(spec)]) == 0
    assert received == [{"model": MODEL}]


# --- engine settings ------------------------------------------------------------------


def test_the_settings_digest_is_stable():
    for model in LANE_MODELS:
        assert runtime.settings_digest(model) == runtime.settings_digest(model)
        assert len(runtime.settings_digest(model)) == 64


def test_the_settings_digest_separates_models_whose_settings_differ():
    """The digest covers the engine settings, so it separates exactly the models whose
    settings differ; models that share a dtype share their settings, and their
    receipts are kept apart by the checkpoint and fixture digests beside it."""
    for first in LANE_MODELS:
        for second in LANE_MODELS:
            same = all(engine_settings(first, c) == engine_settings(second, c)
                       for c in CONDITIONS)
            assert (runtime.settings_digest(first) == runtime.settings_digest(second)) == same
    assert runtime.settings_digest("3b") != runtime.settings_digest("8b")


def test_the_settings_digest_covers_the_capture_switches(monkeypatch):
    before = runtime.settings_digest(MODEL)
    monkeypatch.setattr(runtime, "ATTENTION_ROUNDING", not runtime.ATTENTION_ROUNDING)
    assert runtime.settings_digest(MODEL) != before


def test_an_unknown_model_has_no_settings():
    with pytest.raises(ValueError, match="no vLLM lane"):
        runtime.settings_digest("405b")


# --- fixtures and calibration ----------------------------------------------------------


def test_shipped_fixtures_load_with_their_tolerance(host):
    fixtures, tolerance = runtime.load_fixtures(MODEL)
    assert fixtures.digest == host.fixtures.digest
    assert tolerance.digest == host.tolerance.digest
    assert runtime.fixtures_dir(MODEL) == runtime.FIXTURES_ROOT / MODEL


def test_a_model_without_a_lane_has_no_fixtures(host):
    with pytest.raises(ValueError, match="no vLLM lane"):
        runtime.fixtures_dir("405b")


def test_a_lane_model_without_shipped_fixtures_is_refused(host):
    with pytest.raises(ValueError, match="no conformance fixtures ship for '3b'"):
        runtime.load_fixtures("3b")


def test_the_default_fixture_tree_is_inside_the_package():
    assert runtime.FIXTURES_ROOT == Path(runtime.__file__).parent / "fixtures"


def test_fixtures_shipped_under_another_models_name_are_refused(tmp_path, monkeypatch):
    calib = _calibration(tmp_path / "calib")
    _ship(tmp_path, monkeypatch, _fixture_set(calib, model="3b"), _tolerance("3b"),
          directory=MODEL)
    with pytest.raises(ValueError, match="not '8b'"):
        runtime.load_fixtures(MODEL)


def test_a_tolerance_for_another_model_is_refused(tmp_path, monkeypatch):
    calib = _calibration(tmp_path / "calib")
    _ship(tmp_path, monkeypatch, _fixture_set(calib), _tolerance("3b"))
    with pytest.raises(ValueError, match="not '8b'"):
        runtime.load_fixtures(MODEL)


def test_fixtures_naming_another_lane_are_refused(tmp_path, monkeypatch):
    calib = _calibration(tmp_path / "calib")
    _ship(tmp_path, monkeypatch, _fixture_set(calib, lane="another-lane"), _tolerance())
    with pytest.raises(ValueError, match="name lane another-lane"):
        runtime.load_fixtures(MODEL)


def test_the_calibration_digest_covers_both_artifacts(tmp_path):
    calib = _calibration(tmp_path)
    expected = digest_of_shas({name: hashlib.sha256(name.encode()).hexdigest()
                               for name in runtime.CALIBRATION_FILES})
    assert runtime.calibration_digest(calib) == expected
    assert expected == hashlib.sha256(json.dumps(
        {name: hashlib.sha256(name.encode()).hexdigest()
         for name in runtime.CALIBRATION_FILES}, sort_keys=True).encode()).hexdigest()
    (calib / "pca_model.pkl").write_bytes(b"other")
    assert runtime.calibration_digest(calib) != expected


def test_a_calibration_other_than_the_fixtures_is_refused(host):
    runtime.require_fixture_calibration(host.fixtures, host.calib)
    (host.calib / "positional_means.npz").write_bytes(b"other")
    with pytest.raises(ValueError, match="not the one"):
        runtime.require_fixture_calibration(host.fixtures, host.calib)


def test_a_calibration_missing_an_artifact_is_an_error(tmp_path):
    (tmp_path / "positional_means.npz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        runtime.calibration_digest(tmp_path)


def test_fixture_rows_are_the_engine_steps_rows(host):
    rows = runtime.fixture_rows(host.fixtures)
    assert rows == [dict(generation_id=g, input_ids=[1, 2, 3, 4], prompt_length=1, end=4)
                    for g in (1, 2)]


def test_receipts_are_cached_under_the_output_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ANAMNESIS_OUTPUTS", str(tmp_path))
    assert runtime.default_cache_dir() == tmp_path / "vllm_conformance"


# --- the install check and the receipt a replay needs -----------------------------------


def _fake_steps(monkeypatch, *, drop=None):
    """Replace both steps with stand-ins that write each pass's vectors as a step would.

    ``drop`` maps a check label (``single`` or ``batched``) to generation ids its
    capture leaves out.
    """
    calls = []

    def run_step(step, spec, spec_path):
        spec_path.write_text(json.dumps(spec, default=str))
        calls.append((step, spec))
        if step != "capture":
            return
        out = Path(spec["out"])
        skipped = set((drop or {}).get(out.name, ()))
        for index in range(spec["passes"]):
            directory = out / f"pass-{index}"
            directory.mkdir(parents=True)
            gids = [r["generation_id"] for r in spec["rows"]
                    if r["generation_id"] not in skipped]
            np.savez(directory / "vectors.npz", generation_ids=np.asarray(gids, dtype=np.int64),
                     vectors=np.stack([np.asarray([g, g], dtype=np.float32) for g in gids]))

    monkeypatch.setattr(runtime, "run_step", run_step)
    return calls


def test_the_qualifying_host_decides_identical_and_caches(host, tmp_path, monkeypatch):
    calls = _fake_steps(monkeypatch)
    receipt, cached = runtime.check_install(MODEL, host.model_path, host.calib,
                                            tmp_path / "work", host.cache)
    assert (receipt.tier, cached) == ("identical", False)
    assert receipt.lane_id == lane_id(MODEL)
    assert [(step, spec.get("condition_id")) for step, spec in calls] == [
        ("capture", "full-b1-order0"), ("reduce", None),
        ("capture", "full-b8-order0"), ("reduce", None)]
    assert [spec["passes"] for step, spec in calls if step == "capture"] == [2, 1]
    assert calls[1][1]["feature_names"] == list(NAMES)
    again, cached = runtime.check_install(MODEL, host.model_path, host.calib,
                                          tmp_path / "unused", host.cache)
    assert cached and again == receipt and len(calls) == 4
    assert not (tmp_path / "unused").exists()


def test_a_refresh_runs_the_check_again(host, tmp_path, monkeypatch):
    calls = _fake_steps(monkeypatch)
    runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "w1", host.cache)
    _, cached = runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "w2",
                                      host.cache, refresh=True)
    assert not cached and len(calls) == 8


def test_a_row_missing_from_a_pass_is_refused_for_that_reason(host, tmp_path, monkeypatch):
    _fake_steps(monkeypatch, drop={"batched": (2,)})
    receipt, _ = runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "work",
                                       host.cache)
    assert receipt.tier == "refused"
    assert any("exactly the fixture rows" in r for r in receipt.reasons)


def test_the_check_refuses_an_existing_work_directory(host, tmp_path, monkeypatch):
    _fake_steps(monkeypatch)
    (tmp_path / "work").mkdir()
    with pytest.raises(FileExistsError):
        runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "work", host.cache)


def test_the_check_refuses_another_calibration_before_capturing(host, tmp_path, monkeypatch):
    calls = _fake_steps(monkeypatch)
    (host.calib / "pca_model.pkl").write_bytes(b"other")
    with pytest.raises(ValueError, match="not the one"):
        runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "work", host.cache)
    assert not calls


def test_a_replay_needs_a_cached_receipt_naming_the_command(host):
    with pytest.raises(ValueError, match="anamnesis.scripts.qualify_vllm"):
        runtime.usable_receipt(MODEL, host.model_path, host.cache)


def _stored(host, tier: str, reasons=()) -> ConformanceReceipt:
    receipt = ConformanceReceipt(
        tier=tier, lane_id=None if tier == "refused" else lane_id(MODEL),
        qualified_lane_id=lane_id(MODEL), reasons=tuple(reasons),
        fingerprint=host.fingerprint, rows=(), family_report={})
    ReceiptCache(host.cache).store(receipt)
    return receipt


def test_a_replay_is_refused_on_a_refused_host(host):
    _stored(host, "refused", ("row 1: lane disagrees with itself",))
    with pytest.raises(ValueError, match="refused this host.*disagrees with itself"):
        runtime.usable_receipt(MODEL, host.model_path, host.cache)


@pytest.mark.parametrize("tier", ["identical", "own-lane"])
def test_a_replay_uses_a_receipt_that_did_not_refuse(host, tier):
    stored = _stored(host, tier)
    assert runtime.usable_receipt(MODEL, host.model_path, host.cache) == stored


def test_a_receipt_for_another_fingerprint_is_not_served(host, monkeypatch):
    _stored(host, "identical")
    other = host.fingerprint.model_copy(update={"driver": "another"})
    monkeypatch.setattr(runtime, "host_fingerprint", lambda *a: other)
    with pytest.raises(ValueError, match="no install check is cached"):
        runtime.usable_receipt(MODEL, host.model_path, host.cache)


def test_a_bank_refuses_a_chunk_smaller_than_one_batch(host, tmp_path):
    with pytest.raises(ValueError, match="at least eight"):
        runtime.replay_bank(MODEL, host.model_path, host.calib, [], tmp_path / "out",
                            tmp_path / "work", host.cache, chunk_rows=7, metadata={},
                            provenance={})


def test_a_bank_refuses_a_host_without_a_receipt_before_writing(host, tmp_path):
    with pytest.raises(ValueError, match="qualify_vllm"):
        runtime.replay_bank(MODEL, host.model_path, host.calib, [], tmp_path / "out",
                            tmp_path / "work", host.cache, chunk_rows=8, metadata={},
                            provenance={})
    assert not (tmp_path / "out").exists() and not (tmp_path / "work").exists()


def test_a_reduced_pass_reads_back_by_generation_id(tmp_path):
    vectors = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    np.savez(tmp_path / "vectors.npz", generation_ids=np.asarray([7, 3]), vectors=vectors)
    read = runtime.pass_vectors(tmp_path)
    assert set(read) == {3, 7}
    np.testing.assert_array_equal(read[7], vectors[0])
    np.testing.assert_array_equal(read[3], vectors[1])


# --- capture receipts ---------------------------------------------------------------


def _capture_file(tmp_path: Path, gid: int = 4):
    path = tmp_path / f"row-{gid:05d}.pt"
    path.write_bytes(b"captured substrate")
    receipt = dict(generation_id=gid, hook_noninterference=True, schedule_sha256="d" * 64,
                   raw_sha256=file_sha(path))
    return path, receipt


def test_a_capture_matching_its_receipt_passes(tmp_path):
    path, receipt = _capture_file(tmp_path)
    runtime.verify_capture_receipt(path, receipt, 4)


@pytest.mark.parametrize("change,match", [
    ({"generation_id": 5}, "another generation"),
    ({"generation_id": "4"}, "another generation"),
    ({"generation_id": 4.0}, "another generation"),
    ({"hook_noninterference": False}, "hooks changed"),
    ({"hook_noninterference": "true"}, "hooks changed"),
    ({"schedule_sha256": None}, "schedule digest"),
    ({"schedule_sha256": "d" * 63}, "schedule digest"),
    ({"raw_sha256": "0" * 64}, "bytes differ"),
])
def test_a_capture_its_receipt_does_not_describe_is_refused(tmp_path, change, match):
    path, receipt = _capture_file(tmp_path)
    with pytest.raises(ValueError, match=match):
        runtime.verify_capture_receipt(path, {**receipt, **change}, 4)


def test_a_capture_missing_a_receipt_field_is_refused(tmp_path):
    path, receipt = _capture_file(tmp_path)
    for key in receipt:
        partial = {k: v for k, v in receipt.items() if k != key}
        with pytest.raises(ValueError):
            runtime.verify_capture_receipt(path, partial, 4)


def test_a_capture_whose_bytes_changed_is_refused(tmp_path):
    path, receipt = _capture_file(tmp_path)
    path.write_bytes(b"other substrate")
    with pytest.raises(ValueError, match="bytes differ"):
        runtime.verify_capture_receipt(path, receipt, 4)


# --- replay inputs ------------------------------------------------------------------


def _entries(**spans):
    return {gid: dict(input_ids=list(range(end)), prompt_length=prompt)
            for gid, (prompt, end) in spans.items()}


def test_replay_rows_are_the_engine_steps_rows():
    entries = _entries(**{"3": (2, 6), "9": (1, 4)})
    assert runtime.replay_rows(entries, [9, 3]) == [
        dict(generation_id=9, input_ids=[0, 1, 2, 3], prompt_length=1, end=4),
        dict(generation_id=3, input_ids=[0, 1, 2, 3, 4, 5], prompt_length=2, end=6)]


@pytest.mark.parametrize("prompt,end", [(3, 4), (4, 4), (0, 4)])
def test_a_span_without_a_prompt_and_two_generated_tokens_is_refused(prompt, end):
    with pytest.raises(ValueError, match="generation 1: its span needs"):
        runtime.replay_rows(_entries(**{"1": (prompt, end)}), [1])


def test_a_span_longer_than_the_lanes_context_is_refused():
    limit = SETTINGS["max_model_len"]
    runtime.replay_rows(_entries(**{"1": (10, limit)}), [1])
    with pytest.raises(ValueError, match=f"{limit + 1} tokens exceed the lane's context of {limit}"):
        runtime.replay_rows(_entries(**{"1": (10, limit + 1)}), [1])


def test_source_metadata_reads_either_shape(tmp_path):
    assert runtime.source_metadata(None) == {}
    records = [dict(generation_id=4, mode="a"), dict(generation_id=7, mode="b")]
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps(records))
    wrapped = tmp_path / "metadata.json"
    wrapped.write_text(json.dumps(dict(generations=records, run="x")))
    expected = {4: records[0], 7: records[1]}
    assert runtime.source_metadata(listed) == expected
    assert runtime.source_metadata(wrapped) == expected


def test_source_metadata_without_a_generation_list_is_refused(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(dict(generations={"4": {}})))
    with pytest.raises(ValueError, match="generation list"):
        runtime.source_metadata(path)
    path.write_text(json.dumps("text"))
    with pytest.raises(ValueError, match="generation list"):
        runtime.source_metadata(path)


def test_source_metadata_naming_a_generation_twice_is_refused(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps([dict(generation_id=4), dict(generation_id="4")]))
    with pytest.raises(ValueError, match="generation 4 twice"):
        runtime.source_metadata(path)


def test_the_schema_digest_is_the_fast_lanes(monkeypatch):
    """The vLLM lane stamps the schema digest the fast lane stamps for the same names."""
    import torch
    from test_vllm_readout import make_example

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    lane = make_example(torch.float32)[0]
    assert runtime.feature_schema_sha256(lane.names) == lane.identity["feature_schema_sha256"]


# --- the engine stays out of the readout side --------------------------------------------

READOUT_SIDE = (
    "anamnesis.extraction.vllm",
    "anamnesis.extraction.vllm.envelope",
    "anamnesis.extraction.vllm.conformance",
    "anamnesis.extraction.vllm.rows",
    "anamnesis.extraction.vllm.receipts",
    "anamnesis.extraction.vllm.stats_kernel",
    "anamnesis.extraction.vllm.second_pass",
    "anamnesis.extraction.vllm.step_products",
    "anamnesis.extraction.vllm.adapter",
    "anamnesis.extraction.vllm.readout",
    "anamnesis.extraction.vllm.runtime",
    "anamnesis.extraction.vllm.steps",
    "anamnesis.scripts.qualify_vllm",
    "anamnesis.scripts.run_vllm_replay",
)
"""Every module the readout process and the commands import. The engine-side modules
(``backend``, ``capture``, ``runner``) reach for the engine when it is installed."""

ENGINE_PROBE = """
import importlib, json, sys
attempts = []
class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name == "vllm" or name.startswith("vllm."):
            attempts.append(name)
            raise ModuleNotFoundError(name)
        return None
sys.meta_path.insert(0, Refuse())
for module in json.loads(sys.argv[1]):
    importlib.import_module(module)
print(json.dumps(dict(attempts=attempts,
                      loaded=[m for m in sys.modules if m == "vllm" or m.startswith("vllm.")])))
"""


def _probe(modules) -> dict:
    result = subprocess.run([sys.executable, "-c", ENGINE_PROBE, json.dumps(list(modules))],
                            capture_output=True, text=True, check=False,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_importing_the_readout_side_never_imports_the_engine():
    """Every import is watched, so the claim holds whether or not vLLM is installed."""
    assert _probe(READOUT_SIDE) == dict(attempts=[], loaded=[])


def test_the_engine_probe_sees_an_engine_import():
    """The counterpart: the backend reaches for the engine, and the probe sees it,
    so an empty result above is a property of the import graph."""
    assert _probe(["anamnesis.extraction.vllm.backend"])["attempts"]


def test_a_checkpoint_other_than_the_fixtures_is_refused_before_any_capture(
        host, tmp_path, monkeypatch):
    other = host.fingerprint.model_copy(update={"checkpoint_sha256": "f" * 64})
    monkeypatch.setattr(runtime, "host_fingerprint", lambda *a: other)
    steps = []
    monkeypatch.setattr(runtime, "run_step", lambda *a, **k: steps.append(a))
    with pytest.raises(ValueError, match="not the one the .* fixtures were produced from"):
        runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "work",
                              host.cache)
    assert not steps and not (tmp_path / "work").exists()


def test_the_engine_packages_are_checked_before_the_host_is_fingerprinted(
        host, tmp_path, monkeypatch):
    def missing():
        raise RuntimeError("the vLLM lane needs the pinned engine")

    def fingerprinted(*args):
        raise AssertionError("the fingerprint ran before the package check")

    monkeypatch.setattr(runtime, "require_pinned_packages", missing)
    monkeypatch.setattr(runtime, "host_fingerprint", fingerprinted)
    with pytest.raises(RuntimeError, match="pinned engine"):
        runtime.check_install(MODEL, host.model_path, host.calib, tmp_path / "work",
                              host.cache)
    with pytest.raises(RuntimeError, match="pinned engine"):
        runtime.usable_receipt(MODEL, host.model_path, host.cache)


def test_the_settings_digest_changes_with_the_lane_source(monkeypatch):
    before = runtime.settings_digest(MODEL)
    monkeypatch.setattr(runtime, "lane_source_digest", lambda: "0" * 64)
    assert runtime.settings_digest(MODEL) != before
