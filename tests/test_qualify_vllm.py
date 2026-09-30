"""What ``qualify_vllm.py`` accepts, what it reports, and the exit status it returns.

The command is one call into :func:`anamnesis.extraction.vllm.runtime.check_install`
and a report of the receipt. Its exit status is the contract a script around it
reads — 0 identical or own-lane, 1 refused, 2 the check could not run — so each
is pinned here with the check replaced by a stand-in that returns a receipt or
raises.

What needs a device and the engine is the check itself; its flow, with the two
step processes replaced, is covered in ``tests/test_vllm_runtime.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anamnesis.extraction.vllm import hub, runtime
from anamnesis.extraction.vllm.conformance import ConformanceReceipt, HostFingerprint
from anamnesis.scripts import qualify_vllm

FINGERPRINT = HostFingerprint(gpu_name="g", gpu_uuid="u", driver="d", cuda_runtime="c",
                              torch="t", vllm="v", anamnesis="x", checkpoint_sha256="a" * 64,
                              engine_settings_sha256="e" * 64, fixture_digest="f" * 64,
                              tolerance_digest="0" * 64)


def _receipt(tier, lane="lane-8b", reasons=(), family_report=None):
    return ConformanceReceipt(tier=tier, lane_id=lane, qualified_lane_id="lane-8b",
                              reasons=tuple(reasons), fingerprint=FINGERPRINT, rows=(),
                              family_report=family_report or {})


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--model", "8b", "--model-path", str(tmp_path / "ckpt"),
            "--calib-dir", str(tmp_path / "calib"), "--work-dir", str(tmp_path / "work"),
            *extra]


def _stand_in(monkeypatch, outcome, cached=False):
    calls = []

    def check_install(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, cached

    monkeypatch.setattr(runtime, "check_install", check_install)
    monkeypatch.setattr(runtime, "load_fixtures", lambda model: (None, None))
    monkeypatch.setattr(hub, "verify_calibration", lambda model, directory: None)
    return calls


def test_an_identical_host_exits_zero_and_names_the_qualified_lane(tmp_path, monkeypatch,
                                                                   capsys):
    calls = _stand_in(monkeypatch, _receipt("identical"))
    assert qualify_vllm.main(_argv(tmp_path, "--cache-dir", str(tmp_path / "cache"))) == 0
    out = capsys.readouterr().out
    assert "tier: identical" in out and "lane id: lane-8b" in out and "(cached)" not in out
    args, kwargs = calls[0]
    assert args == ("8b", tmp_path / "ckpt", tmp_path / "calib", tmp_path / "work",
                    tmp_path / "cache")
    assert kwargs == {"refresh": False}


def test_an_own_lane_host_exits_zero_and_reports_its_worst_family(tmp_path, monkeypatch,
                                                                   capsys):
    _stand_in(monkeypatch, _receipt("own-lane", lane="lane-8b+host-1",
                                    family_report={"flow": 0.4, "residual": 0.7}))
    assert qualify_vllm.main(_argv(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "lane id: lane-8b+host-1" in out
    assert "worst family: residual at 0.7" in out


def test_a_refused_host_exits_one_with_its_reasons(tmp_path, monkeypatch, capsys):
    _stand_in(monkeypatch, _receipt("refused", lane=None,
                                    reasons=("checkpoint digest differs from the fixtures'",)))
    assert qualify_vllm.main(_argv(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "tier: refused" in out and "checkpoint digest differs" in out
    assert "lane id" not in out


def test_a_cached_receipt_is_reported_as_cached(tmp_path, monkeypatch, capsys):
    _stand_in(monkeypatch, _receipt("identical"), cached=True)
    assert qualify_vllm.main(_argv(tmp_path)) == 0
    assert "(cached)" in capsys.readouterr().out


def test_refresh_is_passed_through(tmp_path, monkeypatch):
    calls = _stand_in(monkeypatch, _receipt("identical"))
    qualify_vllm.main(_argv(tmp_path, "--refresh"))
    assert calls[0][1] == {"refresh": True}


def test_the_cache_defaults_to_the_output_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANAMNESIS_OUTPUTS", str(tmp_path / "outputs"))
    calls = _stand_in(monkeypatch, _receipt("identical"))
    qualify_vllm.main(_argv(tmp_path))
    assert calls[0][0][4] == tmp_path / "outputs" / "vllm_conformance"
    assert str(tmp_path / "outputs" / "vllm_conformance") in capsys.readouterr().out


@pytest.mark.parametrize("error", [
    ValueError("no conformance fixtures ship for '8b'"),
    RuntimeError("the capture step exited 1"),
    FileExistsError("work"),
])
def test_a_check_that_cannot_run_exits_two(tmp_path, monkeypatch, capsys, error):
    _stand_in(monkeypatch, error)
    assert qualify_vllm.main(_argv(tmp_path)) == 2
    assert "the install check did not run" in capsys.readouterr().err


def test_the_check_runs_without_shipped_fixtures_only_to_refuse(tmp_path, monkeypatch, capsys):
    """With no fixtures under the fixture root, the real check stops before any device work."""
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", tmp_path / "none")
    assert qualify_vllm.main(_argv(tmp_path)) == 2
    assert "no conformance fixtures ship for '8b'" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--model-path", "m", "--calib-dir", "c", "--work-dir", "w"],
    ["--model", "405b", "--model-path", "m", "--calib-dir", "c", "--work-dir", "w"],
    ["--model", "8b", "--calib-dir", "c", "--work-dir", "w"],
    ["--model", "8b", "--model-path", "m", "--calib-dir", "c"],
])
def test_an_incomplete_command_line_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        qualify_vllm.main(argv)
    assert exit_info.value.code == 2


def test_the_command_offers_exactly_the_lane_models():
    from anamnesis.extraction.vllm.envelope import LANE_MODELS

    action = next(a for a in qualify_vllm.parser()._actions if a.dest == "model")
    assert sorted(action.choices) == sorted(LANE_MODELS)


def test_a_model_without_fixtures_is_refused_before_any_calibration_is_fetched(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", tmp_path / "none")
    fetched = []
    monkeypatch.setattr(hub, "fetch_calibration", lambda model: fetched.append(model))
    argv = ["--model", "8b", "--model-path", str(tmp_path / "ckpt"),
            "--work-dir", str(tmp_path / "work")]
    assert qualify_vllm.main(argv) == 2
    assert "no conformance fixtures ship for '8b'" in capsys.readouterr().err
    assert not fetched
