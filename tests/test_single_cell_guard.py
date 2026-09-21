"""The path-of-record guard: a roster walked one model load at a time is refused.

The pattern this exists to stop is a shell loop over a single-cell launcher — the
same script, the same model, a different output directory each time — which pays a
model load per cell while the load-once path sits there documented and unused. The
bar is that the pattern cannot recur silently: it either fails loudly or it says,
in the log, that an escape hatch let it through.

Every case here runs on a CPU against an isolated guard directory with an explicit
job context, so the tests cannot see each other's records and cannot see the
machine's.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from anamnesis.orchestration.gpu import (
    CONTEXT_ENV,
    ESCAPE_ENV,
    GUARD_DIR_ENV,
    TTL_ENV,
    enforce_single_cell_guard,
)

SCRIPT = "anamnesis.scripts.run_replay"
POINTER = "python -m anamnesis.scripts.run_replay --cells-json ... --gpus ..."


@pytest.fixture()
def guard_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(GUARD_DIR_ENV, str(tmp_path / "guard"))
    monkeypatch.setenv(CONTEXT_ENV, "test-job-ctx")
    monkeypatch.delenv(ESCAPE_ENV, raising=False)
    monkeypatch.delenv(TTL_ENV, raising=False)
    return tmp_path


def _call(model: str, out_dir: Path, *, allow: bool = False) -> None:
    enforce_single_cell_guard(
        SCRIPT, model, out_dir, allow_repeat=allow, multicell_pointer=POINTER
    )


def test_second_cell_same_model_refused(guard_env: Path) -> None:
    """The pattern itself: same script and model, a second output directory."""
    _call("/models/gemma", guard_env / "cellA")
    with pytest.raises(SystemExit) as exc:
        _call("/models/gemma", guard_env / "cellB")
    message = str(exc.value)
    assert "PATH-OF-RECORD GUARD" in message
    assert "--cells-json" in message, "the refusal must name the path that loads once"
    assert "--single-cell-ok" in message, "and the way to say the repeat was deliberate"


def test_resume_same_out_dir_passes(guard_env: Path) -> None:
    """Re-running one cell is a resume, which is the normal case and never refused."""
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellA")


def test_different_model_passes(guard_env: Path) -> None:
    """Two models in one job are two passes, not a loop over one roster."""
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/qwen", guard_env / "cellB")


def test_different_script_passes(guard_env: Path) -> None:
    """Generate then replay the same model in one job is a chain, not a loop."""
    enforce_single_cell_guard(
        "anamnesis.scripts.run_gen_tokens", "/models/gemma", guard_env / "cellA",
        allow_repeat=False, multicell_pointer=POINTER,
    )
    _call("/models/gemma", guard_env / "cellB")


def test_escape_hatch_flag(guard_env: Path) -> None:
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellB", allow=True)


def test_escape_hatch_env(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment hatch exists for an invocation buried inside a driver."""
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(ESCAPE_ENV, "1")
    _call("/models/gemma", guard_env / "cellB")


def test_escape_env_zero_still_refuses(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero means off, so a driver can unset the hatch by setting it."""
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(ESCAPE_ENV, "0")
    with pytest.raises(SystemExit):
        _call("/models/gemma", guard_env / "cellB")


def test_separate_job_contexts_isolated(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new job is a new context: yesterday's cell does not refuse today's."""
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(CONTEXT_ENV, "another-job")
    _call("/models/gemma", guard_env / "cellB")


def test_ttl_expiry_allows(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived interactive shell does not trip over work from days before."""
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 13 * 3600)
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setattr(time, "time", real_time)
    _call("/models/gemma", guard_env / "cellB")


def test_third_cell_message_counts_prior(guard_env: Path) -> None:
    """The refusal says how deep the loop already is, which is what names it."""
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellB", allow=True)
    with pytest.raises(SystemExit) as exc:
        _call("/models/gemma", guard_env / "cellC")
    assert "invocation #3" in str(exc.value)


def test_unwritable_guard_dir_degrades_to_warning(
    guard_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing a bookkeeping record must never kill a production launcher."""
    blocked = guard_env / "blocked_file"
    blocked.write_text("not a dir")
    monkeypatch.setenv(GUARD_DIR_ENV, str(blocked / "nope"))
    _call("/models/gemma", guard_env / "cellA")
