"""What ``run_vllm_replay.py`` refuses before it loads anything, and what it hands the bank.

The command reads a replay manifest, selects generations, reads their source
metadata and turns them into the engine step's rows, then makes one call into
:func:`anamnesis.extraction.vllm.runtime.replay_bank`. Everything up to that call
runs without a device, so the refusals — an output that exists, a selection that
is empty, duplicated or unknown, metadata that does not cover it, a span the lane
cannot hold — are exercised here directly, and the bank is replaced by a
stand-in that records what it was given.

What needs a device and the engine is the bank itself: capturing, reducing and
writing the rows. Its refusals that precede any device work are covered in
``tests/test_vllm_runtime.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.extraction.vllm import runtime
from anamnesis.extraction.vllm.envelope import SETTINGS
from anamnesis.provenance import file_sha
from anamnesis.scripts import run_vllm_replay


def _manifest(tmp_path: Path, spans: dict[int, tuple[int, int]]) -> Path:
    entries = {str(gid): dict(input_ids=list(range(end)), prompt_length=prompt,
                              n_gen=end - prompt)
               for gid, (prompt, end) in spans.items()}
    path = tmp_path / "run" / "replay_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(entries=entries, n_ok=len(entries), n_flagged=0,
                                    flagged=[])))
    return path


def _argv(tmp_path: Path, manifest: Path, *extra: str) -> list[str]:
    return ["--model", "8b", "--model-path", str(tmp_path / "ckpt"),
            "--calib-dir", str(tmp_path / "calib"), "--manifest", str(manifest),
            "--output", str(tmp_path / "out"), *extra]


def _stand_in(monkeypatch, outcome=None):
    calls = []

    def replay_bank(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return len(args[3])

    monkeypatch.setattr(runtime, "replay_bank", replay_bank)
    return calls


def test_a_replay_hands_the_bank_its_rows_and_provenance(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANAMNESIS_OUTPUTS", str(tmp_path / "outputs"))
    manifest = _manifest(tmp_path, {3: (2, 6), 9: (1, 4)})
    calls = _stand_in(monkeypatch)
    assert run_vllm_replay.main(_argv(tmp_path, manifest)) == 0
    assert json.loads(capsys.readouterr().out) == dict(model="8b", rows=2,
                                                       output=str(tmp_path / "out"))
    args, kwargs = calls[0]
    model, model_path, calib, rows, output, work, cache = args
    assert (model, model_path, calib) == ("8b", tmp_path / "ckpt", tmp_path / "calib")
    assert [r["generation_id"] for r in rows] == [3, 9]
    assert rows[0] == dict(generation_id=3, input_ids=[0, 1, 2, 3, 4, 5], prompt_length=2,
                           end=6)
    assert output == tmp_path / "out" and work == tmp_path / "out.work"
    assert cache == tmp_path / "outputs" / "vllm_conformance"
    assert kwargs["chunk_rows"] == 32 and kwargs["metadata"] == {}
    provenance = kwargs["provenance"]
    assert provenance["manifest_sha256"] == file_sha(manifest)
    assert provenance["selected_ids"] == [3, 9]
    assert provenance["source_metadata_sha256"] is None
    assert provenance["runner_sha256"] == file_sha(Path(run_vllm_replay.__file__))


def test_the_selection_and_the_bank_settings_are_passed_through(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {3: (2, 6), 9: (1, 4)})
    calls = _stand_in(monkeypatch)
    run_vllm_replay.main(_argv(tmp_path, manifest, "--gen-ids", "9", "--chunk-rows", "8",
                               "--work-dir", str(tmp_path / "w"),
                               "--cache-dir", str(tmp_path / "c")))
    args, kwargs = calls[0]
    assert [r["generation_id"] for r in args[3]] == [9]
    assert args[5] == tmp_path / "w" and args[6] == tmp_path / "c"
    assert kwargs["chunk_rows"] == 8


def test_metadata_beside_the_manifest_is_read_and_stamped(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {3: (2, 6)})
    metadata = manifest.parent / "metadata.json"
    metadata.write_text(json.dumps(dict(generations=[dict(generation_id=3, mode="a")])))
    calls = _stand_in(monkeypatch)
    run_vllm_replay.main(_argv(tmp_path, manifest))
    kwargs = calls[0][1]
    assert kwargs["metadata"] == {3: dict(generation_id=3, mode="a")}
    assert kwargs["provenance"]["source_metadata_sha256"] == file_sha(metadata)


def test_an_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {3: (2, 6)})
    calls = _stand_in(monkeypatch)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run_vllm_replay.main(_argv(tmp_path, manifest))
    assert not calls


@pytest.mark.parametrize("ids", [["4"], ["3", "3"]])
def test_an_unknown_or_duplicated_selection_is_refused(tmp_path, monkeypatch, ids):
    manifest = _manifest(tmp_path, {3: (2, 6)})
    calls = _stand_in(monkeypatch)
    with pytest.raises(ValueError, match="generation selection"):
        run_vllm_replay.main(_argv(tmp_path, manifest, "--gen-ids", *ids))
    assert not calls


def test_an_empty_manifest_is_refused(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {})
    calls = _stand_in(monkeypatch)
    with pytest.raises(ValueError, match="generation selection"):
        run_vllm_replay.main(_argv(tmp_path, manifest))
    assert not calls


def test_metadata_that_misses_a_selected_generation_is_refused(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, {3: (2, 6), 9: (1, 4)})
    metadata = tmp_path / "other.json"
    metadata.write_text(json.dumps([dict(generation_id=3)]))
    calls = _stand_in(monkeypatch)
    with pytest.raises(ValueError, match="missing selected generation"):
        run_vllm_replay.main(_argv(tmp_path, manifest, "--metadata", str(metadata)))
    assert not calls


def test_a_span_longer_than_the_lanes_context_is_refused_before_the_bank(tmp_path,
                                                                        monkeypatch):
    limit = SETTINGS["max_model_len"]
    manifest = _manifest(tmp_path, {3: (2, limit + 1)})
    calls = _stand_in(monkeypatch)
    with pytest.raises(ValueError, match="exceed the lane's context"):
        run_vllm_replay.main(_argv(tmp_path, manifest))
    assert not calls


def test_a_bank_that_refuses_exits_two(tmp_path, monkeypatch, capsys):
    manifest = _manifest(tmp_path, {3: (2, 6)})
    _stand_in(monkeypatch, ValueError("no install check is cached for 8b on this host"))
    assert run_vllm_replay.main(_argv(tmp_path, manifest)) == 2
    assert "the replay did not run: no install check is cached" in capsys.readouterr().err


def test_the_real_bank_refuses_without_shipped_fixtures(tmp_path, monkeypatch, capsys):
    """With no fixtures under the fixture root, the bank stops before any device work."""
    monkeypatch.setattr(runtime, "FIXTURES_ROOT", tmp_path / "none")
    manifest = _manifest(tmp_path, {3: (2, 6)})
    assert run_vllm_replay.main(_argv(tmp_path, manifest, "--chunk-rows", "8")) == 2
    assert "no conformance fixtures ship for '8b'" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("argv", [
    ["--model", "405b", "--model-path", "m", "--calib-dir", "c", "--manifest", "x",
     "--output", "o"],
    ["--model", "8b", "--model-path", "m", "--calib-dir", "c", "--output", "o"],
    ["--model", "8b", "--model-path", "m", "--calib-dir", "c", "--manifest", "x"],
    ["--model", "8b", "--model-path", "m", "--calib-dir", "c", "--manifest", "x",
     "--output", "o", "--chunk-rows", "eight"],
])
def test_an_incomplete_command_line_is_a_usage_error(argv):
    with pytest.raises(SystemExit) as exit_info:
        run_vllm_replay.main(argv)
    assert exit_info.value.code == 2
