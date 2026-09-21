"""The one fast-lane resolution, and the claim that both commands read it.

`run_gpu_replay.py` banks signatures through the lane and `qualify_box.py` measures
whether the lane agrees with the numeric anchor on this machine. Those two are only
worth running if they are about the *same* machine state, and the state is not one
setting: it is the pinned arithmetic, the schema the selected spans resolve to, the
calibration the features are corrected by, and the model the lane reads. Two
commands resolving that separately can qualify one configuration and bank another,
and the only symptom is a receipt describing a machine the run does not reproduce.

So the first case here pins the sharing *by object identity* rather than by
inspection: both commands hold the same function, and a future copy in either one
fails this file rather than being noticed in review. The rest are the refusals and
the arithmetic that need no device — the span checks, the weight digests, and the
environment precondition. Building the lane itself needs CUDA and is covered by the
equivalence suite on a machine that has one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.extraction.fast import runtime
from anamnesis.extraction.fast.runtime import (
    WORKSPACE_ENV,
    WORKSPACE_VALUE,
    FastLaneRuntime,
    LaneSpan,
    require_lane_arithmetic,
    resolve_lane_spans,
    weight_file_digests,
)
from anamnesis.scripts import qualify_box, run_gpu_replay


def test_both_commands_resolve_the_lane_through_one_function() -> None:
    """The consolidation's whole claim, pinned so a second copy cannot pass review."""
    assert qualify_box.resolve_fast_lane is runtime.resolve_fast_lane
    assert run_gpu_replay.resolve_fast_lane is runtime.resolve_fast_lane
    assert qualify_box.require_lane_arithmetic is runtime.require_lane_arithmetic
    assert run_gpu_replay.require_lane_arithmetic is runtime.require_lane_arithmetic


def test_neither_command_builds_a_lane_or_loads_a_model_of_its_own() -> None:
    """Sharing the resolver is worth nothing if a command still has its own path.

    `GpuFeatureLane` and `load_model` are named in exactly one place now, so neither
    entry point can construct a lane on settings the other one has not agreed to.
    """
    for module in (qualify_box, run_gpu_replay):
        source = Path(module.__file__).read_text()
        assert "GpuFeatureLane" not in source, module.__name__
        assert "load_model" not in source, module.__name__
        assert "allow_tf32" not in source, module.__name__


def test_the_arithmetic_is_refused_rather_than_set_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cuBLAS reads its workspace variable at initialisation, so this cannot be fixed up."""
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    with pytest.raises(ValueError, match=WORKSPACE_ENV):
        require_lane_arithmetic()
    monkeypatch.setenv(WORKSPACE_ENV, ":4096:8")
    with pytest.raises(ValueError, match="different lane"):
        require_lane_arithmetic()


def test_the_arithmetic_pins_are_idempotent_and_off_by_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both commands ask for the pins, one of them twice; asking again is not an error."""
    import torch

    monkeypatch.setenv(WORKSPACE_ENV, WORKSPACE_VALUE)
    require_lane_arithmetic()
    require_lane_arithmetic()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)


def _entries(*rows: tuple[int, int, int]) -> dict[str, dict[str, object]]:
    """A manifest's entries: (gen_id, prompt_length, generated token count) each."""
    return {
        str(gen_id): {
            "prompt_length": prompt,
            "input_ids": list(range(prompt + generated)),
        }
        for gen_id, prompt, generated in rows
    }


def test_a_span_derives_its_end_and_its_incremental_step_count() -> None:
    """The prefill produces the first generated token, so n steps is one fewer."""
    span = LaneSpan(gen_id=3, input_ids=list(range(10)), prompt_length=4)
    assert span.end == 10
    assert span.n_steps == 5


def test_spans_come_back_in_the_order_they_were_asked_for() -> None:
    spans = resolve_lane_spans(
        _entries((0, 4, 6), (1, 4, 6), (2, 4, 6)), [2, 0], positions_calibrated=512
    )
    assert [span.gen_id for span in spans] == [2, 0]


def test_a_generation_outside_the_manifest_is_named_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="not in the manifest"):
        resolve_lane_spans(_entries((0, 4, 6)), [0, 9], positions_calibrated=512)


@pytest.mark.parametrize(
    "row",
    [
        (0, 0, 6),  # no prompt at all
        (0, 4, 1),  # one generated token: prefill only, no incremental step
    ],
    ids=["no-prompt", "no-incremental-step"],
)
def test_a_span_the_lane_cannot_replay_is_refused(row: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="outside supported span/calibration"):
        resolve_lane_spans(_entries(row), [0], positions_calibrated=512)


def test_a_span_past_the_calibrated_positions_is_refused() -> None:
    """An uncalibrated position would be corrected against nothing and quietly differ."""
    entries = _entries((0, 4, 20))
    resolve_lane_spans(entries, [0], positions_calibrated=64)
    with pytest.raises(ValueError, match="outside supported span/calibration"):
        resolve_lane_spans(entries, [0], positions_calibrated=8)


def test_weight_digests_name_the_bytes_and_not_the_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    digests = weight_file_digests(tmp_path)
    assert set(digests) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(len(value) == 64 for value in digests.values())
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"three")
    assert weight_file_digests(tmp_path) != digests


def test_a_checkpoint_with_nothing_to_digest_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="local safetensors checkpoint"):
        weight_file_digests(tmp_path)


def test_the_runtime_reports_its_generations_in_span_order() -> None:
    """What a caller iterates, so a bank's rows follow the selection rather than the disk."""
    spans = resolve_lane_spans(
        _entries((7, 4, 6), (2, 4, 6)), [7, 2], positions_calibrated=512
    )
    built = FastLaneRuntime(
        extraction=None,  # type: ignore[arg-type]
        families=None,  # type: ignore[arg-type]
        spans=spans,
        schemas={},
        feature_names=("a", "b"),
        positional_means=np.zeros((2, 4), dtype=np.float32),
        pca_components=None,
        pca_mean=None,
        calibration_files={},
        calibration_sha256="0" * 64,
        model_files=None,
        loaded=None,
        lane=None,
    )
    assert built.gen_ids == (7, 2)


def test_the_qualification_pairs_rows_and_takes_two_by_default(tmp_path: Path) -> None:
    """The command's own selection policy, which the shared resolution does not hold."""
    manifest = tmp_path / "replay_manifest.json"
    manifest.write_text(json.dumps({"entries": _entries((0, 4, 6), (1, 4, 6), (2, 4, 6))}))
    base = ["--model", "3b", "--model-path", "/c", "--calib-dir", str(tmp_path),
            "--manifest", str(manifest)]

    _, ids = qualify_box.select_rows(qualify_box.parser().parse_args(base))
    assert ids == [0, 1], "a qualification takes the first two rows unless told otherwise"

    for bad in (["4", "4"], ["1"]):
        args = qualify_box.parser().parse_args(base + ["--gen-ids", *bad])
        with pytest.raises(ValueError, match="two or more distinct"):
            qualify_box.select_rows(args)

    args = qualify_box.parser().parse_args(base + ["--gen-ids", "0", "9"])
    with pytest.raises(ValueError, match="unknown generation selection"):
        qualify_box.select_rows(args)


def test_the_replay_takes_the_whole_bank_by_default_and_refuses_a_bad_selection() -> None:
    """The other command's policy: one row is enough, but it must be in the manifest."""
    entries = _entries((0, 4, 6), (5, 4, 6))
    assert run_gpu_replay.select_ids(entries, None) == [0, 5]
    assert run_gpu_replay.select_ids(entries, [5]) == [5]
    for bad in ([], [5, 5], [9]):
        with pytest.raises(ValueError, match="empty, duplicated or unknown"):
            run_gpu_replay.select_ids(entries, bad)
