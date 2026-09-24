"""The in-process seam: a resident model, harvested span by span, with no command line.

A server that keeps one model loaded and reads the signature of its own generations
needs four things from the lane, and each is pinned here:

1. **The model it holds is the model the lane reads.** :func:`prepare_fast_lane`
   takes a loaded model and never loads another; the loader is replaced by one that
   fails, and two harvests run anyway. A model the lane would read wrongly — another
   depth, another kernel, no capture hooks, another device — is refused by name
   before any span runs.
2. **The logit series is opt-in and changes nothing else.** Absent by default; when
   asked for, shaped ``[T, k]`` with ``T`` one fewer than the generated tokens,
   matching a direct forward's top-k and entropy; and the feature vector and the
   lane id are byte-identical with and without it. The series is read outside the
   files the lane hashes into its identity, which is pinned by listing those files.
3. **The command line is a thin caller of the same thing.** ``main(argv)`` runs from
   a list, ``--model`` accepts a preset added through ``ANAMNESIS_MODELS`` and
   refuses one no registry holds, and the bank it writes carries the same vector
   :func:`harvest_loaded` returns for the same span.
4. **A checkpoint is digested once per process** while its files are unchanged.

Everything runs on a CPU: the model is the random-weight Llama the equivalence suite
uses, because the lane refuses anything that is not a dense Llama with eager
attention. Whether the lane's numbers on a given accelerator agree with the anchor is
:mod:`anamnesis.scripts.qualify_box`'s question, not this file's.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch
from test_fast_lane_equivalence import tiny_loaded

from anamnesis.config import ModelPreset
from anamnesis.extraction import model_loader
from anamnesis.extraction.fast import runtime
from anamnesis.extraction.fast.harvest import HarvestResult, harvest_loaded
from anamnesis.extraction.fast.runtime import (
    WORKSPACE_ENV,
    WORKSPACE_VALUE,
    PreparedLane,
    check_loaded_model,
    prepare_fast_lane,
    weight_file_digests,
)
from anamnesis.extraction.model_loader import LoadedModel
from anamnesis.scripts import run_gpu_replay

POSITIONS = 48
PROMPT = 7

TINY_ROW: dict[str, Any] = dict(
    name="tiny-llama",
    model_id="tiny-llama",
    torch_dtype="float32",
    num_layers=3,
    hidden_dim=32,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    sampled_layers=[0, 1, 2],
    pca_layers=[0, 1],
    trajectory_layers=[1],
    contrastive_layers=[1],
    early_layer_cutoff=0,
    late_layer_cutoff=2,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=16,
    eos_token_ids=[0],
    calibration_root="outputs",
    calibration_dir="calibration/tiny-llama",
)


def tiny_preset(**overrides: Any) -> ModelPreset:
    return ModelPreset(**{**TINY_ROW, **overrides})


@pytest.fixture
def lane_arithmetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The environment the lane is defined at, and the global pins undone afterwards."""
    monkeypatch.setenv(WORKSPACE_ENV, WORKSPACE_VALUE)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def calib_dir(tmp_path: Path) -> Path:
    """A complete calibration for the tiny model: positional means and a pooled basis."""
    rng = np.random.default_rng(11)
    directory = tmp_path / "calibration"
    directory.mkdir()
    np.savez(
        directory / "positional_means.npz",
        positional_means=rng.normal(0, 0.01, size=(4, POSITIONS, 32)).astype(np.float32),
    )
    with open(directory / "pca_model.pkl", "wb") as stream:
        pickle.dump(
            {
                "components": rng.normal(size=(50, 32)).astype(np.float32),
                "mean": rng.normal(0, 0.01, size=32).astype(np.float32),
            },
            stream,
        )
    return directory


@pytest.fixture
def no_loading(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace the loader with one that records the attempt and fails it."""
    calls: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("the lane loaded a model although one was passed")

    monkeypatch.setattr(model_loader, "load_model", refuse)
    return calls


def tokens(n: int, seed: int = 3) -> list[int]:
    return np.random.default_rng(seed).integers(1, 64, size=n).tolist()


def prepared(calib_dir: Path, **kwargs: Any) -> PreparedLane:
    return prepare_fast_lane(
        tiny_preset(), calib_dir, device="cpu", loaded=tiny_loaded(), **kwargs
    )


def test_a_passed_model_is_reused_and_never_reloaded(
    lane_arithmetic: None, calib_dir: Path, no_loading: list[Any]
) -> None:
    loaded = tiny_loaded()
    lane = prepare_fast_lane(tiny_preset(), calib_dir, device="cpu", loaded=loaded)
    assert lane.loaded is loaded
    first = harvest_loaded(lane, tokens(PROMPT + 20), prompt_len=PROMPT)
    second = harvest_loaded(lane, tokens(PROMPT + 30, seed=4), prompt_len=PROMPT)
    assert no_loading == []
    assert first.feature_names == second.feature_names
    assert first.lane_id == second.lane_id
    assert lane.lane(19)[0] is lane.lane(29)[0], "one schema, one lane, built once"
    assert len(first.features) == len(first.feature_names)
    assert first.receipt["span_start"] == PROMPT
    assert first.receipt["span_end"] == PROMPT + 20
    assert first.extraction_result().block_slices == first.family_slices


def test_the_logit_series_is_absent_unless_asked_for(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    lane = prepared(calib_dir)
    result = harvest_loaded(lane, tokens(PROMPT + 12), prompt_len=PROMPT)
    assert result.logit_series is None
    with pytest.raises(ValueError, match="no logit series"):
        result.series_arrays()


def test_the_logit_series_is_shaped_and_matches_a_direct_forward(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    lane = prepared(calib_dir)
    ids = tokens(PROMPT + 12)
    plain = harvest_loaded(lane, ids, prompt_len=PROMPT)
    result = harvest_loaded(lane, ids, prompt_len=PROMPT, logit_series_top_k=5)
    series = result.logit_series
    assert series is not None
    steps = len(ids) - PROMPT - 1
    assert series.top_k == 5
    assert series.values.shape == (steps, 5) and series.values.dtype == np.float32
    assert series.indices.shape == (steps, 5) and series.indices.dtype == np.int32
    assert series.entropy.shape == (steps,) and series.entropy.dtype == np.float32
    assert series.chosen_ids.tolist() == ids[PROMPT + 1 :]
    assert (np.diff(series.values, axis=1) <= 0).all(), "largest first"

    with torch.no_grad():
        lane.loaded.disable_hooks()
        logits = lane.loaded.model(torch.tensor([ids])).logits[0, PROMPT : len(ids) - 1]
    top = torch.topk(logits.float(), 5, dim=-1)
    np.testing.assert_array_equal(series.indices, top.indices.numpy())
    np.testing.assert_allclose(series.values, top.values.numpy(), rtol=1e-6, atol=1e-6)
    probs = torch.softmax(logits.double(), dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).numpy()
    np.testing.assert_allclose(series.entropy, entropy, rtol=1e-5, atol=1e-6)

    assert result.features.tobytes() == plain.features.tobytes()
    assert result.lane_id == plain.lane_id
    assert not lane.loaded.model._forward_hooks, "the series hook lives only for the call"

    arrays = result.series_arrays()
    assert set(arrays) == {
        "logits_values",
        "logits_indices",
        "logits_entropy",
        "chosen_ids",
        "input_ids",
        "prompt_length",
    }
    assert arrays["input_ids"].tolist() == ids
    assert int(arrays["prompt_length"]) == PROMPT


def test_a_top_k_past_the_vocabulary_keeps_the_vocabulary(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    lane = prepared(calib_dir)
    result = harvest_loaded(lane, tokens(PROMPT + 4), prompt_len=PROMPT, logit_series_top_k=500)
    assert result.logit_series is not None
    assert result.logit_series.top_k == 64
    assert result.logit_series.values.shape == (3, 64)
    with pytest.raises(ValueError, match="must be positive"):
        harvest_loaded(lane, tokens(PROMPT + 4), prompt_len=PROMPT, logit_series_top_k=0)


def test_the_series_is_read_outside_the_files_the_lane_identity_hashes(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    """Why asking for the series cannot move a row's lane id: the reader is not hashed."""
    lane, _ = prepared(calib_dir).lane(10)
    assert set(lane.identity["sources"]) == {
        "features.py",
        "ops.py",
        "attention.py",
        "families.py",
        "batch_layout.py",
    }


@pytest.mark.parametrize(
    ("length", "prompt"),
    [(PROMPT + 5, 0), (PROMPT + 1, PROMPT), (POSITIONS + 2, PROMPT)],
    ids=["no-prompt", "one-generated-token", "past-the-calibration"],
)
def test_a_span_the_lane_cannot_replay_is_refused_before_a_forward(
    lane_arithmetic: None, calib_dir: Path, length: int, prompt: int
) -> None:
    lane = prepared(calib_dir)
    with pytest.raises(ValueError, match="outside what the lane supports"):
        harvest_loaded(lane, tokens(length), prompt_len=prompt)


def test_a_model_the_preset_does_not_describe_is_refused(
    lane_arithmetic: None, calib_dir: Path, no_loading: list[Any]
) -> None:
    with pytest.raises(ValueError, match="num_hidden_layers=3 but preset"):
        prepare_fast_lane(
            tiny_preset(num_layers=4), calib_dir, device="cpu", loaded=tiny_loaded()
        )
    with pytest.raises(ValueError, match="hidden_size=32 but preset"):
        prepare_fast_lane(
            tiny_preset(hidden_dim=64), calib_dir, device="cpu", loaded=tiny_loaded()
        )
    assert no_loading == []


def test_a_model_the_lane_cannot_read_is_refused_by_name() -> None:
    preset = tiny_preset()

    loaded = tiny_loaded()
    loaded.model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        check_loaded_model(loaded, preset, "cpu")

    loaded = tiny_loaded()
    loaded.model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        check_loaded_model(loaded, preset, "cpu")

    loaded = tiny_loaded()
    with pytest.raises(ValueError, match="entirely on cuda:0"):
        check_loaded_model(loaded, preset, "cuda:0")

    loaded = tiny_loaded()
    loaded.remove_hooks()
    with pytest.raises(ValueError, match="carries no capture hook"):
        check_loaded_model(loaded, preset, "cpu")

    loaded = tiny_loaded()
    loaded.model.config.model_type = "mistral"
    with pytest.raises(ValueError, match="dense Llama"):
        check_loaded_model(loaded, preset, "cpu")


def test_preparing_needs_a_model_or_somewhere_to_load_one(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    with pytest.raises(ValueError, match="loaded model or a model_path"):
        prepare_fast_lane(tiny_preset(), calib_dir, device="cpu")
    with pytest.raises(ValueError, match="pass model_path"):
        prepare_fast_lane(
            tiny_preset(),
            calib_dir,
            device="cpu",
            loaded=tiny_loaded(),
            require_local_weights=True,
        )


def test_digests_a_caller_holds_are_taken_as_given(
    lane_arithmetic: None, calib_dir: Path
) -> None:
    held = {"config.json": "c" * 64, "model.safetensors": "d" * 64}
    stamp = prepared(calib_dir, model_files=held).provenance()
    assert stamp["model_files_sha256"] == held
    assert stamp["preset"] == "tiny-llama"
    assert set(stamp["calibration_files"]) == {"positional_means.npz", "pca_model.pkl"}


def test_a_checkpoint_is_digested_once_while_its_files_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []
    real = runtime.file_sha

    def counting(path: Path) -> str:
        reads.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(runtime, "file_sha", counting)
    monkeypatch.setattr(runtime, "_WEIGHT_SHAS", {})
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = weight_file_digests(tmp_path)
    assert weight_file_digests(tmp_path) == first
    assert sorted(reads) == ["config.json", "model.safetensors"]
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    assert weight_file_digests(tmp_path) != first
    assert reads.count("model.safetensors") == 2


def save_tiny_checkpoint(directory: Path) -> LoadedModel:
    """Write the tiny model's weights and config, so the command has bytes to digest."""
    loaded = tiny_loaded()
    loaded.model.save_pretrained(directory)
    return loaded


def test_main_runs_from_argv_on_a_registered_preset_and_banks_what_harvest_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane_arithmetic: None,
    calib_dir: Path,
) -> None:
    """The command end to end, except the loader, which is handed the saved model.

    ``load_model`` places weights through a device map, which needs ``accelerate``,
    and the suite does not install it; the loader's own behaviour is covered where
    a checkpoint is reachable. What this pins is everything around it: the argument
    list, the registry lookup, the calibration and checkpoint digests, the lane and
    the bank it writes.
    """
    checkpoint = tmp_path / "checkpoint"
    loaded = save_tiny_checkpoint(checkpoint)
    loads: list[tuple[str, str, str]] = []

    def load_saved(preset: ModelPreset, model_path: str, device: str) -> LoadedModel:
        loads.append((preset.name, model_path, device))
        return loaded

    monkeypatch.setattr(runtime, "load_lane_model", load_saved)
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"presets": {"tiny-llama": TINY_ROW}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(registry))
    ids = tokens(PROMPT + 16)
    manifest = tmp_path / "replay_manifest.json"
    manifest.write_text(
        json.dumps({"entries": {"4": {"input_ids": ids, "prompt_length": PROMPT}}})
    )
    out = tmp_path / "bank"
    argv = [
        "--model", "tiny-llama",
        "--model-path", str(checkpoint),
        "--calib-dir", str(calib_dir),
        "--manifest", str(manifest),
        "--output", str(out),
        "--device", "cpu",
    ]
    run_gpu_replay.main(argv)
    assert loads == [("tiny-llama", str(checkpoint), "cpu")]

    banked = np.load(out / "gen_004.npz")
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["device"] == "cpu"
    assert set(deployment["model_files_sha256"]) == {"config.json", "model.safetensors"}

    lane = prepare_fast_lane(tiny_preset(), calib_dir, device="cpu", loaded=loaded)
    harvested: HarvestResult = harvest_loaded(lane, ids, prompt_len=PROMPT)
    assert list(banked["feature_names"]) == list(harvested.feature_names)
    np.testing.assert_array_equal(banked["features"], harvested.features)
    assert deployment["lane_id"] == harvested.lane_id


def test_an_unknown_preset_is_refused_by_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = [
        "--model-path", "/model",
        "--calib-dir", "/calibration",
        "--manifest", "/manifest.json",
        "--output", str(tmp_path / "never"),
    ]
    with pytest.raises(SystemExit):
        run_gpu_replay.main(["--model", "no-such-model", *base])
    assert not (tmp_path / "never").exists()

    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"presets": {"tiny-llama": TINY_ROW}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(registry))
    args = run_gpu_replay.parser().parse_args(["--model", "tiny-llama", *base])
    assert args.model == "tiny-llama"
    assert args.device == runtime.DEFAULT_DEVICE
