"""What each command accepts, and what it refuses before a model loads.

A shim's job is to turn a command line into one call into the package, and the part
of that worth testing is where it says no. Every refusal below is a case where
running on would produce something that looks like a result: a roster handed to a
command that cannot fan it out, a recompute whose family set silently widened, a
sampler setting outside the range the arithmetic assumes.

None of these load a model. They exercise the argument surface, the configuration
each command derives from a preset row, and the dispatch decisions made before any
device is touched — which is most of what a shim is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.config import resolve_preset
from anamnesis.scripts import (
    run_calibration,
    run_extraction,
    run_gen_tokens,
    run_persistent_replay,
    run_recompute,
    run_replay,
)

MODEL = "8b"


def test_replay_needs_a_cell_or_a_roster() -> None:
    """A replay with neither a cell nor a roster has nothing to replay.

    The refusal comes before the model loads, which is what makes it worth having:
    the alternative is a checkpoint on a device and then a question.
    """
    args = run_replay.parser().parse_args(
        ["--model", MODEL, "--model-path", "/models/x", "--calib-dir", "/calib"]
    )
    assert args.run_dir is None and args.manifest is None and args.gpus is None


def test_replay_refuses_a_roster_without_devices() -> None:
    """A roster is a thing to fan out; in one process it is a jobs file."""
    with pytest.raises(SystemExit, match="roster to fan out"):
        run_replay.main([
            "--model", MODEL, "--model-path", "/models/x",
            "--calib-dir", "/calib", "--cells-json", "/cells.json",
        ])


def test_replay_refuses_a_jobs_file_alongside_a_cell() -> None:
    """A jobs file is the whole roster, so a cell beside it is two answers."""
    with pytest.raises(SystemExit, match="whole roster"):
        run_replay.main([
            "--model", MODEL, "--model-path", "/models/x", "--calib-dir", "/calib",
            "--jobs-file", "/jobs.json", "--run-dir", "/runs/a",
        ])


def test_generation_refuses_a_roster_without_devices() -> None:
    with pytest.raises(SystemExit, match="roster to fan out"):
        run_gen_tokens.main([
            "--model", MODEL, "--model-path", "/models/x", "--cells-json", "/cells.json",
        ])


def test_generation_refuses_a_non_positive_repetition_penalty() -> None:
    """The penalty divides logits; at or below zero the arithmetic is not defined."""
    with pytest.raises(SystemExit, match="above zero"):
        run_gen_tokens.main([
            "--model", MODEL, "--model-path", "/models/x",
            "--spec-file", "/specs.json", "--out-dir", "/out",
            "--repetition-penalty", "0",
        ])


def test_generation_policy_comes_from_the_preset_row() -> None:
    """Decode settings are a model's, not a constant restated in a launcher."""
    args = run_gen_tokens.parser().parse_args(
        ["--model", MODEL, "--model-path", "/models/x"]
    )
    policy = run_gen_tokens.decode_policy(args)
    preset = resolve_preset(MODEL)
    assert policy.temperature == preset.temperature
    assert tuple(policy.eos_token_ids) == tuple(preset.eos_token_ids)


def test_a_neutral_repetition_penalty_is_withheld_from_the_sampler() -> None:
    """At exactly one the argument is not passed, so the default path is the banked one."""
    args = run_gen_tokens.parser().parse_args(
        ["--model", MODEL, "--model-path", "/models/x"]
    )
    assert run_gen_tokens.decode_policy(args).sampler_extras() == {}
    raised = run_gen_tokens.parser().parse_args(
        ["--model", MODEL, "--model-path", "/models/x", "--repetition-penalty", "1.2"]
    )
    assert run_gen_tokens.decode_policy(raised).sampler_extras() == {
        "repetition_penalty": 1.2
    }


def test_generation_metadata_records_the_policy_it_ran_under() -> None:
    args = run_gen_tokens.parser().parse_args(
        ["--model", MODEL, "--model-path", "/models/x", "--date-string", "12 Jul 2026"]
    )
    block = run_gen_tokens.passthrough(args)
    preset = resolve_preset(MODEL)
    assert block["model"]["model_id"] == preset.model_id
    assert block["generation_config"]["eos_token_ids"] == list(preset.eos_token_ids)
    assert block["template_date_string"] == "12 Jul 2026"
    assert "a5_injection" not in block, "an unsteered run records no intervention"


def test_generation_metadata_records_an_intervention_when_there_is_one() -> None:
    args = run_gen_tokens.parser().parse_args([
        "--model", MODEL, "--model-path", "/models/x",
        "--inject-npz", "/bank.npz", "--inject-key", "V3",
        "--inject-layer", "16", "--inject-alpha", "2.5",
    ])
    block = run_gen_tokens.passthrough(args)
    assert block["a5_injection"]["inject_key"] == "V3"
    assert block["a5_injection"]["inject_layer"] == 16


def test_recompute_family_set_is_the_one_the_banked_vectors_carry() -> None:
    """The replay surface grew families the banked vectors do not have.

    A recompute that inherited the newer set would widen its vector and stop being a
    recompute, so the set is stated here and pinned by this case.
    """
    extraction, families = run_recompute.configs(MODEL)
    preset = resolve_preset(MODEL)
    assert extraction.sampled_layers == list(preset.sampled_layers)
    assert extraction.enable_residual_pca
    on = {
        families.include_core_blocks,
        families.enable_residual_trajectory,
        families.enable_attention_flow,
        families.enable_gate_features,
        families.enable_per_head,
        families.enable_stft,
    }
    assert on == {True}
    off = {
        families.enable_temporal_dynamics,
        families.enable_contrastive_projection,
        families.enable_value_geometry,
        families.enable_qk_geometry,
        families.enable_kv_cka,
        families.enable_expert_routing,
    }
    assert off == {False}
    assert families.trajectory_layers == list(preset.trajectory_layers)


@pytest.mark.parametrize("model", ["3b", "8b"])
def test_recompute_configs_match_the_records_own_construction(model: str) -> None:
    """The shim must drive the arithmetic the banked receipts were produced by.

    The construction below is the frozen record's, transcribed field for field. The
    shim derives the same thing from a preset row instead of restating it, and this
    case is what makes "derived, not restated" a claim about the values rather than
    about the style: a preset row that drifted, or a `from_preset` that filled a
    field differently, fails here before it reaches a recompute.
    """
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, resolve_preset

    preset = resolve_preset(model)
    record_extraction = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_residual_pca=True,
    )
    record_families = FeaturePipelineConfig(
        include_core_blocks=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=True,
        enable_stft=True,
        enable_contrastive_projection=False,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    extraction, families = run_recompute.configs(model)
    assert extraction.model_dump() == record_extraction.model_dump()
    assert families.model_dump() == record_families.model_dump()


def test_extraction_mode_sets_are_the_modes_package_texts() -> None:
    """A mode label in banked data means the exact prompt text in the package."""
    from anamnesis.modes.extended_modes import EXTENDED_MODES
    from anamnesis.modes.run4_modes import RUN4_MODES

    assert run_extraction.mode_prompts("run4") == dict(RUN4_MODES)
    assert run_extraction.mode_prompts("mixed") == dict(EXTENDED_MODES)
    with pytest.raises(ValueError, match="unknown mode set"):
        run_extraction.mode_prompts("run3")


def test_extraction_builds_the_asked_for_count_per_mode() -> None:
    args = run_extraction.parser().parse_args(
        ["--model", MODEL, "--run-name", "unit", "--n-samples", "4"]
    )
    config, n_samples = run_extraction.build_config(args)
    specs = run_extraction.build_specs(args, config, n_samples)
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.mode] = counts.get(spec.mode, 0) + 1
    assert set(counts.values()) == {4}
    assert set(counts) == set(run_extraction.mode_prompts("run4"))


def test_extraction_adds_the_swap_condition_on_request() -> None:
    args = run_extraction.parser().parse_args([
        "--model", MODEL, "--run-name", "unit", "--n-samples", "2", "--include-prompt-swap",
    ])
    config, n_samples = run_extraction.build_config(args)
    specs = run_extraction.build_specs(args, config, n_samples)
    swaps = [spec for spec in specs if spec.mode.startswith("swap_")]
    assert swaps, "the confound condition is what --include-prompt-swap is for"
    assert all(spec.generation_id >= 10000 for spec in swaps), (
        "swap ids sit above a pass's own so the two cannot collide"
    )


def test_extraction_smoke_shortens_the_pass_without_changing_the_model() -> None:
    args = run_extraction.parser().parse_args(
        ["--model", MODEL, "--run-name", "unit", "--smoke-test"]
    )
    config, n_samples = run_extraction.build_config(args)
    preset = resolve_preset(MODEL)
    assert n_samples == 1
    assert config.generation.max_new_tokens == run_extraction.SMOKE_MAX_NEW_TOKENS
    assert config.generation.temperature == preset.temperature
    assert config.model.num_layers == preset.num_layers


def test_calibration_writes_beside_the_presets_directory() -> None:
    args = run_calibration.parser().parse_args(["--model", MODEL])
    preset, means_path, pca_path = run_calibration.resolve_paths(args)
    assert means_path.name == "positional_means.npz"
    assert pca_path.name == "pca_model_corrected.pkl", (
        "the corrected per-layer fit is the default, and does not overwrite a pooled basis"
    )
    assert means_path.parent == pca_path.parent
    assert preset.name == resolve_preset(MODEL).name


def test_calibration_pooled_writes_the_legacy_basis_name() -> None:
    args = run_calibration.parser().parse_args(["--model", MODEL, "--pooled"])
    _, _, pca_path = run_calibration.resolve_paths(args)
    assert pca_path.name == "pca_model.pkl"


def test_persistent_replay_drive_needs_a_roster() -> None:
    with pytest.raises(SystemExit, match="--cells-json"):
        run_persistent_replay.main([
            "--model", MODEL, "--model-path", "/models/x", "--calib-dir", "/calib",
            "--work-dir", "/work", "--drive",
        ])


def test_persistent_replay_parity_needs_a_cell_to_compare_over() -> None:
    with pytest.raises(SystemExit, match="--parity-cell"):
        run_persistent_replay.main([
            "--model", MODEL, "--model-path", "/models/x", "--calib-dir", "/calib",
            "--work-dir", "/work", "--parity",
        ])


def _bank_a_cell(tmp_path: Path) -> Path:
    """A run directory with the two artifacts a replay reads, and nothing else."""
    run_dir = tmp_path / "cell"
    rec_dir = run_dir / "gen_records"
    rec_dir.mkdir(parents=True)
    for gen_id in range(5):
        (rec_dir / f"gen_{gen_id:03d}.json").write_text(json.dumps({
            "generation_id": gen_id,
            "prompt_set": "UNIT",
            "topic": "tides",
            "topic_idx": gen_id,
            "mode": "linear",
            "mode_idx": 0,
            "system_prompt": "",
            "user_prompt": "Write about: tides",
            "seed": 1000 + gen_id,
            "repetition": 0,
            "condition": "standard",
            "generated_text": "text",
            "num_generated_tokens": 3,
            "prompt_length": 4,
            "input_ids": list(range(7)),
        }))
    return run_dir


def test_assembly_and_a_replay_partition_meet_on_the_same_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gateway's two halves, joined on disk with no device involved.

    Generation assembles a run; replay reads that run's manifest and partitions it.
    This is the seam the whole split rests on, and the cheapest place for it to be
    wrong is here — a manifest a replay cannot enumerate turns into a pass that
    quietly does nothing.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run_dir = _bank_a_cell(tmp_path)
    run_gen_tokens.main([
        "--model", MODEL, "--model-path", "/models/x", "--assemble", str(run_dir),
    ])
    assert (run_dir / "metadata.json").exists()

    run_replay.main([
        "--model", MODEL, "--model-path", "/models/x", "--calib-dir", str(tmp_path / "calib"),
        "--run-dir", str(run_dir), "--manifest", str(run_dir / "replay_manifest.json"),
        "--gpus", "0,1", "--workers-per-gpu", "1", "--no-resume", "--dry-run",
    ])
    printed = capsys.readouterr().out
    assert "worker 0 (0): 3 generations" in printed
    assert "worker 1 (1): 2 generations" in printed


def test_a_replay_fan_out_over_a_finished_cell_does_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume is the default, so re-running a finished cell spawns no workers.

    A finished generation is both files a signature is written as, the vector and
    its metadata: `anamnesis.extraction.replay.cell.signature_on_disk` is the one
    predicate the fan-out and a worker's resume filter share, and a fixture that
    laid down only the metadata would be asserting over half a signature.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run_dir = _bank_a_cell(tmp_path)
    run_gen_tokens.main([
        "--model", MODEL, "--model-path", "/models/x", "--assemble", str(run_dir),
    ])
    sig_dir = run_dir / "signatures_v3"
    sig_dir.mkdir()
    for gen_id in range(5):
        (sig_dir / f"gen_{gen_id:03d}.json").write_text("{}")
        (sig_dir / f"gen_{gen_id:03d}.npz").write_bytes(b"")

    run_replay.main([
        "--model", MODEL, "--model-path", "/models/x", "--calib-dir", str(tmp_path / "calib"),
        "--run-dir", str(run_dir), "--manifest", str(run_dir / "replay_manifest.json"),
        "--gpus", "0", "--dry-run",
    ])
    assert "already have signatures" in capsys.readouterr().out
