"""One pass's configuration: derived from a preset, never assembled by hand."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anamnesis.config import paths
from anamnesis.config.experiment import (
    PCA_MODEL_NAME,
    PCA_MODEL_NAMES,
    POSITIONAL_MEANS_NAME,
    CalibrationConfig,
    ExperimentConfig,
    ExtractionConfig,
    FeaturePipelineConfig,
    GenerationConfig,
    GenerationSpec,
)
from anamnesis.config.models import EAGER_ATTENTION, UnknownPresetError, resolve_preset
from anamnesis.extraction import calibration


@pytest.fixture
def data_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point both mutable data roots at a temporary tree."""
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "outputs"))
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    monkeypatch.delenv(paths.RUN_NAME_ENV, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("name", "temperature", "eos"),
    [("3b", 0.7, [128001, 128009]), ("8b", 0.6, [128001, 128008, 128009])],
)
def test_generation_policy_comes_from_the_preset(
    name: str, temperature: float, eos: list[int]
) -> None:
    config = GenerationConfig.from_preset(name)
    assert config.temperature == temperature
    assert config.eos_token_ids == eos
    assert config.top_p == 0.9
    assert config.max_new_tokens == 512
    assert config.do_sample
    assert config.output_hidden_states
    assert config.output_attentions
    assert config.output_logits
    assert config.return_dict_in_generate


def test_generation_policy_takes_a_shorter_budget() -> None:
    config = GenerationConfig.from_preset("8b", max_new_tokens=100)
    assert config.max_new_tokens == 100
    assert config.temperature == 0.6


def test_generation_policy_needs_a_model_named() -> None:
    with pytest.raises(ValidationError):
        GenerationConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("name", "sampled", "pca", "early", "late"),
    [
        ("3b", [0, 7, 14, 18, 21, 24, 27], [7, 14, 18, 21, 24], 7, 21),
        ("8b", [0, 8, 16, 20, 24, 28, 31], [8, 16, 20, 24, 28], 8, 24),
    ],
)
def test_extraction_layer_plan_comes_from_the_preset(
    name: str, sampled: list[int], pca: list[int], early: int, late: int
) -> None:
    config = ExtractionConfig.from_preset(name)
    assert config.sampled_layers == sampled
    assert config.pca_layers == pca
    assert config.early_layer_cutoff == early
    assert config.late_layer_cutoff == late
    assert config.enable_norms_and_output_stats and config.enable_attention_and_deltas and config.enable_cache_and_keys
    assert config.enable_residual_pca and config.enable_knnlm_baseline
    assert not config.save_raw_tensors


def test_extraction_config_stays_mutable_for_a_single_switch() -> None:
    config = ExtractionConfig.from_preset("8b")
    config.enable_residual_pca = False
    assert not config.enable_residual_pca


def test_extraction_layer_plan_is_checked_against_the_model_depth() -> None:
    with pytest.raises(ValueError) as caught:
        ExtractionConfig.from_preset("3b", sampled_layers=[0, 14, 31])
    message = str(caught.value)
    assert "31" in message
    assert "0..27" in message


def test_depth_check_is_available_on_its_own() -> None:
    config = ExtractionConfig.from_preset("8b")
    config.check_depth("8b")
    with pytest.raises(ValueError):
        config.check_depth("3b")


def test_extraction_bands_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        ExtractionConfig.from_preset("8b", early_layer_cutoff=30, late_layer_cutoff=2)


@pytest.mark.parametrize(
    ("name", "layers"),
    [("3b", [7, 14, 18, 21, 24]), ("8b", [8, 16, 20, 24, 28])],
)
def test_family_layers_come_from_the_preset(name: str, layers: list[int]) -> None:
    config = FeaturePipelineConfig.from_preset(name)
    assert config.trajectory_layers == layers
    assert config.contrastive_layers == layers
    assert config.include_core_blocks
    assert not config.enable_attention_flow
    assert not config.enable_path_signature


def test_family_config_needs_a_model_named() -> None:
    with pytest.raises(ValidationError):
        FeaturePipelineConfig()  # type: ignore[call-arg]


def test_path_signature_needs_a_site() -> None:
    with pytest.raises(ValidationError) as caught:
        FeaturePipelineConfig.from_preset("8b", enable_path_signature=True)
    assert "path_signature_layers" in str(caught.value)


def test_path_signature_needs_a_supplied_basis(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        FeaturePipelineConfig.from_preset(
            "8b", enable_path_signature=True, path_signature_layers=[16]
        )
    assert "path_signature_basis_path" in str(caught.value)
    config = FeaturePipelineConfig.from_preset(
        "8b",
        enable_path_signature=True,
        path_signature_layers=[16],
        path_signature_basis_path=tmp_path / "pca.pkl",
    )
    assert config.path_signature_level == 2
    assert config.path_signature_basis_label == "pcaA"


def test_path_signature_sites_are_unique() -> None:
    with pytest.raises(ValidationError):
        FeaturePipelineConfig.from_preset("8b", path_signature_layers=[16, 16])


def test_calibration_artifacts_sit_in_the_presets_directory(data_roots: Path) -> None:
    config = CalibrationConfig.from_preset("8b")
    directory = resolve_preset("8b").resolved_calibration_dir()
    assert config.positional_means_path == directory / POSITIONAL_MEANS_NAME
    assert config.pca_model_path == directory / PCA_MODEL_NAME
    assert config.artifact_paths() == (config.positional_means_path, config.pca_model_path)


def test_the_artifact_names_are_the_ones_the_reader_resolves() -> None:
    """One home for the two filenames, read by the path builders and by the reader.

    Configuration is the base layer of this package and imports nothing else in it,
    so the names sit here and :mod:`anamnesis.extraction.calibration` reads them —
    not the other way round, which would put numpy behind every import of a run's
    description.
    """
    assert (POSITIONAL_MEANS_NAME, PCA_MODEL_NAME) == (
        calibration.POSITIONAL_MEANS_NAME,
        calibration.PCA_MODEL_NAME,
    )
    assert PCA_MODEL_NAMES is calibration.PCA_MODEL_NAMES
    assert PCA_MODEL_NAMES[0] == PCA_MODEL_NAME


def test_configuration_imports_nothing_else_in_the_package() -> None:
    """The claim above, checked rather than stated.

    The filenames can live here only while this package is the layer everything else
    reads and nothing here reads back: one import the other way and the two modules are
    a cycle, and the home for a name that both a path builder and a reader need moves to
    whichever of them imports the other. Function-level imports count, which is why this
    reads the syntax tree rather than the module's namespace.
    """
    package = Path(paths.__file__).parent
    offenders: list[str] = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                named = [node.module or ""]
            else:
                continue
            offenders += [
                f"{module.name}:{node.lineno} imports {name}"
                for name in named
                if name.startswith("anamnesis") and not name.startswith("anamnesis.config")
            ]
    assert offenders == [], f"configuration is no longer the dependency leaf: {offenders}"


def test_legacy_calibration_is_reached_through_the_hatch(data_roots: Path) -> None:
    config = CalibrationConfig.from_preset("3b")
    assert config.positional_means_path == (
        data_roots / "phase_0" / "outputs" / "calibration" / "positional_means.npz"
    )


def test_calibration_can_be_pointed_at_an_explicit_directory(tmp_path: Path) -> None:
    config = CalibrationConfig.in_directory(tmp_path)
    assert config.pca_model_path == tmp_path / PCA_MODEL_NAME
    assert config.positional_means_path == tmp_path / POSITIONAL_MEANS_NAME


def test_experiment_config_assembles_one_agreeing_whole(data_roots: Path) -> None:
    config = ExperimentConfig.from_preset("3b", run_name="3b_fat_01")
    assert config.preset_name == "3b"
    assert config.model.num_layers == 28
    assert config.model.attn_implementation == EAGER_ATTENTION
    assert config.generation.temperature == 0.7
    assert config.generation.eos_token_ids == [128001, 128009]
    assert config.extraction.sampled_layers == [0, 7, 14, 18, 21, 24, 27]
    assert config.calibration.positional_means_path.parent == (
        data_roots / "phase_0" / "outputs" / "calibration"
    )
    run_dir = data_roots / "outputs" / "runs" / "3b_fat_01"
    assert config.outputs_dir == run_dir
    assert config.signatures_dir == run_dir / "signatures"
    assert config.figures_dir == run_dir / "figures"
    assert config.metadata_path == run_dir / "metadata.json"
    assert config.results_path == run_dir / "results.json"
    assert config.prompts_path == paths.prompts_path()


def test_experiment_config_defaults_to_the_run_name_in_the_environment(
    data_roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.RUN_NAME_ENV, "run_from_env")
    config = ExperimentConfig.from_preset("8b")
    assert config.outputs_dir == data_roots / "outputs" / "runs" / "run_from_env"


def test_experiment_config_takes_an_explicit_directory(tmp_path: Path) -> None:
    config = ExperimentConfig.from_preset("8b", outputs_dir=tmp_path / "elsewhere")
    assert config.signatures_dir == tmp_path / "elsewhere" / "signatures"


def test_experiment_overrides_reach_each_section(data_roots: Path, tmp_path: Path) -> None:
    config = ExperimentConfig.from_preset(
        "8b",
        model_overrides={"device_map": "cuda:0"},
        generation_overrides={"max_new_tokens": 100},
        extraction_overrides={"save_raw_tensors": True},
        calibration_overrides={"pca_model_path": tmp_path / "basis.pkl"},
        prompts_path=tmp_path / "prompts.json",
    )
    assert config.model.device_map == "cuda:0"
    assert config.generation.max_new_tokens == 100
    assert config.extraction.save_raw_tensors
    assert config.calibration.pca_model_path == tmp_path / "basis.pkl"
    assert config.prompts_path == tmp_path / "prompts.json"


def test_experiment_config_names_an_unknown_preset(data_roots: Path) -> None:
    with pytest.raises(UnknownPresetError):
        ExperimentConfig.from_preset("llama-4")


def test_ensure_dirs_creates_the_run_and_the_calibration_directories(
    data_roots: Path, tmp_path: Path
) -> None:
    config = ExperimentConfig.from_preset("8b", run_name="fresh")
    config.ensure_dirs()
    assert config.outputs_dir.is_dir()
    assert config.signatures_dir.is_dir()
    assert config.figures_dir.is_dir()
    assert config.calibration.positional_means_path.parent.is_dir()
    config.ensure_dirs()


def test_experiment_config_round_trips_through_json(data_roots: Path) -> None:
    config = ExperimentConfig.from_preset("8b", run_name="round_trip")
    payload = json.loads(config.model_dump_json())
    assert ExperimentConfig.model_validate(payload) == config


def test_generation_spec_describes_one_generation() -> None:
    spec = GenerationSpec(
        generation_id=0,
        prompt_set="set_a",
        topic="tides",
        topic_idx=0,
        mode="socratic→linear",
        mode_idx=2,
        system_prompt="system",
        user_prompt="user",
        seed=1234,
    )
    assert spec.repetition == 0
    assert spec.mode == "socratic→linear"
    with pytest.raises(ValidationError):
        GenerationSpec(
            generation_id=-1,
            prompt_set="set_a",
            topic="tides",
            topic_idx=0,
            mode="linear",
            mode_idx=0,
            system_prompt="system",
            user_prompt="user",
            seed=1,
        )
