"""The preset registry, and the bridge from a preset name to a loader config.

The table in :data:`FACTS_OF_RECORD` is the architecture and decode policy of the
two characterised checkpoints, written out independently of the registry. A test
that read the numbers from the registry it is checking would pass on a typo, so
these are stated here and compared against what ``from_preset`` produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from anamnesis.config import paths
from anamnesis.config.models import (
    ATTENTION_WITHOUT_WEIGHTS,
    EAGER_ATTENTION,
    MODEL_PRESETS,
    PRESET_ALIASES,
    ModelConfig,
    ModelPreset,
    UnknownPresetError,
    preset_names,
    resolve_preset,
)

FACTS_OF_RECORD: dict[str, dict[str, Any]] = {
    "3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3072,
        "num_attention_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "kv_group_size": 3,
        "sampled_layers": (0, 7, 14, 18, 21, 24, 27),
        "pca_layers": (7, 14, 18, 21, 24),
        "trajectory_layers": (7, 14, 18, 21, 24),
        "contrastive_layers": (7, 14, 18, 21, 24),
        "early_layer_cutoff": 7,
        "late_layer_cutoff": 21,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "eos_token_ids": (128001, 128009),
        "torch_dtype": "float16",
        "calibration_root": "legacy",
    },
    "8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "kv_group_size": 4,
        "sampled_layers": (0, 8, 16, 20, 24, 28, 31),
        "pca_layers": (8, 16, 20, 24, 28),
        "trajectory_layers": (8, 16, 20, 24, 28),
        "contrastive_layers": (8, 16, 20, 24, 28),
        "early_layer_cutoff": 8,
        "late_layer_cutoff": 24,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "eos_token_ids": (128001, 128008, 128009),
        "torch_dtype": "bfloat16",
        "calibration_root": "outputs",
    },
}


@pytest.mark.parametrize("name", sorted(FACTS_OF_RECORD))
def test_preset_carries_the_facts_of_record(name: str) -> None:
    preset = resolve_preset(name)
    for field, expected in FACTS_OF_RECORD[name].items():
        assert getattr(preset, field) == expected, field


@pytest.mark.parametrize("name", sorted(FACTS_OF_RECORD))
def test_from_preset_carries_every_field_a_caller_would_copy(name: str) -> None:
    facts = FACTS_OF_RECORD[name]
    config = ModelConfig.from_preset(name)
    assert config.model_id == facts["model_id"]
    assert config.torch_dtype == facts["torch_dtype"]
    assert config.num_layers == facts["num_layers"]
    assert config.hidden_dim == facts["hidden_dim"]
    assert config.num_attention_heads == facts["num_attention_heads"]
    assert config.num_kv_heads == facts["num_kv_heads"]
    assert config.head_dim == facts["head_dim"]
    assert config.kv_group_size == facts["kv_group_size"]
    assert config.attn_implementation == EAGER_ATTENTION
    assert config.device_map == "auto"
    assert config.preset_name == name


def test_from_preset_accepts_a_preset_object() -> None:
    preset = MODEL_PRESETS["8b"]
    assert ModelConfig.from_preset(preset) == ModelConfig.from_preset("8b")


def test_from_preset_takes_overrides_for_a_local_checkpoint(tmp_path: Path) -> None:
    config = ModelConfig.from_preset("3b", model_id=str(tmp_path), device_map="cuda:0")
    assert config.model_id == str(tmp_path)
    assert config.device_map == "cuda:0"
    assert config.num_layers == FACTS_OF_RECORD["3b"]["num_layers"]


def test_from_preset_rejects_an_override_that_names_no_field() -> None:
    with pytest.raises(ValidationError) as caught:
        ModelConfig.from_preset("3b", sampled_layers=[0, 1])
    assert "sampled_layers" in str(caught.value)


def test_unknown_preset_names_every_key_and_alias() -> None:
    with pytest.raises(UnknownPresetError) as caught:
        resolve_preset("llama-4")
    message = str(caught.value)
    assert "llama-4" in message
    for key in MODEL_PRESETS:
        assert key in message
    for alias in PRESET_ALIASES:
        assert alias in message


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("8b", "8b"),
        ("llama31_8b", "8b"),
        ("llama32_3b", "3b"),
        ("DSV2_Lite", "dsv2-lite"),
        ("qwen2.5-7b", "qwen-7b"),
        ("  gemma3-27b  ", "gemma3-27b"),
    ],
)
def test_names_and_aliases_reach_one_row(spelling: str, expected: str) -> None:
    assert resolve_preset(spelling).name == expected


def test_every_alias_points_at_a_registered_preset() -> None:
    for alias, key in PRESET_ALIASES.items():
        assert key in MODEL_PRESETS, alias


def test_registry_keys_match_the_names_inside_them() -> None:
    assert preset_names() == tuple(MODEL_PRESETS)
    for key, preset in MODEL_PRESETS.items():
        assert preset.name == key


@pytest.mark.parametrize("name", sorted(MODEL_PRESETS))
def test_every_preset_is_internally_coherent(name: str) -> None:
    preset = MODEL_PRESETS[name]
    assert preset.num_attention_heads % preset.num_kv_heads == 0
    assert preset.kv_group_size >= 1
    assert preset.is_grouped_query == (preset.kv_group_size > 1)
    for layers in (
        preset.sampled_layers,
        preset.pca_layers,
        preset.trajectory_layers,
        preset.contrastive_layers,
    ):
        assert layers
        assert list(layers) == sorted(set(layers))
        assert max(layers) < preset.num_layers
    assert preset.early_layer_cutoff <= preset.late_layer_cutoff
    assert preset.eos_token_ids


def test_presets_are_immutable() -> None:
    with pytest.raises(ValidationError):
        MODEL_PRESETS["8b"].temperature = 1.0


def test_calibration_directory_resolves_under_its_declared_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.OUTPUTS_ENV, str(tmp_path / "outputs"))
    monkeypatch.setenv(paths.LEGACY_DATA_ENV, str(tmp_path / "phase_0"))
    assert resolve_preset("8b").resolved_calibration_dir() == (
        tmp_path / "outputs" / "calibration" / "llama31_8b"
    )
    assert resolve_preset("3b").resolved_calibration_dir() == (
        tmp_path / "phase_0" / "outputs" / "calibration"
    )


def test_interleaved_attention_is_declared_per_sampled_layer() -> None:
    gemma = resolve_preset("gemma3-27b")
    assert gemma.attention_layer_types is not None
    assert gemma.attention_kind(0) == "local"
    assert gemma.global_layers() == tuple(
        layer for layer in gemma.sampled_layers if layer != 0
    )


def test_full_context_models_report_every_layer_global() -> None:
    llama = resolve_preset("8b")
    assert llama.attention_layer_types is None
    assert llama.global_layers() == llama.sampled_layers
    assert llama.attention_kind(16) == "global"


@pytest.mark.parametrize("implementation", sorted(ATTENTION_WITHOUT_WEIGHTS))
def test_attention_kernels_without_weights_are_refused(implementation: str) -> None:
    with pytest.raises(ValidationError) as caught:
        ModelConfig.from_preset("8b", attn_implementation=implementation)
    message = str(caught.value)
    assert implementation in message
    assert EAGER_ATTENTION in message


def test_grouped_query_split_must_be_exact() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.from_preset("8b", num_kv_heads=7)


def _preset_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "probe",
        "model_id": "vendor/probe",
        "torch_dtype": "bfloat16",
        "num_layers": 8,
        "hidden_dim": 256,
        "num_attention_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 32,
        "sampled_layers": (0, 4, 7),
        "pca_layers": (4,),
        "trajectory_layers": (4,),
        "contrastive_layers": (4,),
        "early_layer_cutoff": 2,
        "late_layer_cutoff": 6,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "eos_token_ids": (1,),
        "calibration_root": "outputs",
        "calibration_dir": "calibration/probe",
    }
    base.update(overrides)
    return base


def test_a_preset_accepts_a_coherent_new_model() -> None:
    preset = ModelPreset(**_preset_kwargs())
    assert preset.kv_group_size == 2


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"sampled_layers": (0, 4, 99)}, "outside"),
        ({"sampled_layers": (4, 0)}, "strictly increasing"),
        ({"sampled_layers": (4, 4)}, "strictly increasing"),
        ({"sampled_layers": ()}, "at least one layer"),
        ({"eos_token_ids": ()}, "token budget"),
        ({"eos_token_ids": (1, 1)}, "repeats"),
        ({"num_kv_heads": 3}, "group size"),
        ({"early_layer_cutoff": 7, "late_layer_cutoff": 3}, "above"),
        ({"late_layer_cutoff": 99}, "outside"),
        ({"attention_layer_types": {0: "local"}}, "omits sampled layers"),
        ({"temperature": 0.0}, "greater than 0"),
        ({"top_p": 1.5}, "less than or equal to 1"),
    ],
)
def test_an_incoherent_preset_is_refused(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError) as caught:
        ModelPreset(**_preset_kwargs(**overrides))
    assert fragment in str(caught.value)
