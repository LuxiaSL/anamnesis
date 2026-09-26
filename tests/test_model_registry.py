"""The model registry file: pinned bytes, additive extension, and the refusals.

The shipped rows are what every banked corpus was produced under, so their bytes
are part of the record the way the prompt sets' are: :data:`MODELS_SHA256` fails on
an edit rather than letting one pass as a shape change.

The other half of this file is the extension path, and it is tested the way a
stranger meets it: a registry file outside the package, named in the environment,
holding a model this package never heard of. A registry nobody has added a row to
is not a registry, so the test adds one — and then checks the refusals that keep an
added row from redefining a shipped one, because merging quietly is how a stored
label comes to mean something other than what produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from anamnesis.config.models import (
    MODELS_ENV,
    MODELS_FILE,
    ModelRegistryError,
    UnknownPresetError,
    layer_counts_by_run_prefix,
    load_registry,
    preset_names,
    registry_paths,
    resolve_preset,
)

MODELS_SHA256 = "a808adb8e06cd8c16168f1a910ab6818e0d215a7d00e40192d1514ada3d93a38"
"""The shipped registry's bytes; an edit to a row fails here."""

SHIPPED_PRESETS = ("8b", "70b", "3b", "olmo2-7b", "gemma3-27b", "qwen-7b", "dsv2-lite")

ADDED_ROW: dict[str, Any] = {
    "name": "tiny-probe",
    "model_id": "vendor/tiny-probe",
    "torch_dtype": "bfloat16",
    "num_layers": 12,
    "hidden_dim": 512,
    "num_attention_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 64,
    "sampled_layers": [0, 3, 6, 9, 11],
    "pca_layers": [3, 6, 9],
    "trajectory_layers": [3, 6, 9],
    "contrastive_layers": [3, 6, 9],
    "early_layer_cutoff": 3,
    "late_layer_cutoff": 9,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "eos_token_ids": [2],
    "calibration_root": "outputs",
    "calibration_dir": "calibration/tiny_probe",
}


def write_registry(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def added(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A user's own registry file, named in the environment."""
    path = write_registry(
        tmp_path / "my_models.json",
        {"presets": {"tiny-probe": ADDED_ROW}, "aliases": {"tiny": "tiny-probe"}},
    )
    monkeypatch.setenv(MODELS_ENV, str(path))
    return path


def test_the_shipped_registry_matches_its_pinned_digest() -> None:
    assert hashlib.sha256(MODELS_FILE.read_bytes()).hexdigest() == MODELS_SHA256


def test_the_shipped_registry_holds_the_rows_of_record() -> None:
    assert preset_names() == SHIPPED_PRESETS


def test_without_an_override_the_shipped_file_is_the_whole_registry() -> None:
    assert registry_paths() == (MODELS_FILE,)


def test_a_user_adds_a_model_by_adding_data(added: Path) -> None:
    """The extension story, end to end: a row in a file, and the model resolves."""
    assert registry_paths() == (MODELS_FILE, added)
    row = resolve_preset("tiny-probe")
    assert row.model_id == "vendor/tiny-probe"
    assert row.num_layers == 12
    assert row.kv_group_size == 4
    assert "tiny-probe" in preset_names()
    assert resolve_preset("tiny").name == "tiny-probe"
    assert layer_counts_by_run_prefix()["tiny-probe"] == 12


def test_an_added_model_is_configurable_like_a_shipped_one(added: Path) -> None:
    from anamnesis.config.models import ModelConfig

    config = ModelConfig.from_preset("tiny-probe")
    assert config.preset_name == "tiny-probe"
    assert config.num_layers == 12


def test_several_override_files_are_read_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    first = write_registry(tmp_path / "a.json", {"presets": {"tiny-probe": ADDED_ROW}})
    second = write_registry(tmp_path / "b.json", {"run_depths": {"other_model": 40}})
    monkeypatch.setenv(MODELS_ENV, os.pathsep.join([str(first), str(second)]))
    assert registry_paths() == (MODELS_FILE, first, second)
    assert layer_counts_by_run_prefix()["other_model"] == 40


def test_an_added_row_cannot_redefine_a_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clash = write_registry(
        tmp_path / "clash.json", {"presets": {"8b": {**ADDED_ROW, "name": "8b"}}}
    )
    monkeypatch.setenv(MODELS_ENV, str(clash))
    with pytest.raises(ModelRegistryError) as caught:
        load_registry()
    message = str(caught.value)
    assert "'8b'" in message
    assert str(MODELS_FILE) in message
    assert str(clash) in message


def test_an_added_depth_cannot_restate_a_shipped_models_layer_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clash = write_registry(tmp_path / "depth.json", {"run_depths": {"8b": 32}})
    monkeypatch.setenv(MODELS_ENV, str(clash))
    with pytest.raises(ModelRegistryError, match="has one home"):
        load_registry()


def test_two_models_cannot_claim_one_run_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clash = write_registry(
        tmp_path / "prefix.json",
        {"presets": {"tiny-probe": {**ADDED_ROW, "run_prefixes": ["dsv2"]}}},
    )
    monkeypatch.setenv(MODELS_ENV, str(clash))
    with pytest.raises(ModelRegistryError, match="resolve to one model"):
        load_registry()


def test_an_alias_pointing_nowhere_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = write_registry(tmp_path / "alias.json", {"aliases": {"tiny": "no-such-model"}})
    monkeypatch.setenv(MODELS_ENV, str(broken))
    with pytest.raises(ModelRegistryError, match="no registry file defines"):
        load_registry()


def test_a_file_named_in_the_environment_that_is_absent_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a fall-through: a silent skip would resolve a shipped name instead."""
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(ModelRegistryError, match="unreadable"):
        load_registry()


def test_invalid_json_says_where(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"presets": ', encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV, str(bad))
    with pytest.raises(ModelRegistryError, match="invalid JSON at line"):
        load_registry()


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"num_kv_heads": 5}, "group size"),
        ({"sampled_layers": [0, 3, 99]}, "outside"),
        ({"eos_token_ids": []}, "token budget"),
        ({"calibration_root": "elsewhere"}, "calibration_root"),
        ({"hidden_dimension": 512}, "hidden_dimension"),
    ],
)
def test_a_malformed_row_refuses_and_names_what_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], fragment: str
) -> None:
    path = write_registry(
        tmp_path / "bad_row.json", {"presets": {"tiny-probe": {**ADDED_ROW, **overrides}}}
    )
    monkeypatch.setenv(MODELS_ENV, str(path))
    with pytest.raises(ModelRegistryError) as caught:
        load_registry()
    message = str(caught.value)
    assert str(path) in message
    assert fragment in message


def test_a_row_whose_name_disagrees_with_its_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_registry(tmp_path / "mismatch.json", {"presets": {"other": ADDED_ROW}})
    monkeypatch.setenv(MODELS_ENV, str(path))
    with pytest.raises(ModelRegistryError, match="are one thing"):
        load_registry()


def test_an_unknown_name_names_the_files_it_looked_in(added: Path) -> None:
    with pytest.raises(UnknownPresetError) as caught:
        resolve_preset("no-such-model")
    message = str(caught.value)
    assert str(MODELS_FILE) in message
    assert str(added) in message
    assert MODELS_ENV in message


def test_an_edit_to_a_registry_file_is_seen_without_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is keyed by each file's stamp, not by the process."""
    path = write_registry(tmp_path / "live.json", {"presets": {"tiny-probe": ADDED_ROW}})
    monkeypatch.setenv(MODELS_ENV, str(path))
    assert resolve_preset("tiny-probe").num_layers == 12
    write_registry(
        tmp_path / "live.json",
        {"presets": {"tiny-probe": {**ADDED_ROW, "num_layers": 16, "sampled_layers": [0, 8, 15]}}},
    )
    assert resolve_preset("tiny-probe").num_layers == 16


def test_the_70b_row_shares_the_8b_depth_rule() -> None:
    row = resolve_preset("70b")
    assert row.num_layers == 80
    assert row.sampled_layers == (0, 20, 40, 50, 60, 70, 79)
    assert row.kv_group_size == 8


def _extending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: dict[str, Any]) -> Path:
    path = write_registry(tmp_path / "variants.json", {"presets": rows})
    monkeypatch.setenv(MODELS_ENV, str(path))
    return path


def test_a_row_extends_a_shipped_row_and_keeps_its_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fine-tune of a shipped model restates only what differs."""
    _extending(tmp_path, monkeypatch, {"70b-variant": {
        "extends": "70b", "model_id": "/models/variant", "eos_token_ids": [128001],
        "calibration_dir": "calibration/variant"}})
    base, row = resolve_preset("70b"), resolve_preset("70b-variant")
    assert row.name == "70b-variant"
    assert (row.model_id, row.eos_token_ids, row.calibration_dir) == (
        "/models/variant", (128001,), "calibration/variant")
    assert row.sampled_layers == base.sampled_layers and row.num_layers == base.num_layers
    assert resolve_preset("70b") == base


def test_identity_fields_are_not_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _extending(tmp_path, monkeypatch, {"8b-variant": {"extends": "8b"}})
    row = resolve_preset("8b-variant")
    assert resolve_preset("8b").stage0_dir is not None
    assert row.stage0_dir is None and row.run_prefixes == ()


def test_extends_cannot_redefine_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _extending(tmp_path, monkeypatch, {"8b": {"extends": "70b"}})
    with pytest.raises(ModelRegistryError, match="already defined"):
        load_registry()


def test_extends_names_its_missing_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _extending(tmp_path, monkeypatch, {"orphan": {"extends": "nope"}})
    with pytest.raises(ModelRegistryError, match="extends 'nope'"):
        load_registry()


def test_extends_resolves_chains_within_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _extending(tmp_path, monkeypatch, {
        "b": {"extends": "a", "temperature": 0.7},
        "a": {"extends": "8b", "model_id": "/models/a"}})
    assert resolve_preset("b").model_id == "/models/a"
    assert resolve_preset("b").temperature == 0.7


def test_an_extending_row_may_not_rename_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _extending(tmp_path, monkeypatch, {"x": {"extends": "8b", "name": "y"}})
    with pytest.raises(ModelRegistryError, match="key and the name are one thing"):
        load_registry()
