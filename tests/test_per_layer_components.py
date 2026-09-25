"""A per-layer basis can keep a different number of components at each layer.

What a layer's samples determine differs by layer — the late layers of the shipped
models share fewer directions between half-fits than the early and middle ones — so a
model's registry row can name a count per PCA layer. The row is the one place the
count is decided: a calibration fits each layer to it, and the extraction projects
each layer onto the rows its basis holds (the lane and the anchor agreeing on a ragged
basis is pinned in ``test_fast_lane_equivalence.py``). The cases below pin the row's
validation, the fit, how the calibration command resolves the count, and the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydantic
import pytest

from test_loaded_model_seam import TINY_ROW
from anamnesis.config import ModelPreset, resolve_preset
from anamnesis.extraction.calibration_fit import (
    CalibrationBuildReceipt,
    CalibrationFitError,
    PromptStates,
    components_at,
    fit_calibration,
    fit_per_layer_basis,
    generation_settings,
)
from anamnesis.scripts import run_calibration

BY_LAYER = {8: 20, 16: 20, 20: 18, 24: 14, 28: 14}


def _row(**overrides) -> ModelPreset:
    base = resolve_preset("8b")
    return ModelPreset(**{**base.model_dump(), **overrides})


# ── the row ────────────────────────────────────────────────────────────────────


def test_a_row_takes_a_count_for_every_pca_layer() -> None:
    assert _row(pca_components_by_layer=BY_LAYER).pca_components_by_layer == BY_LAYER
    assert resolve_preset("8b").pca_components_by_layer is None


def test_a_row_naming_some_pca_layers_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="every PCA layer or none"):
        _row(pca_components_by_layer={8: 20})
    with pytest.raises(pydantic.ValidationError, match="every PCA layer or none"):
        _row(pca_components_by_layer={**BY_LAYER, 31: 5})


def test_a_count_below_one_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="below one"):
        _row(pca_components_by_layer={**BY_LAYER, 8: 0})


# ── the fit ────────────────────────────────────────────────────────────────────


def test_each_layer_keeps_its_own_count() -> None:
    rng = np.random.default_rng(7)
    layers, units = [0, 1, 2], 24
    means = np.zeros((4, 60, units), dtype=np.float32)
    means[:, :50] = rng.normal(0, 0.01, size=(4, 50, units))
    samples = [
        (layer, position, rng.normal(size=units).astype(np.float32))
        for layer in layers for position in range(50)
    ]
    basis = fit_per_layer_basis(samples, means, layers, {0: 6, 1: 3, 2: 9})
    assert {layer: basis[layer]["components"].shape[0] for layer in layers} == {0: 6, 1: 3, 2: 9}
    uniform = fit_per_layer_basis(samples, means, layers, 4)
    assert {layer: uniform[layer]["components"].shape[0] for layer in layers} == {0: 4, 1: 4, 2: 4}


def test_a_mapping_missing_a_layer_is_refused() -> None:
    with pytest.raises(CalibrationFitError, match="no component count for layer 2"):
        components_at({0: 3, 1: 3}, 2)


def test_a_pooled_fit_refuses_a_count_per_layer() -> None:
    prompt = PromptStates(
        prompt_length=2,
        prefill=np.zeros((4, 2, 32), dtype=np.float32),
        steps=tuple(np.zeros((4, 32), dtype=np.float32) for _ in range(3)),
    )
    with pytest.raises(CalibrationFitError, match="one component count"):
        fit_calibration(
            [prompt], preset=ModelPreset(**TINY_ROW),
            settings=generation_settings("3b", max_new_tokens=4),
            n_components={0: 2, 1: 2}, pooled=True, max_positions=8,
        )


# ── the command ────────────────────────────────────────────────────────────────


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"n_components": None, "pooled": False, **overrides})


def test_the_command_resolves_the_count_flag_then_row_then_extraction() -> None:
    row = _row(pca_components_by_layer=BY_LAYER)
    assert run_calibration.component_counts(_args(n_components=7), row, 50) == 7
    assert run_calibration.component_counts(_args(), row, 50) == BY_LAYER
    assert run_calibration.component_counts(_args(pooled=True), row, 50) == 50
    assert run_calibration.component_counts(_args(), resolve_preset("8b"), 50) == 50


def test_a_row_count_past_what_the_extraction_reads_is_refused() -> None:
    row = _row(pca_components_by_layer={**BY_LAYER, 8: 64})
    with pytest.raises(SystemExit, match="above the extraction's 50"):
        run_calibration.component_counts(_args(), row, 50)


def test_the_dry_run_names_the_counts_by_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    row = {**TINY_ROW, "pca_components_by_layer": {0: 3, 1: 2}}
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"presets": {"tiny-llama": row}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(registry))
    run_calibration.main(["--model", "tiny-llama", "--out-dir", str(tmp_path), "--dry-run"])
    assert "components kept: {0: 3, 1: 2} by layer" in capsys.readouterr().out


def test_the_receipt_records_a_count_per_layer() -> None:
    fields = {name: None for name in CalibrationBuildReceipt.model_fields}
    payload = {
        **fields,
        "model": "8b", "model_id": "m", "prompt_set_sha256": "a" * 64, "n_prompts": 1,
        "max_new_tokens": 1, "temperature": 1.0, "top_p": 1.0, "do_sample": True,
        "eos_token_ids": [1], "suppress_eos": False, "chat_template": True, "pooled": False,
        "n_components": BY_LAYER, "means_refitted": True,
        "coverage": {"table_positions": 2, "positions_calibrated": 1, "trailing_zero_rows": 1,
                     "measured_by": "means", "required_through": None},
        "files": {}, "calibration_sha256": "b" * 64,
    }
    receipt = CalibrationBuildReceipt.model_validate(payload)
    again = CalibrationBuildReceipt.model_validate(json.loads(json.dumps(receipt.model_dump(mode="json"))))
    assert again.n_components == BY_LAYER
