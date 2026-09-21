"""Fitting a calibration: the decode policy it runs at, and the arithmetic it does.

Two properties carry the weight here.

**The decode policy is the preset's.** The artifacts a calibration writes are a mean
and a basis over a distribution of states, and that distribution is a function of
how the model sampled. A decode value written as a literal calibrates every model at
one model's setting, and nothing downstream can see it: a mean is a mean whatever
produced it. `gemma3-27b` and `dsv2-lite` decode at a nucleus mass of 0.95 while the
Llama rows decode at 0.9, so those two are where the difference shows.

**The fit is library code.** It takes states, not a checkpoint, so every case below
runs on synthetic arrays with no weights and no device — which is also what makes
the layer indexing testable, and `hidden_states` entry 0 being the embedding output
is the off-by-one that would corrupt every layer-indexed number downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anamnesis.config import ModelPreset, resolve_preset
from anamnesis.extraction.calibration import (
    PCA_MODEL_NAME,
    POSITION_COUNTS_KEY,
    POSITIONAL_MEANS_KEY,
    POSITIONAL_MEANS_NAME,
    load_calibration,
)
from anamnesis.extraction.calibration_fit import (
    CALIBRATION_PROMPTS,
    POSITION_COUNT_FLOOR,
    PROMPT_HEADROOM,
    CalibrationFitError,
    PromptStates,
    fit_calibration,
    generation_settings,
    means_from_totals,
    read_existing_means,
    write_calibration,
)

WIDE_TOP_P_MODELS = ("gemma3-27b", "dsv2-lite")
"""The presets whose nucleus mass is not the 0.9 the Llama rows carry."""


# ── the decode policy ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ("8b", "3b", "olmo2-7b", "qwen-7b", *WIDE_TOP_P_MODELS))
def test_every_decode_value_comes_from_the_preset(model: str) -> None:
    preset = resolve_preset(model)
    settings = generation_settings(preset)
    assert settings.top_p == preset.top_p
    assert settings.temperature == preset.temperature
    assert settings.max_new_tokens == preset.max_new_tokens
    assert list(settings.eos_token_ids) == list(preset.eos_token_ids)


@pytest.mark.parametrize("model", WIDE_TOP_P_MODELS)
def test_a_wide_nucleus_survives_into_the_calibration(model: str) -> None:
    """Pinned by value, so a literal 0.9 anywhere on this path fails here."""
    assert generation_settings(model).top_p == 0.95


def test_max_tokens_comes_from_the_preset_rather_than_a_shared_default() -> None:
    """Every current row happens to budget 512, so the test reads the row, not the number."""
    for model in ("8b", "3b", "gemma3-27b"):
        preset = resolve_preset(model)
        assert generation_settings(model).max_new_tokens == preset.max_new_tokens


def test_an_explicit_argument_wins_over_the_preset() -> None:
    settings = generation_settings("gemma3-27b", max_new_tokens=32, top_p=0.5, temperature=0.1)
    assert (settings.max_new_tokens, settings.top_p, settings.temperature) == (32, 0.5, 0.1)


def test_a_calibration_asks_for_hidden_states_only() -> None:
    """Attention weights and logits are the expensive outputs, and unread here."""
    settings = generation_settings("8b")
    assert settings.output_hidden_states is True
    assert settings.output_attentions is False and settings.output_logits is False


def test_the_prompt_set_is_the_fixed_ruler_it_claims_to_be() -> None:
    assert len(CALIBRATION_PROMPTS) == 50
    assert len(set(CALIBRATION_PROMPTS)) == 50


# ── the arithmetic ────────────────────────────────────────────────────────────


def synthetic_states(
    preset_name: str, n_prompts: int, prompt_length: int, n_steps: int, units: int
) -> list[PromptStates]:
    """One prompt's worth of states per prompt, distinguishable per layer and position."""
    depth = resolve_preset(preset_name).num_layers + 1
    rng = np.random.default_rng(0)
    return [
        PromptStates(
            prompt_length=prompt_length,
            prefill=rng.normal(size=(depth, prompt_length, units)).astype(np.float32),
            steps=tuple(
                rng.normal(size=(depth, units)).astype(np.float32) for _ in range(n_steps)
            ),
        )
        for _ in range(n_prompts)
    ]


def narrow_preset(model: str, units: int) -> ModelPreset:
    """A preset row whose hidden width matches a synthetic state's, for a cheap fit."""
    return resolve_preset(model).model_copy(update={"hidden_dim": units})


def test_the_fit_runs_as_library_code_with_no_checkpoint(tmp_path: Path) -> None:
    """The whole pass, from states to two written files, without a model."""
    units = 12
    preset = narrow_preset("3b", units)
    settings = generation_settings(preset, max_new_tokens=6)
    fit = fit_calibration(
        synthetic_states("3b", n_prompts=8, prompt_length=4, n_steps=6, units=units),
        preset=preset,
        settings=settings,
        n_components=3,
    )
    assert fit.means_refitted is True
    assert fit.positional_means.shape == (
        preset.num_layers + 1,
        settings.max_new_tokens + PROMPT_HEADROOM,
        units,
    )
    assert set(fit.basis) == set(preset.pca_layers), "a corrected fit is one basis per layer"
    for layer in preset.pca_layers:
        assert fit.basis[layer]["components"].shape == (3, units)

    means_path = tmp_path / POSITIONAL_MEANS_NAME
    write_calibration(fit, means_path, tmp_path / PCA_MODEL_NAME)
    archive = np.load(means_path)
    assert POSITIONAL_MEANS_KEY in archive and POSITION_COUNTS_KEY in archive


def test_calibrate_then_read_composes_on_defaults(tmp_path: Path) -> None:
    """The end-to-end name agreement: what a fit writes is what a reader resolves.

    A pooled fit is the shape `load_calibration` projects onto, so this is the pair
    that has to meet on one filename with no argument passed on either side.
    """
    units = 12
    preset = narrow_preset("3b", units)
    fit = fit_calibration(
        synthetic_states("3b", n_prompts=8, prompt_length=4, n_steps=6, units=units),
        preset=preset,
        settings=generation_settings(preset, max_new_tokens=6),
        n_components=3,
        pooled=True,
    )
    write_calibration(fit, tmp_path / POSITIONAL_MEANS_NAME, tmp_path / PCA_MODEL_NAME)

    means, components, mean = load_calibration(tmp_path)
    assert means is not None, "the means a fit wrote are the means a reader finds"
    assert components is not None and components.shape == (3, units)
    assert mean is not None and mean.shape == (units,)


def test_reused_means_are_not_rewritten(tmp_path: Path) -> None:
    """A correction a bank rests on does not move, so its file is left alone."""
    units = 12
    preset = narrow_preset("3b", units)
    settings = generation_settings(preset, max_new_tokens=6)
    states = synthetic_states("3b", n_prompts=8, prompt_length=4, n_steps=6, units=units)
    first = fit_calibration(states, preset=preset, settings=settings, n_components=3)
    means_path = tmp_path / POSITIONAL_MEANS_NAME
    write_calibration(first, means_path, tmp_path / PCA_MODEL_NAME)
    stamp = means_path.stat().st_mtime_ns

    banked = read_existing_means(means_path, refit=False)
    assert banked is not None and np.array_equal(banked, first.positional_means)
    again = fit_calibration(
        states, preset=preset, settings=settings, n_components=3, existing_means=banked
    )
    assert again.means_refitted is False
    write_calibration(again, means_path, tmp_path / "second.pkl")
    assert means_path.stat().st_mtime_ns == stamp


def test_a_refit_reads_nothing_off_disk(tmp_path: Path) -> None:
    means_path = tmp_path / POSITIONAL_MEANS_NAME
    np.savez(means_path, **{POSITIONAL_MEANS_KEY: np.zeros((2, 3, 4), dtype=np.float32)})
    assert read_existing_means(means_path, refit=True) is None
    assert read_existing_means(tmp_path / "absent.npz", refit=False) is None


def test_a_position_under_the_count_floor_is_left_at_zero() -> None:
    """A mean over two states is one of those states, and subtracting it is worse
    than subtracting nothing."""
    counts = np.array([[POSITION_COUNT_FLOOR + 1, POSITION_COUNT_FLOOR]], dtype=np.int64)
    sums = np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=np.float64)
    means = means_from_totals(sums, counts)
    assert means.dtype == np.float32
    assert np.allclose(means[0, 0], 10.0 / (POSITION_COUNT_FLOOR + 1))
    assert np.array_equal(means[0, 1], np.zeros(2, dtype=np.float32))


def test_a_generated_token_is_counted_at_its_absolute_position() -> None:
    """Step *i* sits at ``prompt_length + i``: the prompt's own positions come first."""
    units = 4
    preset = narrow_preset("3b", units)
    fit = fit_calibration(
        synthetic_states("3b", n_prompts=6, prompt_length=5, n_steps=3, units=units),
        preset=preset,
        settings=generation_settings(preset, max_new_tokens=3),
        n_components=2,
    )
    counts = fit.position_counts[0]
    assert list(counts[:8]) == [6, 6, 6, 6, 6, 6, 6, 6]
    assert counts[8] == 0
    assert fit.furthest_position == 7


def test_a_fit_with_nothing_to_fit_refuses_rather_than_writing_an_empty_basis() -> None:
    units = 4
    preset = narrow_preset("3b", units)
    with pytest.raises(CalibrationFitError):
        fit_calibration(
            [],
            preset=preset,
            settings=generation_settings(preset, max_new_tokens=3),
            n_components=2,
            pooled=True,
        )


def test_prompt_states_refuse_a_shape_that_would_misindex_layers() -> None:
    """The [t][l+1] convention is the gotcha this construction check exists for."""
    with pytest.raises(ValueError, match="layer, position, unit"):
        PromptStates(prompt_length=2, prefill=np.zeros((3, 4), dtype=np.float32), steps=())
    with pytest.raises(ValueError, match="layer, unit"):
        PromptStates(
            prompt_length=2,
            prefill=np.zeros((3, 2, 4), dtype=np.float32),
            steps=(np.zeros((3, 2, 4), dtype=np.float32),),
        )
