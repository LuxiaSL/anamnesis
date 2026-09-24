"""Where a per-layer basis draws its samples from, and how to tell what they determine.

A basis of fifty components over thousands of units is only pinned where enough
samples stand behind it. Three samples per prompt left the tail of every banked
per-layer basis to the sample; the per-layer fit now takes
:data:`anamnesis.extraction.calibration_fit.BASIS_STEPS_PER_PROMPT` evenly spread
generated positions per prompt, while the pooled fit keeps the three points its
banked bases were fitted at. The agreement measure is pinned against bases whose
agreement is known by construction, and the stability command end to end against a
real decoder with random weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_calibration_replay import PROMPT_IDS, RULER, _BareTokenizer
from test_fast_lane_equivalence import tiny_loaded
from test_loaded_model_seam import TINY_ROW
from anamnesis.extraction import calibration_fit, model_loader
from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME
from anamnesis.extraction.calibration_fit import (
    BASIS_STEPS_PER_PROMPT,
    PromptStates,
    basis_samples,
    basis_steps,
    determined_components,
    subspace_agreement,
)
from anamnesis.scripts import calibration_stability

# ── which steps ───────────────────────────────────────────────────────────────


def test_a_pooled_fit_keeps_its_three_points_as_they_fall() -> None:
    assert basis_steps(100, pooled=True) == [1, 50, 100]
    assert basis_steps(1, pooled=True) == [1, 1, 1]


def test_a_per_layer_fit_spreads_its_steps_from_first_to_last() -> None:
    steps = basis_steps(500, pooled=False)
    assert len(steps) == BASIS_STEPS_PER_PROMPT == len(set(steps))
    assert steps[0] == 1 and steps[-1] == 500
    gaps = np.diff(steps)
    assert gaps.max() - gaps.min() <= 1, "evenly spread"


def test_a_generation_shorter_than_the_budget_gives_every_step() -> None:
    assert basis_steps(7, pooled=False) == list(range(1, 8))
    assert basis_steps(0, pooled=False) == []


def test_samples_carry_the_absolute_position_of_their_step() -> None:
    depth, units, prompt_length = 4, 6, 10
    prompt = PromptStates(
        prompt_length=prompt_length,
        prefill=np.zeros((depth, prompt_length, units), dtype=np.float32),
        steps=tuple(np.full((depth, units), i, dtype=np.float32) for i in range(40)),
    )
    samples = basis_samples(prompt, [0, 2], pooled=False, per_prompt=5)
    assert {layer for layer, _, _ in samples} == {0, 2}
    for layer, absolute, state in samples:
        step = absolute - prompt_length + 1
        assert state[0] == step - 1, "the state is the one at that step"
    assert len(samples) == 2 * 5


# ── what the samples determine ────────────────────────────────────────────────


def _orthonormal(rows: int, width: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, rows)))
    return q.T.astype(np.float32)


def test_a_basis_agrees_with_itself_everywhere() -> None:
    basis = _orthonormal(10, 40, 1)
    np.testing.assert_allclose(subspace_agreement(basis, basis), 1.0, atol=1e-6)
    assert determined_components(subspace_agreement(basis, basis)) == 10


def test_agreement_holds_through_a_shared_head_and_breaks_at_the_first_foreign_direction() -> None:
    shared = _orthonormal(10, 40, 2)
    other = shared.copy()
    other[6:] = _orthonormal(10, 40, 3)[6:]
    other, _ = np.linalg.qr(other.T.astype(np.float64))
    other = other.T.astype(np.float32)
    curve = subspace_agreement(shared, other)
    np.testing.assert_allclose(curve[:6], 1.0, atol=1e-5)
    assert curve[6] < 0.9
    assert determined_components(curve) == 6


def test_a_sign_flip_or_a_rotation_inside_the_head_is_agreement() -> None:
    basis = _orthonormal(8, 30, 4)
    flipped = basis.copy()
    flipped[0] *= -1
    rotated = basis.copy()
    c, s = np.cos(0.7), np.sin(0.7)
    rotated[[1, 2]] = np.array([[c, s], [-s, c]], dtype=np.float32) @ basis[[1, 2]]
    assert determined_components(subspace_agreement(basis, flipped)) == 8
    curve = subspace_agreement(basis, rotated)
    assert curve[2] > 0.999, "the first three span the same subspace"
    assert curve[1] < 0.9, "the second direction alone differs"


def test_bases_of_different_widths_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot be compared"):
        subspace_agreement(_orthonormal(3, 10, 1), _orthonormal(3, 12, 1))


# ── the command ────────────────────────────────────────────────────────────────


def test_the_stability_command_splits_the_banked_sequences_and_writes_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"presets": {"tiny-llama": TINY_ROW}}))
    monkeypatch.setenv("ANAMNESIS_MODELS", str(registry))

    loaded = tiny_loaded()
    loaded.tokenizer = _BareTokenizer()
    loaded.disable_hooks()
    tokens = calibration_fit.generate_calibration_tokens(
        loaded, RULER, calibration_fit.generation_settings("3b", max_new_tokens=16)
    )
    calib = tmp_path / "calibration"
    calib.mkdir()
    rng = np.random.default_rng(0)
    np.savez(
        calib / POSITIONAL_MEANS_NAME,
        positional_means=rng.normal(0, 0.01, size=(4, 40, 32)).astype(np.float32),
    )
    calibration_fit.tokens_path(calib / PCA_MODEL_NAME).write_text(json.dumps(tokens.model_dump()))

    monkeypatch.setattr(model_loader, "load_model", lambda *a, **k: loaded)
    out = tmp_path / "stability.json"
    calibration_stability.main(
        ["--model", "tiny-llama", "--model-path", "unused", "--calib-dir", str(calib),
         "--n-components", "4", "--json", str(out)]
    )
    receipt = json.loads(out.read_text())
    assert receipt["sequences"] == len(RULER) == len(PROMPT_IDS)
    assert set(receipt["layers"]) == {str(layer) for layer in TINY_ROW["pca_layers"]}
    for layer, halves in receipt["samples_per_half"].items():
        assert all(count > 0 for count in halves), layer
    for entry in receipt["layers"].values():
        assert len(entry["agreement"]) == 4
        assert 0 <= entry["determined"] <= 4
