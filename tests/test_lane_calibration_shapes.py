"""The fast lane reads either basis shape a calibration directory can hold.

``run_calibration`` fits a per-layer basis over corrected states by default, and the
banked calibrations hold a pooled one. The lane projects onto both, so the directory
reader in front of it must hand over both: a lane that refused the per-layer shape
could not read the calibration its own package just wrote. The cases below pin that
a per-layer directory resolves and harvests, that the per-layer basis is the one the
features are projected onto, and that a basis with no mean is refused rather than
centred on nothing.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from test_loaded_model_seam import POSITIONS, PROMPT, tiny_preset, tokens
from anamnesis.extraction.fast.harvest import harvest_loaded
from anamnesis.extraction.fast.runtime import (
    WORKSPACE_ENV,
    WORKSPACE_VALUE,
    prepare_fast_lane,
    read_lane_calibration,
)

UNITS = 32
LAYERS = (0, 1)


@pytest.fixture
def lane_arithmetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(WORKSPACE_ENV, WORKSPACE_VALUE)
    yield
    torch.use_deterministic_algorithms(False)


def _write(directory: Path, basis: Any) -> Path:
    directory.mkdir()
    rng = np.random.default_rng(11)
    np.savez(
        directory / "positional_means.npz",
        positional_means=rng.normal(0, 0.01, size=(4, POSITIONS, UNITS)).astype(np.float32),
    )
    with open(directory / "pca_model.pkl", "wb") as stream:
        pickle.dump(basis, stream)
    return directory


def _per_layer(seed: int, *, with_mean: bool = True) -> dict[int, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return {
        layer: {
            "components": rng.normal(size=(50, UNITS)).astype(np.float32),
            "mean": rng.normal(0, 0.01, size=UNITS).astype(np.float32) if with_mean else None,
        }
        for layer in LAYERS
    }


def test_a_per_layer_directory_resolves_to_per_layer_arrays(tmp_path: Path) -> None:
    basis = _per_layer(3)
    calibration = read_lane_calibration(tiny_preset(), _write(tmp_path / "c", basis))
    assert isinstance(calibration.pca_components, dict)
    assert sorted(calibration.pca_components) == list(LAYERS)
    for layer in LAYERS:
        assert np.array_equal(calibration.pca_components[layer], basis[layer]["components"])
        assert np.array_equal(calibration.pca_mean[layer], basis[layer]["mean"])


def test_a_per_layer_calibration_harvests_and_its_basis_is_the_one_applied(
    lane_arithmetic: None, tmp_path: Path
) -> None:
    ids = tokens(PROMPT + 16)
    harvests = []
    for name, seed in (("a", 3), ("b", 4)):
        lane = prepare_fast_lane(
            tiny_preset(), _write(tmp_path / name, _per_layer(seed)),
            device="cpu", loaded=tiny_loaded(),
        )
        harvests.append(harvest_loaded(lane, ids, prompt_len=PROMPT))
    first, second = harvests
    assert first.feature_names == second.feature_names
    assert np.isfinite(first.features).all()
    assert not np.array_equal(first.features, second.features), (
        "two per-layer bases gave the same features, so the basis is not being applied"
    )


def test_a_per_layer_basis_with_no_mean_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no mean"):
        read_lane_calibration(
            tiny_preset(), _write(tmp_path / "c", _per_layer(3, with_mean=False))
        )


def test_a_pooled_basis_with_no_mean_is_refused(tmp_path: Path) -> None:
    pooled = {"components": np.ones((50, UNITS), dtype=np.float32), "mean": None}
    with pytest.raises(ValueError, match="no mean"):
        read_lane_calibration(tiny_preset(), _write(tmp_path / "c", pooled))
