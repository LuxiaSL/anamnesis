"""The one calibration reader, and what it does about absence.

Two entry points each carried their own copy of this: one logged and one did not,
and both would have drifted. The cases below pin the contract the single reader now
holds — both banked shapes of the basis read to the same arrays, an absent artifact
comes back as ``None`` rather than raising, and the per-layer shape is refused by the
pooled reader rather than silently reduced to one of its layers.

Absence returning ``None`` is the load-bearing part. A caller that cannot proceed
without a calibration says so itself, which is how a pass that needs corrected
features refuses while a pass that does not touch the residual basis still runs.
Raising here would make the second impossible; returning zeros would make the first
invisible.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from anamnesis.extraction.calibration import (
    PCA_MODEL_NAME,
    POSITIONAL_MEANS_NAME,
    load_calibration,
    load_pca_model,
    load_positional_means,
)


class _Estimator:
    """Stands in for a pickled scikit-learn PCA, which is one of the banked shapes."""

    def __init__(self, components: np.ndarray, mean: np.ndarray) -> None:
        self.components_ = components
        self.mean_ = mean


@pytest.fixture()
def calib_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "calibration"
    directory.mkdir()
    np.savez(
        directory / POSITIONAL_MEANS_NAME,
        positional_means=np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4),
    )
    with open(directory / PCA_MODEL_NAME, "wb") as f:
        pickle.dump(
            {
                "components": np.eye(4, dtype=np.float64)[:2],
                "mean": np.ones(4, dtype=np.float64),
            },
            f,
        )
    return directory


def test_positional_means_are_read_as_float32(calib_dir: Path) -> None:
    means = load_positional_means(calib_dir)
    assert means is not None
    assert means.dtype == np.float32 and means.shape == (2, 3, 4)


def test_absent_positional_means_are_none_not_an_error(tmp_path: Path) -> None:
    assert load_positional_means(tmp_path) is None


def test_both_banked_basis_shapes_read_to_the_same_arrays(tmp_path: Path) -> None:
    components = np.eye(4, dtype=np.float64)[:2]
    mean = np.ones(4, dtype=np.float64)
    mapping_path, estimator_path = tmp_path / "a.pkl", tmp_path / "b.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump({"components": components, "mean": mean}, f)
    with open(estimator_path, "wb") as f:
        pickle.dump(_Estimator(components, mean), f)

    from_mapping = load_pca_model(mapping_path)
    from_estimator = load_pca_model(estimator_path)
    for left, right in zip(from_mapping, from_estimator):
        assert left.dtype == np.float32
        assert np.array_equal(left, right)


def test_missing_basis_by_explicit_path_raises(tmp_path: Path) -> None:
    """A caller that named a basis meant that one, so absence is an error there."""
    with pytest.raises(FileNotFoundError):
        load_pca_model(tmp_path / "nope.pkl")


def test_per_layer_basis_is_refused_by_the_pooled_reader(tmp_path: Path) -> None:
    """A mapping of bases would otherwise be reduced to whichever key came first."""
    path = tmp_path / "per_layer.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {14: {"components": np.eye(4)[:2], "mean": np.ones(4)}},
            f,
        )
    with pytest.raises(KeyError):
        load_pca_model(path)


def test_load_calibration_returns_the_triple(calib_dir: Path) -> None:
    means, components, mean = load_calibration(calib_dir, enable_pca=True)
    assert means is not None and components is not None and mean is not None
    assert components.shape == (2, 4) and mean.shape == (4,)


def test_load_calibration_leaves_the_basis_out_when_it_is_off(calib_dir: Path) -> None:
    means, components, mean = load_calibration(calib_dir, enable_pca=False)
    assert means is not None
    assert components is None and mean is None


def test_load_calibration_on_an_empty_directory_is_all_none(tmp_path: Path) -> None:
    assert load_calibration(tmp_path, enable_pca=True) == (None, None, None)
