"""The one calibration reader, and what it does about absence.

The cases below pin the contract the single reader holds: both banked shapes of the
basis read to the same arrays, an absent artifact comes back as ``None`` rather than
raising, the per-layer shape is refused rather than silently reduced to one of its
layers, and a directory holding a basis under either accepted filename resolves.

Absence returning ``None`` is the load-bearing part. A caller that cannot proceed
without a calibration says so itself, which is how a pass that needs corrected
features refuses while a pass that does not touch the residual basis still runs.
Raising here would make the second impossible; returning zeros would make the first
invisible.

Filename resolution is the other load-bearing part, and it points both ways: a
directory a fit just wrote and a directory carrying the other accepted spelling both
resolve with no argument, while the plain name wins where both are present, so no
pass silently changes which basis it projects onto.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from anamnesis.extraction.calibration import (
    PCA_MODEL_NAME,
    PCA_MODEL_NAMES,
    POSITIONAL_MEANS_NAME,
    load_calibration,
    load_pca_model,
    load_positional_means,
    resolve_pca_model,
)

BANKED_PCA_MODEL_NAME = "pca_model_corrected.pkl"
"""The second accepted spelling, written out so the test states the name it pins."""


def _write_pooled(path: Path, components: np.ndarray, mean: np.ndarray) -> None:
    with open(path, "wb") as f:
        pickle.dump({"components": components, "mean": mean}, f)


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


# ── which file a directory's basis is ─────────────────────────────────────────


def test_the_accepted_names_are_declared_and_the_written_one_leads() -> None:
    """Named rather than spelled inline at each reader, so both halves agree."""
    assert PCA_MODEL_NAMES[0] == PCA_MODEL_NAME
    assert BANKED_PCA_MODEL_NAME in PCA_MODEL_NAMES


@pytest.mark.parametrize("name", PCA_MODEL_NAMES)
def test_either_accepted_name_resolves_on_its_own(name: str, tmp_path: Path) -> None:
    (tmp_path / name).write_bytes(b"")
    resolved = resolve_pca_model(tmp_path)
    assert resolved is not None and resolved.name == name


def test_the_written_name_wins_where_a_directory_holds_both(tmp_path: Path) -> None:
    """Otherwise a pass would change which basis it projects onto with no argument."""
    components, mean = np.eye(4, dtype=np.float64)[:2], np.ones(4, dtype=np.float64)
    _write_pooled(tmp_path / PCA_MODEL_NAME, components, mean)
    _write_pooled(tmp_path / BANKED_PCA_MODEL_NAME, components * 2.0, mean * 3.0)

    resolved = resolve_pca_model(tmp_path)
    assert resolved is not None and resolved.name == PCA_MODEL_NAME
    _, read_components, read_mean = load_calibration(tmp_path)
    assert read_components is not None and np.array_equal(read_components, components)
    assert read_mean is not None and np.array_equal(read_mean, mean)


def test_a_directory_with_no_basis_resolves_to_none(tmp_path: Path) -> None:
    assert resolve_pca_model(tmp_path) is None


def test_load_calibration_reads_the_banked_spelling_when_it_is_the_only_one(
    tmp_path: Path,
) -> None:
    """Banked directories are not regenerated, so their spelling stays readable."""
    components, mean = np.eye(4, dtype=np.float64)[:2], np.ones(4, dtype=np.float64)
    _write_pooled(tmp_path / BANKED_PCA_MODEL_NAME, components, mean)
    _, read_components, read_mean = load_calibration(tmp_path)
    assert read_components is not None and read_components.shape == (2, 4)
    assert read_mean is not None and read_mean.shape == (4,)


def test_a_per_layer_basis_names_the_reader_that_takes_it(tmp_path: Path) -> None:
    """The refusal carries where to go, because the shape is legitimate elsewhere."""
    path = tmp_path / PCA_MODEL_NAME
    with open(path, "wb") as f:
        pickle.dump({14: {"components": np.eye(4)[:2], "mean": np.ones(4)}}, f)
    with pytest.raises(KeyError, match="per layer"):
        load_calibration(tmp_path)
