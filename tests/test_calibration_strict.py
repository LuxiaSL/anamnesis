"""One reader for every banked basis shape, and a strict read of a whole directory.

Three shapes of basis file are banked — a pooled mapping, a mapping per layer index,
and a pickled scikit-learn estimator — and three places read them: the pooled
projection in :mod:`anamnesis.extraction.calibration`, the recompute in
:mod:`anamnesis.extraction.feature_pipeline`, and the path-signature family's basis
bank. The cases below pin that all three go through
:func:`anamnesis.extraction.calibration.read_pca_basis` and so agree on every shape,
each casting to its own precision.

:func:`anamnesis.extraction.calibration.load_calibration_strict` is the read for a
pass with no uncorrected fallback. Its load-bearing properties are that absence is a
typed refusal rather than a ``None``, and that the digest it returns names the bytes
it read — the same digest the fast lane stamps for a directory under the written
names, so the two stamps can be compared.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from anamnesis.extraction.calibration import (
    CALIBRATION_ARTIFACT_NAMES,
    PCA_MODEL_NAME,
    PCA_MODEL_NAMES,
    POSITIONAL_MEANS_NAME,
    CalibrationMalformed,
    CalibrationMissing,
    load_calibration_strict,
    load_pca_model,
    read_pca_basis,
)
from anamnesis.extraction.feature_families.path_signature import (
    ProjectionBasisBank,
    ProjectionError,
)
from anamnesis.extraction.feature_pipeline import _load_pca_model
from anamnesis.provenance import digest_of_shas, file_sha

UNITS = 6
RANK = 3
LAYERS = (2, 5)


def _pooled() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "components": rng.normal(size=(RANK, UNITS)),
        "mean": rng.normal(size=UNITS),
    }


def _per_layer() -> dict[int, dict[str, np.ndarray]]:
    rng = np.random.default_rng(2)
    return {
        layer: {"components": rng.normal(size=(RANK, UNITS)), "mean": rng.normal(size=UNITS)}
        for layer in LAYERS
    }


def _estimator() -> PCA:
    return PCA(n_components=RANK).fit(np.random.default_rng(3).normal(size=(20, UNITS)))


def _dump(path: Path, obj: object) -> Path:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    return path


def _calibration_dir(tmp_path: Path, basis: object, name: str = PCA_MODEL_NAME) -> Path:
    directory = tmp_path / "calibration"
    directory.mkdir()
    np.savez(
        directory / POSITIONAL_MEANS_NAME,
        positional_means=np.arange(3 * 4 * UNITS, dtype=np.float64).reshape(3, 4, UNITS),
    )
    _dump(directory / name, basis)
    return directory


# ── the one reader ────────────────────────────────────────────────────────────


def test_each_banked_shape_is_recognised_and_keeps_its_stored_dtype(tmp_path: Path) -> None:
    pooled = read_pca_basis(_dump(tmp_path / "pooled.pkl", _pooled()))
    assert pooled.format == "pooled" and pooled.pooled is not None and not pooled.per_layer
    assert pooled.pooled.components.dtype == np.float64

    per_layer = read_pca_basis(_dump(tmp_path / "per_layer.pkl", _per_layer()))
    assert per_layer.format == "per_layer" and per_layer.pooled is None
    assert sorted(per_layer.per_layer) == list(LAYERS)

    estimator = read_pca_basis(_dump(tmp_path / "estimator.pkl", _estimator()))
    assert estimator.format == "estimator" and estimator.pooled is not None
    assert estimator.pooled.components.shape == (RANK, UNITS)


def test_an_absent_basis_is_a_missing_calibration(tmp_path: Path) -> None:
    with pytest.raises(CalibrationMissing):
        read_pca_basis(tmp_path / "absent.pkl")
    assert issubclass(CalibrationMissing, FileNotFoundError)


@pytest.mark.parametrize(
    "obj",
    [
        {"mean": np.zeros(UNITS)},
        {},
        object(),
        {3: {"components": np.eye(UNITS)}, 4: "not a basis"},
    ],
    ids=["mapping-without-components", "empty-mapping", "bare-object", "ragged-per-layer"],
)
def test_a_file_with_no_components_is_malformed_rather_than_no_basis(
    obj: object, tmp_path: Path
) -> None:
    """Read as "no basis", these would compute uncorrected features under the name."""
    with pytest.raises(CalibrationMalformed):
        read_pca_basis(_dump(tmp_path / "bad.pkl", obj))


def test_a_file_that_does_not_unpickle_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "truncated.pkl"
    path.write_bytes(pickle.dumps(_pooled())[:20])
    with pytest.raises(CalibrationMalformed, match="does not unpickle"):
        read_pca_basis(path)


@pytest.mark.parametrize("shape", ["pooled", "estimator"])
def test_every_pooled_reader_returns_the_same_basis(shape: str, tmp_path: Path) -> None:
    """The projection, the recompute and the path-signature bank agree on one file."""
    obj = _pooled() if shape == "pooled" else _estimator()
    path = _dump(tmp_path / "basis.pkl", obj)
    stored = obj["components"] if isinstance(obj, dict) else obj.components_
    stored_mean = obj["mean"] if isinstance(obj, dict) else obj.mean_

    projected = load_pca_model(path)
    recomputed = _load_pca_model(path)
    bank = ProjectionBasisBank.from_pca_pickle(path)

    for components, mean in (projected, recomputed):
        assert isinstance(components, np.ndarray) and isinstance(mean, np.ndarray)
        assert components.dtype == np.float32
        assert np.array_equal(components, np.asarray(stored, dtype=np.float32))
        assert np.array_equal(mean, np.asarray(stored_mean, dtype=np.float32))
    assert bank.fallback is not None and not bank.per_layer
    assert bank.fallback.components.dtype == np.float64
    assert np.array_equal(bank.fallback.components, np.asarray(stored, dtype=np.float64))


def test_the_per_layer_shape_reaches_the_readers_that_take_it(tmp_path: Path) -> None:
    raw = _per_layer()
    path = _dump(tmp_path / "per_layer.pkl", raw)

    components, mean = _load_pca_model(path)
    assert isinstance(components, dict) and isinstance(mean, dict)
    assert sorted(components) == list(LAYERS)
    for layer in LAYERS:
        assert components[layer].dtype == np.float32
        assert np.array_equal(components[layer], raw[layer]["components"].astype(np.float32))
        assert np.array_equal(mean[layer], raw[layer]["mean"].astype(np.float32))

    bank = ProjectionBasisBank.from_pca_pickle(path)
    assert bank.fallback is None and sorted(bank.per_layer) == list(LAYERS)
    assert np.array_equal(bank.for_layer(LAYERS[0]).components, raw[LAYERS[0]]["components"])

    with pytest.raises(KeyError, match="per layer"):
        load_pca_model(path)


def test_the_recompute_reader_keeps_absence_lenient_and_refuses_a_bad_file(
    tmp_path: Path,
) -> None:
    """Absence runs the recompute without the residual block; a bad file stops it."""
    assert _load_pca_model(tmp_path / "absent.pkl") == (None, None)
    with pytest.raises(CalibrationMalformed):
        _load_pca_model(_dump(tmp_path / "bad.pkl", {"mean": np.zeros(UNITS)}))
    no_mean = {layer: {"components": np.eye(UNITS)[:RANK]} for layer in LAYERS}
    with pytest.raises(CalibrationMalformed, match="no mean"):
        _load_pca_model(_dump(tmp_path / "no_mean.pkl", no_mean))


def test_the_path_signature_bank_reports_a_bad_file_in_its_own_terms(tmp_path: Path) -> None:
    with pytest.raises(ProjectionError, match="not found"):
        ProjectionBasisBank.from_pca_pickle(tmp_path / "absent.pkl")
    with pytest.raises(ProjectionError, match="unusable"):
        ProjectionBasisBank.from_pca_pickle(_dump(tmp_path / "bad.pkl", {}))


def test_the_pooled_projection_refuses_a_basis_with_no_mean(tmp_path: Path) -> None:
    path = _dump(tmp_path / "no_mean.pkl", {"components": np.eye(UNITS)[:RANK]})
    with pytest.raises(CalibrationMalformed, match="no mean"):
        load_pca_model(path)


# ── the strict read of a directory ─────────────────────────────────────────────


@pytest.mark.parametrize("shape", ["pooled", "per_layer", "estimator"])
def test_a_complete_directory_reads_in_every_banked_shape(shape: str, tmp_path: Path) -> None:
    basis = {"pooled": _pooled, "per_layer": _per_layer, "estimator": _estimator}[shape]()
    directory = _calibration_dir(tmp_path, basis)

    calibration = load_calibration_strict(directory)
    assert calibration.basis.format == shape
    assert calibration.positional_means.dtype == np.float32
    assert calibration.positional_means.shape == (3, 4, UNITS)
    if shape == "per_layer":
        assert isinstance(calibration.pca_components, dict)
        assert sorted(calibration.pca_components) == list(LAYERS)
    else:
        assert isinstance(calibration.pca_components, np.ndarray)
        assert calibration.pca_components.shape == (RANK, UNITS)
        assert isinstance(calibration.pca_mean, np.ndarray)


def test_the_digests_name_the_bytes_read(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    calibration = load_calibration_strict(directory)

    expected = {name: file_sha(directory / name) for name in CALIBRATION_ARTIFACT_NAMES}
    assert dict(calibration.files) == expected
    assert calibration.sha256 == digest_of_shas(expected), (
        "the strict read and the fast lane stamp one digest for one directory"
    )
    record = calibration.provenance()
    assert record["calibration_sha256"] == calibration.sha256
    assert record["pca_file"] == PCA_MODEL_NAME and record["pca_format"] == "pooled"


def test_a_basis_under_the_banked_spelling_is_digested_under_that_name(tmp_path: Path) -> None:
    banked = PCA_MODEL_NAMES[1]
    directory = _calibration_dir(tmp_path, _per_layer(), name=banked)
    calibration = load_calibration_strict(directory)
    assert set(calibration.files) == {POSITIONAL_MEANS_NAME, banked}
    assert calibration.basis.path.name == banked


def test_changing_the_basis_bytes_changes_the_digest(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    before = load_calibration_strict(directory).sha256
    moved = _pooled()
    moved["components"] = moved["components"] * 2.0
    _dump(directory / PCA_MODEL_NAME, moved)
    assert load_calibration_strict(directory).sha256 != before


def test_absent_positional_means_are_refused(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    (directory / POSITIONAL_MEANS_NAME).unlink()
    with pytest.raises(CalibrationMissing, match="positional means"):
        load_calibration_strict(directory)


def test_an_absent_basis_is_refused_naming_every_accepted_name(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    (directory / PCA_MODEL_NAME).unlink()
    with pytest.raises(CalibrationMissing) as refusal:
        load_calibration_strict(directory)
    for name in PCA_MODEL_NAMES:
        assert name in str(refusal.value)


def test_an_absent_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CalibrationMissing, match="not a directory"):
        load_calibration_strict(tmp_path / "absent")


def test_an_archive_without_the_means_array_is_malformed(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    np.savez(directory / POSITIONAL_MEANS_NAME, something_else=np.zeros(3))
    with pytest.raises(CalibrationMalformed, match="something_else"):
        load_calibration_strict(directory)


def test_means_not_indexed_by_layer_position_unit_are_malformed(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, _pooled())
    np.savez(directory / POSITIONAL_MEANS_NAME, positional_means=np.zeros((4, UNITS)))
    with pytest.raises(CalibrationMalformed, match="layer, position, unit"):
        load_calibration_strict(directory)


def test_a_basis_without_its_mean_is_refused(tmp_path: Path) -> None:
    directory = _calibration_dir(tmp_path, {"components": np.eye(UNITS)[:RANK]})
    with pytest.raises(CalibrationMalformed, match="no mean"):
        load_calibration_strict(directory)
