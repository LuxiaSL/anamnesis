"""A calibration extended past its base's reach keeps the base's identity, and only then.

:mod:`anamnesis.extraction.calibration_extension` lets a table that fills more
positions stand under the base's calibration digest, on evidence: the same basis
bytes, and the same means and counts bytes at every position the base covers. These
tests hold that contract on the tiny random-weight Llama the fast-lane suite uses:

1. an extension verifies, and its identity is the base's while the digest of the
   bytes it read is its own;
2. a span within the base's reach gives byte-identical vectors under the same lane id
   under either table, and the receipt and provenance name the table actually read;
3. a span past the base's reach and within the extension's is admitted by the
   extension and refused by the base, on the harvest path and the external path;
4. every break in the evidence is refused by name, when writing and when loading;
5. an extension of an extension resolves to the first base.
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_fast_lane_equivalence import tiny_loaded
from test_loaded_model_seam import PROMPT, lane_arithmetic, tiny_preset, tokens  # noqa: F401

from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME
from anamnesis.extraction.calibration_extension import (
    CALIBRATION_EXTENSION_NAME,
    CalibrationExtensionRefused,
    read_calibration_identity,
    write_calibration_extension,
)
from anamnesis.extraction.fast.external import ExternalCaptureError, ExternalSpan
from anamnesis.extraction.fast.harvest import harvest_loaded
from anamnesis.extraction.fast.runtime import (
    PreparedLane,
    prepare_fast_lane,
    read_lane_calibration,
)
from anamnesis.provenance import digest_of_shas, file_sha

LAYERS, WIDTH, UNITS = 4, 64, 32
BASE_REACH, EXTENDED_REACH = 20, 40


def _tables(reach: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Means and counts filled at positions ``[0, reach)``, zeros past it.

    Rows come from one fixed stream per position, so two tables with different
    reaches agree byte for byte on the rows both fill.
    """
    means = np.zeros((LAYERS, WIDTH, UNITS), dtype=np.float32)
    counts = np.zeros((LAYERS, WIDTH), dtype=np.int64)
    for position in range(reach):
        rng = np.random.default_rng((seed, position))
        means[:, position] = rng.normal(0, 0.01, size=(LAYERS, UNITS)).astype(np.float32)
        counts[:, position] = 10
    return means, counts


def _basis(seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "components": rng.normal(size=(50, UNITS)).astype(np.float32),
        "mean": rng.normal(0, 0.01, size=UNITS).astype(np.float32),
    }


def _write(directory: Path, means: np.ndarray, counts: np.ndarray | None,
           basis_from: Path | None = None) -> Path:
    directory.mkdir(parents=True)
    arrays = {"positional_means": means}
    if counts is not None:
        arrays["pos_counts"] = counts
    np.savez(directory / POSITIONAL_MEANS_NAME, **arrays)
    if basis_from is not None:
        shutil.copyfile(basis_from / PCA_MODEL_NAME, directory / PCA_MODEL_NAME)
    else:
        with open(directory / PCA_MODEL_NAME, "wb") as stream:
            pickle.dump(_basis(), stream)
    return directory


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return _write(tmp_path / "base", *_tables(BASE_REACH))


@pytest.fixture
def extended(tmp_path: Path, base: Path) -> Path:
    directory = _write(tmp_path / "extended", *_tables(EXTENDED_REACH), basis_from=base)
    write_calibration_extension(base, directory)
    return directory


def _prepared(calib_dir: Path) -> PreparedLane:
    return prepare_fast_lane(tiny_preset(), calib_dir, device="cpu", loaded=tiny_loaded())


def test_an_extension_keeps_the_base_identity_and_records_its_own_bytes(
    base: Path, extended: Path
) -> None:
    base_files = {name: file_sha(base / name) for name in (POSITIONAL_MEANS_NAME, PCA_MODEL_NAME)}
    plain = read_calibration_identity(base)
    assert plain.sha256 == plain.files_sha256 == digest_of_shas(base_files)
    assert plain.extension is None

    identity = read_calibration_identity(extended)
    assert identity.sha256 == plain.sha256
    assert identity.files_sha256 != plain.files_sha256
    assert identity.files[PCA_MODEL_NAME] == base_files[PCA_MODEL_NAME]
    assert identity.extension is not None
    assert identity.extension.base_files == base_files
    assert identity.extension.base_positions_calibrated == BASE_REACH
    assert identity.extension.positions_calibrated == EXTENDED_REACH

    lane = read_lane_calibration(tiny_preset(), extended)
    assert lane.calibration_sha256 == plain.sha256
    assert lane.positions_calibrated == EXTENDED_REACH
    assert write_calibration_extension(base, extended) == identity.extension, (
        "writing the same record again is a no-op"
    )


def test_a_span_within_the_base_reach_is_byte_identical_under_one_lane_id(
    lane_arithmetic: None, base: Path, extended: Path
) -> None:
    ids = tokens(PROMPT + 12)
    under_base = harvest_loaded(_prepared(base), ids, prompt_len=PROMPT)
    prepared = _prepared(extended)
    under_extension = harvest_loaded(prepared, ids, prompt_len=PROMPT)

    assert under_base.features.tobytes() == under_extension.features.tobytes()
    assert under_base.feature_names == under_extension.feature_names
    assert under_base.lane_id == under_extension.lane_id
    assert under_base.receipt["calibration_sha256"] == under_extension.receipt["calibration_sha256"]

    actual = read_calibration_identity(extended).files_sha256
    assert under_extension.receipt["calibration_files_sha256"] == actual
    assert under_base.receipt["calibration_files_sha256"] != actual
    assert under_extension.receipt["positions_calibrated"] == EXTENDED_REACH
    assert under_base.receipt["positions_calibrated"] == BASE_REACH

    stamp = prepared.provenance()
    assert stamp["calibration_files_sha256"] == actual
    assert stamp["positions_calibrated"] == EXTENDED_REACH
    assert stamp["calibration_extension"] == json.loads(
        (extended / CALIBRATION_EXTENSION_NAME).read_text()
    )
    assert _prepared(base).provenance()["calibration_extension"] is None


def test_a_span_past_the_base_reach_is_admitted_only_by_the_extension(
    lane_arithmetic: None, base: Path, extended: Path
) -> None:
    ids = tokens(PROMPT + 25)
    end = len(ids)
    assert BASE_REACH <= end - 2 < EXTENDED_REACH

    with pytest.raises(ValueError, match="outside what the lane supports"):
        harvest_loaded(_prepared(base), ids, prompt_len=PROMPT)
    result = harvest_loaded(_prepared(extended), ids, prompt_len=PROMPT)
    assert np.isfinite(result.features).all()

    steps = end - PROMPT - 1
    base_lane, _ = _prepared(base).lane(steps)
    extended_lane, _ = _prepared(extended).lane(steps)
    assert base_lane.lane_id == extended_lane.lane_id
    with pytest.raises(ExternalCaptureError, match="positional calibration covers"):
        ExternalSpan(base_lane, start=PROMPT, end=end, num_layers=3)
    ExternalSpan(extended_lane, start=PROMPT, end=end, num_layers=3)


def test_a_different_basis_is_refused(tmp_path: Path, base: Path) -> None:
    directory = _write(tmp_path / "refit", *_tables(EXTENDED_REACH))
    with open(directory / PCA_MODEL_NAME, "wb") as stream:
        pickle.dump(_basis(seed=6), stream)
    with pytest.raises(CalibrationExtensionRefused, match="basis"):
        write_calibration_extension(base, directory)
    assert not (directory / CALIBRATION_EXTENSION_NAME).exists()


@pytest.mark.parametrize("array", ["means", "counts"])
def test_one_ulp_inside_the_base_reach_is_refused(
    tmp_path: Path, base: Path, array: str
) -> None:
    means, counts = _tables(EXTENDED_REACH)
    if array == "means":
        means[2, BASE_REACH - 1, 7] = np.nextafter(means[2, BASE_REACH - 1, 7], np.float32(1))
    else:
        counts[2, BASE_REACH - 1] += 1
    directory = _write(tmp_path / "moved", means, counts, basis_from=base)
    with pytest.raises(CalibrationExtensionRefused, match="differ from"):
        write_calibration_extension(base, directory)


def test_an_extension_reaching_less_than_its_base_is_refused(tmp_path: Path, base: Path) -> None:
    directory = _write(tmp_path / "short", *_tables(BASE_REACH - 5), basis_from=base)
    with pytest.raises(CalibrationExtensionRefused, match="at least as far"):
        write_calibration_extension(base, directory)


@pytest.mark.parametrize("case", ["dtype", "width", "counts"])
def test_a_different_layout_is_refused(tmp_path: Path, base: Path, case: str) -> None:
    means, counts = _tables(EXTENDED_REACH)
    held: np.ndarray | None = counts
    if case == "dtype":
        means = means.astype(np.float64)
    elif case == "width":
        means = np.concatenate([means, means[:, :, :1]], axis=2)
    else:
        held = None
    directory = _write(tmp_path / case, means, held, basis_from=base)
    with pytest.raises(CalibrationExtensionRefused, match="only the position axis|counts"):
        write_calibration_extension(base, directory)


def test_a_tampered_record_is_refused_at_load(extended: Path) -> None:
    record = extended / CALIBRATION_EXTENSION_NAME
    fields = json.loads(record.read_text())
    record.write_text(json.dumps({**fields, "base_prefix_sha256": "0" * 64}))
    with pytest.raises(CalibrationExtensionRefused, match="not the base's bytes"):
        read_lane_calibration(tiny_preset(), extended)


def test_a_table_edited_inside_the_base_reach_after_its_record_is_refused_at_load(
    extended: Path,
) -> None:
    means, counts = _tables(EXTENDED_REACH)
    means[0, 0, 0] += np.float32(1e-3)
    np.savez(extended / POSITIONAL_MEANS_NAME, positional_means=means, pos_counts=counts)
    with pytest.raises(CalibrationExtensionRefused, match="positions 0..19"):
        read_lane_calibration(tiny_preset(), extended)


def test_a_basis_replaced_after_the_record_is_refused_at_load(extended: Path) -> None:
    with open(extended / PCA_MODEL_NAME, "wb") as stream:
        pickle.dump(_basis(seed=6), stream)
    with pytest.raises(CalibrationExtensionRefused, match="basis"):
        read_lane_calibration(tiny_preset(), extended)


def test_an_extension_of_an_extension_resolves_to_the_first_base(
    tmp_path: Path, base: Path, extended: Path
) -> None:
    further = _write(tmp_path / "further", *_tables(EXTENDED_REACH + 10), basis_from=base)
    record = write_calibration_extension(extended, further)
    root = read_calibration_identity(base).sha256
    assert record.base_calibration_sha256 == root
    assert record.base_positions_calibrated == EXTENDED_REACH
    assert record.base_files == read_calibration_identity(extended).files
    assert read_calibration_identity(further).sha256 == root
