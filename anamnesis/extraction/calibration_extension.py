"""A calibration whose positional means reach further than its base's, under the base's identity.

A fast-lane id hashes the calibration digest, :func:`anamnesis.provenance.digest_of_shas`
over the bytes of :data:`~anamnesis.extraction.calibration.CALIBRATION_ARTIFACT_NAMES`,
so any change to either file is a new lane. One change is not: a table that fills
positions the base left at zeros, while every byte a span within the base's reach
reads stays as it was. Such a span reads the same basis and the same rows of the same
means, so it computes the same vector under either table; the extension only admits
spans the base refused. No row banked under the base can conflict with one computed
under the extension, which is why the extension keeps the base's identity.

The evidence is checked, not assumed. :func:`write_calibration_extension` compares a
base directory with its extension and writes :data:`CALIBRATION_EXTENSION_NAME` into
the extension only when

* the residual basis is the same file, byte for byte;
* the positional means at every position the base covers are the same bytes on every
  layer, at the same dtype, layer count and width, and so are the per-position counts
  when the base carries them;
* the extension's reach is at least the base's.

The record keeps the base's canonical digest, its file digests, its reach, a digest
of the protected prefix of its means, and the extension's own reach. A base that is
itself an extension contributes its canonical digest, so a chain of extensions
resolves to the calibration it started from.

:func:`read_calibration_identity` is what every reader that stamps an identity
calls. With no record, the identity is the digest of the files, as it is for any
calibration. With a record, the evidence is re-checked from the extended directory
alone — the basis digest against the recorded one, the prefix of the means against
the recorded digest, the reach against the recorded reach — and the identity is the
base's canonical digest. A record that does not verify is refused: falling back to
the files' own digest would give a table that was edited after its record a fresh
lane id instead of an error. The files' own digests and the record stay in the
provenance, so a reader can see which table corrected a row.

The protected prefix is every position a pass under the base reads: up to the
larger of the base's reach by its counts and by its nonzero rows, since
:class:`~anamnesis.extraction.fast.features.GpuFeatureLane` reads coverage off the
rows alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.config.experiment import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME
from anamnesis.extraction.calibration import (
    CALIBRATION_ARTIFACT_NAMES,
    POSITION_COUNT_FLOOR,
    POSITION_COUNTS_KEY,
    POSITIONAL_MEANS_KEY,
    positions_calibrated,
)
from anamnesis.provenance import digest_of_shas, file_sha

CALIBRATION_EXTENSION_NAME = "calibration_extension.json"
"""The record, inside an extended calibration directory, that names its base."""

HEX64 = r"^[0-9a-f]{64}$"


class CalibrationExtensionRefused(ValueError):
    """An extension's evidence does not hold, at the time of writing or of reading."""


class CalibrationExtension(BaseModel):
    """What an extended calibration directory records about the base it extends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_calibration_sha256: str = Field(
        pattern=HEX64,
        description="The identity the extension keeps: the base's canonical digest, "
        "which is the root's when the base is itself an extension",
    )
    base_files: dict[str, str] = Field(
        description="The base's file digests, by artifact name"
    )
    base_positions_calibrated: int = Field(
        ge=0, description="How many leading positions the protected prefix covers"
    )
    base_prefix_sha256: str = Field(
        pattern=HEX64,
        description="prefix_digest of the means, and the counts if any, over the "
        "protected positions",
    )
    positions_calibrated: int = Field(
        ge=0, description="The extension's reach, by its counts when it carries them"
    )
    position_count_floor: int = Field(
        ge=0, description="The per-position count floor the reaches were read at"
    )


@dataclass(frozen=True)
class CalibrationIdentity:
    """The digests a calibration directory is stamped with.

    ``sha256`` is the identity a lane hashes: the files' own digest, or the base's
    canonical digest for a verified extension. ``files`` and ``files_sha256`` are
    the bytes actually read, and ``extension`` the record that licensed the
    difference, when there is one.
    """

    files: dict[str, str]
    files_sha256: str
    sha256: str
    extension: CalibrationExtension | None = None


def _read_means(directory: Path) -> tuple[NDArray[Any], NDArray[Any] | None]:
    """The stored means and counts, in the dtype the archive holds them.

    Raises
    ------
    CalibrationExtensionRefused
        When the archive is absent, unreadable or holds no means.
    """
    path = directory / POSITIONAL_MEANS_NAME
    try:
        with np.load(path) as archive:
            held = {name: archive[name] for name in (POSITIONAL_MEANS_KEY, POSITION_COUNTS_KEY)
                    if name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise CalibrationExtensionRefused(f"{path} does not read as an archive: {exc}") from exc
    means, counts = held.get(POSITIONAL_MEANS_KEY), held.get(POSITION_COUNTS_KEY)
    if means is None:
        raise CalibrationExtensionRefused(f"{path} holds no {POSITIONAL_MEANS_KEY!r}")
    if means.ndim != 3:
        raise CalibrationExtensionRefused(
            f"{path} holds positional means of shape {means.shape}; they are indexed "
            "[layer, position, unit]"
        )
    return means, counts


def protected_reach(means: NDArray[Any], counts: NDArray[Any] | None) -> int:
    """Every leading position a pass under these means may read.

    The larger of :func:`~anamnesis.extraction.calibration.positions_calibrated` read
    by the counts and read by the rows alone, because one reader takes each.
    """
    by_rows = positions_calibrated(means)
    return by_rows if counts is None else max(by_rows, positions_calibrated(means, counts))


def prefix_digest(means: NDArray[Any], counts: NDArray[Any] | None, reach: int) -> str:
    """SHA-256 over the means, and the counts if any, at positions ``[0, reach)``.

    Each array is restricted to its first ``reach`` positions on every layer, made
    C-contiguous and little-endian in its stored dtype, and hashed after a header
    naming the dtype and shape, so two prefixes digest alike only when their bytes,
    dtype and layout all agree. An absent counts array is hashed as absent.
    """
    digest = hashlib.sha256()
    for name, array in (("means", means), ("counts", counts)):
        if array is None:
            digest.update(json.dumps({"array": name, "absent": True}).encode() + b"\n")
            continue
        part = np.ascontiguousarray(array[:, :reach])
        part = part.astype(part.dtype.newbyteorder("<"), copy=False)
        header = {"array": name, "dtype": part.dtype.str, "shape": list(part.shape)}
        digest.update(json.dumps(header, sort_keys=True).encode() + b"\n")
        digest.update(part.tobytes())
    return digest.hexdigest()


def _files(directory: Path) -> dict[str, str]:
    """The artifact digests a calibration identity is taken over.

    Raises
    ------
    CalibrationExtensionRefused
        When an artifact is absent under its written name.
    """
    missing = [name for name in CALIBRATION_ARTIFACT_NAMES if not (directory / name).is_file()]
    if missing:
        raise CalibrationExtensionRefused(f"{directory} holds no {missing}")
    return {name: file_sha(directory / name) for name in CALIBRATION_ARTIFACT_NAMES}


def read_extension(directory: Path) -> CalibrationExtension | None:
    """The record in ``directory``, or ``None`` when it holds none.

    Raises
    ------
    CalibrationExtensionRefused
        When the record is present but does not parse as one.
    """
    path = Path(directory) / CALIBRATION_EXTENSION_NAME
    if not path.is_file():
        return None
    try:
        return CalibrationExtension.model_validate_json(path.read_text())
    except ValueError as exc:
        raise CalibrationExtensionRefused(f"{path} is not a calibration extension: {exc}") from exc


def read_calibration_identity(calib_dir: Path) -> CalibrationIdentity:
    """The digests a calibration directory is stamped with, its extension re-verified.

    Raises
    ------
    CalibrationExtensionRefused
        When an artifact is absent, or the directory holds an extension record whose
        basis digest, prefix digest or reach does not match the files beside it.
    """
    directory = Path(calib_dir)
    files = _files(directory)
    files_sha256 = digest_of_shas(files)
    extension = read_extension(directory)
    if extension is None:
        return CalibrationIdentity(files=files, files_sha256=files_sha256, sha256=files_sha256)
    record = directory / CALIBRATION_EXTENSION_NAME
    if files[PCA_MODEL_NAME] != extension.base_files.get(PCA_MODEL_NAME):
        raise CalibrationExtensionRefused(
            f"{record}: the basis {PCA_MODEL_NAME} is not the base's bytes, so the extension "
            "is a new calibration and cannot keep the base's identity"
        )
    means, counts = _read_means(directory)
    if prefix_digest(means, counts, extension.base_positions_calibrated) != (
        extension.base_prefix_sha256
    ):
        raise CalibrationExtensionRefused(
            f"{record}: the positional means at positions "
            f"0..{extension.base_positions_calibrated - 1} are not the base's bytes"
        )
    reach = positions_calibrated(means, counts)
    if reach != extension.positions_calibrated or (
        extension.position_count_floor != POSITION_COUNT_FLOOR
    ):
        raise CalibrationExtensionRefused(
            f"{record}: the means reach {reach} positions at count floor "
            f"{POSITION_COUNT_FLOOR}, and the record says {extension.positions_calibrated} at "
            f"{extension.position_count_floor}; it was written for another table"
        )
    return CalibrationIdentity(
        files=files,
        files_sha256=files_sha256,
        sha256=extension.base_calibration_sha256,
        extension=extension,
    )


def write_calibration_extension(base_dir: Path, extended_dir: Path) -> CalibrationExtension:
    """Check that ``extended_dir`` extends ``base_dir`` and record it there.

    Writing the same record again is a no-op; an existing record that differs is
    refused rather than replaced.

    Raises
    ------
    CalibrationExtensionRefused
        Naming the check that failed: the two are one directory; an artifact is
        absent; the base's own record does not verify; the basis bytes differ; the
        means differ in dtype, layer count or width; one carries counts and the other
        does not; the extension reaches fewer positions than the base; the protected
        prefix of the means or counts differs in any byte; or a different record is
        already there.
    """
    base, extended = Path(base_dir), Path(extended_dir)
    if base.resolve() == extended.resolve():
        raise CalibrationExtensionRefused(f"{base} cannot extend itself")
    base_identity = read_calibration_identity(base)
    extended_files = _files(extended)
    if extended_files[PCA_MODEL_NAME] != base_identity.files[PCA_MODEL_NAME]:
        raise CalibrationExtensionRefused(
            f"the basis in {extended} is not the bytes of the basis in {base}; a refitted "
            "basis is a new calibration"
        )
    base_means, base_counts = _read_means(base)
    means, counts = _read_means(extended)
    if means.dtype != base_means.dtype or (
        means.shape[0], means.shape[2]) != (base_means.shape[0], base_means.shape[2]
    ):
        raise CalibrationExtensionRefused(
            f"the means in {extended} are {means.dtype} {means.shape} and the base's are "
            f"{base_means.dtype} {base_means.shape}; only the position axis may grow"
        )
    if (counts is None) != (base_counts is None):
        raise CalibrationExtensionRefused(
            f"one of {base} and {extended} carries per-position counts and the other does not"
        )
    if counts is not None and base_counts is not None and counts.dtype != base_counts.dtype:
        raise CalibrationExtensionRefused(
            f"the counts in {extended} are {counts.dtype} and the base's are {base_counts.dtype}"
        )
    base_reach = protected_reach(base_means, base_counts)
    reach = positions_calibrated(means, counts)
    if reach < base_reach:
        raise CalibrationExtensionRefused(
            f"{extended} reaches {reach} positions and the base {base_reach}; an extension "
            "reaches at least as far"
        )
    base_prefix = prefix_digest(base_means, base_counts, base_reach)
    if prefix_digest(means, counts, base_reach) != base_prefix:
        raise CalibrationExtensionRefused(
            f"the positional means or counts in {extended} differ from {base} at positions "
            f"0..{base_reach - 1}; every byte a span within the base's reach reads must "
            "be the base's"
        )
    extension = CalibrationExtension(
        base_calibration_sha256=base_identity.sha256,
        base_files=base_identity.files,
        base_positions_calibrated=base_reach,
        base_prefix_sha256=base_prefix,
        positions_calibrated=reach,
        position_count_floor=POSITION_COUNT_FLOOR,
    )
    existing = read_extension(extended)
    if existing is not None and existing != extension:
        raise CalibrationExtensionRefused(
            f"{extended / CALIBRATION_EXTENSION_NAME} already records a different base"
        )
    (extended / CALIBRATION_EXTENSION_NAME).write_text(
        extension.model_dump_json(indent=2) + "\n"
    )
    return extension
