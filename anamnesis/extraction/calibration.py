"""What a calibration directory is called, and how the two artifacts in it are read.

Positional decomposition needs per-position means to subtract, and the residual
PCA needs a fitted basis to project onto. Both are model-specific files written
once per model by :mod:`anamnesis.extraction.calibration_fit`, and both are read
by every path that computes features: the in-process extraction pass, the replay
worker, the fast lane, and the offline recompute.

The filenames themselves are :data:`anamnesis.config.experiment.PCA_MODEL_NAME` and
its siblings, imported here rather than spelled again. They sit in the
configuration layer because that layer builds the paths as well, and because it is
the base of this package and imports nothing else in it — the other direction would
put numpy behind every import of a run's description. What lives here is the
resolution: which of the accepted names a directory answers with, and what each file
turns into.

Three on-disk shapes for the PCA model are banked, and :func:`read_pca_basis` is
the one reader of all three: a plain mapping with ``components`` and ``mean`` keys,
a pickled scikit-learn estimator with ``components_`` and ``mean_`` attributes, and
the per-layer basis a corrected fit writes, a mapping from layer index to one such
mapping. Every other reader here and elsewhere in the package goes through it, so a
shape is recognised the same way wherever a basis is read. Every extraction path —
the hook path, the fast lane, the offline recompute — projects onto either the
pooled or the per-layer shape, so :func:`load_calibration` returns whichever the
file holds. :func:`load_pca_model` is the one reader that promises a single basis,
and it refuses the per-layer shape by name rather than reducing it to whichever
layer a key happened to hold.

A calibration corrects only the positions its fit reached. :func:`require_positions_covered`
is the refusal a pass makes before reading a position past them, since the correction
there would subtract a row of zeros.

Absence is returned, not raised, by :func:`load_positional_means` and
:func:`load_calibration`. A caller that cannot proceed without a calibration says so
itself — :mod:`anamnesis.scripts.run_gpu_replay` refuses a partial one outright,
while a pass whose features do not touch the residual PCA runs without it. Silently
handing back uncorrected features under the name of corrected ones is the failure
this split avoids, so a missing positional means is logged as a warning at the point
of reading. A caller that needs both artifacts and a record of which bytes it read
takes :func:`load_calibration_strict` instead, which refuses either absence with
:class:`CalibrationMissing` and returns the arrays together with their digests.

This module is pure numpy, pickle and hashing. Fitting a basis needs scikit-learn, a
model runtime and weights on a device, so it sits in a sibling module that the CPU
recompute lane and the fast-lane readers never import.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from anamnesis.config.experiment import (
    PCA_MODEL_NAME,
    PCA_MODEL_NAMES,
    POSITIONAL_MEANS_NAME,
)
from anamnesis.provenance import digest_of_shas, file_sha

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

POSITIONAL_MEANS_KEY = "positional_means"
"""The array key inside the positional-means archive."""

POSITION_COUNTS_KEY = "pos_counts"
"""How many states each position's mean was taken over, in the same archive."""

POSITION_COUNT_FLOOR = 5
"""A position whose mean is an average of this many states or fewer is left at
zero: a mean over one or two prompts is not a mean, it is one of the states. The fit
in :mod:`anamnesis.extraction.calibration_fit` writes by it, and
:func:`positions_calibrated` reads coverage by it."""

I64 = NDArray[np.int64]

CALIBRATION_ARTIFACT_NAMES: tuple[str, ...] = (POSITIONAL_MEANS_NAME, PCA_MODEL_NAME)
"""The pair a provenance digest over a calibration directory hashes.

The written names, not :data:`anamnesis.config.experiment.PCA_MODEL_NAMES`: a digest
names the bytes it read, so a directory that answers :func:`resolve_pca_model` under
the other accepted spelling has no digest under this one and the pass that wanted a
provenance stamp fails on the missing file rather than stamping a different set.
"""


PCAFormat = Literal["pooled", "per_layer", "estimator"]
"""Which banked shape a basis file holds: one mapping with ``components`` and
``mean``, one such mapping per layer index, or a fitted scikit-learn estimator."""


class CalibrationMissing(FileNotFoundError):
    """A calibration artifact a strict read needs is not on disk.

    A subclass of :class:`FileNotFoundError`, so a caller that already handles an
    absent file handles this one.
    """


class CalibrationMalformed(ValueError):
    """A calibration artifact is on disk but holds no shape this package reads."""


@dataclass(frozen=True)
class BasisArrays:
    """One basis as stored: components ``[k, units]`` and, when stored, the mean
    ``[units]`` it was fitted around. The arrays keep the dtype the file holds, so
    each reader casts to the precision its own arithmetic runs at."""

    components: NDArray[Any]
    mean: NDArray[Any] | None


@dataclass(frozen=True)
class PCABasis:
    """What :func:`read_pca_basis` found in one basis file.

    Exactly one of ``pooled`` and ``per_layer`` is populated: ``pooled`` for the
    mapping and estimator shapes, ``per_layer`` (keyed by layer index) for the
    corrected fit's shape.
    """

    path: Path
    format: PCAFormat
    pooled: BasisArrays | None = None
    per_layer: Mapping[int, BasisArrays] = field(default_factory=dict)

    @property
    def has_every_mean(self) -> bool:
        """Whether every basis in the file carries the mean it was fitted around."""
        if self.pooled is not None:
            return self.pooled.mean is not None
        return all(basis.mean is not None for basis in self.per_layer.values())

    def float32_arrays(self) -> tuple[F32 | dict[int, F32], F32 | dict[int, F32] | None]:
        """Components and mean as float32, the precision the projection runs at.

        A pooled basis comes back as two arrays, with the mean ``None`` when the
        file stores none. A per-layer basis comes back as two mappings keyed by
        layer index, the form :mod:`anamnesis.extraction.feature_pipeline` takes.

        Raises
        ------
        CalibrationMalformed
            When a per-layer basis leaves out a layer's mean: the recompute
            centres every layer, so a missing one has no stand-in.
        """
        if self.pooled is not None:
            mean = self.pooled.mean
            return (
                np.asarray(self.pooled.components, dtype=np.float32),
                None if mean is None else np.asarray(mean, dtype=np.float32),
            )
        missing = sorted(layer for layer, basis in self.per_layer.items() if basis.mean is None)
        if missing:
            raise CalibrationMalformed(
                f"{self.path} holds a per-layer basis with no mean at layers {missing}"
            )
        return (
            {
                layer: np.asarray(basis.components, dtype=np.float32)
                for layer, basis in self.per_layer.items()
            },
            {
                layer: np.asarray(basis.mean, dtype=np.float32)
                for layer, basis in self.per_layer.items()
            },
        )


def read_pca_basis(path: Path) -> PCABasis:
    """The basis in one file, in whichever of the three banked shapes it holds.

    This is the package's one reader of a basis file. A mapping whose first value is
    itself a mapping with ``components`` is the per-layer shape; any other mapping
    must carry ``components`` itself; anything else must carry ``components_``. A
    mean is optional at this level and comes back as ``None`` when the file stores
    none, since whether a caller can project without one is the caller's rule.

    Raises
    ------
    FileNotFoundError
        When the file is absent — a caller that asked for a basis by path meant
        that one. The raised type is :class:`CalibrationMissing`.
    CalibrationMalformed
        When the file does not unpickle, or holds an object with no components in
        any of the three shapes. Reading it as "no basis" would compute
        uncorrected features under the name of corrected ones.
    """
    target = Path(path)
    if not target.is_file():
        raise CalibrationMissing(f"no PCA model at {target}")
    try:
        with open(target, "rb") as handle:
            model = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError) as exc:
        raise CalibrationMalformed(f"{target} does not unpickle as a PCA model: {exc}") from exc

    if isinstance(model, dict):
        values = list(model.values())
        if values and isinstance(values[0], dict) and "components" in values[0]:
            per_layer: dict[int, BasisArrays] = {}
            for key, entry in model.items():
                if not isinstance(entry, dict) or "components" not in entry:
                    raise CalibrationMalformed(
                        f"{target} holds a per-layer basis whose entry {key!r} has no components"
                    )
                per_layer[int(key)] = BasisArrays(
                    components=np.asarray(entry["components"]),
                    mean=None if entry.get("mean") is None else np.asarray(entry["mean"]),
                )
            return PCABasis(path=target, format="per_layer", per_layer=per_layer)
        if model.get("components") is None:
            raise CalibrationMalformed(
                f"{target} holds a mapping keyed {sorted(map(str, model))} with no components"
            )
        return PCABasis(
            path=target,
            format="pooled",
            pooled=BasisArrays(
                components=np.asarray(model["components"]),
                mean=None if model.get("mean") is None else np.asarray(model["mean"]),
            ),
        )

    components = getattr(model, "components_", None)
    if components is None:
        raise CalibrationMalformed(
            f"{target} holds a {type(model).__name__} with no components_ attribute"
        )
    mean = getattr(model, "mean_", None)
    return PCABasis(
        path=target,
        format="estimator",
        pooled=BasisArrays(
            components=np.asarray(components),
            mean=None if mean is None else np.asarray(mean),
        ),
    )


def resolve_pca_model(calib_dir: Path) -> Path | None:
    """The basis in a calibration directory, or ``None`` when it holds none.

    Tries :data:`PCA_MODEL_NAMES` in order and returns the first that exists, so a
    directory written by a fit and a banked directory both resolve without the
    caller naming a file.
    """
    directory = Path(calib_dir)
    for name in PCA_MODEL_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def load_positional_means(calib_dir: Path) -> F32 | None:
    """Per-position means from a calibration directory, or ``None`` if absent."""
    path = Path(calib_dir) / POSITIONAL_MEANS_NAME
    if not path.exists():
        logger.warning(
            f"no positional means at {path}; features that subtract them would be wrong"
        )
        return None
    means: F32 = np.load(path)[POSITIONAL_MEANS_KEY].astype(np.float32)
    logger.info(f"positional_means {means.shape}")
    return means


def load_position_counts(calib_dir: Path) -> I64 | None:
    """How many states each position's mean was taken over, or ``None``.

    ``None`` when the directory holds no positional means or the archive carries no
    counts; :func:`positions_calibrated` then reads coverage off the means alone.
    """
    path = Path(calib_dir) / POSITIONAL_MEANS_NAME
    if not path.is_file():
        return None
    with np.load(path) as archive:
        if POSITION_COUNTS_KEY not in archive.files:
            return None
        counts: I64 = archive[POSITION_COUNTS_KEY].astype(np.int64)
    return counts


def positions_calibrated(positional_means: F32, counts: I64 | None = None) -> int:
    """How many leading positions the means actually correct: the last filled row, plus one.

    The width of the table is not its coverage. A fit allocates rows for every
    position it might reach and leaves a row it did not reach, or reached too few
    times, at exact zeros, so subtracting it corrects nothing. A position past the
    last filled row is therefore uncalibrated whatever the table's width says, and
    a pass that treats it as calibrated computes an uncorrected feature under the
    corrected feature's name.

    A position is filled when every layer's row is. With ``counts`` (the
    ``pos_counts`` array a fit writes, indexed ``[layer, position]``) that means a
    count above :data:`POSITION_COUNT_FLOOR`, the rule the fit writes rows by;
    without it, a row that is not all zeros. Interior gaps do not shorten coverage;
    only the tail does.

    Returns
    -------
    int
        The index of the last filled position plus one, or 0 when no row is filled.
        A position ``p`` is covered exactly when ``p < positions_calibrated(...)``.

    Raises
    ------
    ValueError
        When the means are not indexed ``[layer, position, unit]``, or when
        ``counts`` does not have the means' ``[layer, position]`` shape.
    """
    means = np.asarray(positional_means)
    if means.ndim != 3:
        raise ValueError(
            f"positional means of shape {means.shape} are not indexed [layer, position, unit]"
        )
    if counts is not None:
        table = np.asarray(counts)
        if table.shape != means.shape[:2]:
            raise ValueError(
                f"position counts of shape {table.shape} do not match positional means "
                f"of shape {means.shape}; they are indexed [layer, position]"
            )
        filled = (table > POSITION_COUNT_FLOOR).all(axis=0)
    else:
        filled = np.any(means != 0, axis=2).all(axis=0)
    rows = np.flatnonzero(filled)
    return int(rows[-1]) + 1 if rows.size else 0


def load_pca_model(path: Path) -> tuple[F32, F32]:
    """One pooled PCA basis and its mean, as float32, read by :func:`read_pca_basis`.

    Raises
    ------
    FileNotFoundError
        When the file is absent — a caller that asked for a specific PCA by path
        meant that one.
    KeyError
        When the object is a mapping of bases keyed by layer index. A pass that
        loads a model projects onto one basis, so a per-layer basis is refused
        here rather than reduced to whichever layer a key happened to hold; the
        recompute in :mod:`anamnesis.extraction.feature_pipeline` takes that
        shape, and the pooled shape comes from a pooled fit.
    CalibrationMalformed
        When the file holds none of the banked shapes, or a pooled basis with no
        mean to centre on.
    """
    basis = read_pca_basis(path)
    if basis.pooled is None:
        raise KeyError(
            f"{basis.path} holds a basis per layer, keyed {sorted(basis.per_layer)}, and this "
            "reader returns one basis; load_calibration returns either shape, and every "
            "extraction path projects onto both"
        )
    if basis.pooled.mean is None:
        raise CalibrationMalformed(f"{basis.path} holds a basis with no mean to centre on")
    return (
        np.asarray(basis.pooled.components, dtype=np.float32),
        np.asarray(basis.pooled.mean, dtype=np.float32),
    )


Basis = F32 | dict[int, F32]
"""A residual basis as the extraction takes it: one array for every layer, or one per layer."""


def load_calibration(
    calib_dir: Path, enable_pca: bool = True
) -> tuple[F32 | None, Basis | None, Basis | None]:
    """Positional means and, when the residual PCA is on, its basis and mean.

    Returns the triple every feature-computing entry point takes: ``(positional
    means, PCA components, PCA mean)``, each of which is ``None`` when the
    artifact is absent or, for the PCA pair, when it is switched off. The basis
    comes back in the shape it is stored in — one pooled pair of arrays, or a
    mapping from layer index to each layer's own — because every extraction path
    projects onto either.

    The basis is resolved by :func:`resolve_pca_model`, so :data:`PCA_MODEL_NAME`
    wins over the banked spelling beside it when a directory holds both.

    Raises
    ------
    CalibrationMalformed
        When the resolved file holds none of the banked shapes, or a basis that
        stores no mean. A directory that holds no basis at all is the ``None``
        case; one that holds an unusable basis is a refusal, because a pass that
        asked for corrected features would otherwise compute uncorrected ones
        under their name.
    """
    directory = Path(calib_dir)
    positional_means = load_positional_means(directory)
    components: Basis | None = None
    mean: Basis | None = None
    pca_path = resolve_pca_model(directory) if enable_pca else None
    if pca_path is not None:
        basis = read_pca_basis(pca_path)
        components, mean = basis.float32_arrays()
        if mean is None:
            raise CalibrationMalformed(f"{pca_path} holds a basis with no mean to centre on")
        logger.info(f"{basis.format} basis from {pca_path.name}")
    return positional_means, components, mean


def require_positional_means(positional_means: F32 | None, calib_dir: Path) -> F32:
    """The means a command's features are corrected by, or a refusal naming how to get them.

    Every corrected feature subtracts these means. A command that went on without them
    would write uncorrected quantities under the corrected names, and nothing
    downstream could tell, so a missing table stops the command rather than warning.

    Raises
    ------
    CalibrationMissing
        When ``positional_means`` is ``None``.
    """
    if positional_means is None:
        raise CalibrationMissing(
            f"no positional means in {calib_dir}; every corrected feature subtracts them, so "
            f"calibrate this model first with run_calibration"
        )
    return positional_means


class PositionsUncovered(ValueError):
    """A pass would read a position its positional means do not fill."""


def require_positions_covered(
    positional_means: F32 | None,
    last_position: int,
    *,
    what: str,
    counts: I64 | None = None,
) -> None:
    """Refuse a pass whose last read position is past the rows the means fill.

    The correction clamps a position to the table's width, and a row no fit
    reached is zeros, so a position past :func:`positions_calibrated` is
    corrected by nothing and its feature is an uncorrected quantity under the
    corrected name. Nothing to check when there are no means: that pass computes
    uncorrected features and says so where the means are read.

    Raises
    ------
    PositionsUncovered
        Naming the position, the filled extent, and the calibration that would
        cover it.
    """
    if positional_means is None:
        return
    reach = positions_calibrated(positional_means, counts)
    if last_position >= reach:
        raise PositionsUncovered(
            f"{what} reads position {last_position}, and the positional means fill "
            f"positions 0..{reach - 1}; past that the correction subtracts zeros and the "
            f"feature is uncorrected under the corrected name. Calibrate with "
            f"run_calibration --required-through {last_position}, or --reach-from the "
            f"replay manifest of the runs it will correct"
        )


@dataclass(frozen=True)
class Calibration:
    """Both artifacts of a calibration directory, with the digests of the bytes read.

    ``files`` maps each artifact's filename to the SHA-256 of its bytes, and
    ``sha256`` is :func:`anamnesis.provenance.digest_of_shas` over that mapping —
    the same digest :mod:`anamnesis.extraction.fast.runtime` stamps for a
    directory holding the written names. A result stamped with it names the
    correction it was computed under, which a path does not: a directory is
    replaced in place, and two bases of the same shape give features with the same
    names and different values.
    """

    directory: Path
    positional_means: F32
    basis: PCABasis
    files: Mapping[str, str]
    sha256: str

    @property
    def pca_components(self) -> F32 | dict[int, F32]:
        """The basis components as float32, per layer for a per-layer basis."""
        return self.basis.float32_arrays()[0]

    @property
    def pca_mean(self) -> F32 | dict[int, F32]:
        """The basis mean as float32, per layer for a per-layer basis."""
        mean = self.basis.float32_arrays()[1]
        if mean is None:
            raise CalibrationMalformed(f"{self.basis.path} holds a basis with no mean")
        return mean

    def provenance(self) -> dict[str, Any]:
        """The record a result computed under this calibration carries beside it."""
        return {
            "calibration_dir": str(self.directory),
            "calibration_sha256": self.sha256,
            "files": dict(self.files),
            "pca_file": self.basis.path.name,
            "pca_format": self.basis.format,
        }


def load_calibration_strict(calib_dir: Path) -> Calibration:
    """Positional means and basis from ``calib_dir``, refusing anything missing.

    For a pass that has no uncorrected fallback. The basis may be any of the three
    banked shapes and is resolved by :func:`resolve_pca_model`; the digests are
    taken over the files actually read, under the names they were read by.

    Raises
    ------
    CalibrationMissing
        When the directory, the positional means, or a basis under every accepted
        name is absent.
    CalibrationMalformed
        When the means archive holds no ``positional_means`` array, holds one that
        is not indexed ``[layer, position, unit]``, or when the basis is
        unreadable or leaves out a mean.
    """
    directory = Path(calib_dir).expanduser()
    if not directory.is_dir():
        raise CalibrationMissing(f"calibration directory {directory} is not a directory")

    means_path = directory / POSITIONAL_MEANS_NAME
    if not means_path.is_file():
        raise CalibrationMissing(
            f"no positional means at {means_path}; features that subtract them would be wrong"
        )
    try:
        with np.load(means_path) as archive:
            held = sorted(archive.files)
            stored = archive[POSITIONAL_MEANS_KEY] if POSITIONAL_MEANS_KEY in held else None
    except (OSError, ValueError, EOFError) as exc:
        raise CalibrationMalformed(f"{means_path} does not read as an archive: {exc}") from exc
    if stored is None:
        raise CalibrationMalformed(
            f"{means_path} holds no {POSITIONAL_MEANS_KEY!r} array (it holds {held})"
        )
    positional_means: F32 = stored.astype(np.float32)
    if positional_means.ndim != 3:
        raise CalibrationMalformed(
            f"{means_path} holds positional means of shape {positional_means.shape}; "
            "they are indexed [layer, position, unit]"
        )

    pca_path = resolve_pca_model(directory)
    if pca_path is None:
        raise CalibrationMissing(
            f"no PCA model in {directory}; looked for {list(PCA_MODEL_NAMES)}"
        )
    basis = read_pca_basis(pca_path)
    if not basis.has_every_mean:
        raise CalibrationMalformed(f"{pca_path} holds a basis with no mean to centre on")

    files = {means_path.name: file_sha(means_path), pca_path.name: file_sha(pca_path)}
    calibration = Calibration(
        directory=directory,
        positional_means=positional_means,
        basis=basis,
        files=files,
        sha256=digest_of_shas(files),
    )
    logger.info(
        f"calibration {directory}: positional_means {positional_means.shape}, "
        f"{basis.format} basis from {pca_path.name}, sha256 {calibration.sha256}"
    )
    return calibration
