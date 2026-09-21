"""Banked trajectories to path-signature design matrices, and the null beside them.

A path signature is computed over a trajectory: the sequence of states a generation
passed through, projected to a handful of coordinates. The projection is expensive
and belongs to a calibration, so it happens once and the result is banked — a
``paths_*.npz`` holding every generation's ``[T, k]`` trajectory, ragged-packed into
one array with row offsets.

This module is the marshaller between that bank and the family that computes the
numbers. Every value it produces comes out of
:mod:`anamnesis.extraction.feature_families.path_signature`; what it adds is the
loading, the stacking, and one decision the family cannot make for a bank: a
trajectory too short for the requested level is **dropped and counted**, never
zero-filled, so the caller sees the denominator it actually has.

The basis handed to the family is the identity on the already-projected
coordinates. That is not a way around the family's rule against fitting a basis —
the real basis is the banked per-layer calibration, applied upstream and recorded in
the bank's sidecar. Routing through the family's own basis object is what keeps the
time-augmentation, the integration and the permutation null identical to the
family's, rather than a second implementation that agrees by inspection.

The **increment-permutation null** is the control that makes a level-2 result
readable: shuffling the increments of a path preserves its endpoint and its level-1
signature exactly while destroying the order the level-2 terms are about. Several
seeds of it on the *real* paths are the comparison a level-2 number is read against.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.extraction.feature_families.path_signature import (
    PathSignatureConfig,
    ProjectionBasis,
    ShortPathError,
    feature_names,
    signature_features_from_path,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

MIN_NULL_SEEDS = 3
"""Seeds the null is computed over, at minimum. One shuffle is a draw, not a null."""

DEFAULT_BASIS_LABEL = "pcaA"
"""What the banked projection is called in the family's own vocabulary, recorded on
every feature name so a matrix cannot be read as one built under another basis."""


class PathBank(BaseModel):
    """A loaded trajectory bank: variable-length ``[T, k]`` paths, ragged-packed.

    The packing is one array plus offsets rather than a list of arrays because the
    bank is written once and read many times, and a single contiguous array is what
    makes a read of one path a slice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    paths: Any = Field(description="[sum_T, k] float32, concatenated in gen_ids order")
    offsets: Any = Field(description="[n+1] int64 row offsets into paths")
    gen_ids: Any = Field(description="[n] int64 generation ids, the bank's order")
    lengths: Any = Field(description="[n] int64 path lengths")
    prompt_lengths: Any = Field(description="[n] int64 prompt lengths")
    source: str = ""
    variant: str = "pc"

    @property
    def n(self) -> int:
        return int(np.asarray(self.gen_ids).shape[0])

    @property
    def k_max(self) -> int:
        """The banked rank: the widest projection a caller can ask for."""
        return int(np.asarray(self.paths).shape[1])

    def path(self, index: int) -> F64:
        """One generation's trajectory, contiguous and in float64."""
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.ascontiguousarray(np.asarray(self.paths)[start:stop].astype(np.float64))

    @classmethod
    def load(cls, npz_path: Path | str, variant: Literal["pc", "nopc"] = "pc") -> PathBank:
        """Read a bank, checking that its offsets describe its own rows.

        The two variants are the projected trajectory and the unprojected one; a bank
        holding only one of them refuses the other by name rather than returning an
        empty array.
        """
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"path bank not found: {path}")
        with np.load(path) as data:
            key = "paths_pc" if variant == "pc" else "paths_nopc"
            if key not in data.files:
                raise KeyError(f"{path}: no {key!r} (has {sorted(data.files)})")
            bank = cls(
                paths=np.asarray(data[key]),
                offsets=np.asarray(data["offsets"], dtype=np.int64),
                gen_ids=np.asarray(data["gen_ids"], dtype=np.int64),
                lengths=np.asarray(data["lengths"], dtype=np.int64),
                prompt_lengths=np.asarray(data["prompt_lengths"], dtype=np.int64),
                source=str(path),
                variant=variant,
            )
        if int(bank.offsets[-1]) != int(np.asarray(bank.paths).shape[0]):
            raise ValueError(
                f"{path}: offsets end at {int(bank.offsets[-1])} but the bank holds "
                f"{int(np.asarray(bank.paths).shape[0])} rows"
            )
        if int(np.asarray(bank.offsets).shape[0]) != bank.n + 1:
            raise ValueError(f"{path}: {bank.n} paths need {bank.n + 1} offsets")
        return bank

    @classmethod
    def from_paths(
        cls,
        paths: list[NDArray[Any]],
        gen_ids: list[int],
        prompt_lengths: list[int] | None = None,
        source: str = "",
        variant: str = "pc",
    ) -> PathBank:
        """Pack a list of trajectories into a bank, computing the offsets."""
        if not paths:
            raise ValueError("a bank needs at least one path")
        if len(gen_ids) != len(paths):
            raise ValueError(f"{len(paths)} paths but {len(gen_ids)} generation ids")
        lengths = [int(np.asarray(p).shape[0]) for p in paths]
        offsets = np.zeros(len(paths) + 1, dtype=np.int64)
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        return cls(
            paths=np.concatenate([np.asarray(p, dtype=np.float32) for p in paths], axis=0),
            offsets=offsets,
            gen_ids=np.asarray(gen_ids, dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.int64),
            prompt_lengths=np.asarray(
                prompt_lengths if prompt_lengths is not None else [0] * len(paths),
                dtype=np.int64,
            ),
            source=source,
            variant=variant,
        )


def identity_basis(k: int, label: str = DEFAULT_BASIS_LABEL) -> ProjectionBasis:
    """The identity on already-projected coordinates, labelled with the real basis."""
    return ProjectionBasis(components=np.eye(k, dtype=np.float64), mean=None, label=label)


class PathDesignMatrix(BaseModel):
    """A design matrix over a bank, with the paths it covers and the ones it dropped.

    Named for what it is rather than for the family that filled it, because
    :class:`anamnesis.analysis.audit_lib.SignatureMatrix` is a different object — a
    merged signature corpus — and one name for both would be a trap.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    X: Any = Field(description="[n_kept, P] float32 design matrix")
    names: list[str]
    kept: list[int] = Field(description="Indices into the bank of the paths that entered")
    n_dropped: int = Field(ge=0, description="Paths too short for the requested level")

    @property
    def n_kept(self) -> int:
        return len(self.kept)


def signature_matrix(
    bank: PathBank,
    *,
    layer: int,
    k: int,
    level: Literal[1, 2],
    time_augment: bool = True,
    permutation_seed: int | None = None,
) -> PathDesignMatrix:
    """Stack one design matrix over every path in a bank.

    A path too short for the requested level is dropped rather than imputed, and the
    count comes back with the matrix: the family refuses a short path by design, and
    the refusal is caught once here, at the layer where the caller can see how many
    rows it lost.
    """
    if k > bank.k_max:
        raise ValueError(f"k={k} exceeds the banked rank {bank.k_max} ({bank.source})")
    config = PathSignatureConfig(
        layer_indices=(layer,),
        n_components=k,
        level=level,
        time_augment=time_augment,
        permute_increments=permutation_seed is not None,
        permutation_seed=permutation_seed,
        basis_label=DEFAULT_BASIS_LABEL,
    )
    basis = identity_basis(k)
    rows: list[F64] = []
    kept: list[int] = []
    dropped = 0
    for index in range(bank.n):
        try:
            features, _length = signature_features_from_path(
                bank.path(index)[:, :k], basis, config
            )
        except ShortPathError:
            dropped += 1
            continue
        rows.append(features)
        kept.append(index)
    if not rows:
        raise RuntimeError(
            f"every path in {bank.source} was too short for level {level} "
            f"({dropped} dropped)"
        )
    if dropped:
        logger.info(f"{bank.source}: {dropped}/{bank.n} paths dropped as too short")
    return PathDesignMatrix(
        X=np.stack(rows).astype(np.float32),
        names=list(feature_names(config)),
        kept=kept,
        n_dropped=dropped,
    )


def null_matrices(
    bank: PathBank,
    *,
    layer: int,
    k: int,
    level: Literal[1, 2],
    seeds: tuple[int, ...],
    time_augment: bool = True,
) -> list[PathDesignMatrix]:
    """The increment-permutation null over the real paths, one matrix per seed.

    The null is computed on the banked trajectories rather than on synthetic ones,
    because what it has to hold constant is everything about the path except the
    order of its increments.
    """
    if len(seeds) < MIN_NULL_SEEDS:
        raise ValueError(
            f"the null takes at least {MIN_NULL_SEEDS} shuffle seeds per cell; got {len(seeds)}"
        )
    return [
        signature_matrix(
            bank, layer=layer, k=k, level=level,
            time_augment=time_augment, permutation_seed=int(seed),
        )
        for seed in seeds
    ]


__all__ = [
    "DEFAULT_BASIS_LABEL",
    "MIN_NULL_SEEDS",
    "PathBank",
    "PathDesignMatrix",
    "identity_basis",
    "null_matrices",
    "signature_matrix",
]
