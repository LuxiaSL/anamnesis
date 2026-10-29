"""Arithmetic-lane safety for scientific inputs, not equivalence validation.

All-untagged historical inputs remain readable but are not certified as one
known arithmetic backend. They cannot be combined with explicitly tagged data.

The gate has two widths, and they are the same rule read at two distances.
:func:`require_single_lane` takes the metadata mappings a caller already holds.
:func:`gate_banked_signatures` takes the ``gen_NNN.npz`` paths a reader is about
to stack and finds the mappings itself, in the ``gen_NNN.json`` sidecar that
`anamnesis/extraction/feature_pipeline.py` writes beside every vector. Every
reader that turns a bank into a matrix calls the second one, because a matrix is
a join and a join across lanes is the failure this module exists to prevent.

Bypassing the gate is possible and has to be named: ``allow_mixed_lanes=True``
downgrades the refusal to a logged warning and reports no lane. The default is
the gate, so a reader that says nothing gets it.

A row computed under a registry row that binds its weights carries a weights record
(:mod:`anamnesis.extraction.weights_binding`). A lane id does not name the weights, so
one lane id over two weights digests is refused like two lanes.

Declared contrasts
------------------
Two contrasts across lanes are admitted, each only when the caller declares it, and
each returns a :class:`ContrastRecord` that a report carries:

* :class:`LineageContrast` — same tokens, different weights in a recorded lineage.
  Every row carries a weights record; the records' presets trace to one common
  ancestor through the registry's ``parent`` links; each set of weights is computed
  under one lane; and the declaration names the calibration policy: ``"own"`` (each
  member under its own calibration) or ``"reference:<calibration sha256>"`` (every
  member under that one).
* :class:`BridgeContrast` — same weights, different computors (a replay lane beside a
  decode engine's lane). Every row carries a weights record and all name one digest.
  The declaration supplies the measured bridge floor between the lanes and where it
  was measured, and an effect counts only beyond that floor
  (:meth:`ContrastRecord.relative_to_floor`).

Different weights under a bridge declaration are refused, and so is one set of
weights under two lanes in a lineage declaration: each is the other mode's question.
:func:`require_declared_contrast` takes metadata mappings and
:func:`gate_declared_contrast` takes banked signature paths, as the ordinary gate does.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anamnesis.config.models import ModelRegistry, UnknownPresetError, load_registry

logger = logging.getLogger(__name__)


class MixedLaneError(ValueError):
    """A scientific input combines incompatible or missing lane identities."""


DRAWN_LANE_MARKER = "synthetic"
"""The mark a lane identity carries when its numbers are drawn rather than measured.

A bank of drawn vectors is the same shape as a measured one, so nothing downstream can
tell them apart except the lane identity — which is why a drawn bank carries this mark
and a reading over one can say what it is about. `anamnesis/synthetic_bank.py` writes
the one such bank this package produces.
"""


def lane_is_drawn(lane_id: str | None) -> bool:
    """Whether a lane identity says its numbers came from a generator.

    A reader that reports a verdict over drawn vectors has to state that the verdict
    is about the generator. ``None`` — an untagged bank — is not drawn: an absent lane
    is unknown provenance, which :func:`require_single_lane` treats as its own case.
    """
    return lane_id is not None and DRAWN_LANE_MARKER in lane_id.lower()


def metadata_lane(metadata: Mapping[str, Any]) -> str | None:
    values = []
    if "lane_id" in metadata:
        values.append(metadata["lane_id"])
    nested = metadata.get("extraction_lane")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise MixedLaneError("extraction_lane must be a metadata mapping")
        if "lane_id" in nested:
            values.append(nested["lane_id"])
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise MixedLaneError("empty or malformed arithmetic lane ID")
    if len(set(values)) > 1:
        raise MixedLaneError("conflicting arithmetic lane IDs within one signature")
    return values[0] if values else None


WEIGHTS_KEY = "weights"
"""Where a row's weights record sits: in its ``extraction_lane`` mapping or at the top
level of its metadata (:mod:`anamnesis.extraction.weights_binding` writes it)."""


def metadata_weights(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The weights record one row carries, or None for a row that carries none.

    Raises
    ------
    MixedLaneError
        When a record is malformed, or the row carries two that disagree.
    """
    records = []
    if WEIGHTS_KEY in metadata:
        records.append(metadata[WEIGHTS_KEY])
    nested = metadata.get("extraction_lane")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise MixedLaneError("extraction_lane must be a metadata mapping")
        if WEIGHTS_KEY in nested:
            records.append(nested[WEIGHTS_KEY])
    for record in records:
        if not isinstance(record, Mapping) or not all(
            isinstance(record.get(field), str) and record[field]
            for field in ("preset", "checkpoint_sha256")
        ):
            raise MixedLaneError(
                "malformed weights record; it names a preset and a checkpoint_sha256"
            )
    if len({(r["preset"], r["checkpoint_sha256"]) for r in records}) > 1:
        raise MixedLaneError("conflicting weights records within one signature")
    return records[0] if records else None


def require_single_lane(metadata: Sequence[Mapping[str, Any]]) -> str | None:
    lanes = {metadata_lane(m) for m in metadata}
    if len(lanes) > 1:
        raise MixedLaneError(
            "mixed arithmetic lanes (including tagged/untagged inputs); do not combine them inside a scientific contrast"
        )
    checkpoints = {
        record["checkpoint_sha256"]
        for record in (metadata_weights(m) for m in metadata)
        if record is not None
    }
    if len(checkpoints) > 1:
        raise MixedLaneError(
            f"one arithmetic lane over {len(checkpoints)} weights digests "
            f"({', '.join(sorted(checkpoints))}); different weights are not one lane's rows, "
            "and comparing them is a declared LineageContrast"
        )
    lane = next(iter(lanes), None)
    if lane is not None:
        for field in ("calibration_sha256", "feature_schema_sha256"):
            values = [(m.get("extraction_lane") or {}).get(field) for m in metadata]
            if any(v is not None and (not isinstance(v, str) or not v) for v in values):
                raise MixedLaneError(f"malformed {field}")
            if len(set(values)) > 1:
                raise MixedLaneError(f"inconsistent {field} within arithmetic lane")
    return lane


def sidecar_path(signature: Path) -> Path:
    """The per-generation metadata json beside one banked signature vector."""
    return signature.with_suffix(".json")


def _sidecar_metadata(
    signatures: Sequence[Path],
) -> tuple[list[Mapping[str, Any]], list[Path]]:
    """Readable sidecar mappings, and the sidecars that yielded none.

    A sidecar is unreadable when it is absent, unparseable, or not a mapping. All
    three leave the lane of that vector unknown, which is a different thing from
    untagged, so they are counted rather than defaulted.
    """
    metadata: list[Mapping[str, Any]] = []
    unknown: list[Path] = []
    for signature in signatures:
        path = sidecar_path(signature)
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            unknown.append(path)
            continue
        if not isinstance(document, Mapping):
            unknown.append(path)
            continue
        metadata.append(document)
    return metadata, unknown


def gate_banked_signatures(
    signatures: Sequence[Path], *, allow_mixed_lanes: bool = False
) -> str | None:
    """The arithmetic lane the given banked signature vectors share.

    Parameters
    ----------
    signatures
        Paths of the ``gen_NNN.npz`` vectors that are about to be stacked into
        one matrix. The lane lives in the ``gen_NNN.json`` sidecar beside each.
    allow_mixed_lanes
        Accept a bank the gate would refuse. The conflict is logged at error
        level and the return is ``None``, because a set of rows with no single
        lane has no lane to report.

    Returns
    -------
    The shared lane id, or ``None`` for an all-untagged historical bank and for
    a refusal the caller chose to accept.

    Raises
    ------
    MixedLaneError
        When the rows carry more than one lane, when a tagged bank contains rows
        whose lane cannot be read, or when two rows in one lane were normalized
        against different calibration or feature schemas. A tagged bank with an
        unreadable sidecar refuses because the unreadable row may belong to
        another lane; an all-untagged bank with one does not, because there is no
        lane for it to contaminate.
    """
    metadata, unknown = _sidecar_metadata(signatures)
    try:
        lane = require_single_lane(metadata)
        if lane is not None and unknown:
            raise MixedLaneError(
                f"lane {lane} declared, but {len(unknown)} of {len(signatures)} "
                "signatures carry no readable lane metadata"
            )
    except MixedLaneError as error:
        if not allow_mixed_lanes:
            raise
        logger.error(f"reading {len(signatures)} signatures across lanes anyway: {error}")
        return None
    return lane


CALIBRATION_OWN = "own"
"""Lineage calibration policy: every member under its own calibration."""

CALIBRATION_REFERENCE_PREFIX = "reference:"
"""Lineage calibration policy prefix: every member under the calibration whose sha256
follows it."""

_REFERENCE = re.compile(r"^reference:([0-9a-f]{64})$")


class LineageContrast(BaseModel):
    """A declared contrast across weights that stand in a recorded lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["lineage"] = "lineage"
    calibration: str = Field(
        description=(
            "'own' (each member under its own calibration) or 'reference:<sha256>' "
            "(every member under the calibration with that calibration_sha256)"
        )
    )

    @field_validator("calibration")
    @classmethod
    def _policy(cls, value: str) -> str:
        if value != CALIBRATION_OWN and not _REFERENCE.match(value):
            raise ValueError(
                f"calibration policy {value!r} is neither {CALIBRATION_OWN!r} nor "
                f"'{CALIBRATION_REFERENCE_PREFIX}<64 hex calibration_sha256>'"
            )
        return value

    @property
    def reference_calibration(self) -> str | None:
        """The one calibration every member reads under, or None for ``own``."""
        match = _REFERENCE.match(self.calibration)
        return match.group(1) if match else None


class BridgeContrast(BaseModel):
    """A declared contrast across computors of the same weights, against a measured floor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["bridge"] = "bridge"
    floor: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="The measured distance between the two lanes on identical tokens",
    )
    floor_provenance: str = Field(
        min_length=1,
        description="Where the floor was measured: the lanes, the tokens, the receipt",
    )


DeclaredContrast = LineageContrast | BridgeContrast
"""What a caller declares to join rows across lanes."""


class ContrastMember(BaseModel):
    """One lane and set of weights inside a declared contrast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane_id: str
    preset: str
    checkpoint_sha256: str
    adapter_sha256: str | None = None
    calibration_sha256: str | None
    feature_schema_sha256: str | None
    rows: int


class ContrastRecord(BaseModel):
    """What a declared contrast admitted, for a report to carry beside its effects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["lineage", "bridge"]
    members: tuple[ContrastMember, ...]
    calibration: str | None = Field(
        default=None, description="Lineage: the declared calibration policy"
    )
    root: str | None = Field(
        default=None, description="Lineage: the common ancestor's registry key"
    )
    ancestry: dict[str, tuple[tuple[str, str], ...]] | None = Field(
        default=None,
        description=(
            "Lineage: each member preset's chain to the root, as (registry key, lineage "
            "kind of that key's link to its parent) pairs; the root's kind is 'root'"
        ),
    )
    bridge_floor: float | None = Field(default=None, description="Bridge: the declared floor")
    bridge_floor_provenance: str | None = Field(
        default=None, description="Bridge: where the floor was measured"
    )

    def relative_to_floor(self, effect: float) -> float:
        """A bridge contrast's effect as a multiple of its declared floor.

        Raises
        ------
        ValueError
            For a lineage contrast, which declares no floor.
        """
        if self.bridge_floor is None:
            raise ValueError("a lineage contrast declares no bridge floor")
        return effect / self.bridge_floor


def _member_rows(
    metadata: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """Rows grouped by (lane id, weights digest), refusing a row that carries either not."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in metadata:
        lane = metadata_lane(row)
        if lane is None:
            raise MixedLaneError("an untagged row cannot enter a declared contrast")
        weights = metadata_weights(row)
        if weights is None:
            raise MixedLaneError(
                f"a row on lane {lane} carries no weights record; a declared contrast reads "
                "rows computed under a preset that binds its weights (checkpoint_sha256)"
            )
        groups.setdefault((lane, weights["checkpoint_sha256"]), []).append(row)
    return groups


def _member(rows: list[Mapping[str, Any]]) -> ContrastMember:
    """One member's rows as a single lane, and what it is."""
    lane = require_single_lane(rows)
    assert lane is not None
    records = {(r["preset"], r.get("adapter_sha256")): r
               for r in (metadata_weights(m) for m in rows) if r is not None}
    if len(records) > 1:
        raise MixedLaneError(
            f"lane {lane} names one weights digest under presets "
            f"{sorted(str(p) for p, _ in records)}; a member is one registry row"
        )
    record = next(iter(records.values()))
    nested = rows[0].get("extraction_lane") or {}
    return ContrastMember(
        lane_id=lane,
        preset=record["preset"],
        checkpoint_sha256=record["checkpoint_sha256"],
        adapter_sha256=record.get("adapter_sha256"),
        calibration_sha256=nested.get("calibration_sha256"),
        feature_schema_sha256=nested.get("feature_schema_sha256"),
        rows=len(rows),
    )


def _same_ruler(members: Sequence[ContrastMember],
                groups: Mapping[tuple[str, str], list[Mapping[str, Any]]]) -> None:
    """Members whose vectors name the same coordinates, computed by the same method."""
    schemas = {m.feature_schema_sha256 for m in members}
    if len(schemas) > 1:
        raise MixedLaneError(
            "the members emit different feature schemas; their columns are not one quantity"
        )
    methods = {
        (row.get("extraction_lane") or {}).get("eff_dim", "svd")
        for rows in groups.values()
        for row in rows
    }
    if len(methods) > 1:
        raise MixedLaneError(
            f"the members compute the effective dimension differently ({sorted(methods)})"
        )


def _lineage(contrast: LineageContrast, members: Sequence[ContrastMember],
             registry: ModelRegistry) -> ContrastRecord:
    lanes_by_weights: dict[str, set[str]] = {}
    for member in members:
        lanes_by_weights.setdefault(member.checkpoint_sha256, set()).add(member.lane_id)
    for checkpoint, lanes in lanes_by_weights.items():
        if len(lanes) > 1:
            raise MixedLaneError(
                f"weights {checkpoint} appear under lanes {sorted(lanes)}; one set of weights "
                "under two computors is a declared BridgeContrast, not a lineage contrast"
            )
    if len(lanes_by_weights) < 2:
        raise MixedLaneError(
            "a lineage contrast compares at least two sets of weights; these rows carry one"
        )
    chains: dict[str, tuple[str, ...]] = {}
    for member in members:
        try:
            row = registry.resolve(member.preset)
        except UnknownPresetError as error:
            raise MixedLaneError(
                f"lane {member.lane_id} names preset {member.preset!r}, which the registry "
                "does not hold, so its lineage cannot be traced"
            ) from error
        if row.checkpoint_sha256 != member.checkpoint_sha256:
            raise MixedLaneError(
                f"lane {member.lane_id} records weights {member.checkpoint_sha256} under "
                f"preset {member.preset!r}, which binds {row.checkpoint_sha256}"
            )
        chains[row.name] = registry.ancestry(row.name)
    roots = {chain[-1] for chain in chains.values()}
    if len(roots) > 1:
        raise MixedLaneError(
            f"presets {sorted(chains)} do not trace to one another through recorded parent "
            f"links: their lineages end at {sorted(roots)}"
        )
    reference = contrast.reference_calibration
    calibrations = [m.calibration_sha256 for m in members]
    if reference is not None:
        stray = sorted({c for c in calibrations if c != reference}, key=str)
        if stray:
            raise MixedLaneError(
                f"the contrast declares reference calibration {reference}, and members read "
                f"under {stray}"
            )
    else:
        if any(c is None for c in calibrations):
            raise MixedLaneError("a member under 'own' calibration records no calibration_sha256")
        if len(set(calibrations)) != len(calibrations):
            raise MixedLaneError(
                "two members read under one calibration while the contrast declares 'own'; "
                "declare 'reference:<sha256>' when one calibration serves every member"
            )
    ancestry = {
        key: tuple((link, registry.presets[link].lineage or "root") for link in chain)
        for key, chain in chains.items()
    }
    return ContrastRecord(kind="lineage", members=tuple(members),
                          calibration=contrast.calibration, root=roots.pop(),
                          ancestry=ancestry)


def _bridge(contrast: BridgeContrast, members: Sequence[ContrastMember]) -> ContrastRecord:
    checkpoints = {m.checkpoint_sha256 for m in members}
    if len(checkpoints) > 1:
        raise MixedLaneError(
            f"a bridge contrast joins lanes of one weights digest, and these rows carry "
            f"{len(checkpoints)}; comparing different weights is a declared LineageContrast"
        )
    return ContrastRecord(kind="bridge", members=tuple(members),
                          bridge_floor=contrast.floor,
                          bridge_floor_provenance=contrast.floor_provenance)


def require_declared_contrast(
    metadata: Sequence[Mapping[str, Any]],
    contrast: DeclaredContrast,
    *,
    registry: ModelRegistry | None = None,
) -> ContrastRecord:
    """Admit rows across lanes under a declared contrast, or refuse them.

    Parameters
    ----------
    metadata
        Each row's metadata mapping, carrying its ``lane_id`` and its weights record.
    contrast
        The declaration: a :class:`LineageContrast` or a :class:`BridgeContrast`.
    registry
        Where a lineage contrast traces ``parent`` links; the merged registry when
        omitted. A bridge contrast does not read it.

    Returns
    -------
    The record of what was admitted: every member's lane, preset, weights and
    calibration, and the lineage traced or the bridge floor declared.

    Raises
    ------
    MixedLaneError
        When a row is untagged or carries no weights record; when one member's rows
        disagree on calibration or schema; when members emit different feature
        schemas or effective-dimension methods; when the rows hold a single member;
        for a lineage contrast, when one set of weights spans two lanes, a recorded
        digest disagrees with the registry, the presets do not trace to a common
        ancestor, or the calibrations break the declared policy; for a bridge
        contrast, when the rows carry more than one weights digest.
    """
    if not isinstance(contrast, (LineageContrast, BridgeContrast)):
        raise TypeError(f"not a declared contrast: {type(contrast).__name__}")
    if not metadata:
        raise MixedLaneError("a declared contrast over no rows")
    groups = _member_rows(metadata)
    members = [_member(rows) for rows in groups.values()]
    if len(members) < 2:
        raise MixedLaneError(
            "these rows are one lane under one set of weights; an ordinary contrast needs "
            "no declaration"
        )
    _same_ruler(members, groups)
    if isinstance(contrast, BridgeContrast):
        if not math.isfinite(contrast.floor):
            raise MixedLaneError("the bridge floor is not finite")
        return _bridge(contrast, members)
    return _lineage(contrast, members, registry if registry is not None else load_registry())


def gate_declared_contrast(
    signatures: Sequence[Path],
    contrast: DeclaredContrast,
    *,
    registry: ModelRegistry | None = None,
) -> ContrastRecord:
    """:func:`require_declared_contrast` over banked ``gen_NNN.npz`` paths.

    Raises
    ------
    MixedLaneError
        When a sidecar is unreadable, since its row's lane and weights are unknown;
        otherwise as :func:`require_declared_contrast`.
    """
    metadata, unknown = _sidecar_metadata(signatures)
    if unknown:
        raise MixedLaneError(
            f"{len(unknown)} of {len(signatures)} signatures carry no readable metadata, "
            "so their lanes and weights are unknown"
        )
    return require_declared_contrast(metadata, contrast, registry=registry)
