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
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MixedLaneError(ValueError):
    """A scientific input combines incompatible or missing lane identities."""


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


def require_single_lane(metadata: Sequence[Mapping[str, Any]]) -> str | None:
    lanes = {metadata_lane(m) for m in metadata}
    if len(lanes) > 1:
        raise MixedLaneError(
            "mixed arithmetic lanes (including tagged/untagged inputs); do not combine them inside a scientific contrast"
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
