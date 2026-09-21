"""Arithmetic-lane safety for scientific inputs, not equivalence validation.

All-untagged historical inputs remain readable but are not certified as one
known arithmetic backend. They cannot be combined with explicitly tagged data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


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
