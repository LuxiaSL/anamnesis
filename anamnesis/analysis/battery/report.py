"""Typed per-cell results + map-level rollup (§6b) — Wave-1 implementation.

Raw-artifacts-next-to-claims: every reported cell carries the path of the raw
artifact backing it. The rollup is the visibility map itself: per arm × model,
which cells carry, which are blind (N4 rows), the channel column, and the
dissociation column — each number stamped (n, M, law, floor-type).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.analysis.battery.channel import ChannelSplit
from anamnesis.analysis.battery.decomp import CellVerdict
from anamnesis.analysis.battery.dissoc import DissociationRow
from anamnesis.analysis.battery.manifest import BatteryCell


class CellResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell: BatteryCell
    verdicts: list[CellVerdict]
    channel: ChannelSplit | None = None
    dissociation: DissociationRow | None = None
    raw_artifacts: list[str] = Field(default_factory=list)


class VisibilityMap(BaseModel):
    """The map-level rollup — the paper's Part-I object."""

    cells: list[CellResult] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
