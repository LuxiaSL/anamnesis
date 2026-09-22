"""Family decomposition with a floor-ruler and a mass correction.

For each (source × method × dynamic × depth) cell, an arm effect COUNTS only if it
is ≥ k× the matching floor in that cell (the floor-ruler), with a per-cell
feature-mass correction so a big cell cannot win by size alone. Localization speaks
feature_map only: a cell, never a coarse feature block. The visibility map records
BLINDNESS rows (``N4_family_level``) beside carrier rows, because "moves
symmetrically" and "fails the ruler" are results rather than misses.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.floors import FloorReport
from anamnesis.analysis.battery.stats import StampedValue


class CellVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell: str                      # feature_map cell key
    effect: StampedValue           # floor-ruled, mass-corrected effect
    passes_ruler: bool
    ruler_k: float
    confirmatory: bool             # True = counted in the law's m; False = exploratory


def decompose(
    deltas: object,
    floor: FloorReport,
    ruler_k: float = 2.0,
) -> list[CellVerdict]:
    raise NotImplementedError("Wave-1: family decomposition w/ floor-ruler (prereg §6b decomp.py)")
