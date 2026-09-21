"""Family decomposition with floor-ruler + mass correction (§6b) — Wave-1 implementation.

Generalizes kv-rotation/experiments/exp11_family_decomp.py over feature_map cells:
for each (source × method × dynamic × depth) cell, an arm effect COUNTS only if
≥ k× the matching floor in that cell (floor-ruler), with per-cell feature-mass
correction so big cells can't win by size. Localization speaks feature_map only —
no tier vocabulary. The visibility map records BLINDNESS rows (N4) alongside
carrier rows: "moves symmetrically / fails the ruler" is a result, not a miss.
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
