"""anamnesis.analysis.battery — the one analysis template for every arm × model.

prereg-vmb-v1 §6b. Typed (pydantic) throughout; prereg-locked readouts; no tier
vocabulary in code or output strings; every emitted number carries
(n, M, law, floor-type). Stage 0 ships manifest + floors + stats functional;
deltas / decomp / channel / dissoc / report are typed Wave-1 stubs.
"""
from anamnesis.analysis.battery.manifest import (
    Arm,
    BatteryCell,
    BatteryManifest,
    CellType,
    FloorType,
)
from anamnesis.analysis.battery.floors import (
    ALPHA_GRID,
    FloorCell,
    FloorReport,
    LawParams,
    compute_faithfulness_floors,
    compute_stochastic_floors,
)
from anamnesis.analysis.battery.stats import ResultStamp, StampedValue, bh_fdr, permutation_pvalue

__all__ = [
    "ALPHA_GRID",
    "Arm",
    "BatteryCell",
    "BatteryManifest",
    "CellType",
    "FloorCell",
    "FloorReport",
    "FloorType",
    "LawParams",
    "ResultStamp",
    "StampedValue",
    "bh_fdr",
    "compute_faithfulness_floors",
    "compute_stochastic_floors",
    "permutation_pvalue",
]
