"""anamnesis.analysis.battery — the one analysis template for every arm × model.

prereg-vmb-v1 §6b. Typed (pydantic) throughout; prereg-locked readouts; localization
in feature_map cells, in code and in output strings alike; every emitted number carries
(n, M, law, floor-type). Stage 0 ships manifest + floors + stats functional;
deltas / decomp / channel / dissoc / report are typed Wave-1 stubs.

Two modules are the protocols the metrology is collected and read through:
``stage0`` is the faithfulness-replay plan and the floors-to-n-min law table —
the floors are battery metrology, so the protocol that collects them lives here
rather than in an entry point — and ``census`` is the class object over banked arm
records, with the judge-defense gate it cannot be produced without. Both are
addressed by module rather than re-exported here, because a caller wants the
protocol by name.
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
