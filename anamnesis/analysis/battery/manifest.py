"""Battery manifest — the typed registry a prediction block compiles into.

One analysis template for every arm × model. A BatteryCell is the unit of the
visibility map: (arm, model, dose, cell type), each declaring the floor it is ruled
against. Localization speaks feature_map (source × method
× dynamic × depth); a coarse feature block never stands in for a cell.

Every emitted number downstream carries (n, M, law, floor-type); the manifest is
where those stamps originate.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Arm(str, Enum):
    """The perturbation classes (``A``-prefixed) and the nulls (``N``) read beside them.

    A null is built to move nothing, so a signature that moves under one is
    measuring the instrument.
    """

    A1_sampling = "A1_sampling"
    A2_instruction_vs_execution = "A2_instruction_vs_execution"
    A3_processing_strategy = "A3_processing_strategy"
    A4_state_surgery = "A4_state_surgery"
    A5_activation_write = "A5_activation_write"
    A5_inv_map_sourced = "A5_inv_map_sourced"
    A6_weight_delta = "A6_weight_delta"
    A7_routing_perturbation = "A7_routing_perturbation"
    N1_context_prefix = "N1_context_prefix"
    N1b_unexecuted_instruction = "N1b_unexecuted_instruction"
    N3_wrong_channel = "N3_wrong_channel"       # runs inside A6_weight_delta
    N4_family_level = "N4_family_level"          # localization-row blindness inside visible arms
    N5_source_side_dose_zero = "N5_source_side_dose_zero"  # also inside A6_weight_delta
    stage0_floor = "stage0_floor"                # the floors themselves (not an arm)


class CellType(str, Enum):
    free_gen = "free_gen"
    matched_token = "matched_token"   # replay channel
    floor = "floor"
    null = "null"


class FloorType(str, Enum):
    """The two floor designs, with faithfulness split by where its replays ran."""

    stochastic = "stochastic"                     # matched-history, different-seed pairs
    faithfulness = "faithfulness"                 # replay-vs-replay identity (pooled)
    faithfulness_within_device = "faithfulness_within_device"   # pinned-device repeats
    faithfulness_cross_device = "faithfulness_cross_device"     # operational jitter


class ModelMeta(BaseModel):
    """Analyzer-side metadata per onboarded model label.

    `label` is the model's preset key, and what an arm corpus directory is named
    after (vmb_a1_{label}_{dose}). `stage0_dir` is the banked Stage-0 run name,
    which spells some labels differently (qwen7b, not qwen-7b) because those
    directories exist on disk and are never renamed.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    n_layers: int
    stage0_dir: str          # outputs/battery/<stage0_dir> holds the model's floors
    native_temperature: float


MODEL_META: dict[str, ModelMeta] = {
    "3b": ModelMeta(label="3b", n_layers=28, stage0_dir="vmb_stage0_3b",
                    native_temperature=0.7),
    "8b": ModelMeta(label="8b", n_layers=32, stage0_dir="vmb_stage0_8b",
                    native_temperature=0.6),
    "qwen-7b": ModelMeta(label="qwen-7b", n_layers=28, stage0_dir="vmb_stage0_qwen7b",
                         native_temperature=0.7),
    "olmo2-7b": ModelMeta(label="olmo2-7b", n_layers=32,
                          stage0_dir="vmb_stage0_olmo2_7b", native_temperature=0.7),
    "gemma3-27b": ModelMeta(label="gemma3-27b", n_layers=62,
                            stage0_dir="vmb_stage0_gemma3_27b", native_temperature=1.0),
    "dsv2-lite": ModelMeta(label="dsv2-lite", n_layers=27,
                           stage0_dir="vmb_stage0_dsv2_lite", native_temperature=0.3),
}


class BatteryCell(BaseModel):
    """One (arm × model × dose × channel) cell of the visibility map."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    model: str                                  # preset key: "3b", "8b", "qwen-7b", ...
    cell_type: CellType
    floor_type: FloorType
    dose: Optional[str] = None                  # e.g. "T=0.9", "alpha=2", "evict=0.5"; None for floors
    description: str = ""
    n_planned: Optional[int] = Field(
        default=None,
        description="Sample count for this cell: law_multiplier × the Stage-0 law's n_min.",
    )
    law_multiplier: float = Field(
        default=2.0,
        description="Battery n as a multiple of the Stage-0 law n_min (A2 cells: 4.0).",
    )
    confirmatory_cells: Optional[list[str]] = Field(
        default=None,
        description=(
            "The confirmatory family-cells declared for this arm (feature_map keys). "
            "THESE and only these count toward the law's m; every other decomposition "
            "cell is exploratory / hypothesis-generating."
        ),
    )

    def cell_id(self) -> str:
        dose = self.dose or "-"
        return f"{self.arm.value}|{self.model}|{self.cell_type.value}|{dose}"


class BatteryManifest(BaseModel):
    """The registry a prediction block compiles into. Duplicate cell_ids are rejected."""

    cells: list[BatteryCell] = Field(default_factory=list)

    def add(self, cell: BatteryCell) -> None:
        if any(c.cell_id() == cell.cell_id() for c in self.cells):
            raise ValueError(f"duplicate battery cell: {cell.cell_id()}")
        self.cells.append(cell)

    def by_arm(self, arm: Arm) -> list[BatteryCell]:
        return [c for c in self.cells if c.arm == arm]

    def by_model(self, model: str) -> list[BatteryCell]:
        return [c for c in self.cells if c.model == model]

    def confirmatory_m(self) -> int:
        """Total declared confirmatory cell count → the law's Bonferroni-style m.

        Counts (cell × confirmatory family-cell) pairs across the manifest. Cells
        with no confirmatory_cells contribute 0, so an exploratory-only cell can
        never inflate m.
        """
        return sum(len(c.confirmatory_cells or []) for c in self.cells)
