"""Stage 0: the protocol that measures a box's floors, and the law it yields.

Before any arm runs, two questions have answers or nothing else means anything:
**how large a difference counts**, and **how many samples it takes to see one**.
Stage 0 answers them in two passes over one model.

* The **stochastic floor** asks how far two signatures drift apart when nothing
  about the computation was meant to differ except the sampler's own randomness.
  Its corpus is the floor run: a fixed set of continuations, many seeds each.
* The **faithfulness floor** asks how far a signature drifts from *itself* — the
  same continuation, teacher-forced through the same weights, replayed. That
  number should be zero, and on replay-deterministic hardware it is exactly
  zero; the pass exists to find out, on this box, rather than to assume it.

The faithfulness pass is stratified, because there are two ways a replay can
differ and they are not the same finding: repeats on **one** device measure
replay determinism, and repeats spread across **other** devices measure
operational jitter. Ten replays per continuation, four pinned and six spread, is
what separates them — and the same stratification is why the plan is an object
here rather than a loop in a launcher: a signature has to be traceable back to
the device and the component it belongs to, which is what ``replay_index.json``
records and :func:`~anamnesis.analysis.battery.floors.compute_faithfulness_floors`
reads.

The replays are ordinary replays: the plan writes a **synthetic manifest** whose
ten entries per continuation all carry the same banked token ids, so the replay
command needs no notion of this protocol and the protocol needs no replay code of
its own. Fanning those entries out across devices is
:mod:`anamnesis.orchestration.launch`'s job, which is why nothing here spawns a
process.

This module is the protocol and the law table. The floors themselves — the cells,
the deltas, the n-min arithmetic — are :mod:`anamnesis.analysis.battery.floors`,
and that is the seam: floors *are* battery metrology, so the protocol that
collects them lives beside them rather than in an entry point.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anamnesis.analysis.battery.floors import (
    FloorReport,
    LawParams,
    compute_faithfulness_floors,
    compute_stochastic_floors,
    load_signature_matrix,
    robust_scale,
)

logger = logging.getLogger(__name__)

N_TOPICS = 20
"""Continuations in the faithfulness set: one per topic of the floor corpus."""

TOPICS_PER_STRATUM = 5
"""Topics a task stratum covers. The stratum a topic belongs to is arithmetic —
``topic_idx // TOPICS_PER_STRATUM`` — so the selection is reproducible from the
topic index alone and needs no table."""

SEEDS_PER_CLASS = 10
"""Seeds the floor corpus banks per class, which is the stride of its gid layout."""

N_REPLAYS = 10
"""Replays per continuation: the two components together."""

N_PINNED = 4
"""Replays of each continuation that stay on one device — the within-device
component. The remaining ``N_REPLAYS - N_PINNED`` spread over the other devices
and measure cross-device jitter."""

WITHIN = "within"
CROSS = "cross"

MANIFEST_NAME = "replay_manifest.json"
INDEX_NAME = "replay_index.json"
SIGNATURES_SUBDIR = "signatures_v3"


def floor_gid(topic_idx: int, *, seed_idx: int = 0) -> int:
    """The floor corpus's generation id for one topic's seed.

    The floor run lays its generations out as ``(stratum · N_TOPICS + topic) ·
    SEEDS_PER_CLASS + seed``, so seed 0 of each topic — the continuation this
    protocol replays — is addressable without reading the corpus.
    """
    if not 0 <= topic_idx < N_TOPICS:
        raise ValueError(f"topic_idx must be in [0, {N_TOPICS}), got {topic_idx}")
    stratum = topic_idx // TOPICS_PER_STRATUM
    return (stratum * N_TOPICS + topic_idx) * SEEDS_PER_CLASS + seed_idx


def select_continuations(floor_manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """One continuation per topic, taken from the floor run's own manifest.

    The choice is deterministic rather than sampled: seed 0 of every topic, which
    spreads the twenty continuations evenly over the four task strata. A manifest
    missing one of those ids is an error and not a smaller set, because a floor
    measured over a different number of continuations is a different floor.
    """
    entries = floor_manifest["entries"]
    selected: dict[int, dict[str, Any]] = {}
    for topic_idx in range(N_TOPICS):
        gid = floor_gid(topic_idx)
        entry = entries.get(str(gid))
        if entry is None:
            stratum = topic_idx // TOPICS_PER_STRATUM
            raise KeyError(
                f"floor manifest has no generation {gid} "
                f"(topic {topic_idx}, stratum {stratum})"
            )
        selected[topic_idx] = dict(entry)
    return selected


class ReplayInstance(BaseModel):
    """One replay of one continuation: where it ran, and which component it is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gen_id: int = Field(ge=0, description="Id this replay carries in the synthetic manifest")
    continuation_id: int = Field(ge=0, description="Which continuation was replayed")
    replay_idx: int = Field(ge=0, description="Which repeat of that continuation")
    device: str = Field(min_length=1, description="Physical device the replay ran on")
    component: str = Field(description="`within` for the pinned device, `cross` otherwise")

    @model_validator(mode="after")
    def _known_component(self) -> Self:
        if self.component not in (WITHIN, CROSS):
            raise ValueError(f"component must be {WITHIN!r} or {CROSS!r}, got {self.component!r}")
        return self

    @property
    def signature_name(self) -> str:
        """The signature file's stem, which is the key the replay index is read by."""
        return f"gen_{self.gen_id:03d}"

    def index_row(self) -> dict[str, Any]:
        """This instance as `replay_index.json` carries it.

        The device is written with its ``gpu`` prefix because the index is read as
        a label, not as a device to open — two replays agree on their device when
        their labels are equal, and nothing downstream reopens one.
        """
        return {
            "sig": self.signature_name,
            "continuation_id": self.continuation_id,
            "replay_idx": self.replay_idx,
            "device": f"gpu{self.device}",
            "component": self.component,
        }


class StratifiedReplayPlan(BaseModel):
    """Every replay the faithfulness pass will run, and the manifest it reads.

    ``entries`` is the synthetic manifest: each replay instance gets its own id
    and its continuation's banked ``input_ids``, so ten entries describe ten
    replays of one sequence and a replay command sees nothing unusual.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instances: tuple[ReplayInstance, ...] = Field(min_length=1)
    entries: dict[str, dict[str, Any]] = Field(description="Synthetic manifest, keyed by gen id")
    pinned_device: str = Field(min_length=1)
    spread_devices: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_entry_per_instance(self) -> Self:
        if {str(i.gen_id) for i in self.instances} != set(self.entries):
            raise ValueError("every replay instance has exactly one manifest entry")
        return self

    @property
    def n_continuations(self) -> int:
        return len({i.continuation_id for i in self.instances})

    def gen_ids_by_device(self) -> dict[str, list[int]]:
        """Which generation ids each device is responsible for, ascending."""
        out: dict[str, list[int]] = {}
        for instance in self.instances:
            out.setdefault(instance.device, []).append(instance.gen_id)
        return {device: sorted(ids) for device, ids in sorted(out.items())}

    def manifest_document(self) -> dict[str, Any]:
        """The manifest as written: the same shape a generation pass banks."""
        return {
            "entries": self.entries,
            "n_ok": len(self.entries),
            "n_flagged": 0,
            "flagged": [],
        }

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        """Bank the manifest and the index; return where each landed."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / MANIFEST_NAME
        index_path = out_dir / INDEX_NAME
        manifest_path.write_text(json.dumps(self.manifest_document()), encoding="utf-8")
        index_path.write_text(
            json.dumps([i.index_row() for i in self.instances], indent=1), encoding="utf-8"
        )
        logger.info(
            f"{self.n_continuations} continuations x {N_REPLAYS} replays "
            f"= {len(self.entries)} instances -> {out_dir}"
        )
        return manifest_path, index_path


def plan_stratified_replays(
    continuations: Mapping[int, Mapping[str, Any]],
    *,
    pinned_device: str,
    spread_devices: Sequence[str],
    n_replays: int = N_REPLAYS,
    n_pinned: int = N_PINNED,
) -> StratifiedReplayPlan:
    """Lay out the replays: ``n_pinned`` on one device, the rest round-robin.

    A replay instance's id is ``continuation · n_replays + replay_idx``, so the id
    says which continuation and which repeat it is without consulting the index —
    the index exists for the device and the component, which arithmetic cannot
    recover.

    Raises
    ------
    ValueError
        When the pinned device also appears among the spread devices, which would
        report cross-device jitter measured on one device, or when the counts
        leave no replays for one of the two components.
    """
    if not continuations:
        raise ValueError("no continuations to replay")
    if not 0 < n_pinned < n_replays:
        raise ValueError(
            f"n_pinned must leave replays for both components: got {n_pinned} of {n_replays}"
        )
    spread = [str(d) for d in spread_devices]
    if not spread:
        raise ValueError("the cross-device component needs at least one spread device")
    if str(pinned_device) in spread:
        raise ValueError(
            f"device {pinned_device} is both pinned and spread — the cross-device "
            f"component would be measured within one device"
        )

    instances: list[ReplayInstance] = []
    entries: dict[str, dict[str, Any]] = {}
    for continuation_id, source in sorted(continuations.items()):
        for replay_idx in range(n_replays):
            gen_id = continuation_id * n_replays + replay_idx
            pinned = replay_idx < n_pinned
            instances.append(
                ReplayInstance(
                    gen_id=gen_id,
                    continuation_id=continuation_id,
                    replay_idx=replay_idx,
                    device=str(pinned_device) if pinned else spread[(replay_idx - n_pinned) % len(spread)],
                    component=WITHIN if pinned else CROSS,
                )
            )
            entries[str(gen_id)] = dict(source)
    return StratifiedReplayPlan(
        instances=tuple(instances),
        entries=entries,
        pinned_device=str(pinned_device),
        spread_devices=tuple(spread),
    )


def law_table_md(reports: Sequence[FloorReport], model: str) -> str:
    """The law table a planner reads: per cell, the floor and the n it implies.

    One section per floor, one row per cell. ``PLAN`` is the conservative reading
    — the larger of the σ-based and MAD-based n at α=0.05 — because a floor
    distribution with heavy tails makes the σ-based number optimistic, and a cell
    planned too small cannot be rescued after the fact. A cell whose floor is
    bitwise zero has no n to compute: its row says so, and its arm's n is set by
    the effect side instead.
    """
    lines = [
        f"# Stage-0 law table — {model}",
        "",
        "Battery n per cell = 2 x n_min (4x for A2_instruction_vs_execution cells).",
        "alpha_test = 0.05 / m, with m = the confirmatory cell count declared for that arm",
        "before it runs.",
        "Shift reading: an arm sits at k=2x the floor median, so the effect it must resolve",
        "is (k-1) * median / sigma_floor. A rank test needs n / 0.955 of these.",
        "",
    ]
    for report in reports:
        lines.append(
            f"## {report.floor_type.value}  (n_gens={report.n_gens}, "
            f"pairs={report.n_pairs_total}, M={report.model})"
        )
        lines.append("")
        lines.append(
            "| cell | n_feat | floor median | sigma | sigma_rob | d | d_rob | "
            + " | ".join(f"n_min@alpha={a}" for a in report.law.alpha_grid)
            + " | n_min_rob@0.05 | PLAN n@0.05 |"
        )
        lines.append("|" + "---|" * (9 + len(report.law.alpha_grid)))
        for cell in sorted(report.cells, key=lambda c: c.cell):
            if cell.exact_zero:
                blank = " | ".join("—" for _ in report.law.alpha_grid)
                lines.append(
                    f"| {cell.cell} | {cell.n_features} | 0 (EXACT) | 0 | 0 | inf | inf | "
                    f"{blank} | — | EXACT* |"
                )
                continue
            ns = " | ".join(str(cell.n_min_by_alpha[str(a)]) for a in report.law.alpha_grid)
            plan = max(cell.n_min_by_alpha["0.05"], cell.n_min_by_alpha_robust["0.05"])
            lines.append(
                f"| {cell.cell} | {cell.n_features} | {cell.median:.4f} | {cell.std:.4f} | "
                f"{cell.mad:.4f} | {cell.effect_d:.2f} | {cell.effect_d_robust:.2f} | {ns} | "
                f"{cell.n_min_by_alpha_robust['0.05']} | {plan} |"
            )
        lines.append("")
        lines.append(
            "PLAN column = max(sigma-based, MAD-based) n_min at alpha=0.05 — conservative "
            "under heavy tails; battery n = 2x PLAN (4x for A2_instruction_vs_execution)."
        )
        lines.append("")
        lines.append(
            "*EXACT = the floor distribution is bitwise ZERO on replay-deterministic "
            "hardware (identical signature digests across devices and repeats). Any "
            "nonzero matched-token delta then exceeds the floor, the cell's "
            "n is set by the effect side (dose-ladder resolution), and the matched-token "
            "prediction sharpens to literal bitwise equality."
        )
        lines.append("")
    return "\n".join(lines)


class Stage0Law(BaseModel):
    """What a law pass produced: the floor reports, and where they were written."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    reports: list[FloorReport]
    report_paths: list[Path]
    law_table_path: Path

    @property
    def stochastic(self) -> FloorReport:
        """The stochastic report, which every other floor is standardized against."""
        return self.reports[0]


def compute_stage0_law(
    *,
    model: str,
    n_layers: int,
    floor_sig_dir: Path,
    floor_metadata: Path,
    out_dir: Path,
    faith_sig_dir: Path | None = None,
    faith_index: Path | None = None,
    law: LawParams | None = None,
) -> Stage0Law:
    """Floors to n-min law for one model, with every artifact banked beside it.

    The faithfulness deltas are standardized on the **stochastic** corpus's scale,
    not their own: the two floors are compared to each other, and a z computed
    against two different rulers is not a comparison. That is why the stochastic
    pass runs first here rather than being a separate call a caller could skip.
    """
    law = law or LawParams()
    out_dir = Path(out_dir)
    reports: list[FloorReport] = []
    paths: list[Path] = []

    stochastic = compute_stochastic_floors(
        Path(floor_sig_dir), Path(floor_metadata), model=model, n_layers=n_layers, law=law
    )
    stochastic_path = out_dir / f"floors_stochastic_{model}.json"
    stochastic.save(stochastic_path)
    reports.append(stochastic)
    paths.append(stochastic_path)

    if faith_sig_dir is not None and faith_index is not None:
        X, _names, _ids = load_signature_matrix(Path(floor_sig_dir))
        scale = robust_scale(X)
        for report in compute_faithfulness_floors(
            Path(faith_sig_dir),
            Path(faith_index),
            model=model,
            n_layers=n_layers,
            law=law,
            scale_from=scale,
        ):
            path = out_dir / f"floors_{report.floor_type.value}_{model}.json"
            report.save(path)
            reports.append(report)
            paths.append(path)

    table_path = out_dir / f"law_table_{model}.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(law_table_md(reports, model), encoding="utf-8")
    logger.info(f"law table -> {table_path}")
    return Stage0Law(reports=reports, report_paths=paths, law_table_path=table_path)


def whole_vector_reading(report: FloorReport) -> str:
    """The one-line headline of a floor: the whole vector's cell, with its n.

    Every number a floor is quoted by travels with the n it rests on and the law
    it was computed under, which is why this is a function rather than a print
    statement in an entry point.
    """
    cells = [c for c in report.cells if c.cell == "whole_vector"]
    if not cells:
        raise ValueError(f"{report.floor_type.value}: no whole_vector cell in the report")
    cell = cells[0]
    if cell.exact_zero:
        return (
            f"[{report.model}] whole-vector {report.floor_type.value} floor: EXACT ZERO "
            f"(n={cell.n_pairs} pairs, M={report.model})"
        )
    return (
        f"[{report.model}] whole-vector {report.floor_type.value} floor: "
        f"median={cell.median:.4f} d={cell.effect_d:.2f} "
        f"n_min@0.05={cell.n_min_by_alpha['0.05']} "
        f"@1e-4={cell.n_min_by_alpha['0.0001']} "
        f"(n={cell.n_pairs} pairs, M={report.model}, law k={report.law.k} "
        f"power={report.law.power}, floor={report.floor_type.value})"
    )


__all__ = [
    "CROSS",
    "INDEX_NAME",
    "MANIFEST_NAME",
    "N_PINNED",
    "N_REPLAYS",
    "N_TOPICS",
    "SEEDS_PER_CLASS",
    "SIGNATURES_SUBDIR",
    "TOPICS_PER_STRATUM",
    "WITHIN",
    "ReplayInstance",
    "Stage0Law",
    "StratifiedReplayPlan",
    "compute_stage0_law",
    "floor_gid",
    "law_table_md",
    "plan_stratified_replays",
    "select_continuations",
    "whole_vector_reading",
]
