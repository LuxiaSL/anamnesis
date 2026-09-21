"""Fan a pass out across devices, or across cells, and wait for it.

A pass over a corpus is embarrassingly parallel in the generation it processes:
each generation is seeded from its own coordinates, and each replay is
teacher-forced from its own banked token sequence, so an output never depends on
which worker produced it or in what order. What a launcher does is therefore
arithmetic and process bookkeeping, not science — and the science is what makes
it safe.

Two shapes, and the second is the one that wins.

**Fan-out.** Partition the work round-robin across ``devices x
workers_per_device`` subprocesses, one device per worker, and wait. More than one
worker per device is deliberate: the model forward is cheap next to the feature
arithmetic, so packing workers overlaps one worker's CPU math with another's
device time.

**Multicell, the path of record.** A roster of cells fanned out one cell per
invocation reloads the model once per cell, and at the sizes this instrument runs
that reload dominates. So a worker instead receives *its slice of every cell* in
one job file, loads the model once, and loops. Output is identical either way —
that is what makes the load-once path legitimate rather than a shortcut — and
:mod:`anamnesis.orchestration.gpu`'s guard refuses the per-cell loop that
rediscovers the slow path by accident. :func:`plan_multicell` is the one
partition both the generation and the replay side use; a cell's payload is opaque
here, because what an injection or a routing perturbation *means* is the worker's
business and not the launcher's.

**Assembly.** A generation pass banks one record per generation and a run is
assembled from them afterwards, which is what makes it resumable: a killed pass
leaves every completed record on disk. :func:`assemble_run` writes the two run
artifacts — ``metadata.json`` with its generations, and the replay manifest —
and it writes the manifest through
:mod:`anamnesis.extraction.replay.manifest`, which is the manifest's one home.
A launcher that formats that file itself is a second answer to a question that
already has one.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anamnesis.extraction.replay.manifest import (
    FlaggedGeneration,
    ReplayEntry,
    entry_from_ids,
    manifest_from_entries,
    manifest_path,
    write_replay_manifest,
)
from anamnesis.orchestration.gpu import resolve_physical_gpus, worker_environment

logger = logging.getLogger(__name__)

T = TypeVar("T")

METADATA_NAME = "metadata.json"
"""The run artifact holding the pass's configuration and its generations."""

RECORDS_SUBDIR = "gen_records"
"""Where a generation worker banks one record per generation, before assembly."""


def round_robin(items: Sequence[T], buckets: int) -> list[list[T]]:
    """Deal ``items`` into ``buckets`` lists, in order.

    Item *i* goes to bucket ``i % buckets``, so every bucket's own order is the
    input's and the assignment depends only on the input length. Buckets may be
    empty; a caller skips those rather than spawning idle workers.

    Raises
    ------
    ValueError
        When ``buckets`` is not positive, which would silently drop the work.
    """
    if buckets <= 0:
        raise ValueError(f"buckets must be positive, got {buckets}")
    out: list[list[T]] = [[] for _ in range(buckets)]
    for index, item in enumerate(items):
        out[index % buckets].append(item)
    return out


class LaunchPlan(BaseModel):
    """How many workers, on which devices, logging where.

    Build it with :meth:`resolve`, which translates logical device slots through
    the scheduler's assignment; the constructor takes devices that are already
    physical, which is what a test wants.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    devices: tuple[str, ...] = Field(min_length=1, description="Physical device ids, one per slot")
    workers_per_device: int = Field(gt=0, description="Workers packed onto each device")
    log_dir: Path = Field(description="Directory one log file per worker is written into")

    @classmethod
    def resolve(
        cls,
        devices: str | Iterable[str],
        workers_per_device: int,
        log_dir: Path,
    ) -> LaunchPlan:
        """A plan over logical slots, resolved against ``CUDA_VISIBLE_DEVICES``.

        ``devices`` may be the comma-separated form a command line carries.
        """
        slots = devices.split(",") if isinstance(devices, str) else list(devices)
        wanted = [s.strip() for s in slots if str(s).strip()]
        if not wanted:
            raise ValueError("no device slots given")
        return cls(
            devices=tuple(resolve_physical_gpus(wanted)),
            workers_per_device=workers_per_device,
            log_dir=Path(log_dir),
        )

    @property
    def n_workers(self) -> int:
        return len(self.devices) * self.workers_per_device

    def device_for(self, worker: int) -> str:
        """The device worker ``worker`` is confined to."""
        return self.devices[worker % len(self.devices)]

    def partition(self, items: Sequence[T]) -> list[list[T]]:
        """This plan's share of the work, per worker."""
        return round_robin(items, self.n_workers)

    def log_path(self, worker: int, stem: str) -> Path:
        """Where worker ``worker``'s output is captured."""
        return self.log_dir / f"{stem}_w{worker}_gpu{self.device_for(worker)}.log"


@dataclass
class _Spawned:
    """One live worker: its process, and the log file handle it writes into."""

    worker: int
    device: str
    process: subprocess.Popen
    log_path: Path
    stream: Any


class LaunchResult(BaseModel):
    """What a fan-out did: who ran, who failed, how long it took."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    returncodes: dict[int, int] = Field(description="Exit status per worker index")
    log_paths: dict[int, Path] = Field(description="Where each worker's output landed")
    seconds: float = Field(ge=0.0, description="Wall time from first spawn to last exit")

    @model_validator(mode="after")
    def _one_log_per_worker(self) -> Self:
        if set(self.returncodes) != set(self.log_paths):
            raise ValueError("every worker that ran has exactly one log path")
        return self

    @property
    def n_workers(self) -> int:
        return len(self.returncodes)

    @property
    def failed(self) -> tuple[int, ...]:
        """Workers that exited non-zero, ascending."""
        return tuple(sorted(w for w, rc in self.returncodes.items() if rc != 0))

    @property
    def ok(self) -> bool:
        return not self.failed

    def raise_on_failure(self) -> None:
        """Stop the pass when any worker failed, naming where to look.

        A launcher that returns zero while a worker died reports a complete pass
        over an incomplete corpus, which is the one failure mode a downstream
        contrast cannot detect.

        Raises
        ------
        SystemExit
            When at least one worker exited non-zero.
        """
        if self.ok:
            return
        logs = ", ".join(str(self.log_paths[w]) for w in self.failed)
        raise SystemExit(
            f"{len(self.failed)} of {self.n_workers} workers failed "
            f"(rc={[self.returncodes[w] for w in self.failed]}); see {logs}"
        )


def launch(
    plan: LaunchPlan,
    command_for: Callable[[int], Sequence[str]],
    *,
    stem: str = "worker",
    workers: Iterable[int] | None = None,
    extra_env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> LaunchResult:
    """Spawn one subprocess per worker, wait for all of them, and report.

    ``command_for`` builds a worker's argv; ``workers`` names which worker
    indices to spawn, defaulting to every slot in the plan — a caller that
    partitioned the work first passes only the indices whose share is non-empty.
    Each worker is confined to its own device with single-threaded numeric pools
    (:func:`anamnesis.orchestration.gpu.worker_environment`) and its combined
    output is captured to its own log file.

    Every worker is waited on even after one fails, because a launcher that
    returns while siblings still hold devices leaves the machine in a state the
    next job cannot reason about. The verdict is in the result.
    """
    indices = list(range(plan.n_workers)) if workers is None else list(workers)
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    live: list[_Spawned] = []
    started = time.time()
    for worker in indices:
        device = plan.device_for(worker)
        log_path = plan.log_path(worker, stem)
        stream = open(log_path, "w")
        process = popen(
            list(command_for(worker)),
            env=worker_environment(device, extra=extra_env),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        live.append(_Spawned(worker, device, process, log_path, stream))
        logger.info(f"  worker {worker} on device {device} -> {log_path}")

    returncodes: dict[int, int] = {}
    for spawned in live:
        returncodes[spawned.worker] = int(spawned.process.wait())
        spawned.stream.close()
        if returncodes[spawned.worker] != 0:
            logger.error(
                f"worker {spawned.worker} (device {spawned.device}) exited "
                f"rc={returncodes[spawned.worker]} — see {spawned.log_path}"
            )
    result = LaunchResult(
        returncodes=returncodes,
        log_paths={s.worker: s.log_path for s in live},
        seconds=time.time() - started,
    )
    logger.info(
        f"{result.n_workers - len(result.failed)}/{result.n_workers} workers OK "
        f"in {result.seconds:.0f}s"
    )
    return result


def write_worker_inputs(
    directory: Path, stem: str, payloads: Mapping[int, Any]
) -> dict[int, Path]:
    """Write one JSON input file per worker; return where each landed.

    A worker's share travels as a file rather than as arguments because a slice
    of a corpus does not fit on a command line, and because a file is what a
    resumed pass can be read against.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for worker in sorted(payloads):
        path = directory / f"{stem}_w{worker}.json"
        path.write_text(json.dumps(payloads[worker]))
        out[worker] = path
    return out


class Cell(BaseModel):
    """One cell of a roster: where its output goes, and what its worker needs.

    ``payload`` rides through to the worker untouched. The launcher does not
    interpret it — an injection site, a sampler setting or a routing
    perturbation is a property of the pass the worker runs, and a launcher that
    understood them would have to be changed every time one is added.
    """

    model_config = ConfigDict(extra="forbid")

    target: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields naming this cell's inputs and outputs, merged into every job",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Per-cell fields the worker interprets"
    )


def plan_multicell(
    cells: Sequence[Cell],
    items_for: Callable[[Cell], Sequence[Any]],
    n_workers: int,
    *,
    items_key: str,
) -> dict[int, list[dict[str, Any]]]:
    """Every cell's work, sliced per worker, grouped so each worker loads once.

    Worker *w* receives one job per cell it has a share of, in roster order, and
    each job carries that cell's target fields, its payload, and the worker's
    slice of the cell's items under ``items_key``. A worker with no share of any
    cell is absent from the result rather than present and idle.

    The slice is :func:`round_robin`, so which worker runs which item is fixed by
    the roster alone — and since an item's output does not depend on its worker,
    the assignment is free to be arithmetic.
    """
    jobs: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        shares = round_robin(items_for(cell), n_workers)
        for worker, share in enumerate(shares):
            if not share:
                continue
            jobs.setdefault(worker, []).append(
                {**cell.target, **cell.payload, items_key: list(share)}
            )
    return jobs


@dataclass
class AssembledRun:
    """What assembly found: the generations it covered, and what it could not."""

    n_generations: int
    metadata_path: Path
    manifest_path: Path
    flagged: list[FlaggedGeneration] = field(default_factory=list)


def assemble_run(
    out_run_dir: Path,
    passthrough: Mapping[str, Any],
    *,
    records_subdir: str = RECORDS_SUBDIR,
) -> AssembledRun:
    """Turn a directory of banked generation records into a run.

    Reads every ``gen_*.json`` record in generation-id order and writes two
    files. ``metadata.json`` carries the pass's configuration — whatever
    ``passthrough`` holds — and its generations, with the realized token ids
    stripped out. The replay manifest carries those ids, written through
    :mod:`anamnesis.extraction.replay.manifest` so that the run this pass banked
    and a run whose manifest was reconstructed are the same object on disk.

    A record whose prompt/generated split does not partition a replayable
    sequence — a generation that produced no tokens, for instance — is flagged
    rather than written, because a manifest row that cannot be replayed would
    turn into a failure at replay time with nothing naming its cause.
    """
    run_dir = Path(out_run_dir)
    rec_dir = run_dir / records_subdir
    paths = sorted(rec_dir.glob("gen_*.json"), key=lambda p: int(p.stem.split("_")[1]))

    generations: list[dict[str, Any]] = []
    entries: dict[str, ReplayEntry] = {}
    flagged: list[FlaggedGeneration] = []
    for path in paths:
        record = json.loads(path.read_text())
        gen_id = int(record["generation_id"])
        try:
            entries[str(gen_id)] = entry_from_ids(
                record["input_ids"], int(record["prompt_length"])
            )
        except (KeyError, ValueError) as exc:
            flagged.append(FlaggedGeneration(gen_id=gen_id, reason=f"not replayable: {exc}"))
            logger.warning(f"gen_{gen_id:03d}: excluded from the manifest ({exc})")
            continue
        generations.append({k: v for k, v in record.items() if k != "input_ids"})

    metadata = {
        "total_generations": len(generations),
        "failed_ids": [],
        **dict(passthrough),
        "generations": generations,
    }
    meta_path = run_dir / METADATA_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    written = write_replay_manifest(
        manifest_path(run_dir), manifest_from_entries(entries, flagged)
    )
    logger.info(
        f"assembled {len(generations)} generations -> {meta_path.name} + {written.name} "
        f"({len(entries)} manifest entries, {len(flagged)} flagged)"
    )
    return AssembledRun(
        n_generations=len(generations),
        metadata_path=meta_path,
        manifest_path=written,
        flagged=flagged,
    )
