"""Workers that stay loaded, and a file queue that survives losing them.

A fan-out pays one model load per subprocess. When the load is the expensive part
— and at the checkpoint sizes this instrument reaches, it is — the answer is a
worker that loads once and then serves jobs until told to stop. The queue is
plain files, which is what makes the arrangement crash-safe: a job file is
unlinked only *after* its result has landed, so a fleet that dies leaves its
remaining work on disk and respawning workers over the same directory continues
exactly where it stopped. Every write is a temporary file followed by a rename,
so a reader never sees a half-written job or a half-written result.

The protocol, under one work directory on node-local disk::

    jobs/w<ID>/job_<TAG>.json     driver -> worker, consumed after its result lands
    results/w<ID>_job_<TAG>.npz   worker -> driver
    ready/w<ID>                   worker readiness marker
    STOP                          drain marker; workers exit at the next poll

Nothing about the queue knows what a job is: the worker side takes a handler over
a parsed job dictionary and returns arrays to save. Driver-side state — a search's
own checkpoint, say — stays the driver's.

:class:`ReplayJob` and :func:`replay_handler` are the job type this instrument
runs on it. The handler calls :func:`anamnesis.extraction.replay.cell.replay_cell`
— the same function the one-shot command runs — so a signature produced through
the queue is identical to one produced without it, and
:func:`signature_mismatches` is how that is checked rather than asserted. A
handler that raises is fatal for its worker, deliberately: a silently skipped job
would leave the driver waiting forever, and leaving the job file in place is what
lets a respawned fleet retry it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.extraction.interventions import resolve_injection, attach_injection
from anamnesis.extraction.replay.cell import (
    DEFAULT_SIGNATURES_SUBDIR,
    ReplaySurface,
    replay_cell,
)
from anamnesis.orchestration.gpu import worker_environment

logger = logging.getLogger(__name__)

DEFAULT_POLL_S = 0.05
STOP_MARKER = "STOP"
READY_DIR = "ready"
JOBS_DIR = "jobs"
RESULTS_DIR = "results"


@dataclass
class PersistentWorker:
    """Poll one worker's job directory until STOP; one result per job, atomically.

    ``handler`` consumes the parsed job and returns the arrays to save. ``on_stop``
    runs after the drain, which is where a worker detaches whatever it armed.
    """

    work_dir: Path
    worker_id: int
    handler: Callable[[dict], Mapping[str, np.ndarray]]
    poll_s: float = DEFAULT_POLL_S
    on_stop: Callable[[], None] | None = None

    @property
    def jobs_dir(self) -> Path:
        return self.work_dir / JOBS_DIR / f"w{self.worker_id}"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / RESULTS_DIR

    def mark_ready(self) -> None:
        """Announce that this worker has loaded and is polling."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ready = self.work_dir / READY_DIR
        ready.mkdir(exist_ok=True)
        (ready / f"w{self.worker_id}").touch()

    def run(self) -> int:
        """Process jobs until STOP appears; return how many completed."""
        self.mark_ready()
        logger.info(f"worker {self.worker_id} READY ({self.jobs_dir})")
        n_done = 0
        try:
            while not (self.work_dir / STOP_MARKER).exists():
                jobs = sorted(self.jobs_dir.glob("job_*.json"))
                if not jobs:
                    time.sleep(self.poll_s)
                    continue
                job_file = jobs[0]
                tag = job_file.stem[len("job_"):]
                job = json.loads(job_file.read_text())
                arrays = self.handler(job)
                out = self.results_dir / f"w{self.worker_id}_job_{tag}.npz"
                tmp = out.with_suffix(".tmp.npz")
                np.savez(tmp, **arrays)
                tmp.rename(out)
                job_file.unlink()  # consume only once the result has landed
                n_done += 1
        finally:
            if self.on_stop is not None:
                self.on_stop()
        logger.info(f"worker {self.worker_id} STOP after {n_done} jobs")
        return n_done


@dataclass
class WorkerFleet:
    """Driver side: spawn, dispatch, collect, drain."""

    work_dir: Path
    worker_ids: Sequence[int]
    procs: list[subprocess.Popen] = field(default_factory=list)

    def clear_stale_state(self) -> None:
        """Remove what a previous, crashed fleet left behind on this directory.

        A stale STOP kills fresh workers on their first poll, a stale ready marker
        impersonates a worker that is not there, and a stale result corrupts a
        collect. Pending *job* files are kept: they are the resume state.
        """
        (self.work_dir / STOP_MARKER).unlink(missing_ok=True)
        for directory, pattern in ((READY_DIR, "w*"), (RESULTS_DIR, "*.npz")):
            base = self.work_dir / directory
            if base.is_dir():
                for path in base.glob(pattern):
                    path.unlink()

    def spawn(
        self,
        cmd_for_worker: Callable[[int], list[str]],
        gpu_for_worker: Callable[[int], str] | None = None,
        log_dir: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> list[subprocess.Popen]:
        """Start one subprocess per worker id, each pinned to its own device."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.clear_stale_state()
        logs = log_dir or (self.work_dir / "logs")
        logs.mkdir(parents=True, exist_ok=True)
        for worker in self.worker_ids:
            device = None if gpu_for_worker is None else gpu_for_worker(worker)
            stream = open(logs / f"worker_{worker}.log", "w")
            self.procs.append(
                popen(
                    cmd_for_worker(worker),
                    env=worker_environment(device, extra=extra_env),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )
        return self.procs

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        """Block until every worker has marked itself ready.

        Raises
        ------
        SystemExit
            When a worker exits before announcing readiness — which is a failed
            model load, not a slow one — or when the timeout passes.
        """
        started = time.time()
        while True:
            ready = [
                (self.work_dir / READY_DIR / f"w{worker}").exists()
                for worker in self.worker_ids
            ]
            if all(ready):
                return
            dead = [p for p in self.procs if p.poll() is not None]
            if dead:
                raise SystemExit(
                    f"{len(dead)} workers exited before READY "
                    f"(rc={[p.returncode for p in dead]}); see the worker logs"
                )
            if time.time() - started > timeout_s:
                raise SystemExit(f"workers not ready after {timeout_s}s: {ready}")
            time.sleep(0.5)

    def submit(self, worker_id: int, tag: str, payload: dict) -> Path:
        """Place one job atomically; return the result path to wait on."""
        job_file = self.work_dir / JOBS_DIR / f"w{worker_id}" / f"job_{tag}.json"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = job_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.rename(job_file)
        return self.work_dir / RESULTS_DIR / f"w{worker_id}_job_{tag}.npz"

    def collect(
        self,
        expected: Sequence[Path],
        timeout_s: float = 3600.0,
        poll_s: float = DEFAULT_POLL_S,
        consume: bool = True,
    ) -> dict[Path, dict[str, np.ndarray]]:
        """Wait for every expected result and read it.

        Raises
        ------
        SystemExit
            When the whole fleet has exited with results still pending, or when
            the timeout passes — naming what is missing, since a collect that
            returned short would be read as a complete pass.
        """
        pending = list(expected)
        out: dict[Path, dict[str, np.ndarray]] = {}
        started = time.time()
        while pending:
            done = [p for p in pending if p.exists()]
            for path in done:
                with np.load(path, allow_pickle=False) as archive:
                    out[path] = {key: archive[key] for key in archive.files}
                if consume:
                    path.unlink()
                pending.remove(path)
            if pending:
                dead = [p for p in self.procs if p.poll() is not None]
                if dead and len(dead) == len(self.procs):
                    raise SystemExit(
                        f"all workers exited with {len(pending)} results pending; "
                        f"see the logs under {self.work_dir / 'logs'}"
                    )
                if time.time() - started > timeout_s:
                    raise SystemExit(
                        f"collect timed out after {timeout_s}s; missing: "
                        f"{[str(p) for p in pending[:5]]}"
                    )
                time.sleep(poll_s)
        return out

    def stop(self, timeout_s: float = 120.0) -> None:
        """Drain the fleet, idempotently: signal STOP and reap what remains."""
        (self.work_dir / STOP_MARKER).touch()
        for process in self.procs:
            try:
                process.wait(timeout=timeout_s)
            except Exception:  # noqa: BLE001 — a stuck worker must not block teardown
                process.kill()
        self.procs = []


class ReplayJob(BaseModel):
    """One cell for a resident replay worker to run.

    The same fields a multi-cell job file carries, so a roster can be dispatched
    through the queue or handed to a load-once process without being rewritten.
    """

    model_config = ConfigDict(extra="allow")

    run_dir: Path = Field(description="The cell's output directory, holding its metadata")
    manifest: Path = Field(description="The manifest whose token sequences are replayed")
    gen_ids: list[int] | None = Field(
        default=None, description="This job's slice of the cell, or all of it"
    )
    sig_subdir: str = Field(
        default=DEFAULT_SIGNATURES_SUBDIR, description="Where the signatures land in the cell"
    )
    no_resume: bool = Field(
        default=False, description="Recompute a signature that is already on disk"
    )
    inject_from_metadata: bool = Field(
        default=False, description="Read the write from the cell's own run metadata"
    )

    def injection_fields(self) -> dict[str, Any]:
        """The explicit ``inject_*`` fields this job carries, if any."""
        extra = self.model_extra or {}
        return {k: v for k, v in extra.items() if k.startswith("inject_")}


def replay_handler(
    surface: ReplaySurface,
    calibration: tuple[Any, Any, Any],
    *,
    worker_id: int = 0,
    logits_top_k: int = 50,
) -> tuple[Callable[[dict], Mapping[str, np.ndarray]], Callable[[], None]]:
    """A handler that replays one cell per job, plus the cleanup to pass as ``on_stop``.

    The model and the calibration are loaded once by the caller and closed over.
    Each job re-arms its own intervention after removing the previous one, so
    writes never stack across cells, and raw tensors are not banked: a resident
    worker's job is signatures.

    A job with any failed generation raises, which kills the worker and leaves the
    job file queued. That is the intended behaviour — a partially replayed cell
    that reported success would enter a bank short.
    """
    state: dict[str, Any] = {"handle": None}
    label = f"w{worker_id}"

    def handle(payload: dict) -> Mapping[str, np.ndarray]:
        if state["handle"] is not None:
            state["handle"].remove()
            state["handle"] = None
        job = ReplayJob.model_validate(payload)
        injection = resolve_injection(
            job.run_dir,
            from_metadata=job.inject_from_metadata,
            fields=job.injection_fields(),
        )
        state["handle"] = attach_injection(surface.loaded, injection, label)
        result = replay_cell(
            surface,
            calibration,
            job.run_dir,
            job.manifest,
            gen_ids=job.gen_ids,
            signatures_subdir=job.sig_subdir,
            save_raw=False,
            resume=not job.no_resume,
            logits_top_k=logits_top_k,
            write_handle=state["handle"],
            injection=injection,
            label=label,
        )
        if not result.ok:
            raise RuntimeError(
                f"{result.n_failed} generations failed in {job.run_dir}; failing loud so "
                "the job stays queued for retry"
            )
        return {
            "n_done": np.array([result.n_done]),
            "run_dir": np.array([str(job.run_dir)]),
        }

    def cleanup() -> None:
        if state["handle"] is not None:
            state["handle"].remove()
            state["handle"] = None

    return handle, cleanup


def dispatch(
    fleet: WorkerFleet,
    jobs: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 3600.0,
) -> int:
    """Submit one job per item round-robin over a ready fleet, and collect.

    Returns the total the workers reported done. Which worker gets which job does
    not affect any output, so the dispatch is arithmetic.
    """
    n_workers = len(fleet.worker_ids)
    expected = [
        fleet.submit(index % n_workers, f"{index:05d}", dict(job))
        for index, job in enumerate(jobs)
    ]
    collected = fleet.collect(expected, timeout_s=timeout_s)
    return sum(int(arrays["n_done"][0]) for arrays in collected.values())


def signature_mismatches(
    left: Path, right: Path, gen_ids: Sequence[int], suffixes: Sequence[str] = ("npz", "json")
) -> list[str]:
    """Names of the signature files that are not byte-identical between two dirs.

    This is the parity check every load-once path is allowed on the strength of:
    the fast arrangement is legitimate exactly when its output cannot be
    distinguished from the plain one. A file missing on either side counts as a
    mismatch, so a leg that silently produced nothing cannot read as agreement.
    """
    mismatched: list[str] = []
    for gen_id in gen_ids:
        for suffix in suffixes:
            name = f"gen_{gen_id:03d}.{suffix}"
            a, b = Path(left) / name, Path(right) / name
            if not (a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()):
                mismatched.append(name)
    return mismatched
