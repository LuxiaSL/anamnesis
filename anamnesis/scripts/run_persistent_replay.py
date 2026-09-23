"""Replay a roster through workers that stay loaded, and prove they agree.

``run_replay.py`` fanned out pays one model load per worker. That is the right
trade for one roster; it is the wrong one for a driver that dispatches cells as it
decides on them, or for a checkpoint whose load is measured in minutes. Here the
workers load once and then serve jobs from a file queue until told to stop, so a
campaign's dispatch policy is free of its load cost, and a fleet that dies leaves
its remaining work on disk for the next one.

Three modes:

* ``--drive`` with ``--cells-json`` — spawn the fleet, submit one job per cell,
  collect, drain.
* the default — *be* a worker: load once, poll ``--work-dir`` for jobs, exit on
  STOP. This is what ``--drive`` spawns; it is rarely run by hand.
* ``--parity`` with ``--parity-cell`` — the gate that licenses all of it. Replay a
  few of one cell's generations through a one-worker fleet and again through
  ``run_replay.py``, into different signature directories, and compare the files
  byte for byte. Agreement is the claim that the queue is a scheduling decision
  and not a different experiment, and it is checked rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from anamnesis.config import preset_names

MODULE = "anamnesis.scripts.run_persistent_replay"
REPLAY_MODULE = "anamnesis.scripts.run_replay"
PARITY_STANDARD_SUBDIR = "signatures_parity_direct"
PARITY_QUEUE_SUBDIR = "signatures_parity_queue"

logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(preset_names()), required=True)
    p.add_argument("--model-path", required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True, help="Queue root, on node-local disk")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--drive", action="store_true", help="Spawn a fleet and dispatch a roster")
    p.add_argument("--cells-json", type=Path, default=None, help="The roster to dispatch")
    p.add_argument("--gpus", default="0", help="Device slots for the fleet, comma-separated")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--sig-subdir", default="signatures_v3")
    p.add_argument("--parity", action="store_true", help="Run the byte-identity gate and stop")
    p.add_argument(
        "--parity-cell", type=Path, default=None, help="A banked cell to run the gate over"
    )
    p.add_argument("--gen-ids", type=int, nargs="+", default=[0, 1, 10, 11])
    p.add_argument(
        "--inject-from-metadata",
        action="store_true",
        help="Cells read their write from their own metadata",
    )
    return p


def _resident_worker_command(args: argparse.Namespace, worker: int) -> list[str]:
    return [
        sys.executable, "-m", MODULE,
        "--model", args.model, "--model-path", args.model_path,
        "--calib-dir", str(args.calib_dir), "--work-dir", str(args.work_dir),
        "--worker-id", str(worker),
    ]


def _surface(args: argparse.Namespace):
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.replay.cell import load_replay_model

    surface = load_replay_model(resolve_preset(args.model), args.model_path)
    return surface, load_calibration(args.calib_dir, enable_pca=True)


def serve(args: argparse.Namespace) -> None:
    """Load once, then run one cell per job until STOP."""
    from anamnesis.orchestration.workers import PersistentWorker, replay_handler

    logger.info(f"[w{args.worker_id}] loading once: {args.model_path}")
    surface, calibration = _surface(args)
    handle, cleanup = replay_handler(surface, calibration, worker_id=args.worker_id)
    PersistentWorker(
        work_dir=args.work_dir,
        worker_id=args.worker_id,
        handler=handle,
        on_stop=cleanup,
    ).run()


def drive(args: argparse.Namespace) -> None:
    """Spawn the fleet, dispatch every cell, collect, drain."""
    from anamnesis.orchestration.gpu import resolve_physical_gpus
    from anamnesis.orchestration.workers import WorkerFleet, dispatch

    if args.cells_json is None:
        raise SystemExit("--drive needs --cells-json")
    cells = json.loads(args.cells_json.read_text())["cells"]
    devices = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(devices) * args.workers_per_gpu
    fleet = WorkerFleet(work_dir=args.work_dir, worker_ids=list(range(n_workers)))
    fleet.spawn(
        lambda w: _resident_worker_command(args, w),
        gpu_for_worker=lambda w: devices[w % len(devices)],
    )
    fleet.wait_ready()
    logger.info(f"{n_workers} replay workers resident; no reloads from here")
    try:
        total = dispatch(fleet, cells, timeout_s=args.timeout_s)
        logger.info(f"{len(cells)} cells done, {total} generations replayed")
    finally:
        fleet.stop()


def parity(args: argparse.Namespace) -> None:
    """Replay one cell both ways and compare the signatures byte for byte."""
    from anamnesis.orchestration.workers import WorkerFleet, dispatch, signature_mismatches

    if args.parity_cell is None:
        raise SystemExit("--parity needs --parity-cell")
    cell = args.parity_cell
    manifest = cell / "replay_manifest.json"
    direct = [
        sys.executable, "-m", REPLAY_MODULE,
        "--model", args.model, "--model-path", args.model_path,
        "--calib-dir", str(args.calib_dir), "--run-dir", str(cell),
        "--manifest", str(manifest), "--no-raw", "--no-resume",
        "--sig-subdir", PARITY_STANDARD_SUBDIR,
        "--gen-ids", *[str(g) for g in args.gen_ids],
    ]
    if args.inject_from_metadata:
        direct.append("--inject-from-metadata")
    if subprocess.run(direct).returncode != 0:
        raise SystemExit("the direct leg failed, so there is nothing to compare against")

    fleet = WorkerFleet(work_dir=args.work_dir, worker_ids=[0])
    fleet.spawn(
        lambda w: _resident_worker_command(args, w),
        gpu_for_worker=lambda w: args.gpus.split(",")[0].strip(),
    )
    fleet.wait_ready()
    try:
        dispatch(
            fleet,
            [{
                "run_dir": str(cell),
                "manifest": str(manifest),
                "gen_ids": list(args.gen_ids),
                "sig_subdir": PARITY_QUEUE_SUBDIR,
                "no_resume": True,
                "inject_from_metadata": bool(args.inject_from_metadata),
            }],
            timeout_s=args.timeout_s,
        )
    finally:
        fleet.stop()

    mismatched = signature_mismatches(
        cell / PARITY_STANDARD_SUBDIR, cell / PARITY_QUEUE_SUBDIR, args.gen_ids
    )
    for name in mismatched:
        print(f"MISMATCH {name}")
    print("PERSISTENT_REPLAY_PARITY:", "FAIL" if mismatched else "PASS")
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} signature files differ between the queue and the direct path; "
            "the resident-worker arrangement is not a scheduling decision until they agree"
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    if args.parity:
        parity(args)
    elif args.drive:
        drive(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
