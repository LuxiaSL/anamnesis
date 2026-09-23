"""Replay one manifest through a series of adapter checkpoints, on one model load.

A checkpoint series asks what changed between training steps, so the same banked token
sequences are replayed through each checkpoint. Loading the base model once per worker and
swapping adapters is what makes that affordable: the hooks stay attached across the whole
series because merging writes into the wrapped layers' own weights and leaves the modules
alive.

Three ways to run it, and the default is the launcher:

* **fanned out** (the default) — partitions the manifest's generation ids across
  ``gpus x workers-per-gpu`` and re-invokes this command once per worker, each confined to
  one device and handed the whole checkpoint list.
* **one worker** — ``--worker`` with ``--gen-ids``, which is what the launcher spawns.
* **dry run** — prints the partition and stops.

Pristine restore is on by default and **required** for more than one checkpoint: repeated
merge and unmerge drifts along the series, and the drift reads as a training effect.

A full-weight checkpoint is not an adapter. Replay one of those with ``run_replay.py
--model-path=<checkpoint>``, which pays a model load per checkpoint because that is what
it costs.

    python -m anamnesis.scripts.run_replay_multickpt --model 3b --model-path <base> \\
        --calib-dir outputs/calibration/3b --manifest <run>/replay_manifest.json \\
        --checkpoints-json cells_cat_dpo.json --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anamnesis.config import preset_names

logger = logging.getLogger(__name__)

MODULE = "anamnesis.scripts.run_replay_multickpt"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_replay_multickpt.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(preset_names()), required=True)
    p.add_argument("--model-path", required=True, help="The base checkpoint every adapter wraps")
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="The fixed manifest every checkpoint replays")
    p.add_argument(
        "--checkpoints-json", type=Path, required=True,
        help='{"checkpoints": [{label, adapter_path, run_dir}, ...]} in replay order',
    )
    p.add_argument("--worker", action="store_true", help="Run one worker's share rather than fanning out")
    p.add_argument("--gen-ids", type=int, nargs="+", default=None, help="This worker's share")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Device slots to fan out over")
    p.add_argument("--workers-per-gpu", type=int, default=8)
    p.add_argument("--sig-subdir", default="signatures_v3")
    p.add_argument("--no-raw", action="store_true", help="Signatures only")
    p.add_argument("--no-resume", action="store_true", help="Recompute existing signatures")
    p.add_argument("--no-pca", action="store_true", help="Skip the residual-PCA features")
    p.add_argument(
        "--pristine-restore", action=argparse.BooleanOptionalAction, default=True,
        help="Restore the base weights before each merge. On by default, and required "
             "for more than one checkpoint",
    )
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--label", default="w", help="Worker label carried into the logs")
    p.add_argument("--dry-run", action="store_true", help="Print the partition and stop")
    return p


def replay_worker(args: argparse.Namespace) -> None:
    """Load the base once, wrap it, and walk the series over this worker's share."""
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.replay.cell import load_replay_model
    from anamnesis.extraction.replay.checkpoint_series import (
        CheckpointSeries,
        replay_series,
        require_pristine_restore,
        wrap_with_first_adapter,
    )

    series = CheckpointSeries.load(args.checkpoints_json)
    require_pristine_restore(len(series.checkpoints), args.pristine_restore)
    logger.info(f"[{args.label}] loading the base once: {args.model_path}")
    surface = load_replay_model(
        resolve_preset(args.model), args.model_path, enable_pca=not args.no_pca
    )
    calibration = load_calibration(args.calib_dir, enable_pca=not args.no_pca)
    surface = wrap_with_first_adapter(surface, series)
    results = replay_series(
        surface,
        series,
        calibration,
        args.manifest,
        gen_ids=args.gen_ids,
        signatures_subdir=args.sig_subdir,
        save_raw=not args.no_raw,
        resume=not args.no_resume,
        pristine_restore=args.pristine_restore,
        label=args.label,
    )
    failed = sum(result.n_failed for result in results)
    if failed:
        raise SystemExit(
            f"[{args.label}] {failed} generations failed across "
            f"{len(series.checkpoints)} checkpoints"
        )


def fan_out_series(args: argparse.Namespace) -> None:
    """Partition the manifest across devices and re-invoke this command per worker."""
    from anamnesis.extraction.replay.checkpoint_series import (
        CheckpointSeries,
        require_pristine_restore,
    )
    from anamnesis.orchestration.launch import LaunchPlan, launch

    series = CheckpointSeries.load(args.checkpoints_json)
    require_pristine_restore(len(series.checkpoints), args.pristine_restore)
    ids = sorted(
        int(key)
        for key in json.loads(args.manifest.read_text(encoding="utf-8"))["entries"]
    )
    log_dir = args.log_dir or (args.checkpoints_json.parent / f"multickpt_logs_{args.label}")
    plan = LaunchPlan.resolve(args.gpus, args.workers_per_gpu, log_dir)
    shares = plan.partition(ids)
    logger.info(
        f"{len(series.checkpoints)} checkpoints x {len(ids)} generations over "
        f"{plan.n_workers} workers; one base load per worker"
    )
    if args.dry_run:
        for worker, share in enumerate(shares):
            if share:
                logger.info(
                    f"  worker {worker} ({plan.device_for(worker)}): {len(share)} "
                    f"generations {share[:5]}"
                )
        return

    def command_for(worker: int) -> list[str]:
        command = [
            sys.executable, "-m", MODULE, "--worker",
            "--model", args.model, "--model-path", args.model_path,
            "--calib-dir", str(args.calib_dir), "--manifest", str(args.manifest),
            "--checkpoints-json", str(args.checkpoints_json),
            "--sig-subdir", args.sig_subdir,
            "--label", f"w{worker}g{plan.device_for(worker)}",
            "--gen-ids", *[str(g) for g in shares[worker]],
        ]
        for flag, present in (
            ("--no-raw", args.no_raw),
            ("--no-resume", args.no_resume),
            ("--no-pca", args.no_pca),
        ):
            if present:
                command.append(flag)
        command.append("--pristine-restore" if args.pristine_restore else "--no-pristine-restore")
        return command

    result = launch(
        plan,
        command_for,
        stem="multickpt",
        workers=[w for w, share in enumerate(shares) if share],
    )
    result.raise_on_failure()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    if args.worker:
        replay_worker(args)
    else:
        fan_out_series(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
