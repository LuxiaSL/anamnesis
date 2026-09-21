"""Replay banked token sequences back into signatures.

Replay is the instrument's gateway: a generation pass banks text as the token ids
it actually realized, and every signature of record comes from teacher-forcing
those ids through the capture surface. Because the ids are banked, re-running this
reproduces a signature exactly, and changing the capture surface costs a replay
rather than a re-generation.

Four ways to say the same thing, differing only in how much hardware is in play:

* **One cell** — ``--run-dir`` and ``--manifest``. Loads the model, replays the
  cell's generations (or the ``--gen-ids`` slice of them), writes signatures.
* **A roster, one model load** — ``--jobs-file``. Loads once and walks cells,
  re-arming each cell's intervention. This is the path of record for a roster.
* **Fanned out over devices** — add ``--gpus``. Partitions the work and re-invokes
  this command once per worker, each confined to one device.
* **A roster fanned out** — ``--gpus`` with ``--cells-json``. Each worker gets its
  slice of *every* cell in one job file, so the roster costs one model load per
  worker rather than one per cell.

The last two produce what the first two would. That is not an aspiration: the
resident-worker path is checked byte-for-byte against this one by
``run_persistent_replay.py --parity``, and the single-cell guard refuses a roster
being walked one invocation at a time — the arrangement that quietly pays a model
load per cell.

**This command fails closed.** Every cell it walks reports what it was asked for
and what landed, and a cell short of its request exits
``anamnesis.shortfall.EXIT_SHORT`` naming the generations that are missing and the
ones that raised. ``--allow-partial`` accepts the short cell and exits
``anamnesis.shortfall.EXIT_SHORT_SANCTIONED`` instead, still non-zero; either way
a receipt lands in the signature directory it describes. A resumed pass that
computes three signatures because seventeen were already on disk has produced
twenty and exits zero, and so does a worker that produced exactly its ``--gen-ids``
share.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.interventions import injection_fields

MODULE = "anamnesis.scripts.run_replay"
MULTICELL_POINTER = f"python -m {MODULE} --cells-json <cells.json> --gpus <devices>"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument("--model-path", required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=None, help="One cell's output directory")
    p.add_argument("--manifest", type=Path, default=None, help="That cell's replay manifest")
    p.add_argument(
        "--jobs-file",
        type=Path,
        default=None,
        help="A roster as JSON jobs, replayed under one model load",
    )
    p.add_argument(
        "--cells-json",
        type=Path,
        default=None,
        help="A roster to fan out: {\"cells\": [{run_dir, manifest, gen_ids?, inject_*?}, ...]}",
    )
    p.add_argument("--gen-ids", type=int, nargs="+", default=None, help="Slice of one cell")
    p.add_argument("--sig-subdir", default="signatures_v3")
    p.add_argument("--raw-subdir", default="raw_tensors_v3")
    p.add_argument(
        "--raw-dir", type=Path, default=None, help="Absolute raw output directory, e.g. a scratch disk"
    )
    p.add_argument("--no-raw", action="store_true", help="Signatures only")
    p.add_argument(
        "--logits-top-k",
        type=int,
        default=50,
        help="Logits kept per position in raw; raise it for cells that need logit mass",
    )
    p.add_argument("--no-pca", action="store_true", help="Skip the residual-PCA features")
    p.add_argument("--no-resume", action="store_true", help="Recompute existing signatures")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Accept a cell short of its manifest; the receipt is written either way and "
             "the status stays non-zero",
    )
    p.add_argument("--label", default="w", help="Worker label carried into the logs")
    p.add_argument("--adapter-path", default=None, help="Adapter merged before hooks are placed")
    p.add_argument("--inject-npz", type=Path, default=None, help="Vector bank for a residual write")
    p.add_argument("--inject-key", default=None)
    p.add_argument("--inject-layer", type=int, default=None)
    p.add_argument("--inject-alpha", type=float, default=None, help="Absolute magnitude")
    p.add_argument("--inject-alpha-frac", type=float, default=None, help="Recorded as bookkeeping")
    p.add_argument(
        "--inject-from-metadata",
        action="store_true",
        help="Read the write from the cell's own metadata, so generation and replay cannot differ",
    )
    p.add_argument("--gpus", default=None, help="Device slots to fan out over, comma-separated")
    p.add_argument("--workers-per-gpu", type=int, default=3)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the partition and stop")
    p.add_argument(
        "--single-cell-ok",
        action="store_true",
        help="Allow repeat single-cell invocations against one model in one job context",
    )
    return p


def _replay_worker_command(args: argparse.Namespace, *, label: str, **overrides: Any) -> list[str]:
    """This same command, for one worker, with its share substituted in."""
    cmd = [
        sys.executable, "-m", MODULE,
        "--model", args.model, "--model-path", args.model_path,
        "--calib-dir", str(args.calib_dir),
        "--sig-subdir", args.sig_subdir, "--raw-subdir", args.raw_subdir,
        "--logits-top-k", str(args.logits_top_k), "--label", label,
    ]
    for flag, present in (
        ("--no-raw", args.no_raw),
        ("--no-pca", args.no_pca),
        ("--no-resume", args.no_resume),
        ("--allow-partial", args.allow_partial),
        ("--inject-from-metadata", args.inject_from_metadata),
    ):
        if present:
            cmd.append(flag)
    if args.raw_dir:
        cmd += ["--raw-dir", str(args.raw_dir)]
    if args.adapter_path:
        cmd += ["--adapter-path", str(args.adapter_path)]
    if not args.inject_from_metadata and args.inject_npz is not None:
        cmd += [
            "--inject-npz", str(args.inject_npz),
            "--inject-key", str(args.inject_key),
            "--inject-layer", str(args.inject_layer),
            "--inject-alpha", str(args.inject_alpha),
        ]
        if args.inject_alpha_frac is not None:
            cmd += ["--inject-alpha-frac", str(args.inject_alpha_frac)]
    for flag, value in overrides.items():
        cmd += [flag, *([str(v) for v in value] if isinstance(value, list) else [str(value)])]
    return cmd


def fan_out_replay(args: argparse.Namespace) -> None:
    """Partition the work over devices and re-invoke this command per worker.

    A worker that came up short exits with a shortfall status rather than
    crashing, and ``LaunchResult.raise_on_failure`` inherits that status, so a
    fan-out whose workers were all short is short rather than generically failed.
    """
    from anamnesis.extraction.replay.cell import signature_on_disk
    from anamnesis.orchestration.gpu import enforce_single_cell_guard
    from anamnesis.orchestration.launch import LaunchPlan, fan_out_roster, launch

    log_dir = args.log_dir or (
        (args.cells_json.parent if args.cells_json else args.run_dir) / "replay_logs"
    )
    plan = LaunchPlan.resolve(args.gpus, args.workers_per_gpu, log_dir)

    if args.cells_json is not None:
        fan_out_roster(
            plan,
            json.loads(args.cells_json.read_text())["cells"],
            _manifest_ids,
            target_fields=("run_dir", "manifest"),
            item_fields=("gen_ids",),
            items_key="gen_ids",
            jobs_dir=log_dir / "jobs",
            command_for=lambda worker, jobs_file: _replay_worker_command(
                args, label=f"w{worker}g{plan.device_for(worker)}",
                **{"--jobs-file": jobs_file},
            ),
            stem="replay",
            dry_run=args.dry_run,
        )
        return

    if args.run_dir is None or args.manifest is None:
        raise SystemExit("fanning out one cell needs --run-dir and --manifest, or --cells-json")
    ids = _manifest_ids({"manifest": args.manifest, "gen_ids": args.gen_ids})
    sig_dir = args.run_dir / args.sig_subdir
    # The same predicate a worker resumes on, so the fan-out cannot call a cell
    # finished that a worker would have found work in.
    todo = ids if args.no_resume else [g for g in ids if not signature_on_disk(sig_dir, g)]
    if not todo:
        print(f"all {len(ids)} generations already have signatures in {sig_dir}")
        return
    shares = plan.partition(todo)
    if args.dry_run:
        for worker, share in enumerate(shares):
            if share:
                print(f"worker {worker} ({plan.device_for(worker)}): {len(share)} generations")
        return
    enforce_single_cell_guard(
        MODULE, args.model_path, args.run_dir,
        allow_repeat=args.single_cell_ok, multicell_pointer=MULTICELL_POINTER,
    )
    result = launch(
        plan,
        lambda w: _replay_worker_command(
            args, label=f"w{w}g{plan.device_for(w)}",
            **{"--run-dir": args.run_dir, "--manifest": args.manifest, "--gen-ids": shares[w]},
        ),
        stem="replay",
        workers=[w for w, share in enumerate(shares) if share],
    )
    result.raise_on_failure()


def _manifest_ids(cell: dict[str, Any]) -> list[int]:
    """A cell's replayable generation ids, narrowed to its own slice if it names one."""
    entries = json.loads(Path(cell["manifest"]).read_text())["entries"]
    ids = sorted(int(k) for k in entries)
    wanted = cell.get("gen_ids")
    if wanted is None:
        return ids
    keep = {int(g) for g in wanted}
    return [g for g in ids if g in keep]


def replay(args: argparse.Namespace) -> None:
    """Load the model once and replay one cell, or a roster of them.

    Each cell's accounting is collected and the verdict comes at the end, so an
    invocation over a roster names every short cell rather than dying on the
    first one and leaving the rest unattempted.
    """
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.calibration import load_calibration
    from anamnesis.extraction.interventions import armed_interventions, resolve_injection
    from anamnesis.extraction.replay.cell import cell_shortfall, load_replay_model, replay_cell
    from anamnesis.shortfall import Shortfall, refuse_unless_complete

    surface = load_replay_model(
        resolve_preset(args.model),
        args.model_path,
        enable_pca=not args.no_pca,
        adapter_path=args.adapter_path,
    )
    calibration = load_calibration(args.calib_dir, enable_pca=not args.no_pca)

    def run_one(
        run_dir: Path,
        manifest: Path,
        gen_ids: list[int] | None,
        fields: dict[str, Any],
        from_metadata: bool,
        perturb: dict[str, Any] | None,
        label: str,
    ) -> Shortfall:
        injection = resolve_injection(run_dir, from_metadata=from_metadata, fields=fields)
        with armed_interventions(
            surface.loaded, surface.loaded.model,
            injection=injection, perturbation=perturb, label=label,
        ) as handle:
            result = replay_cell(
                surface, calibration, run_dir, manifest,
                gen_ids=gen_ids,
                signatures_subdir=args.sig_subdir,
                raw_dir=args.raw_dir,
                raw_subdir=args.raw_subdir,
                save_raw=not args.no_raw,
                resume=not args.no_resume,
                logits_top_k=args.logits_top_k,
                write_handle=handle,
                injection=injection,
                label=label,
            )
        return cell_shortfall(result, manifest, command=MODULE, label=label)

    if args.jobs_file is not None:
        jobs = json.loads(args.jobs_file.read_text())
        shortfalls = [
            run_one(
                Path(job["run_dir"]), Path(job["manifest"]), job.get("gen_ids"),
                {k: v for k, v in job.items() if k.startswith("inject_")},
                bool(job.get("inject_from_metadata", False)),
                job.get("perturb"),
                f"{args.label}c{index}",
            )
            for index, job in enumerate(jobs)
        ]
        refuse_unless_complete(shortfalls, allow_partial=args.allow_partial)
        return

    if args.run_dir is None or args.manifest is None:
        raise SystemExit("replaying one cell needs --run-dir and --manifest (or --jobs-file)")
    refuse_unless_complete(
        [run_one(
            args.run_dir, args.manifest, args.gen_ids, injection_fields(
                args.inject_npz, args.inject_key, args.inject_layer,
                args.inject_alpha, args.inject_alpha_frac,
            ),
            args.inject_from_metadata, None, args.label,
        )],
        allow_partial=args.allow_partial,
    )


def main(argv: list[str] | None = None) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    if args.jobs_file is not None and (args.run_dir or args.manifest or args.cells_json):
        raise SystemExit("--jobs-file is the whole roster; it takes no cell of its own")
    if args.gpus is not None:
        fan_out_replay(args)
    elif args.cells_json is not None:
        raise SystemExit("--cells-json is a roster to fan out; give --gpus, or --jobs-file")
    else:
        replay(args)


if __name__ == "__main__":
    main()
