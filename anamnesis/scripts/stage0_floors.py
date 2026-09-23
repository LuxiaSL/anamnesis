"""Stage 0: collect the faithfulness replays, then turn the floors into the law.

Two subcommands, in the order they run.

``replays`` plans the stratified faithfulness set and fans it out. It reads the floor
run's own manifest, takes seed 0 of every topic, and writes a synthetic manifest whose
ten entries per continuation all carry the same banked tokens — four pinned to one
device, six spread over the others. Each device then gets exactly its own generation
ids through the ordinary replay command, so nothing about this protocol lives in the
replay path.

``law`` computes the floors and the n-min table on a CPU: the stochastic floor from the
floor corpus, the faithfulness floors from the stratified replays, and the law table
that says how large a difference has to be and how many samples it takes to see one.
The faithfulness deltas are standardized on the stochastic corpus's scale, which is why
one command does both rather than two commands doing one each.

    python -m anamnesis.scripts.stage0_floors replays --model 3b \\
        --model-path /models/llama-3.2-3b-instruct \\
        --floor-run-dir outputs/battery/vmb_stage0_3b \\
        --calib-dir outputs/calibration/3b \\
        --out-dir outputs/battery/vmb_stage0_3b/faithfulness \\
        --pinned-gpu 0 --spread-gpus 1,2,3

    python -m anamnesis.scripts.stage0_floors law --model 3b --n-layers 28 \\
        --floor-sig-dir outputs/battery/vmb_stage0_3b/signatures_v3 \\
        --floor-metadata outputs/battery/vmb_stage0_3b/metadata.json \\
        --faith-sig-dir outputs/battery/vmb_stage0_3b/faithfulness/signatures_v3 \\
        --faith-index outputs/battery/vmb_stage0_3b/faithfulness/replay_index.json \\
        --out-dir outputs/battery/vmb_stage0_3b/floors
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anamnesis.config import preset_names

logger = logging.getLogger(__name__)

REPLAY_MODULE = "anamnesis.scripts.run_replay"
"""The command each device's worker runs. A faithfulness replay is an ordinary replay of
a synthetic manifest, so this protocol invokes the replay command rather than holding a
replay loop of its own."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stage0_floors.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)

    replays = sub.add_parser("replays", help="plan and run the stratified faithfulness replays")
    replays.add_argument("--model", choices=list(preset_names()), required=True)
    replays.add_argument("--model-path", required=True, help="Local checkpoint directory")
    replays.add_argument(
        "--floor-run-dir", type=Path, required=True,
        help="The stochastic floor run, whose replay_manifest.json the continuations come from",
    )
    replays.add_argument("--calib-dir", type=Path, required=True)
    replays.add_argument("--out-dir", type=Path, required=True)
    replays.add_argument("--pinned-gpu", default="0", help="Device the within-device component runs on")
    replays.add_argument(
        "--spread-gpus", default="1,2,3",
        help="Devices the cross-device component is spread over, comma-separated",
    )
    replays.add_argument("--dry-run", action="store_true", help="Write the plan and stop")

    law = sub.add_parser("law", help="floors to the n-min law table")
    law.add_argument("--model", required=True)
    law.add_argument("--n-layers", type=int, required=True, help="Depth, for the layer-band cells")
    law.add_argument("--floor-sig-dir", type=Path, required=True)
    law.add_argument("--floor-metadata", type=Path, required=True)
    law.add_argument("--faith-sig-dir", type=Path, default=None)
    law.add_argument("--faith-index", type=Path, default=None)
    law.add_argument("--out-dir", type=Path, required=True)
    law.add_argument("--k", type=float, default=2.0, help="Where an arm sits, in floor medians")
    law.add_argument("--power", type=float, default=0.9)
    return p


def run_replays(args: argparse.Namespace) -> int:
    """Plan the stratified set, bank it, and give each device its own share."""
    from anamnesis.analysis.battery.stage0 import (
        SIGNATURES_SUBDIR,
        plan_stratified_replays,
        select_continuations,
    )
    from anamnesis.orchestration.gpu import resolve_physical_gpus
    from anamnesis.orchestration.launch import LaunchPlan, launch

    manifest = json.loads(
        (args.floor_run_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    continuations = select_continuations(manifest)
    pinned = resolve_physical_gpus([args.pinned_gpu.strip()])[0]
    spread = resolve_physical_gpus(
        [g.strip() for g in args.spread_gpus.split(",") if g.strip()]
    )
    plan = plan_stratified_replays(
        continuations, pinned_device=pinned, spread_devices=spread
    )
    manifest_path, index_path = plan.write(args.out_dir)
    by_device = plan.gen_ids_by_device()
    logger.info(
        f"pinned device {pinned} x{len([i for i in plan.instances if i.device == pinned])}, "
        f"spread over {spread}"
    )

    if args.dry_run:
        for device, ids in by_device.items():
            logger.info(f"  device {device}: {len(ids)} replays {ids[:6]}")
        logger.info(f"plan banked: {manifest_path}, {index_path}")
        return 0

    devices = list(by_device)
    launch_plan = LaunchPlan(
        devices=tuple(devices), workers_per_device=1, log_dir=args.out_dir / "logs"
    )

    def command_for(worker: int) -> list[str]:
        device = launch_plan.device_for(worker)
        return [
            sys.executable, "-m", REPLAY_MODULE,
            "--model", args.model, "--model-path", args.model_path,
            "--calib-dir", str(args.calib_dir),
            "--run-dir", str(args.out_dir), "--manifest", str(manifest_path),
            "--gen-ids", *[str(g) for g in by_device[device]],
            "--no-raw", "--sig-subdir", SIGNATURES_SUBDIR,
            "--label", f"faith-gpu{device}",
        ]

    result = launch(launch_plan, command_for, stem="faithfulness")
    signatures = len(list((args.out_dir / SIGNATURES_SUBDIR).glob("gen_*.npz")))
    logger.info(
        f"{signatures}/{len(plan.entries)} signatures after {result.seconds:.0f}s "
        f"({len(result.failed)} workers failed)"
    )
    result.raise_on_failure()
    return 0


def run_law(args: argparse.Namespace) -> int:
    """Compute the floors, bank the reports and the table, print the headline."""
    from anamnesis.analysis.battery.floors import LawParams
    from anamnesis.analysis.battery.stage0 import compute_stage0_law, whole_vector_reading

    if (args.faith_sig_dir is None) != (args.faith_index is None):
        raise SystemExit(
            "the faithfulness floors need both --faith-sig-dir and --faith-index: "
            "a replay corpus with no index cannot be split into its two components"
        )
    result = compute_stage0_law(
        model=args.model,
        n_layers=args.n_layers,
        floor_sig_dir=args.floor_sig_dir,
        floor_metadata=args.floor_metadata,
        out_dir=args.out_dir,
        faith_sig_dir=args.faith_sig_dir,
        faith_index=args.faith_index,
        law=LawParams(k=args.k, power=args.power),
    )
    for report in result.reports:
        logger.info(whole_vector_reading(report))
    logger.info(f"law table -> {result.law_table_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    return run_replays(args) if args.stage == "replays" else run_law(args)


if __name__ == "__main__":
    raise SystemExit(main())
