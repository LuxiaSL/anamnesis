"""Generate text and bank the token ids it was realized as.

This is phase one of the replay gateway: sample, write down the full ``prompt +
generated`` token sequence per generation, and stop. No hooks are placed and no
features are computed — those come from ``run_replay.py`` over exactly these ids,
which is why the capture surface can change without the corpus changing.

Three ways to run it, and one thing to do afterwards:

* **One worker** — ``--spec-file`` and ``--out-dir``. Generates that list of specs
  into that directory, skipping specs whose record is already there.
* **Fanned out over devices** — add ``--gpus``. Partitions the spec list and
  re-invokes this command once per worker, each confined to one device.
* **A roster, one model load** — ``--cells-json`` with ``--gpus``, or
  ``--jobs-file`` directly. Each worker receives its slice of every cell, loads
  once and walks them, re-arming each cell's intervention.
* **Assembly** — ``--assemble`` turns a directory of banked records into a run:
  ``metadata.json`` plus the replay manifest replay reads. It is safe to re-run
  and it runs at the end of a generating pass that produced every spec it was
  given. A short pass refuses before assembling, so a manifest is never stamped
  over a corpus in the same breath as the report that it is incomplete;
  ``--assemble`` on its own, with no specs, assembles whatever is there.

Per-generation seeding is what licenses all of this: a generation's output is a
function of its own coordinates, so a worker's identity and a cell's position in a
roster cannot reach it.

**This command fails closed.** A cell that banked fewer records than it had specs
exits ``anamnesis.shortfall.EXIT_SHORT``, naming the generation ids that are
missing and the ones that raised; ``--allow-partial`` accepts the short cell and
exits ``anamnesis.shortfall.EXIT_SHORT_SANCTIONED`` instead, still non-zero. Either
way a receipt lands in the records directory it describes. A resumed pass that
generates three records because seventeen were already banked has produced twenty
and exits zero, and so does a worker that produced exactly its share of the specs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.interventions import injection_fields
from anamnesis.orchestration.gpu import THREAD_LIMIT_ENV

MODULE = "anamnesis.scripts.run_gen_tokens"
logger = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    p.add_argument("--model-path", required=True, help="Local checkpoint directory")
    p.add_argument("--spec-file", type=Path, default=None, help="JSON list of generation specs")
    p.add_argument("--out-dir", type=Path, default=None, help="Where the records for those specs go")
    p.add_argument(
        "--jobs-file",
        type=Path,
        default=None,
        help="A roster as JSON jobs [{out_dir, specs, inject_*?, perturb?}], under one model load",
    )
    p.add_argument(
        "--cells-json",
        type=Path,
        default=None,
        help="A roster to fan out: {\"cells\": [{out_dir, spec_file, inject_*?}, ...]}",
    )
    p.add_argument(
        "--assemble",
        type=Path,
        default=None,
        help="Assemble this run directory's banked records into metadata plus a replay manifest",
    )
    p.add_argument("--temperature", type=float, default=None, help="Default: the preset's")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Above one discourages repeating context tokens; at exactly one the argument is "
             "withheld from the sampler, so the default path is the banked corpus's",
    )
    p.add_argument(
        "--attn",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention kernel for generation only. Replay is eager regardless, so this "
             "changes which tokens get sampled and nothing about their signatures",
    )
    p.add_argument(
        "--date-string",
        default=None,
        help="Pin the chat template's rendered date, so prompt tokens are not a function of "
             "the wall clock across a pass",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Accept a cell short of its specs; the receipt is written either way and "
             "the status stays non-zero",
    )
    p.add_argument("--label", default="w")
    p.add_argument("--inject-npz", default=None, help="Vector bank for a residual write")
    p.add_argument("--inject-key", default=None)
    p.add_argument("--inject-layer", type=int, default=None)
    p.add_argument("--inject-alpha", type=float, default=None, help="Absolute magnitude")
    p.add_argument("--inject-alpha-frac", type=float, default=None, help="Recorded as bookkeeping")
    p.add_argument(
        "--perturb-json", type=Path, default=None, help="A routing perturbation, applied model-wide"
    )
    p.add_argument("--gpus", default=None, help="Device slots to fan out over, comma-separated")
    p.add_argument("--workers-per-gpu", type=int, default=4)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the partition and stop")
    p.add_argument(
        "--single-cell-ok",
        action="store_true",
        help="Allow repeat single-cell invocations against one model in one job context",
    )
    return p


def decode_policy(args: argparse.Namespace) -> Any:
    """The sampling policy this invocation runs under, from the preset and the flags."""
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.token_generation import DecodePolicy

    preset = resolve_preset(args.model)
    return DecodePolicy(
        temperature=args.temperature if args.temperature is not None else preset.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=tuple(preset.eos_token_ids),
        repetition_penalty=args.repetition_penalty,
        date_string=args.date_string,
    )


def passthrough(args: argparse.Namespace) -> dict[str, Any]:
    """What a run's metadata records about how it was produced."""
    from anamnesis.config import resolve_preset

    preset = resolve_preset(args.model)
    policy = decode_policy(args)
    block: dict[str, Any] = {
        "model": {
            "model_id": preset.model_id,
            "torch_dtype": preset.torch_dtype,
            "num_layers": preset.num_layers,
            "hidden_dim": preset.hidden_dim,
        },
        "generation_config": {
            "max_new_tokens": policy.max_new_tokens,
            "temperature": policy.temperature,
            "top_p": policy.top_p,
            "repetition_penalty": policy.repetition_penalty,
            "do_sample": True,
            "eos_token_ids": list(policy.eos_token_ids),
        },
    }
    if policy.date_string:
        block["template_date_string"] = policy.date_string
    fields = injection_fields(
        args.inject_npz, args.inject_key, args.inject_layer,
        args.inject_alpha, args.inject_alpha_frac,
    )
    if fields["inject_npz"] is not None:
        block["a5_injection"] = {k: v for k, v in fields.items()}
    return block


def _generation_worker_command(args: argparse.Namespace, *, label: str, **overrides: Any) -> list[str]:
    """This same command, for one worker, with its share substituted in."""
    policy = decode_policy(args)
    cmd = [
        sys.executable, "-m", MODULE,
        "--model", args.model, "--model-path", args.model_path,
        "--temperature", str(policy.temperature), "--top-p", str(policy.top_p),
        "--max-new-tokens", str(policy.max_new_tokens),
        "--attn", args.attn, "--label", label,
    ]
    if args.allow_partial:
        cmd.append("--allow-partial")
    if policy.repetition_penalty != 1.0:
        cmd += ["--repetition-penalty", str(policy.repetition_penalty)]
    if policy.date_string:
        cmd += ["--date-string", policy.date_string]
    fields = injection_fields(
        args.inject_npz, args.inject_key, args.inject_layer,
        args.inject_alpha, args.inject_alpha_frac,
    )
    if fields["inject_npz"] is not None:
        cmd += [
            "--inject-npz", str(fields["inject_npz"]),
            "--inject-key", str(fields["inject_key"]),
            "--inject-layer", str(fields["inject_layer"]),
            "--inject-alpha", str(fields["inject_alpha"]),
        ]
        if fields["inject_alpha_frac"] is not None:
            cmd += ["--inject-alpha-frac", str(fields["inject_alpha_frac"])]
    if args.perturb_json:
        cmd += ["--perturb-json", str(args.perturb_json)]
    for flag, value in overrides.items():
        cmd += [flag, str(value)]
    return cmd


def fan_out_generation(args: argparse.Namespace) -> None:
    """Partition the specs over devices and re-invoke this command per worker."""
    from anamnesis.orchestration.gpu import enforce_single_cell_guard
    from anamnesis.orchestration.launch import (
        RECORDS_SUBDIR,
        Cell,
        LaunchPlan,
        assemble_run,
        launch,
        plan_multicell,
        write_worker_inputs,
    )

    if args.cells_json is not None:
        roster = json.loads(args.cells_json.read_text())["cells"]
        log_dir = args.log_dir or (args.cells_json.parent / "_multicell_jobs")
        plan = LaunchPlan.resolve(args.gpus, args.workers_per_gpu, log_dir)
        cells = [
            Cell(
                target={"out_dir": str(Path(row["out_dir"]) / RECORDS_SUBDIR)},
                payload={
                    k: v for k, v in row.items() if k not in ("out_dir", "spec_file", "specs")
                },
            )
            for row in roster
        ]
        specs_per_cell = {
            id(cell): _cell_specs(row) for cell, row in zip(cells, roster)
        }
        jobs = plan_multicell(
            cells, lambda cell: specs_per_cell[id(cell)], plan.n_workers, items_key="specs"
        )
        if args.dry_run:
            for worker in sorted(jobs):
                print(f"worker {worker} ({plan.device_for(worker)}): {len(jobs[worker])} cells")
            return
        files = write_worker_inputs(log_dir, "jobs", jobs)
        launch(
            plan,
            lambda w: _generation_worker_command(args, label=f"w{w}g{plan.device_for(w)}",
                                      **{"--jobs-file": files[w]}),
            stem="gen",
            workers=sorted(files),
        ).raise_on_failure()
        for row in roster:
            assemble_run(Path(row["out_dir"]), passthrough(args))
        return

    if args.spec_file is None or args.out_dir is None:
        raise SystemExit("fanning out needs --spec-file and --out-dir, or --cells-json")
    specs = json.loads(args.spec_file.read_text())
    todo = [
        s for s in specs
        if not (args.out_dir / f"gen_{s['generation_id']:03d}.json").exists()
    ]
    log_dir = args.log_dir or (args.out_dir.parent / "gen_logs")
    plan = LaunchPlan.resolve(args.gpus, args.workers_per_gpu, log_dir)
    shares = plan.partition(todo)
    if args.dry_run:
        for worker, share in enumerate(shares):
            if share:
                print(f"worker {worker} ({plan.device_for(worker)}): {len(share)} specs")
        return
    if todo:
        enforce_single_cell_guard(
            MODULE, args.model_path, args.out_dir,
            allow_repeat=args.single_cell_ok,
            multicell_pointer=f"python -m {MODULE} --cells-json <cells.json> --gpus <devices>",
        )
        files = write_worker_inputs(
            log_dir, "specs", {w: share for w, share in enumerate(shares) if share}
        )
        launch(
            plan,
            lambda w: _generation_worker_command(
                args, label=f"w{w}g{plan.device_for(w)}",
                **{"--spec-file": files[w], "--out-dir": args.out_dir},
            ),
            stem="gen",
            workers=sorted(files),
        ).raise_on_failure()
    if args.assemble is not None:
        assemble_run(args.assemble, passthrough(args))


def _cell_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A roster row's specs, inline or from the file it names."""
    if "specs" in row:
        return list(row["specs"])
    if "spec_file" not in row:
        raise SystemExit(f"cell {row.get('out_dir')!r} names neither specs nor a spec_file")
    return json.loads(Path(row["spec_file"]).read_text())


def generate(args: argparse.Namespace) -> None:
    """Load the model once and generate one cell's specs, or a roster of them.

    Each cell's accounting is collected and the verdict comes at the end, so an
    invocation over a roster names every short cell rather than dying on the first
    one and leaving the rest ungenerated.
    """
    import os

    # Pin the numeric thread pools before torch is imported. A launcher already
    # does this in the worker's environment; a hand-run worker gets it here, and
    # either way generation is device-bound so one CPU thread costs nothing and
    # avoids the thread explosion of many workers sizing pools to the machine.
    for name in THREAD_LIMIT_ENV:
        os.environ.setdefault(name, "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)

    from anamnesis.config import resolve_preset
    from anamnesis.extraction.interventions import (
        attach_injection,
        attach_perturbation,
        resolve_injection,
    )
    from anamnesis.extraction.token_generation import generate_specs, generation_shortfall
    from anamnesis.shortfall import Shortfall, refuse_unless_complete

    preset = resolve_preset(args.model)
    dtype = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    }.get(str(preset.torch_dtype), torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=dtype, attn_implementation=args.attn
        )
        .to("cuda")
        .eval()
    )
    policy = decode_policy(args)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else policy.eos_token_ids[0]
    )

    def run_one(
        specs: list[dict[str, Any]],
        out_dir: Path,
        fields: dict[str, Any],
        perturb: dict[str, Any] | None,
        penalty: float | None,
        label: str,
    ) -> Shortfall:
        injection = resolve_injection(None, fields=fields)
        handle = attach_injection(model, injection, label)
        perturb_handle = attach_perturbation(model, perturb, label)
        try:
            result = generate_specs(
                model, tokenizer, specs, out_dir,
                policy.with_repetition_penalty(penalty),
                pad_token_id=pad_id,
                write_handle=handle,
                injection=injection,
                perturbation=perturb,
                label=label,
            )
        finally:
            for armed in (handle, perturb_handle):
                if armed is not None:
                    armed.remove()
        return generation_shortfall(result, command=MODULE, label=label)

    if args.jobs_file is not None:
        shortfalls = [
            run_one(
                list(job["specs"]), Path(job["out_dir"]),
                {k: v for k, v in job.items() if k.startswith("inject_")},
                job.get("perturb"), job.get("repetition_penalty"),
                f"{args.label}c{index}",
            )
            for index, job in enumerate(json.loads(args.jobs_file.read_text()))
        ]
        refuse_unless_complete(shortfalls, allow_partial=args.allow_partial)
        return

    if args.spec_file is None or args.out_dir is None:
        raise SystemExit("generating needs --spec-file and --out-dir (or --jobs-file)")
    perturb = json.loads(args.perturb_json.read_text()) if args.perturb_json else None
    refuse_unless_complete(
        [run_one(
            json.loads(args.spec_file.read_text()), args.out_dir,
            injection_fields(
                args.inject_npz, args.inject_key, args.inject_layer,
                args.inject_alpha, args.inject_alpha_frac,
            ), perturb, None, args.label,
        )],
        allow_partial=args.allow_partial,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    if args.repetition_penalty <= 0:
        raise SystemExit(f"--repetition-penalty must be above zero, got {args.repetition_penalty}")
    if args.gpus is not None:
        fan_out_generation(args)
    elif args.cells_json is not None:
        raise SystemExit("--cells-json is a roster to fan out; give --gpus, or --jobs-file")
    elif args.spec_file is None and args.jobs_file is None and args.assemble is not None:
        from anamnesis.orchestration.launch import assemble_run

        assemble_run(args.assemble, passthrough(args))
    else:
        generate(args)
        if args.assemble is not None:
            from anamnesis.orchestration.launch import assemble_run

            assemble_run(args.assemble, passthrough(args))


if __name__ == "__main__":
    main()
