"""Check that a fine-tune of a qualified model can use its base's vLLM lane.

**What the check asks.** A shipped lane (a key of
:data:`anamnesis.extraction.vllm.envelope.LANE_MODELS`) was qualified against the
numeric anchor, and the lane's effects were retained at the
deviations that qualification measured. A fine-tune of the base has the same
architecture, so the kernels, the determinism and the feature schema carry over.
What its weights can change is how far the vLLM lane drifts from the fast lane on
a row. This command measures that drift on a sample of the fine-tune's own rows
and scores it against the base's recorded tolerance. Inside it, the fine-tune
inherits the base's evidence that effects survive the lane; outside it, the
fine-tune needs a qualification of its own.

What it needs: the fine-tune's checkpoint; its calibration, fitted at the base's
dtype; a registry preset for it that extends the base's preset with the same
architecture and layer plan (``ANAMNESIS_MODELS``); and a replay manifest of 44 to
60 rows the fine-tune generated. One GPU holds the vLLM engine first and the
Hugging Face model after it.

What it does (:func:`anamnesis.extraction.vllm.transfer_run.run_transfer`):

1. Captures the sample through the vLLM lane under the base's engine settings,
   twice one at a time and once in batches of eight: all three must agree byte
   for byte.
2. Replays it through the fast lane for the reference vectors, and through the
   numeric anchor's two execution paths for the fine-tune's own σ_cal and each
   row's path floor.
3. Scores each row's vLLM deviation from the fast lane against the base's
   ceilings and family maxima, and 16 ordinary rows, fixed from the generation ids
   before anything is measured, against the base's p90 (median) and p99 (at most
   one row over it). Rows whose path floor exceeds the base's limit are named and
   left out of the scoring.

Written into ``--out`` (:func:`anamnesis.extraction.vllm.transfer_run.write_transfer`):
the transfer receipt always; on a pass also the fine-tune's fixture set and
tolerance, for its hosts' install checks, and a lane file declaring the extension.
Name that file in ``ANAMNESIS_VLLM_LANES`` and ``qualify_vllm`` and
``run_vllm_replay`` accept the key.

Exit status: 0 pass, 1 refuse, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from anamnesis.extraction.vllm.envelope import LANE_MODELS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transfer_vllm.py", description=__doc__.splitlines()[0])
    p.add_argument("--key", required=True, help="The extension lane's key; a new name")
    p.add_argument("--extends", choices=sorted(LANE_MODELS), required=True,
                   help="The shipped lane the checkpoint is a fine-tune of")
    p.add_argument("--preset", required=True,
                   help="The fine-tune's registry preset, extending the base's")
    p.add_argument("--model-path", type=Path, required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path, required=True,
                   help="The fine-tune's calibration, fitted at the base's dtype")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Replay manifest of the fine-tune's own banked rows")
    p.add_argument("--gen-ids", type=int, nargs="+",
                   help="The sample's generations; the whole manifest by default")
    p.add_argument("--out", type=Path, required=True, help="New directory for the outputs")
    p.add_argument("--work-dir", type=Path,
                   help="New directory for the vLLM captures; defaults beside --out")
    p.add_argument("--device", default="cuda:0",
                   help="Device the Hugging Face model is loaded on")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from anamnesis.extraction.replay.manifest import ReplayManifest
    from anamnesis.extraction.vllm.transfer_run import run_transfer

    try:
        manifest = ReplayManifest.model_validate_json(args.manifest.read_text())
        ids = sorted(manifest.gen_ids()) if args.gen_ids is None else list(args.gen_ids)
        if any(str(i) not in manifest.entries for i in ids):
            raise ValueError("the sample names generations the manifest does not hold")
        entries = {k: entry.model_dump() for k, entry in manifest.entries.items()}
        result, written = run_transfer(
            key=args.key, extends=args.extends, preset=args.preset,
            model_path=args.model_path, calib_dir=args.calib_dir, entries=entries, ids=ids,
            out=args.out, work_dir=args.work_dir or args.out.with_name(args.out.name + ".work"),
            device=args.device)
    except (ValueError, RuntimeError, OSError, ImportError, NotImplementedError) as exc:
        print(f"the transfer check did not run: {exc}", file=sys.stderr)
        return 2
    receipt = result.receipt
    print(f"key: {receipt.key}  extends: {receipt.extends}  verdict: {receipt.verdict}")
    print(f"lane id: {receipt.lane_id}")
    if receipt.fragile_rows:
        print(f"rows over the path-floor limit {receipt.max_floor}, not scored: "
              f"{list(receipt.fragile_rows)}")
    for reason in receipt.reasons[:10]:
        print(f"  {reason}")
    print(json.dumps({role: str(path) for role, path in written.items()}))
    return 0 if receipt.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
