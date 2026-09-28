"""Check this host's vLLM lane install against the shipped fixtures, and cache the result.

**What the check asks.** Not whether the vLLM lane agrees with the numeric anchor
(one comparison per engine release and model measured that, and its result is the
shipped fixtures and tolerance) but the smaller question a host can answer on its
own: does *this* install compute what the fixtures record? It needs no second model and no reference bank, only the checkpoint,
its calibration and one GPU. The calibration is the one the fixtures were reduced
with; without ``--calib-dir`` it is fetched once and verified against its pinned
digests.

What it does, per model:

1. **Checkpoint.** The weights' digest must be the fixtures', or the host is
   refused.
2. **Determinism.** Every fixture row is captured twice one at a time and once
   in batches of eight. All three vectors must be byte-identical, because a lane
   that disagrees with itself cannot be compared with anything.
3. **Deviation.** The vectors are compared with the fixtures:

   * **identical** — every vector byte-identical. The host runs the fixtures' lane
     and its outputs carry that lane id.
   * **conformant** — within the recorded tolerance: every row's distance under
     its recorded ceiling, every continuous coordinate within its family's
     recorded maximum, and the ordinary rows' median and tail within the recorded
     p90 and p99. The host is a lane of its own, with an id derived from its
     fingerprint, and fully usable; its outputs are never combined with another
     lane's inside one contrast.
   * **refused** — anything else, with the reasons.

An extension lane (:mod:`anamnesis.extraction.vllm.extensions`) is checked against
its own fixtures, with its calibration verified against its declared pins.

The receipt is cached under the output root against the host's fingerprint (GPU,
driver, CUDA runtime, torch, vLLM and anamnesis versions, checkpoint, fixtures,
tolerance, engine settings and the lane's source), and reused only while every field
is equal.
`run_vllm_replay.py` refuses to run on a host without one.

Exit status: 0 identical or conformant, 1 refused, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from anamnesis.extraction.vllm.extensions import lane_keys


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qualify_vllm.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=lane_keys(), required=True)
    p.add_argument("--model-path", type=Path, required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path,
                   help="The lane's calibration; fetched and verified when omitted")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="New directory for the captures and vectors of this check")
    p.add_argument("--cache-dir", type=Path,
                   help="Receipt cache; defaults to vllm_conformance under the output root")
    p.add_argument("--refresh", action="store_true",
                   help="Run the check even when a receipt is cached for this host")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        command = parser()
    except ValueError as exc:
        print(f"the install check did not run: {exc}", file=sys.stderr)
        return 2
    args = command.parse_args(argv)
    from anamnesis.extraction.vllm import extensions
    from anamnesis.extraction.vllm.hub import fetch_calibration, verify_calibration
    from anamnesis.extraction.vllm.runtime import (
        check_install,
        default_cache_dir,
        load_fixtures,
    )

    cache_dir = args.cache_dir or default_cache_dir()
    try:
        load_fixtures(args.model)
        if extensions.declared_lane(args.model) is not None:
            calib_dir = extensions.verify_calibration(args.model, args.calib_dir)
        else:
            if args.calib_dir is not None:
                verify_calibration(args.model, args.calib_dir)
            calib_dir = args.calib_dir or fetch_calibration(args.model)
        receipt, cached = check_install(args.model, args.model_path, calib_dir,
                                        args.work_dir, cache_dir, refresh=args.refresh)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"the install check did not run: {exc}", file=sys.stderr)
        return 2
    print(f"model: {args.model}  tier: {receipt.tier}{'  (cached)' if cached else ''}")
    if receipt.lane_id is not None:
        print(f"lane id: {receipt.lane_id}")
    for reason in receipt.reasons[:10]:
        print(f"  {reason}")
    if receipt.family_report:
        worst = max(receipt.family_report.items(), key=lambda kv: kv[1])
        print(f"worst family: {worst[0]} at {worst[1]:.3g} of its recorded maximum")
    print(f"receipt: {receipt.digest}  cache: {cache_dir}")
    return 1 if receipt.tier == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
