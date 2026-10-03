"""Check this host's vLLM lane install against the shipped fixtures, and cache its tier.

**Different hardware gives different numbers, and that is expected rather than
wrong.** A lane is self-consistency plus provenance: on one declared model,
engine, arithmetic and host, the same tokens give the same signature every time.
The check asks what a host can answer on its own, with the checkpoint, its
calibration and one GPU: is this install a lane, and is it the qualified one? It
needs no second model and no reference bank. The calibration is the one the
fixtures were reduced with; without ``--calib-dir`` it is fetched once and
verified against its pinned digests.

What it does, per model:

1. **Checkpoint.** The weights' digest must be the fixtures', or the host is
   refused.
2. **Self-consistency.** Every fixture row is captured twice one at a time and
   once in batches of eight. All three vectors must be byte-identical, because a
   lane that disagrees with itself cannot be compared with anything.
3. **Tier.** The vectors are compared with the fixtures:

   * **identical** — every vector byte-identical. The host runs the qualified
     lane and its outputs carry that lane id.
   * **own-lane** — the vectors differ, every feature is finite, and per
     component the median ratio over the fixture rows is at or below the maximum
     the qualification recorded. The host is a lane of its own, with an id
     derived from its fingerprint, and fully usable; its outputs are never
     combined with another lane's inside one contrast.
   * **refused** — anything else, with the reasons.

The receipt also reports where the host sits against every recorded ceiling (per
row, per family, and the ordinary rows' median and tail); those readings gate
nothing. Whether this lane agrees with another is an audit run when a claim needs
it (``transfer_vllm.py``).

An extension lane (:mod:`anamnesis.extraction.vllm.extensions`) is checked against
its own fixtures, with its calibration verified against its declared pins.

The receipt is cached under the output root against the host's fingerprint (GPU,
driver, CUDA runtime, torch, vLLM and anamnesis versions, NumPy's CPU dispatch,
checkpoint, fixtures, tolerance, engine settings and the lane's source), and reused
only while every field is equal.
`run_vllm_replay.py` refuses to run on a host without one.

**A pass here is not a certification.** It says this host repeats itself, which
lane it is, and how far that lane sits from the qualified one.

Exit status: 0 identical or own-lane, 1 refused, 2 the check could not run.
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
    if receipt.component_medians:
        print("median ratio: " + ", ".join(f"{c} {m:.3g}"
                                           for c, m in receipt.component_medians.items()))
    if receipt.readings:
        print(f"readings past a recorded ceiling (reported, not gated): "
              f"{len(receipt.readings)}")
    if receipt.family_report:
        worst = max(receipt.family_report.items(), key=lambda kv: kv[1])
        print(f"worst family: {worst[0]} at {worst[1]:.3g} of its recorded maximum")
    print(f"receipt: {receipt.digest}  cache: {cache_dir}")
    return 1 if receipt.tier == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
