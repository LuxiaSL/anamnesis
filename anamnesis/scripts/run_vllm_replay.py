"""Replay a banked run through the vLLM lane and write its signatures.

**What this entry point covers.** A model with a vLLM lane (``--model`` offers
exactly those), whole-prompt prefill of spans that fit the lane's context, one
GPU, and only on a host whose install check (`qualify_vllm.py`) is cached and did
not refuse it. The command refuses before building an engine otherwise.

**What an output is.** The same banked format the fast lane writes: a feature
vector and a metadata sidecar per generation, no raw tensors, beside a deployment
record of what produced them. Every row carries the lane id the install check assigned
(the fixtures' lane id on an ``identical`` host, the host's own otherwise) and
an ``extraction_lane`` receipt naming the tier and the receipt digest.
:mod:`anamnesis.analysis.lane_guard` refuses to combine rows of different lanes
inside one contrast, and a vLLM lane is never the fast lane.

Rows are captured and reduced in chunks, because a captured row's substrate is
large; ``--work-dir`` holds one chunk's captures at a time and ``--chunk-rows``
sets its size.

Exit status: 0 banked, 2 refused before or while banking, with the reason on
stderr. An existing ``--output`` is refused, never reused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from anamnesis.extraction.vllm.envelope import LANE_MODELS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_vllm_replay.py", description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(LANE_MODELS), required=True)
    p.add_argument("--model-path", type=Path, required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path,
                   help="The lane's calibration; fetched and verified when omitted")
    p.add_argument("--manifest", type=Path, required=True, help="Replay manifest of a banked run")
    p.add_argument("--output", type=Path, required=True,
                   help="New directory; existing outputs are never overwritten")
    p.add_argument("--work-dir", type=Path,
                   help="New directory for chunk captures; defaults beside --output")
    p.add_argument("--gen-ids", type=int, nargs="+",
                   help="Generations to replay; the whole manifest by default")
    p.add_argument("--metadata", type=Path,
                   help="Source generation metadata; defaults to metadata.json beside the "
                        "manifest when present")
    p.add_argument("--chunk-rows", type=int, default=32,
                   help="Rows captured and reduced together (at least 8)")
    p.add_argument("--cache-dir", type=Path,
                   help="Install-check receipt cache; defaults to vllm_conformance under "
                        "the output root")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from anamnesis.extraction.replay.manifest import ReplayManifest
    from anamnesis.extraction.vllm.hub import fetch_calibration, verify_calibration
    from anamnesis.extraction.vllm.runtime import (
        default_cache_dir,
        load_fixtures,
        replay_bank,
        replay_rows,
        source_metadata,
    )
    from anamnesis.provenance import file_sha

    work_dir = args.work_dir or args.output.with_name(args.output.name + ".work")
    try:
        if args.output.exists():
            raise FileExistsError(f"{args.output} exists; outputs are never overwritten")
        load_fixtures(args.model)
        manifest = ReplayManifest.model_validate_json(args.manifest.read_text())
        ids = list(manifest.gen_ids()) if args.gen_ids is None else args.gen_ids
        if not ids or len(set(ids)) != len(ids) or any(str(i) not in manifest.entries
                                                       for i in ids):
            raise ValueError("empty, duplicated or unknown generation selection")
        metadata_path = args.metadata
        if metadata_path is None and (args.manifest.parent / "metadata.json").exists():
            metadata_path = args.manifest.parent / "metadata.json"
        metadata = source_metadata(metadata_path)
        if metadata_path is not None and any(i not in metadata for i in ids):
            raise ValueError("source metadata is missing selected generation ids")
        entries = {key: entry.model_dump() for key, entry in manifest.entries.items()}
        rows = replay_rows(entries, ids)
        provenance = dict(
            model=args.model, model_path=str(args.model_path),
            manifest_sha256=file_sha(args.manifest), selected_ids=ids,
            source_metadata_sha256=file_sha(metadata_path) if metadata_path else None,
            runner_sha256=file_sha(Path(__file__)))
        if args.calib_dir is not None:
            verify_calibration(args.model, args.calib_dir)
        calib_dir = args.calib_dir or fetch_calibration(args.model)
        written = replay_bank(args.model, args.model_path, calib_dir, rows, args.output,
                              work_dir, args.cache_dir or default_cache_dir(),
                              chunk_rows=args.chunk_rows, metadata=metadata,
                              provenance=provenance)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"the replay did not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(model=args.model, rows=written, output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
