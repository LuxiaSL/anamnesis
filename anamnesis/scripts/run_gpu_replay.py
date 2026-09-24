"""Replay a banked run through the fast lane and write its signatures.

**What this entry point covers.** A dense Llama named by any registry preset
(``--model`` offers every key :func:`anamnesis.config.preset_names` holds, so a row
added through ``ANAMNESIS_MODELS`` is accepted as soon as it is readable), one full
teacher-forced pass per span, the complete probe-free battery, on a single device.
That is the configuration the lane tests and the equivalence suite exercise. A
checkpoint whose architecture, depth or widths are not what the preset declares is
refused once loaded, before any span runs.

**What it does not cover, and refuses rather than approximates.** Adapters,
activation interventions and batched submission each change what a forward pass
is, so each needs its own qualification against the numeric anchor before its
numbers mean anything. None of the three is an argument here; a command line
that names one is rejected instead of being reinterpreted as the covered case.

**What an output is.** Features and metadata, no raw tensors, so a bank written
here cannot be re-featurised later — it records the battery as configured at
replay time and nothing else. Every row carries the lane identity, and rows
carrying different identities must not be combined inside one contrast
(:mod:`anamnesis.analysis.lane_guard` enforces that on the read side).
Agreement with the anchor is a property of the machine, measured by
`qualify_box.py`, not something a run inherits from this file. The two commands
build the lane through one resolution,
:func:`anamnesis.extraction.fast.runtime.resolve_fast_lane`, so the configuration
a receipt describes is the configuration a bank was produced under.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from anamnesis.config import preset_names
from anamnesis.extraction.fast.runtime import (
    DEFAULT_DEVICE,
    require_lane_arithmetic,
    resolve_fast_lane,
)
from anamnesis.provenance import file_sha


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", choices=preset_names(), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing outputs are never overwritten",
    )
    p.add_argument("--gen-ids", type=int, nargs="+")
    p.add_argument(
        "--metadata",
        type=Path,
        help="Source generation metadata; defaults to metadata.json beside manifest when present",
    )
    p.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device the lane and the model run on; every row is about this device",
    )
    return p


def read_generation_metadata(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    document = json.loads(path.read_text())
    generations = document["generations"] if isinstance(document, dict) else document
    if not isinstance(generations, list):
        raise ValueError("source metadata must contain a generation list")
    result = {}
    for record in generations:
        key = int(record["generation_id"])
        if key in result:
            raise ValueError("duplicate generation metadata")
        result[key] = dict(record)
    return result


def select_ids(entries: dict, requested: list[int] | None) -> list[int]:
    """The generations to replay: the whole manifest, or exactly what was asked for.

    Raises
    ------
    ValueError
        When the selection is empty, names an id twice, or names one the manifest
        does not hold. A silently narrowed selection would bank a partial cell under
        the name of a whole one.
    """
    ids = sorted(int(k) for k in entries) if requested is None else requested
    if not ids or len(set(ids)) != len(ids) or any(str(i) not in entries for i in ids):
        raise ValueError("empty, duplicated or unknown generation selection")
    return ids


def main(argv: Sequence[str] | None = None) -> None:
    """Replay the selected generations and bank their signatures.

    ``argv`` is the argument list without the program name; ``None`` reads the
    process's own command line.
    """
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    require_lane_arithmetic()
    import torch
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.state_extractor import ExtractionResult
    from anamnesis.extraction.feature_pipeline import save_features

    entries = json.loads(args.manifest.read_text())["entries"]
    source_path = args.metadata
    if source_path is None and (args.manifest.parent / "metadata.json").exists():
        source_path = args.manifest.parent / "metadata.json"
    source_metadata = read_generation_metadata(source_path)
    ids = select_ids(entries, args.gen_ids)
    if source_path is not None and any(i not in source_metadata for i in ids):
        raise ValueError("source metadata is missing selected generation IDs")
    runtime = resolve_fast_lane(
        preset=resolve_preset(args.model),
        model_path=args.model_path,
        calib_dir=args.calib_dir,
        entries=entries,
        gen_ids=ids,
        device=args.device,
        require_local_weights=True,
    )
    lane, loaded = runtime.lane, runtime.loaded
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        lane=lane.identity,
        lane_id=lane.lane_id,
        manifest_sha256=file_sha(args.manifest),
        calibration_files=runtime.calibration_files,
        model_path=args.model_path,
        device=args.device,
        model_files_sha256=runtime.model_files,
        runner_sha256=file_sha(Path(__file__)),
        configuration_source_sha256=file_sha(
            Path(__file__).parents[1] / "extraction/replay_config.py"
        ),
        schema_source_sha256=file_sha(
            Path(__file__).parents[1] / "extraction/fast/schema.py"
        ),
        # The lane's runtime — its determinism pins, span resolution, calibration load
        # and model construction — decides what the numbers below were produced by, so
        # the receipt covers it alongside the schema it resolves against. Without this
        # digest the stamp attests to a configuration it does not describe.
        lane_runtime_source_sha256=file_sha(
            Path(__file__).parents[1] / "extraction/fast/runtime.py"
        ),
        model_config=loaded.model.config.to_dict(),
        selected_ids=ids,
        source_metadata_sha256=file_sha(source_path)
        if source_path is not None
        else None,
        raw_tensors_saved=False,
        certified=False,
    )
    (args.output / "deployment.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    # Kernel launches return before the kernels finish, so a CUDA timing without a
    # barrier on each side measures the launch rather than the replay.
    on_cuda = torch.device(args.device).type == "cuda"
    for span in runtime.spans:
        i = span.gen_id
        if on_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = lane.replay_span(loaded, span.input_ids, span.prompt_length, span.end)
        if on_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        converted = ExtractionResult(
            result.features,
            list(result.feature_names),
            runtime.schemas[i].family_slices,
            result.knnlm_baseline,
        )
        metadata = dict(source_metadata.get(i, {}))
        metadata.update(
            generation_id=i,
            lane_id=lane.lane_id,
            extraction_lane=result.metadata,
            replay_seconds=elapsed,
            mean_logprob=result.mean_logprob,
            raw_tensors_saved=False,
            certified=False,
        )
        save_features(
            i,
            converted,
            metadata,
            args.output,
        )
        print(
            json.dumps(
                dict(
                    generation_id=i,
                    seconds=elapsed,
                    lane_id=lane.lane_id,
                    certified=False,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
