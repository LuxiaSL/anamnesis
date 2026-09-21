"""Replay a banked run through the fast lane and write its signatures.

**What this entry point covers.** Dense 3B/8B Llama, one full teacher-forced
pass per span, the complete probe-free battery, on a single CUDA device. That is
the configuration the lane tests and the equivalence suite exercise, and it is
the configuration this script accepts.

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
`qualify_box.py`, not something a run inherits from this file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from anamnesis.provenance import digest_of_shas, file_sha


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", choices=("3b", "8b"), required=True)
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


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":16:8":
        raise ValueError(
            "this lane identity requires CUBLAS_WORKSPACE_CONFIG=:16:8 before Python startup"
        )
    import torch
    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.calibration import CALIBRATION_ARTIFACT_NAMES, load_calibration
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_config import native_replay_configs
    from anamnesis.extraction.fast.schema import resolve_gpu_schema
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.state_extractor import ExtractionResult
    from anamnesis.extraction.feature_pipeline import save_features

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preset = MODEL_PRESETS[args.model]
    ec, fc = native_replay_configs(preset)
    entries = json.loads(args.manifest.read_text())["entries"]
    source_path = args.metadata
    if source_path is None and (args.manifest.parent / "metadata.json").exists():
        source_path = args.manifest.parent / "metadata.json"
    source_metadata = read_generation_metadata(source_path)
    ids = sorted(int(k) for k in entries) if args.gen_ids is None else args.gen_ids
    if not ids or len(set(ids)) != len(ids) or any(str(i) not in entries for i in ids):
        raise ValueError("empty, duplicated or unknown generation selection")
    if source_path is not None and any(i not in source_metadata for i in ids):
        raise ValueError("source metadata is missing selected generation IDs")
    files = {
        name: file_sha(args.calib_dir / name)
        for name in CALIBRATION_ARTIFACT_NAMES
    }
    model_root = Path(args.model_path)
    weights = sorted(model_root.glob("*.safetensors"))
    if not weights:
        raise ValueError(
            "local safetensors checkpoint required for explicit provenance"
        )
    model_files = {p.name: file_sha(p) for p in [model_root / "config.json", *weights]}
    calibration_sha = digest_of_shas(files)
    pm, components, mean = load_calibration(args.calib_dir, True)
    if any(value is None for value in (pm, components, mean)):
        raise ValueError("complete positional/PCA calibration required")
    schemas = {}
    for i in ids:
        row = entries[str(i)]
        start, end = int(row["prompt_length"]), len(row["input_ids"])
        if not 0 < start < end - 1 or end - 2 >= pm.shape[1]:
            raise ValueError(f"generation {i} outside supported span/calibration")
        schemas[i] = resolve_gpu_schema(
            preset.num_layers, end - start - 1, ec, fc, components
        )
    if len({s.feature_names for s in schemas.values()}) != 1:
        raise ValueError("selected spans have different feature schemas; do not mix")
    cfg = ModelConfig(
        model_id=args.model_path,
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    layers = list(range(preset.num_layers))
    loaded = load_model(
        cfg,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=layers,
        value_layers=layers,
        query_layers=layers,
        attn_output_layers=layers,
    )
    lane = GpuFeatureLane(
        ec,
        fc,
        list(schemas[ids[0]].feature_names),
        pm,
        components,
        mean,
        device="cuda:0",
        calibration_sha256=calibration_sha,
        replay_path="full",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        lane=lane.identity,
        lane_id=lane.lane_id,
        manifest_sha256=file_sha(args.manifest),
        calibration_files=files,
        model_path=args.model_path,
        model_files_sha256=model_files,
        runner_sha256=file_sha(Path(__file__)),
        configuration_source_sha256=file_sha(
            Path(__file__).parents[1] / "extraction/replay_config.py"
        ),
        schema_source_sha256=file_sha(
            Path(__file__).parents[1] / "extraction/fast/schema.py"
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
    for i in ids:
        row = entries[str(i)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = lane.replay_span(
            loaded, row["input_ids"], int(row["prompt_length"]), len(row["input_ids"])
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        converted = ExtractionResult(
            result.features,
            list(result.feature_names),
            schemas[i].family_slices,
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
