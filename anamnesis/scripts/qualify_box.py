"""Qualify this machine: does its fast lane agree with the numeric anchor?

**Different hardware gives different numbers, and that is expected rather than
wrong.** Floating-point reductions are not associative, so a different BLAS
build, GPU architecture, dtype policy or thread count changes the last digits of
a feature. Nothing in the program claims otherwise. What the program does claim
is that on *one* machine the fast lane and the numeric anchor compute the same
features, and that a set of signatures compared inside one contrast came from one
machine and one lane. The first claim is what this script measures. The second is
why the measurement matters: **results from different boxes must not be mixed
inside one contrast**, and this is how you learn which box yours is.

What it does, over a small sample of a banked run's spans:

1. **Repeatability.** Replays each span through the lane twice and compares the
   feature vectors byte for byte. A box that is not repeatable cannot be
   qualified, and nothing downstream of that is worth measuring.
2. **Agreement.** Computes the same spans through the anchor
   (:mod:`anamnesis.extraction.state_extractor`, via the feature pipeline) and
   hands both to :mod:`anamnesis.extraction.equivalence.fidelity`, which renders
   the verdicts against a ruler this script states explicitly.
3. **A path bound.** Prices each row's bound with
   :mod:`anamnesis.extraction.equivalence.path_floor`, from the first incremental
   step alone, and hands it over as the declared lower bound it is.

The ruler is unit-scaled: every feature weighted 1, every sigma 1, so a distance
is in raw feature units. That is a deliberately plain choice and it is stated in
the receipt, because a standardizing ruler belongs to a cohort and a cohort is a
scientific object this script does not have. A deployment that has one passes its
own sigma to `fidelity` directly.

**A pass here is not a certification.** Certification belongs to a deployment,
with its own cohort, its own ruler and its own recorded scope. This prints what
is true of one machine: whether it repeats, how far its two paths are apart, and
the lane identity its outputs will carry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.config import MODEL_PRESETS, ModelConfig
from anamnesis.extraction.equivalence.fidelity import (
    FidelityError,
    ReplayBatch,
    Ruler,
    verify_vectors,
)
from anamnesis.extraction.equivalence.path_floor import (
    coordinate_lower_bound,
    first_incremental_coordinates,
    first_position_coordinates,
)
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.extraction.calibration import load_calibration
from anamnesis.provenance import digest_of_shas, file_sha

ANCHOR_LANE = "numpy-anchor"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qualify_box.py", description=__doc__.splitlines()[0]
    )
    p.add_argument("--model", choices=tuple(MODEL_PRESETS), required=True)
    p.add_argument("--model-path", required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, required=True, help="Replay manifest of a banked run"
    )
    p.add_argument(
        "--gen-ids",
        type=int,
        nargs="+",
        help="Generations to qualify over; the first two of the manifest by default",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Device the lane and the model run on; the verdict is about this device",
    )
    p.add_argument("--json", type=Path, help="Write the full receipt here")
    return p


def _digest(*arrays: np.ndarray) -> str:
    running = hashlib.sha256()
    for array in arrays:
        running.update(np.ascontiguousarray(array).tobytes())
    return running.hexdigest()


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    """Run the three legs and return the receipt. Requires a model on `--device`.

    The arithmetic settings are pinned to the ones `run_gpu_replay.py` runs under.
    They are part of the lane identity, so a verdict reached under different
    settings would be a verdict about a lane the user is not going to use.
    """
    import torch

    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.fast.schema import resolve_gpu_schema
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay.extract import replay_extract

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":16:8":
        raise ValueError(
            "set CUBLAS_WORKSPACE_CONFIG=:16:8 before Python startup; it is part of "
            "the lane identity, so a verdict without it is about a different lane"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preset = MODEL_PRESETS[args.model]
    extraction, families = native_replay_configs(preset)
    entries = json.loads(args.manifest.read_text())["entries"]
    ids = args.gen_ids if args.gen_ids else sorted(int(k) for k in entries)[:2]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("two or more distinct generations are required to pair rows")
    if any(str(i) not in entries for i in ids):
        raise ValueError("unknown generation selection")
    positional_means, components, pca_mean = load_calibration(args.calib_dir, True)
    if any(v is None for v in (positional_means, components, pca_mean)):
        raise ValueError("complete positional/PCA calibration required")
    calibration_sha256 = digest_of_shas(
        {
            name: file_sha(args.calib_dir / name)
            for name in ("positional_means.npz", "pca_model.pkl")
        }
    )

    spans = []
    for i in ids:
        row = entries[str(i)]
        start, end = int(row["prompt_length"]), len(row["input_ids"])
        if not 0 < start < end - 1 or end - 2 >= positional_means.shape[1]:
            raise ValueError(f"generation {i} outside supported span/calibration")
        spans.append((i, list(row["input_ids"]), start, end))
    schemas = {
        i: resolve_gpu_schema(
            preset.num_layers, end - start - 1, extraction, families, components
        )
        for i, _, start, end in spans
    }
    names = {s.feature_names for s in schemas.values()}
    if len(names) != 1:
        raise ValueError("selected spans have different feature schemas; do not mix")
    feature_names = names.pop()

    loaded = load_model(
        ModelConfig.from_preset(
            preset, model_id=args.model_path, device_map=args.device
        ),
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=list(range(preset.num_layers)),
        value_layers=list(range(preset.num_layers)),
        query_layers=list(range(preset.num_layers)),
        attn_output_layers=list(range(preset.num_layers)),
    )
    lane = GpuFeatureLane(
        extraction,
        families,
        list(feature_names),
        positional_means,
        components,
        pca_mean,
        device=args.device,
        calibration_sha256=calibration_sha256,
        replay_path="full",
    )

    keys: list[tuple[str, str, str, int, str]] = []
    anchor: list[np.ndarray] = []
    candidate: list[np.ndarray] = []
    again: list[np.ndarray] = []
    inputs: list[str] = []
    bounds: dict[tuple[str, str, str, int, str], float] = {}
    proofs: dict[tuple[str, str, str, int, str], str] = {}
    prefixes: dict[tuple[str, str, str, int, str], int] = {}
    for i, tokens, start, end in spans:
        key = ("qualify", str(i), "full", 0, f"{start}:{end}")
        raw = replay_extract(loaded, tokens, start, positional_means)
        reference = compute_features_v2_from_data(
            raw, extraction, families, components, pca_mean
        )
        if tuple(reference.feature_names) != feature_names:
            raise ValueError(f"generation {i}: anchor schema differs from the lane's")
        first = lane.replay_span(loaded, tokens, start, end)
        repeat = lane.replay_span(loaded, tokens, start, end)
        reference_coordinates, _ = first_position_coordinates(
            np.ascontiguousarray(raw.hidden_states[0]),
            positional_means,
            start,
            extraction,
        )
        incremental, _ = first_incremental_coordinates(
            loaded, tokens, start, end, positional_means, extraction
        )
        # Unit ruler: the bound comes back in raw feature units, which is the only
        # scale available without a cohort standardizer.
        ones = np.ones(len(incremental), dtype=np.float64)
        bound = coordinate_lower_bound(incremental, reference_coordinates, ones, ones)
        keys.append(key)
        anchor.append(reference.features)
        candidate.append(first.features)
        again.append(repeat.features)
        inputs.append(first.metadata["input_tokens_sha256"])
        bounds[key] = bound["floor_b_lower_bound"]
        proofs[key] = _digest(incremental, reference_coordinates)
        prefixes[key] = start
        if first.metadata["replay_id"] == repeat.metadata["replay_id"]:
            raise ValueError("two replays of one span shared a replay id")

    rows = tuple(keys)
    stack_sha256 = hashlib.sha256(
        json.dumps(lane.identity, sort_keys=True).encode()
    ).hexdigest()
    schema_sha256 = hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest()

    def batch(features, lane_id, tag, stack):
        return ReplayBatch(
            np.stack(features).astype(np.float32),
            rows,
            feature_names,
            (lane_id,) * len(rows),
            tuple(inputs),
            tuple(f"{tag}-{i}" for i in range(len(rows))),
            stack,
            calibration_sha256,
            schema_sha256,
        )

    anchor_batch = batch(anchor, ANCHOR_LANE, "anchor", schema_sha256)
    candidate_batch = batch(candidate, lane.lane_id, "candidate", stack_sha256)
    repeat_batch = batch(again, lane.lane_id, "repeat", stack_sha256)
    unit = np.ones(len(feature_names), dtype=np.float64)
    pairs = {k: (k, rows[(i + 1) % len(rows)]) for i, k in enumerate(rows)}
    distances = {
        k: float(
            np.linalg.norm(
                anchor_batch.features[rows.index(a)].astype(np.float64)
                - anchor_batch.features[rows.index(b)].astype(np.float64)
            )
        )
        for k, (a, b) in pairs.items()
    }
    ruler = Ruler(
        unit,
        unit,
        _digest(unit, unit),
        feature_names,
        bounds,
        distances,
        pairs,
        prefixes,
        rows,
        calibration_sha256,
        schema_sha256,
        _digest(np.stack(anchor)),
        floor_kinds=dict.fromkeys(rows, "lower_bound"),
        floor_proof_sha256=proofs,
    )
    vectors = verify_vectors(anchor_batch, candidate_batch, repeat_batch, ruler)
    return dict(
        repeatable=vectors["G0"],
        agreement=vectors["G1"],
        lane_id=lane.lane_id,
        device=args.device,
        model=args.model,
        rows=len(rows),
        ruler="unit-scaled; a cohort standardizer is a deployment's own",
        path_bound_kind="lower_bound",
        certified=False,
        vectors=vectors,
        lane_identity=lane.identity,
    )


def render(receipt: dict[str, Any]) -> str:
    """One paragraph a reader can act on, and the identity their outputs will carry."""
    agreement = receipt["agreement"]
    verdict = {True: "AGREES", False: "DISAGREES", None: "UNDECIDED"}[agreement]
    lines = [
        f"box: {receipt['device']}  model: {receipt['model']}  rows: {receipt['rows']}",
        f"repeatable: {'yes' if receipt['repeatable'] else 'NO'}",
        f"fast lane vs numeric anchor: {verdict}",
        f"lane id: {receipt['lane_id']}",
    ]
    if not receipt["repeatable"]:
        lines.append(
            "this box does not reproduce its own vectors; nothing else here is measured"
        )
    elif agreement is None:
        lines.append(
            "a declared lower bound was insufficient for at least one row: measure the "
            "full path bound for those rows, which is more work rather than a failure"
        )
    lines.append(
        "different hardware gives different numbers; that is expected. Signatures "
        "carrying different lane ids must not be combined inside one contrast."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        receipt = qualify(args)
    except (FidelityError, ValueError, OSError) as exc:
        print(f"qualification did not run: {exc}", file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(render(receipt))
    return 0 if receipt["repeatable"] and receipt["agreement"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
