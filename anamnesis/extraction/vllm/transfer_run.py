"""Running the transfer check on a device: a fine-tune's sample through every path.

:mod:`anamnesis.extraction.vllm.transfer` decides from vectors; this module
produces them, and writes what a pass yields. :func:`run_transfer` takes a
fine-tune's checkpoint, its calibration at its base's dtype, and a sample of its
own banked rows, and in order:

1. refuses, before any device work, a key that is already a lane, a base that is
   not shipped, a registry preset that is not a structural copy of the base's, a
   sample outside the allowed size, a span the lane's context or schema does not
   hold, and an output or work directory that exists;
2. captures and reduces the sample through the vLLM lane under the base's engine
   settings, twice one at a time and once in batches of eight
   (:func:`anamnesis.extraction.vllm.runtime.capture_repeats`), in the lane's own
   processes;
3. loads the checkpoint in this process through the fast lane
   (:func:`anamnesis.extraction.fast.runtime.resolve_fast_lane`) for the reference
   vectors, and through the numeric anchor twice, one teacher-forced forward
   (:func:`anamnesis.extraction.replay.extract.replay_extract`) and prefill plus
   one-token steps (:func:`anamnesis.extraction.replay.cached.replay_extract_incremental`),
   for the fine-tune's own σ_cal and path floors;
4. decides (:func:`anamnesis.extraction.vllm.transfer.check_transfer`) and writes the
   receipt, and on a pass the fixtures, the tolerance and a lane-file entry
   (:func:`write_transfer`).

The vLLM steps run first because they run in child processes that need the
device to themselves; the fast lane then holds the model in this one.

The engine steps run under the base's key: an extension inherits every engine
setting, and its own key is not a lane until its receipt exists. The capture
records in the work directory therefore name the base's lane id; the receipt and
the fixtures name the extension's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anamnesis.extraction.vllm.envelope import LANE_MODELS
from anamnesis.extraction.vllm.extensions import (
    CalibrationPin,
    calibration_pins,
    check_preset,
    lane_entry,
    lane_keys,
)
from anamnesis.extraction.vllm.runtime import (
    calibration_digest,
    capture_repeats,
    load_fixtures,
    replay_rows,
    span_schemas,
)
from anamnesis.extraction.vllm.transfer import (
    BASE_MAX_FLOOR,
    SAMPLE_ROWS,
    SampleRow,
    TransferResult,
    TransferSample,
    check_transfer,
    path_ruler,
    select_strata,
)
from anamnesis.provenance import digest_of_shas, file_sha

RECEIPT_FILE = "transfer_receipt.json"
FIXTURES_DIR = "fixtures"
ENTRY_FILE = "lane-entry.json"


@dataclass(frozen=True)
class HfSample:
    """The sample's vectors from the fast lane and from the anchor's two paths."""

    feature_names: tuple[str, ...]
    reference: dict[int, np.ndarray]
    replay: dict[int, np.ndarray]
    incremental: dict[int, np.ndarray]


def hf_sample(preset: str, model_path: Path, calib_dir: Path,
              entries: Mapping[str, Any], ids: Sequence[int], device: str) -> HfSample:
    """Replay ``ids`` through the fast lane and through both anchor paths.

    The model is loaded once, by the fast lane's own resolution, and released
    before returning.

    Raises
    ------
    ValueError
        From the fast lane's resolution, or when an anchor path's schema differs
        from the lane's.
    """
    import gc

    import torch

    from anamnesis.config import resolve_preset
    from anamnesis.extraction.fast.runtime import resolve_fast_lane
    from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
    from anamnesis.extraction.replay.cached import replay_extract_incremental
    from anamnesis.extraction.replay.extract import replay_extract

    runtime = resolve_fast_lane(preset=resolve_preset(preset), model_path=str(model_path),
                                calib_dir=Path(calib_dir), entries=entries, gen_ids=list(ids),
                                device=device, require_local_weights=True)
    reference, replay, incremental = {}, {}, {}
    try:
        for span in runtime.spans:
            tokens, start, end = span.input_ids, span.prompt_length, span.end
            reference[span.gen_id] = np.asarray(
                runtime.lane.replay_span(runtime.loaded, tokens, start, end).features,
                dtype=np.float32)
            for path, target in ((replay_extract, replay),
                                 (replay_extract_incremental, incremental)):
                raw = path(runtime.loaded, tokens, start, runtime.positional_means)
                result = compute_features_with_families_from_data(
                    raw, runtime.extraction, runtime.families, runtime.pca_components,
                    runtime.pca_mean)
                if tuple(result.feature_names) != tuple(runtime.feature_names):
                    raise ValueError(f"generation {span.gen_id}: the anchor's schema differs "
                                     "from the fast lane's")
                target[span.gen_id] = np.asarray(result.features, dtype=np.float32)
        return HfSample(feature_names=tuple(runtime.feature_names), reference=reference,
                        replay=replay, incremental=incremental)
    finally:
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_transfer(out: Path, result: TransferResult, *, calib_dir: Path, preset: str,
                   pins: Mapping[str, CalibrationPin]) -> dict[str, Path]:
    """Write the receipt, and on a pass the fixtures, the tolerance and a lane entry.

    ``out`` must not exist. The entry names the fixtures and the receipt relative
    to ``out`` and the calibration directory absolutely, so ``out``'s entry file
    can be named in :data:`anamnesis.extraction.vllm.extensions.LANES_ENV` as it is.
    Returns what was written, by role.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    written = {"receipt": out / RECEIPT_FILE}
    if result.fixtures is not None and result.tolerance is not None:
        fixtures_dir = result.fixtures.save(out / FIXTURES_DIR)
        (fixtures_dir / "tolerance.json").write_text(
            result.tolerance.model_dump_json(indent=2) + "\n")
        written["fixtures"] = fixtures_dir
    written["receipt"].write_text(result.receipt.model_dump_json(indent=2) + "\n")
    if "fixtures" in written:
        receipt = result.receipt
        entry = lane_entry(key=receipt.key, extends=receipt.extends, preset=preset,
                           checkpoint_sha256=receipt.checkpoint_sha256,
                           calibration_dir=Path(calib_dir).resolve(), calibration=pins,
                           fixtures_dir=FIXTURES_DIR, transfer_receipt=RECEIPT_FILE,
                           transfer_receipt_sha256=file_sha(written["receipt"]))
        written["entry"] = out / ENTRY_FILE
        written["entry"].write_text(json.dumps(entry, indent=2) + "\n")
    return written


def run_transfer(*, key: str, extends: str, preset: str, model_path: Path, calib_dir: Path,
                 entries: Mapping[str, Any], ids: Sequence[int], out: Path, work_dir: Path,
                 device: str) -> tuple[TransferResult, dict[str, Path]]:
    """Run the transfer check for one fine-tune and write its outputs.

    Returns the result and the files written.

    Raises
    ------
    ValueError
        For every refusal that precedes a verdict: the pre-flight checks listed in
        this module's description, a row any capture pass lost, or a selection
        rule under which every row ties.
    RuntimeError
        When an engine or readout step fails.
    """
    from anamnesis.extraction.fast.runtime import weight_file_digests

    if key in lane_keys():
        raise ValueError(f"{key!r} is already a lane; an extension takes a key of its own")
    if extends not in LANE_MODELS:
        raise ValueError(f"{extends!r} is not a shipped vLLM lane (shipped: "
                         f"{', '.join(sorted(LANE_MODELS))})")
    check_preset(preset, extends)
    for directory in (out, work_dir):
        if Path(directory).exists():
            raise FileExistsError(f"{directory} exists; outputs are never overwritten")
    low, high = SAMPLE_ROWS
    if not low <= len(ids) <= high or len(set(ids)) != len(ids):
        raise ValueError(f"a transfer sample is {low} to {high} distinct rows, not {len(ids)}")
    base_fixtures, base_tolerance = load_fixtures(extends)
    names = list(base_fixtures.feature_names)
    rows = replay_rows(entries, ids)
    span_schemas(extends, calib_dir, rows, names)
    checkpoint = digest_of_shas(weight_file_digests(model_path))
    pins = calibration_pins(calib_dir)

    captured = capture_repeats(extends, model_path, calib_dir, rows, names, Path(work_dir))
    lost = sorted(set(ids) - {c.generation_id for c in captured})
    if lost:
        raise ValueError(f"rows {lost[:5]} are missing from a capture pass")
    hf = hf_sample(preset, model_path, calib_dir, entries, ids, device)
    order = sorted(ids)
    sigma, floors = path_ruler(np.stack([hf.replay[g] for g in order]),
                               np.stack([hf.incremental[g] for g in order]))
    by_id = {r["generation_id"]: r for r in rows}
    sample = TransferSample(
        feature_names=hf.feature_names, sigma_cal=sigma,
        weights=np.ones_like(sigma),
        rows=tuple(SampleRow(generation_id=g, input_ids=tuple(by_id[g]["input_ids"]),
                             prompt_length=by_id[g]["prompt_length"], end=by_id[g]["end"],
                             floor_b=float(f)) for g, f in zip(order, floors)),
        reference=hf.reference,
        candidate={c.generation_id: c.first for c in captured})
    result = check_transfer(
        key=key, extends=extends, base_tolerance=base_tolerance, base_feature_names=names,
        checkpoint_sha256=checkpoint, calibration_sha256=calibration_digest(calib_dir),
        sample=sample, strata=select_strata(sample, base_tolerance), determinism=captured,
        max_floor=BASE_MAX_FLOOR[extends])
    return result, write_transfer(out, result, calib_dir=calib_dir, preset=preset, pins=pins)
