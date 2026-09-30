"""Audit how far a fine-tune's vLLM lane sits from its fast lane, against its base's regime.

A lane-agreement audit (:mod:`anamnesis.extraction.vllm.transfer`): run it when a
claim needs the two lanes to agree, before a finding is said to hold beyond one
lane, or when a serving change should have left the computation alone. Its
verdict is information, not a condition of using either lane.

The fine-tune needs a registry preset extending the base's with the same
architecture and layer plan (``ANAMNESIS_MODELS``), its own calibration fitted at
that preset, and a replay manifest of 44 to 60 rows it generated. On one GPU the
command captures the sample through the vLLM lane under the base's settings (twice
alone, once batched; all must agree byte for byte), then through the fast lane and
the numeric anchor's two paths for the fine-tune's own σ_cal and path floors, and
scores it with :func:`anamnesis.extraction.vllm.transfer.check_transfer`. The vLLM
steps run first, in child processes that need the device to themselves; their
capture records name the base's lane id, because the key is not a lane until it
is declared.

``--out`` receives the transfer receipt and, whenever the sample was scored,
whatever the verdict, the fine-tune's fixtures and tolerance and a lane file
declaring the extension with the receipt as its evidence, to be named in
``ANAMNESIS_VLLM_LANES``.

Exit status: 0 pass, 1 refuse, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from anamnesis.extraction.vllm.envelope import LANE_MODELS
from anamnesis.extraction.vllm.transfer import TransferResult

RECEIPT_FILE = "transfer_receipt.json"
FIXTURES_DIR = "fixtures"
ENTRY_FILE = "lane-entry.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transfer_vllm.py", description=__doc__.splitlines()[0])
    p.add_argument("--key", required=True, help="The extension lane's key; a new name")
    p.add_argument("--extends", choices=sorted(LANE_MODELS), required=True)
    p.add_argument("--preset", required=True, help="The fine-tune's registry preset")
    p.add_argument("--model-path", type=Path, required=True, help="Local checkpoint directory")
    p.add_argument("--calib-dir", type=Path, required=True, help="The fine-tune's calibration")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Replay manifest of the fine-tune's own rows")
    p.add_argument("--gen-ids", type=int, nargs="+", help="The sample; the whole manifest by "
                                                          "default")
    p.add_argument("--out", type=Path, required=True, help="New directory for the outputs")
    p.add_argument("--work-dir", type=Path, help="New directory for the vLLM captures")
    p.add_argument("--device", default="cuda:0", help="Device for the Hugging Face model")
    return p


def hf_vectors(preset: str, model_path: Path, calib_dir: Path, entries: Mapping[str, Any],
               ids: Sequence[int], device: str, *, lane_hidden: Path | None = None,
               rules: Any = None) -> tuple[tuple[str, ...], dict, dict, dict, dict]:
    """Feature names, and per row the fast-lane vector, the anchor's replay and
    incremental vectors, and, where ``lane_hidden`` holds the row's lane block outputs,
    the bifurcation diagnostic against the anchor's one-forward replay, reading from ``rules``' depth
    (each kept file is deleted once read); the model is released before returning."""
    import torch

    from anamnesis.config import resolve_preset
    from anamnesis.extraction.fast.runtime import resolve_fast_lane
    from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
    from anamnesis.extraction.replay.cached import replay_extract_incremental
    from anamnesis.extraction.replay.extract import replay_extract
    from anamnesis.extraction.vllm.transfer import bifurcation_report

    lane = resolve_fast_lane(preset=resolve_preset(preset), model_path=str(model_path),
                             calib_dir=Path(calib_dir), entries=entries, gen_ids=list(ids),
                             device=device, require_local_weights=True)
    out: tuple[dict, dict, dict] = ({}, {}, {})
    reports: dict[int, Any] = {}
    try:
        for span in lane.spans:
            args = (span.input_ids, span.prompt_length)
            out[0][span.gen_id] = np.asarray(lane.lane.replay_span(
                lane.loaded, *args, span.end).features, dtype=np.float32)
            for path, target in ((replay_extract, out[1]), (replay_extract_incremental, out[2])):
                raw = path(lane.loaded, *args, lane.positional_means)
                kept = None if lane_hidden is None or rules is None \
                    or path is not replay_extract \
                    else lane_hidden / f"row-{span.gen_id:05d}.hidden.pt"
                if kept is not None and kept.is_file():
                    blocks = torch.load(kept, map_location="cpu", weights_only=True)
                    reports[span.gen_id] = bifurcation_report(
                        blocks.float().numpy(), np.stack(raw.hidden_states),
                        rules.diagnostic_from_depth)
                    kept.unlink()
                result = compute_features_with_families_from_data(
                    raw, lane.extraction, lane.families, lane.pca_components, lane.pca_mean)
                if tuple(result.feature_names) != tuple(lane.feature_names):
                    raise ValueError(f"generation {span.gen_id}: the anchor's schema differs "
                                     "from the fast lane's")
                target[span.gen_id] = np.asarray(result.features, dtype=np.float32)
        return (tuple(lane.feature_names), *out, reports)
    finally:
        del lane
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_transfer(out: Path, result: TransferResult, *, calib_dir: Path,
                   preset: str) -> dict[str, Path]:
    """Write the receipt, and for a scored sample the fixtures, tolerance and a lane
    file whose entry names them relative to ``out``; ``out`` must not exist."""
    from anamnesis.extraction.vllm.extensions import calibration_pins
    from anamnesis.provenance import file_sha

    out.mkdir(parents=True, exist_ok=False)
    written = {"receipt": out / RECEIPT_FILE}
    written["receipt"].write_text(result.receipt.model_dump_json(indent=2) + "\n")
    if result.fixtures is None or result.tolerance is None:
        return written
    written["fixtures"] = result.fixtures.save(out / FIXTURES_DIR)
    (written["fixtures"] / "tolerance.json").write_text(result.tolerance.model_dump_json())
    r = result.receipt
    entry = dict(extends=r.extends, preset=preset, checkpoint_sha256=r.checkpoint_sha256,
                 calibration_dir=str(Path(calib_dir).resolve()),
                 calibration={n: p.model_dump() for n, p in calibration_pins(calib_dir).items()},
                 fixtures_dir=FIXTURES_DIR, transfer_receipt=RECEIPT_FILE,
                 transfer_receipt_sha256=file_sha(written["receipt"]))
    written["entry"] = out / ENTRY_FILE
    written["entry"].write_text(json.dumps({"lanes": {r.key: entry}}, indent=2) + "\n")
    return written


def run_transfer(*, key: str, extends: str, preset: str, model_path: Path, calib_dir: Path,
                 entries: Mapping[str, Any], ids: Sequence[int], out: Path, work_dir: Path,
                 device: str) -> tuple[TransferResult, dict[str, Path]]:
    """Run the check and write its outputs.

    Raises
    ------
    ValueError
        Before any device work: a key that is already a lane, a preset that is not a
        structural copy of the base's, a sample outside 44 to 60 distinct rows, or a
        span the lane's context or schema does not hold; after it, a row a capture
        pass lost or a selection rule under which every row ties.
    FileExistsError
        When ``out`` or ``work_dir`` exists.
    """
    from anamnesis.config import resolve_preset
    from anamnesis.extraction.fast.runtime import (
        read_lane_calibration, resolve_lane_spans, weight_file_digests)
    from anamnesis.extraction.vllm.extensions import check_preset, lane_keys
    from anamnesis.extraction.vllm.runtime import (
        calibration_digest, capture_repeats, load_fixtures, replay_rows, span_schemas)
    from anamnesis.extraction.vllm.transfer import (
        BASE_MAX_FLOOR, SAMPLE_ROWS, TransferSample, check_transfer, load_transfer_rules,
        path_ruler, select_strata)
    from anamnesis.extraction.vllm.conformance import FixtureRow
    from anamnesis.provenance import digest_of_shas

    if key in lane_keys():
        raise ValueError(f"{key!r} is already a lane; an extension takes a key of its own")
    check_preset(preset, extends)
    for directory in (out, work_dir):
        if Path(directory).exists():
            raise FileExistsError(f"{directory} exists; outputs are never overwritten")
    if not SAMPLE_ROWS[0] <= len(set(ids)) == len(ids) <= SAMPLE_ROWS[1]:
        raise ValueError(f"a transfer sample is {SAMPLE_ROWS[0]} to {SAMPLE_ROWS[1]} distinct "
                         f"rows, not {len(ids)}")
    base_fixtures, base_tolerance = load_fixtures(extends)
    rules = load_transfer_rules(extends, base_tolerance)
    names = list(base_fixtures.feature_names)
    rows = replay_rows(entries, ids)
    span_schemas(extends, calib_dir, rows, names)
    # The anchor's side holds a span only up to the positions the calibration fills;
    # checked here, before any capture, rather than after the engine has run.
    resolve_lane_spans(entries, ids, positions_calibrated=read_lane_calibration(
        resolve_preset(preset), Path(calib_dir)).positions_calibrated)
    checkpoint = digest_of_shas(weight_file_digests(model_path))

    kept = Path(work_dir) / "single" / "pass-0"
    try:
        captured = capture_repeats(extends, model_path, calib_dir, rows, names, Path(work_dir),
                                   keep_hidden=True)
        lost = sorted(set(ids) - {c.generation_id for c in captured})
        if lost:
            raise ValueError(f"rows {lost[:5]} are missing from a capture pass")
        feature_names, reference, replay, incremental, reports = hf_vectors(
            preset, model_path, calib_dir, entries, ids, device, lane_hidden=kept, rules=rules)
    finally:
        # The kept block outputs are a working copy for the diagnostic, never a record.
        for path in kept.glob("row-*.hidden.pt"):
            path.unlink()
    order = sorted(ids)
    sigma, floors = path_ruler(np.stack([replay[g] for g in order]),
                               np.stack([incremental[g] for g in order]))
    by_id = {r["generation_id"]: r for r in rows}
    sample = TransferSample(
        feature_names=feature_names, sigma_cal=sigma, weights=np.ones_like(sigma),
        rows=tuple(FixtureRow(generation_id=g, population="native",
                              input_ids=tuple(by_id[g]["input_ids"]),
                              prompt_length=by_id[g]["prompt_length"], end=by_id[g]["end"],
                              floor_b=float(f), selected_by="sampled")
                   for g, f in zip(order, floors)),
        reference=reference, candidate={c.generation_id: c.first for c in captured})
    result = check_transfer(
        key=key, extends=extends, base_tolerance=base_tolerance, base_feature_names=names,
        checkpoint_sha256=checkpoint, calibration_sha256=calibration_digest(calib_dir),
        sample=sample, strata=select_strata(sample, base_tolerance), determinism=captured,
        max_floor=BASE_MAX_FLOOR[extends], rules=rules, diagnostics=reports)
    return result, write_transfer(out, result, calib_dir=calib_dir, preset=preset)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from anamnesis.extraction.replay.manifest import ReplayManifest

    try:
        manifest = ReplayManifest.model_validate_json(args.manifest.read_text())
        ids = sorted(manifest.gen_ids()) if args.gen_ids is None else list(args.gen_ids)
        if any(str(i) not in manifest.entries for i in ids):
            raise ValueError("the sample names generations the manifest does not hold")
        result, written = run_transfer(
            key=args.key, extends=args.extends, preset=args.preset,
            model_path=args.model_path, calib_dir=args.calib_dir,
            entries={k: e.model_dump() for k, e in manifest.entries.items()}, ids=ids,
            out=args.out, work_dir=args.work_dir or args.out.with_name(args.out.name + ".work"),
            device=args.device)
    except (ValueError, RuntimeError, OSError, ImportError, NotImplementedError) as exc:
        print(f"the transfer check did not run: {exc}", file=sys.stderr)
        return 2
    receipt = result.receipt
    print(f"{receipt.key} extends {receipt.extends}: {receipt.verdict}  lane id: "
          f"{receipt.lane_id}")
    for reason in receipt.reasons[:10]:
        print(f"  {reason}")
    print(json.dumps({role: str(path) for role, path in written.items()}))
    return 0 if receipt.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
