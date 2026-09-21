"""One cell's replay: the loop that turns banked tokens back into signatures.

A cell is a run directory, a manifest of token sequences, and optionally an
intervention. Replaying it means, per generation: teacher-force the banked
sequence, save the raw tensors when they are wanted, compute the feature vector,
and write it beside metadata that says what produced it. That loop is here rather
than in an entry point because three callers run it and must run *the same* one:
the single-cell command, the multi-cell command that loads the model once and
walks a roster, and the persistent worker that keeps a model resident across
jobs. Signatures from all three are identical by construction, not by
coincidence, and that is the only reason the faster paths are allowed to exist.

:func:`load_replay_model` is the other half: the capture surface a replay reads.
Keys, values, queries and attention outputs are hooked at *every* layer while the
feature families still consume the preset's sampled layers, so the vector is
unchanged and the extra layers are banked raw — depth stays available as an axis
to measure later without re-running a pass.

A generation that fails is counted and logged with its traceback, and the loop
continues; a cell reports how many succeeded and how many did not. A caller that
cannot accept a partial cell — the persistent worker, whose job must be retried
rather than silently half-done — checks the count and fails on it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig, ModelPreset
from anamnesis.extraction.interventions import InjectionSpec, check_injection_gating
from anamnesis.extraction.replay_config import native_replay_configs
from anamnesis.extraction.state_extractor import STORED_BLOCK_SLICES_KEY

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

DEFAULT_SIGNATURES_SUBDIR = "signatures_v3"
DEFAULT_RAW_SUBDIR = "raw_tensors_v3"
EXTRACTION_VERSION = 3
"""The capture-surface generation these signatures belong to. It is written into
every signature's metadata so a bank can never be read as an earlier one."""

ROUTING_FAMILY = "expert_routing"
"""The family whose presence distinguishes a routing-aware vector. Its version
rider is recorded per signature: absent means a bank predating the surface, zero
means a dense checkpoint that supplies no routing, and a positive value names the
enriched family. A vector's width differs across those cases too, which the
loader's modal-width check catches independently."""
ROUTING_VERSION = 2


@dataclass(frozen=True)
class ReplaySurface:
    """A loaded model, and the two configs the features are computed under."""

    loaded: Any
    extraction: ExtractionConfig
    families: FeaturePipelineConfig


def load_replay_model(
    preset: ModelPreset,
    model_path: str,
    *,
    enable_pca: bool = True,
    adapter_path: str | None = None,
) -> ReplaySurface:
    """Load a model with the full replay capture surface.

    ``model_path`` overrides the preset's identifier, because a replay runs
    against a local checkpoint whose bytes are the provenance; everything else —
    depth, head counts, dtype, the layer plan, the band cutoffs — comes from the
    preset row, so a replay cannot describe a model other than the one it loads.
    """
    extraction, families = native_replay_configs(preset, enable_pca=enable_pca)
    config = ModelConfig.from_preset(preset, model_id=model_path)
    all_layers = list(range(preset.num_layers))

    from anamnesis.extraction.model_loader import load_model

    loaded = load_model(
        config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=all_layers,
        value_layers=all_layers,
        query_layers=all_layers,
        attn_output_layers=all_layers,
        adapter_path=adapter_path,
    )
    return ReplaySurface(loaded=loaded, extraction=extraction, families=families)


@dataclass
class CellResult:
    """How a cell went: what landed, and what did not."""

    n_done: int
    n_failed: int
    signatures_dir: Path
    seconds: float

    @property
    def ok(self) -> bool:
        return self.n_failed == 0


def _source_metadata(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Per-generation metadata from a run, keyed by id; empty when absent.

    A replay copies the source generation's record onto its signature so a
    signature knows the topic, mode and seed it belongs to. A cell assembled
    without metadata still replays — the manifest is what a replay needs — and
    its signatures carry their id and nothing more.
    """
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    document = json.loads(meta_path.read_text())
    generations = (
        document["generations"]
        if isinstance(document, dict) and "generations" in document
        else document
    )
    return {int(g["generation_id"]): g for g in generations}


def replay_cell(
    surface: ReplaySurface,
    calibration: tuple[F32 | None, F32 | None, F32 | None],
    run_dir: Path,
    manifest_path: Path,
    *,
    gen_ids: Sequence[int] | None = None,
    signatures_subdir: str = DEFAULT_SIGNATURES_SUBDIR,
    raw_dir: Path | None = None,
    raw_subdir: str = DEFAULT_RAW_SUBDIR,
    save_raw: bool = True,
    resume: bool = True,
    logits_top_k: int = 50,
    write_handle: Any | None = None,
    injection: InjectionSpec | None = None,
    label: str = "w",
) -> CellResult:
    """Replay a cell's generations to signatures, in manifest order.

    ``gen_ids`` narrows the cell to a slice, which is how a fan-out gives each
    worker its share. ``resume`` skips a generation whose signature is already on
    disk, so a killed pass re-run over the same directory continues rather than
    recomputing; a caller that means to recompute says ``resume=False``.

    ``write_handle`` is an armed residual write and ``injection`` its spec. The
    write's start position is set per generation from that generation's prompt
    length — where the prompt ends is a property of the sequence — and after each
    forward the gating is checked against the number of generated tokens, unless
    the magnitude is zero.
    """
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3
    from anamnesis.extraction.replay.extract import replay_extract

    positional_means, pca_components, pca_mean = calibration

    entries = json.loads(Path(manifest_path).read_text())["entries"]
    src_meta = _source_metadata(run_dir)

    raw_target = raw_dir if raw_dir is not None else (run_dir / raw_subdir)
    sig_dir = run_dir / signatures_subdir
    sig_dir.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_target.mkdir(parents=True, exist_ok=True)

    available = sorted(int(k) for k in entries)
    wanted = None if gen_ids is None else set(int(g) for g in gen_ids)
    todo = [g for g in available if wanted is None or g in wanted]
    if resume:
        todo = [g for g in todo if not (sig_dir / f"gen_{g:03d}.json").exists()]
    logger.info(
        f"[{label}] {len(todo)} generations to process -> {sig_dir} "
        f"({len(available)} in the manifest)"
    )

    n_done = n_failed = 0
    started = time.time()
    for index, gen_id in enumerate(todo):
        try:
            entry = entries[str(gen_id)]
            input_ids = entry["input_ids"]
            prompt_length = int(entry["prompt_length"])
            if write_handle is not None:
                write_handle.spec.start_pos = prompt_length
                write_handle.reset_stats()
            raw_data = replay_extract(
                surface.loaded, input_ids, prompt_length, positional_means=positional_means
            )
            if write_handle is not None and injection is not None and injection.alpha != 0.0:
                check_injection_gating(
                    write_handle, len(input_ids) - prompt_length, f"gen_{gen_id:03d}"
                )
            if save_raw:
                save_raw_tensors_v3(
                    raw_data,
                    gen_id,
                    raw_target,
                    prompt_length=prompt_length,
                    input_ids=input_ids,
                    top_k_logits=logits_top_k,
                )
            result = compute_features_v2_from_data(
                raw_data, surface.extraction, surface.families, pca_components, pca_mean
            )
            metadata = dict(src_meta.get(gen_id, {"generation_id": gen_id}))
            if injection is not None:
                metadata["injection"] = injection.metadata()
            metadata["num_features"] = int(len(result.features))
            metadata[STORED_BLOCK_SLICES_KEY] = {
                k: list(v) for k, v in result.block_slices.items()
            }
            metadata["extraction_version"] = EXTRACTION_VERSION
            metadata["xrt_version"] = (
                ROUTING_VERSION if ROUTING_FAMILY in result.block_slices else 0
            )
            save_features(gen_id, result, metadata, sig_dir)
            n_done += 1
            if (index + 1) % 10 == 0 or index == 0:
                elapsed = time.time() - started
                rate = (index + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(todo) - index - 1) / rate if rate > 0 else 0.0
                logger.info(
                    f"[{label}] {index + 1}/{len(todo)} gen_{gen_id:03d}: "
                    f"{len(result.features)} features, {elapsed:.0f}s, ETA {eta:.0f}s"
                )
        except Exception as exc:  # noqa: BLE001 — one generation's failure is not the cell's
            n_failed += 1
            logger.error(f"[{label}] gen_{gen_id:03d} FAILED: {exc}", exc_info=True)
    seconds = time.time() - started
    logger.info(f"[{label}] cell done: {n_done} ok, {n_failed} failed in {seconds:.0f}s")
    return CellResult(
        n_done=n_done, n_failed=n_failed, signatures_dir=sig_dir, seconds=seconds
    )
