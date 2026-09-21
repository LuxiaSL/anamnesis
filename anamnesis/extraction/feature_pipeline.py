"""Layer 2: Pluggable feature computation from saved raw tensors.

Reads raw per-token tensors from disk and computes feature vectors
using configurable feature extraction. With no family enabled it reproduces
exactly the four blocks `anamnesis/extraction/state_extractor.py` builds.

Usage:
    # Recompute features for all raw tensor files
    python -m anamnesis.extraction.feature_pipeline \\
        --model 8b \\
        --raw-dir outputs/runs/smoke_test/raw_tensors/ \\
        --output-dir outputs/runs/smoke_test/signatures_v2/ \\
        --pca-model outputs/calibration/llama31_8b/pca_model.pkl

    # Or use programmatically:
    from anamnesis.extraction.feature_pipeline import recompute_all_features
    recompute_all_features(raw_dir, output_dir, config)

A sample that raises is named in the `RecomputeCount` this returns and the pass
continues, because one unreadable tensor file is not a reason to abandon the rest
of the bank. The command above fails closed on that count: a pass that wrote four
vectors for five raw tensors exits `anamnesis.shortfall.EXIT_SHORT` naming the
missing generation, and `--allow-partial` exits
`anamnesis.shortfall.EXIT_SHORT_SANCTIONED` instead — non-zero either way, with a
receipt in the output directory. The alternative is a signature directory one
vector short of the bank it was computed from, which nothing downstream can see.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import (
    MODEL_PRESETS,
    ExtractionConfig,
    FeaturePipelineConfig,
)
from anamnesis.extraction.raw_saver import list_raw_tensor_ids, load_raw_tensors
from anamnesis.extraction.state_extractor import (
    STORED_BLOCK_SLICES_KEY,
    ExtractionResult,
    RawGenerationData,
    extract_all_features,
)
from anamnesis.shortfall import Shortfall, ids_present

logger = logging.getLogger(__name__)

MODULE = "anamnesis.extraction.feature_pipeline"

F32 = NDArray[np.float32]


def compute_features_from_raw(
    raw_dir: Path,
    gen_id: int,
    config: ExtractionConfig,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    positional_means: F32 | None = None,
    *,
    surfaces: Sequence[str] | None = None,
    attn_layers: Sequence[int] | None = None,
) -> ExtractionResult:
    """Load raw tensors for one generation and compute features.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw tensor npz files.
    gen_id : int
        Generation ID to process.
    config : ExtractionConfig
        Feature extraction configuration.
    pca_components, pca_mean : arrays, optional
        Pre-fitted PCA model for the residual-PCA block.
    positional_means : array, optional
        Injected only if the loaded raw_data has none (v3 deduped raw stores
        positional_means once in the calibration dir, not per-gen).
    surfaces, attn_layers : optional (keyword-only)
        Lean-load passthrough to load_raw_tensors. Defaults (None) load
        everything — the exact historical behavior. CAUTION: attn_entropy and
        head_agreement read ALL attention layers present, so restricting
        attn_layers changes those features on v3 all-layer banks. Only restrict
        when every enabled consumer reads sampled layers only.

    Returns
    -------
    ExtractionResult with features, names and block slices.
    """
    raw_data = load_raw_tensors(gen_id, raw_dir, surfaces=surfaces, attn_layers=attn_layers)
    if raw_data.positional_means is None and positional_means is not None:
        raw_data.positional_means = positional_means
    return extract_all_features(raw_data, config, pca_components, pca_mean)


def compute_features_v2(
    raw_dir: Path,
    gen_id: int,
    config: ExtractionConfig,
    family_config: FeaturePipelineConfig,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    positional_means: F32 | None = None,
    *,
    surfaces: Sequence[str] | None = None,
    attn_layers: Sequence[int] | None = None,
) -> ExtractionResult:
    """Load raw tensors for one generation and compute v2 features.

    Thin wrapper over compute_features_v2_from_data: loads the saved raw tensors,
    injects calibration positional_means when the npz lacks them (v3 raw is deduped —
    pos_means lives once in the run's calibration dir, not per-gen), then computes.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw tensor npz files.
    gen_id : int
        Generation ID to process.
    config, family_config, pca_components, pca_mean :
        See compute_features_v2_from_data.
    positional_means : array, optional
        Injected only if the loaded raw_data has none (v3 deduped raw).
    surfaces, attn_layers : optional (keyword-only)
        Lean-load passthrough to load_raw_tensors (see compute_features_from_raw
        for the all-layer attention caveat). Defaults preserve exact behavior.
    """
    raw_data = load_raw_tensors(gen_id, raw_dir, surfaces=surfaces, attn_layers=attn_layers)
    if raw_data.positional_means is None and positional_means is not None:
        raw_data.positional_means = positional_means
    return compute_features_v2_from_data(
        raw_data, config, family_config, pca_components, pca_mean,
    )


def compute_features_v2_from_data(
    raw_data: RawGenerationData,
    config: ExtractionConfig,
    family_config: FeaturePipelineConfig,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
) -> ExtractionResult:
    """Compute the core blocks plus the pluggable feature families from in-memory raw_data.

    The GPU-free feature loop: the core blocks, then the enabled families, concatenated.
    Used by replay-extract (in-memory raw_data) and compute_features_v2 (from disk).
    Caller is responsible for setting raw_data.positional_means, which the cache-and-keys
    block, the residual-PCA block and the per-head and residual families all correct with.
    """

    # The blocks the numeric anchor builds
    if family_config.include_core_blocks:
        core = extract_all_features(raw_data, config, pca_components, pca_mean)
        all_features = [core.features]
        all_names = list(core.feature_names)
        all_slices = dict(core.block_slices)
        offset = len(core.features)
        knnlm = core.knnlm_baseline
    else:
        all_features = []
        all_names = []
        all_slices = {}
        offset = 0
        knnlm = None

    # Pluggable feature families
    if family_config.enable_residual_trajectory:
        from anamnesis.extraction.feature_families.residual_stream import (
            extract_residual_trajectory,
        )
        result = extract_residual_trajectory(
            raw_data,
            layer_indices=family_config.trajectory_layers,
            positional_means=raw_data.positional_means,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_contrastive_projection and family_config.contrastive_model_path:
        from anamnesis.extraction.feature_families.contrastive_projection import (
            extract_contrastive_projection,
        )
        try:
            result = extract_contrastive_projection(
                raw_data,
                projection_model_path=family_config.contrastive_model_path,
                layer_indices=family_config.contrastive_layers,
                temporal_samples=family_config.contrastive_temporal_samples,
            )
            if len(result) > 0:
                all_slices[result.family_name] = (offset, offset + len(result))
                all_features.append(result.features)
                all_names.extend(result.feature_names)
                offset += len(result)
        except Exception as e:
            logger.warning(f"Contrastive projection failed: {e}")

    if family_config.enable_attention_flow:
        from anamnesis.extraction.feature_families.attention_flow import (
            extract_attention_flow,
        )
        result = extract_attention_flow(
            raw_data,
            sampled_layers=config.sampled_layers,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_gate_features:
        from anamnesis.extraction.feature_families.gate_features import (
            extract_gate_features,
        )
        result = extract_gate_features(
            raw_data,
            sampled_layers=config.sampled_layers,
            sparsity_threshold=family_config.gate_sparsity_threshold,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_temporal_dynamics:
        from anamnesis.extraction.feature_families.temporal_dynamics import (
            extract_temporal_dynamics,
        )
        result = extract_temporal_dynamics(
            raw_data,
            sampled_layers=config.sampled_layers,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_per_head:
        from anamnesis.extraction.feature_families.per_head import extract_per_head
        result = extract_per_head(
            raw_data,
            sampled_layers=config.sampled_layers,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_value_geometry:
        from anamnesis.extraction.feature_families.value_geometry import extract_value_geometry
        result = extract_value_geometry(
            raw_data,
            sampled_layers=config.sampled_layers,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_qk_geometry:
        from anamnesis.extraction.feature_families.qk_geometry import extract_qk_geometry
        result = extract_qk_geometry(
            raw_data,
            sampled_layers=config.sampled_layers,
            n_windows=family_config.temporal_n_windows,
            include_stft=family_config.enable_stft,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    if family_config.enable_kv_cka:
        from anamnesis.extraction.feature_families.key_cka import extract_key_cka
        result = extract_key_cka(
            raw_data,
            sampled_layers=config.sampled_layers,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    # Path signature (level-2 log-signature of the residual trajectory; spec 2026-09-11).
    # Unlike the other families this one needs an EXTERNAL artefact (the calibration PCA that
    # supplies the projection basis) and deliberately does NOT degrade to a skip when it is
    # missing — a silently-absent family is exactly the failure the spec's discipline forbids.
    if family_config.enable_path_signature:
        from anamnesis.extraction.feature_families.path_signature import (
            PathSignatureConfig,
            ProjectionBasisBank,
            extract_path_signature,
        )
        if family_config.path_signature_basis_path is None:
            raise ValueError(
                "enable_path_signature=True requires path_signature_basis_path (the banked "
                "calibration PCA). This family never fits a basis on evaluation data."
            )
        ps_bank = ProjectionBasisBank.from_pca_pickle(
            family_config.path_signature_basis_path,
            label=family_config.path_signature_basis_label,
        )
        ps_config = PathSignatureConfig(
            layer_indices=tuple(family_config.path_signature_layers),
            n_components=family_config.path_signature_k,
            level=family_config.path_signature_level,  # type: ignore[arg-type]
            time_augment=family_config.path_signature_time_augment,
            basis_label=family_config.path_signature_basis_label,
            permute_increments=family_config.path_signature_permute_seed is not None,
            permutation_seed=family_config.path_signature_permute_seed,
        )
        result = extract_path_signature(raw_data, ps_bank, ps_config)
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    # Attention-residual routing — the reference optional per-architecture hook: the gate is on the
    # SUBSTRATE's presence, never on a model name, so an architecture that routes over committed
    # blocks gets the family and one that does not leaves attn_res_* None and is skipped.
    if family_config.enable_attn_res and raw_data.attn_res_routing is not None:
        from anamnesis.extraction.feature_families.attn_res import extract_attn_res
        result = extract_attn_res(
            raw_data.attn_res_routing,
            committed=raw_data.attn_res_committed,
            sampled_layers=config.sampled_layers,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    # MoE expert routing (vmb arm A7, M6 DeepSeek-V2-Lite class; needs router_dist capture — dense
    # models leave it None → skipped). MoE layers are those the capture actually banked a router for,
    # so dense prefix layers (e.g. DeepSeek layer 0) are excluded data-drivenly, not by a magic index.
    if family_config.enable_expert_routing and raw_data.router_dist:
        from anamnesis.extraction.feature_families.expert_routing import (
            extract_expert_routing_features,
        )
        moe_layers = [l for l in config.sampled_layers if l in raw_data.router_dist]
        result = extract_expert_routing_features(
            raw_data,
            sampled_layers=moe_layers,
            top_k=family_config.expert_routing_top_k,
        )
        if len(result) > 0:
            all_slices[result.family_name] = (offset, offset + len(result))
            all_features.append(result.features)
            all_names.extend(result.feature_names)
            offset += len(result)

    # Concatenate
    if all_features:
        combined = np.concatenate(all_features)
    else:
        combined = np.array([], dtype=np.float32)

    logger.info(f"V2 features: {len(combined)} total ({len(all_slices)} families)")

    return ExtractionResult(
        features=combined,
        feature_names=all_names,
        block_slices=all_slices,
        knnlm_baseline=knnlm,
    )


def save_features(
    gen_id: int,
    result: ExtractionResult,
    metadata: dict | None,
    output_dir: Path,
) -> Path:
    """Save computed features to npz + json (same format as generation_runner)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"gen_{gen_id:03d}.npz"

    save_dict: dict[str, object] = {
        "features": result.features,
        "feature_names": np.array(result.feature_names),
    }
    if result.knnlm_baseline is not None:
        save_dict["knnlm_baseline"] = result.knnlm_baseline

    for block_name, (start, end) in result.block_slices.items():
        save_dict[f"features_{block_name}"] = result.features[start:end]

    np.savez_compressed(npz_path, **save_dict)

    # Save metadata json if available
    if metadata is not None:
        json_path = output_dir / f"gen_{gen_id:03d}.json"
        meta_copy = metadata.copy()
        if STORED_BLOCK_SLICES_KEY in meta_copy:
            meta_copy[STORED_BLOCK_SLICES_KEY] = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in meta_copy[STORED_BLOCK_SLICES_KEY].items()
            }
        # Bounds come from this result, not from the metadata being copied
        meta_copy[STORED_BLOCK_SLICES_KEY] = {
            k: list(v) for k, v in result.block_slices.items()
        }
        meta_copy["num_features"] = len(result.features)
        with open(json_path, "w") as f:
            json.dump(meta_copy, f, indent=2, default=str)

    return npz_path


def _process_one_sample(args: tuple) -> tuple[int, int | None, str | None]:
    """Process a single sample — top-level function for pickling by ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (gen_id, raw_dir, output_dir, metadata_dir, config, family_config,
         pca_components, pca_mean, use_v2)

    Returns
    -------
    (gen_id, n_features, error_message)
    """
    (gen_id, raw_dir, output_dir, metadata_dir, config, family_config,
     pca_components, pca_mean, use_v2, positional_means) = args
    try:
        if use_v2:
            result = compute_features_v2(
                raw_dir, gen_id, config, family_config,
                pca_components, pca_mean, positional_means,
            )
        else:
            result = compute_features_from_raw(
                raw_dir, gen_id, config, pca_components, pca_mean, positional_means,
            )

        # Load original metadata if available
        metadata: dict | None = None
        json_path = metadata_dir / f"gen_{gen_id:03d}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        save_features(gen_id, result, metadata, output_dir)
        return (gen_id, len(result.features), None)
    except Exception as e:
        return (gen_id, None, str(e))


@dataclass
class RecomputeCount:
    """How a recompute went: the tensors asked for, and which of them raised.

    ``requested`` is every generation the raw directory offered, which is the whole
    set this pass was asked for — a recompute reads the bank it is pointed at
    rather than a subset of it. ``failed`` maps a generation id to the message its
    exception carried, which is what lets a refusal say *which* vector is absent
    instead of leaving a reader to infer it from a count.
    """

    requested: tuple[int, ...]
    n_done: int
    failed: dict[int, str]
    output_dir: Path
    seconds: float

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    @property
    def ok(self) -> bool:
        return self.n_failed == 0


def recompute_shortfall(
    result: RecomputeCount, *, command: str, label: str = ""
) -> Shortfall:
    """State a recompute as expected-versus-produced, for a command to refuse on.

    Produced means a vector npz on disk for a requested generation, so a pass that
    wrote four of five vectors reports four of five rather than reporting nothing
    at all.
    """
    requested = tuple(str(gen_id) for gen_id in result.requested)
    return Shortfall(
        command=command,
        unit="generation",
        target=result.output_dir,
        requested=requested,
        produced=ids_present(
            requested,
            lambda name: (result.output_dir / f"gen_{int(name):03d}.npz").exists(),
        ),
        failures={str(gen_id): reason for gen_id, reason in result.failed.items()},
        label=label,
    )


def _default_n_workers() -> int:
    """Sensible parallel default for recompute_all_features.

    Leaves 2 cores for the OS/caller and caps at 32 (each worker holds one
    loaded sample: ~300 MB for v2 banks, more for v3 all-layer banks).
    """
    cpus = os.cpu_count() or 1
    return max(1, min(32, cpus - 2))


def _pin_blas_single_thread() -> None:
    """Pool-worker initializer: pin BLAS to 1 thread per worker.

    Unpinned pools oversubscribe cores (node1 convention: OMP/OPENBLAS/MKL
    _NUM_THREADS=1 per worker). Env vars cover spawn-started workers;
    threadpoolctl (when installed) also covers fork-started workers, whose
    BLAS pools were already initialized before the fork.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:  # noqa: BLE001 — best-effort; env vars remain in effect
        pass


def recompute_all_features(
    raw_dir: Path,
    output_dir: Path,
    config: ExtractionConfig,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    metadata_dir: Path | None = None,
    family_config: FeaturePipelineConfig | None = None,
    n_workers: int | None = None,
    positional_means: F32 | None = None,
) -> RecomputeCount:
    """Recompute features for all raw tensor files in a directory.

    This is the core iteration loop: change feature config → run this →
    run analysis. No GPU needed.

    Parameters
    ----------
    raw_dir : Path
        Directory with raw tensor npz files.
    output_dir : Path
        Where to save recomputed feature npz files.
    config : ExtractionConfig
        Feature extraction configuration.
    pca_components, pca_mean : arrays, optional
        Pre-fitted PCA model.
    metadata_dir : Path, optional
        Directory with original metadata json files (for copying metadata).
        If None, uses the signatures dir adjacent to raw_dir.
    family_config : FeaturePipelineConfig, optional
        When provided, uses compute_features_v2() with pluggable families.
        When None, uses compute_features_from_raw(), which runs the core blocks alone.
    n_workers : int, optional
        Number of parallel workers. None (default) resolves to a cpu-based
        count (cpu_count - 2, capped at 32); pass 1 to force sequential.
        Each worker loads one sample (~280 MB), so memory ≈ n_workers × 300 MB.
        Pool workers pin BLAS to 1 thread (node1 shared-box convention).

    Returns
    -------
    RecomputeCount
        The generations the raw directory offered, the count that landed, and each
        id that raised with the reason. One sample's failure does not stop the
        pass — the other four hundred vectors are worth having — so the caller
        decides the pass's fate, through :func:`recompute_shortfall`.
    """
    if n_workers is None:
        n_workers = _default_n_workers()

    t_start = time.perf_counter()
    gen_ids = list_raw_tensor_ids(raw_dir)
    if not gen_ids:
        logger.warning(f"No raw tensor files found in {raw_dir}")
        return RecomputeCount(
            requested=(),
            n_done=0,
            failed={},
            output_dir=output_dir,
            seconds=time.perf_counter() - t_start,
        )

    # Try to find metadata json files
    if metadata_dir is None:
        # Convention: raw_tensors/ is next to signatures/
        metadata_dir = raw_dir.parent / "signatures"

    output_dir.mkdir(parents=True, exist_ok=True)

    use_v2 = family_config is not None
    label = "v2" if use_v2 else "v1 (core blocks only)"
    logger.info(f"Recomputing features ({label}) for {len(gen_ids)} generations...")
    logger.info(f"  Raw tensors: {raw_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Workers: {n_workers}")
    if use_v2:
        enabled = []
        if family_config.include_core_blocks:
            enabled.append("core blocks")
        if family_config.enable_residual_trajectory:
            enabled.append("trajectory")
        if family_config.enable_attention_flow:
            enabled.append("attention_flow")
        if family_config.enable_gate_features:
            enabled.append("gate")
        if family_config.enable_temporal_dynamics:
            enabled.append("temporal_dynamics")
        if family_config.enable_contrastive_projection:
            enabled.append("contrastive")
        logger.info(f"  Families: {', '.join(enabled)}")

    # Build argument tuples for each sample
    work_args = [
        (gen_id, raw_dir, output_dir, metadata_dir, config, family_config,
         pca_components, pca_mean, use_v2, positional_means)
        for gen_id in gen_ids
    ]

    processed: list[int] = []
    failed: dict[int, str] = {}

    if n_workers <= 1:
        # Sequential path
        for i, args in enumerate(work_args):
            gen_id, n_features, error = _process_one_sample(args)
            if error:
                failed[gen_id] = error
                logger.error(f"Failed to process gen_{gen_id:03d}: {error}")
            else:
                processed.append(gen_id)
                if (i + 1) % 10 == 0 or i == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = (i + 1) / elapsed
                    remaining = (len(gen_ids) - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        f"  [{i+1}/{len(gen_ids)}] gen_{gen_id:03d}: "
                        f"{n_features} features, "
                        f"{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining"
                    )
    else:
        # Parallel path
        from concurrent.futures import ProcessPoolExecutor, as_completed

        done_count = 0
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_pin_blas_single_thread,
        ) as executor:
            futures = {
                executor.submit(_process_one_sample, args): args[0]
                for args in work_args
            }
            for future in as_completed(futures):
                gen_id = futures[future]
                try:
                    gen_id_result, n_features, error = future.result()
                    if error:
                        failed[gen_id_result] = error
                        logger.error(f"Failed gen_{gen_id_result:03d}: {error}")
                    else:
                        processed.append(gen_id_result)
                except Exception as e:  # noqa: BLE001 — a dead worker is one sample's loss
                    # A worker that died in the pool never returned a row, so the
                    # accounting has to be made here or the sample would vanish
                    # from both the successes and the failures.
                    failed[gen_id] = f"{type(e).__name__}: {e}"
                    logger.error(f"Worker exception for gen_{gen_id:03d}: {e}")

                done_count += 1
                if done_count % 20 == 0 or done_count == len(gen_ids):
                    elapsed = time.perf_counter() - t_start
                    rate = done_count / elapsed
                    remaining = (len(gen_ids) - done_count) / rate if rate > 0 else 0
                    logger.info(
                        f"  [{done_count}/{len(gen_ids)}] "
                        f"{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining "
                        f"({rate:.1f} samples/s)"
                    )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Done: {len(processed)}/{len(gen_ids)} processed in {elapsed:.1f}s"
    )

    return RecomputeCount(
        requested=tuple(gen_ids),
        n_done=len(processed),
        failed=failed,
        output_dir=output_dir,
        seconds=elapsed,
    )


def _load_pca_model(
    pca_path: Path,
) -> tuple[F32 | None, F32 | None]:
    """Load PCA components and mean from pickle file."""
    if not pca_path.exists():
        logger.warning(f"PCA model not found: {pca_path}")
        return None, None

    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)

    if isinstance(pca_data, dict):
        # C5 per-layer format: {layer_idx: {"components", "mean", ...}} → return dicts
        vals = list(pca_data.values())
        if vals and isinstance(vals[0], dict) and "components" in vals[0]:
            comp = {int(k): np.asarray(v["components"], dtype=np.float32) for k, v in pca_data.items()}
            mean = {int(k): np.asarray(v["mean"], dtype=np.float32) for k, v in pca_data.items()}
            return comp, mean  # type: ignore[return-value]
        components = pca_data.get("components")
        mean = pca_data.get("mean")
    else:
        components = getattr(pca_data, "components_", None)
        mean = getattr(pca_data, "mean_", None)

    if components is not None:
        components = np.asarray(components, dtype=np.float32)
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)

    return components, mean


def main() -> None:
    """CLI entry point for batch feature recomputation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Recompute features from saved raw tensors (no GPU needed)",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing raw tensor npz files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to save recomputed feature npz files",
    )
    parser.add_argument(
        "--pca-model", type=Path, default=None,
        help="Path to PCA model pickle (for the residual-PCA block)",
    )
    parser.add_argument(
        "--no-residual-pca", action="store_true",
        help="Disable the residual-PCA block",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Directory with original metadata json files",
    )

    # v2 feature family flags
    parser.add_argument(
        "--v2", action="store_true",
        help="Enable v2 pipeline with pluggable feature families",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_PRESETS.keys()), required=True,
        help="Model preset the layer plan comes from (sampled_layers, pca_layers, "
             "trajectory_layers, contrastive_layers, depth bands). Required: a layer "
             "index means a different fraction of the network in each model, so there "
             "is no correct default.",
    )
    parser.add_argument(
        "--no-trajectory", action="store_true",
        help="Disable residual stream trajectory features (v2 only)",
    )
    parser.add_argument(
        "--no-attention-flow", action="store_true",
        help="Disable attention flow features (v2 only)",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Disable SwiGLU gate features (v2 only)",
    )
    parser.add_argument(
        "--no-temporal-dynamics", action="store_true",
        help="Disable temporal decomposition of the attention and cache metrics (v2 only)",
    )
    parser.add_argument(
        "--no-stft", action="store_true",
        help="Disable STFT spectral features in temporal operators (v2 only)",
    )
    parser.add_argument(
        "--no-core-blocks", action="store_true",
        help="Compute only the families, without the four blocks the numeric anchor "
             "builds (v2 only, for ablation)",
    )
    parser.add_argument(
        "--contrastive-model", type=Path, default=None,
        help="Path to trained contrastive projection model (.npz). "
             "Enables contrastive projection features when provided (v2 only).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 2, capped at 32; "
             "pass 1 to force sequential). Each worker uses ~300 MB RAM for "
             "loading raw tensors; BLAS is pinned to 1 thread per worker.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Accept fewer vectors than there are raw tensors; the receipt is written "
             "either way and the status stays non-zero",
    )

    args = parser.parse_args()

    # Build ExtractionConfig with model-specific layers
    config = ExtractionConfig.from_preset(args.model)
    if args.no_residual_pca:
        config.enable_residual_pca = False

    pca_components: F32 | None = None
    pca_mean: F32 | None = None
    if args.pca_model and not args.no_residual_pca:
        pca_components, pca_mean = _load_pca_model(args.pca_model)
        if pca_components is not None:
            logger.info(f"Loaded PCA model: {pca_components.shape}")

    # Build v2 family config if requested
    family_config: FeaturePipelineConfig | None = None
    if args.v2:
        family_kwargs: dict = {
            "include_core_blocks": not args.no_core_blocks,
            "enable_residual_trajectory": not args.no_trajectory,
            "enable_attention_flow": not args.no_attention_flow,
            "enable_gate_features": not args.no_gate,
            "enable_temporal_dynamics": not args.no_temporal_dynamics,
            "enable_stft": not args.no_stft,
            "enable_contrastive_projection": args.contrastive_model is not None,
            "contrastive_model_path": args.contrastive_model,
        }
        family_config = FeaturePipelineConfig.from_preset(args.model, **family_kwargs)

    from anamnesis.shortfall import refuse_unless_complete

    result = recompute_all_features(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=config,
        pca_components=pca_components,
        pca_mean=pca_mean,
        metadata_dir=args.metadata_dir,
        family_config=family_config,
        n_workers=args.workers,
    )
    refuse_unless_complete(
        [recompute_shortfall(result, command=MODULE)],
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
