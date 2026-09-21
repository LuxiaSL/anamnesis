"""The gauntlet: every standing analysis, run over one set of signatures.

A signature directory goes in and eleven sections come out, each reading the
same matrices for a different kind of structure. The point of running them as
one pass rather than as eleven scripts is that a claim about signatures is
usually a claim about how several of these sections agree: a classification
accuracy means one thing beside a clean semantic-orthogonality result and
another beside a length-only baseline that reaches the same number.

The eleven sections:
  1. Data integrity & descriptive statistics
  2. Classification (5-way mode discrimination)
  3. Block ablation & feature importance
  4. Intrinsic dimension profiling
  5. CCGP (cross-condition generalization)
  6. Topology & hyperbolicity
  7. Silhouette & clustering
  8. Contrastive projection (optional, requires torch)
  9. Semantic independence (optional, requires sentence-transformers)
  10. Prediction scorecard
  11. Manifold geometry (curvature, geodesic, anisotropy)

Sections are declared in the ``SECTIONS`` registry below and dispatched by a
single loop in ``run_full_analysis``, which imports each section module only
when its section runs — so a pass that skips the contrastive and semantic
sections never imports torch or sentence-transformers.

A full pass is long enough that losing it to a crash in section 9 would matter,
so results are checkpointed after each section completes and ``resume`` reads
the checkpoint back, validates each section against its schema, and skips what
is already there. An error stub does not count as completed.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from anamnesis.config.paths import outputs_root

from .signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    EVERYTHING,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    AnalysisData,
    load_analysis_data,
)
from .schemas import (
    AnalysisResults,
    CCGPResult,
    ClassificationResult,
    ClusteringResult,
    ContrastiveResult,
    IntegrityResult,
    IntrinsicDimensionResult,
    ManifoldGeometryResult,
    ScorecardResult,
    SemanticResult,
    LegacyBinReadoutResult,
    TopologyResult,
    migrate_banked_results,
)
from .utils import clean_for_json


@dataclass(frozen=True)
class SectionSpec:
    """Declarative orchestration entry for a single analysis section.

    Attributes
    ----------
    number : int
        User-facing section number (1-11); accepted by ``--skip``.
    name : str
        Printed section header.
    key : str
        Results-dict key; also the schema model key in ``SECTION_MODELS``.
    runner : Callable[[dict[str, Any]], Any]
        Closure that imports the section module and invokes its ``run_*``
        function. Lazy — nothing is loaded until the section actually runs.
    requires_text : bool
        True if the section consumes ``data.generated_texts`` (section 9).
        Controls ``load_text`` on ``load_analysis_data``.
    always_rerun : bool
        True for sections that consume other sections' results (section 10).
        Never treated as "already completed" on resume; no per-section timing
        or checkpoint save inside the loop.
    """

    number: int
    name: str
    key: str
    runner: Callable[[dict[str, Any]], Any]
    requires_text: bool = False
    always_rerun: bool = False


def _lazy_call(module: str, fn_name: str, *args: Any, **kwargs: Any) -> Any:
    """Import ``module`` relative to this package and call ``fn_name``.

    Called inside section runners so heavy imports (``dadapy``,
    ``sentence_transformers``, ``torch``) only happen when their section runs.
    """
    mod = importlib.import_module(f".{module}", package=__name__)
    return getattr(mod, fn_name)(*args, **kwargs)


def _run_data_only(module: str, fn_name: str) -> Callable[[dict[str, Any]], Any]:
    """Runner factory for sections that only consume ``ctx['data']``."""

    def run(ctx: dict[str, Any]) -> Any:
        return _lazy_call(module, fn_name, ctx["data"])

    return run


def _run_semantic(ctx: dict[str, Any]) -> Any:
    return _lazy_call(
        "semantic",
        "run_semantic",
        ctx["data"],
        signature_dir=ctx["signature_dir"],
        addon_dirs=ctx["addon_dirs"],
    )


def _run_scorecard(ctx: dict[str, Any]) -> Any:
    return _lazy_call("scorecard", "run_scorecard", ctx["results"])


SECTIONS: list[SectionSpec] = [
    SectionSpec(1, "Data Integrity", "integrity",
                _run_data_only("integrity", "run_integrity_checks")),
    SectionSpec(2, "Classification", "classification",
                _run_data_only("classification", "run_classification")),
    SectionSpec(3, "Legacy Bin Readout", "legacy_bin_readout",
                _run_data_only("legacy_bin_readout", "run_legacy_bin_readout")),
    SectionSpec(4, "Intrinsic Dimension", "intrinsic_dimension",
                _run_data_only("geometry", "run_intrinsic_dimension")),
    SectionSpec(5, "CCGP", "ccgp",
                _run_data_only("geometry", "run_ccgp")),
    SectionSpec(6, "Topology", "topology",
                _run_data_only("geometry", "run_topology")),
    SectionSpec(7, "Clustering", "clustering",
                _run_data_only("clustering", "run_clustering")),
    SectionSpec(8, "Contrastive Projection", "contrastive",
                _run_data_only("contrastive", "run_contrastive")),
    SectionSpec(9, "Semantic Independence", "semantic",
                _run_semantic, requires_text=True),
    SectionSpec(10, "Prediction Scorecard", "scorecard",
                _run_scorecard, always_rerun=True),
    SectionSpec(11, "Manifold Geometry", "manifold_geometry",
                _run_data_only("geometry", "run_manifold_geometry")),
]

# Lookup tables derived from SECTIONS for any external consumers and for
# readability when scanning the module.
SECTION_KEYS: dict[int, str] = {spec.number: spec.key for spec in SECTIONS}
SECTION_NAMES: dict[int, str] = {spec.number: spec.name for spec in SECTIONS}

# Registry mapping section-results key → pydantic model class. Used by
# ``_rehydrate_section`` to validate checkpointed dicts back into typed
# models so downstream consumers can use attribute access.
SECTION_MODELS: dict[str, type[BaseModel]] = {
    "integrity": IntegrityResult,
    "classification": ClassificationResult,
    "legacy_bin_readout": LegacyBinReadoutResult,
    "intrinsic_dimension": IntrinsicDimensionResult,
    "ccgp": CCGPResult,
    "topology": TopologyResult,
    "clustering": ClusteringResult,
    "contrastive": ContrastiveResult,
    "semantic": SemanticResult,
    "scorecard": ScorecardResult,
    "manifold_geometry": ManifoldGeometryResult,
}


def _is_error_value(value: object) -> bool:
    """True if a result dict/model carries an error stub (skip-on-resume)."""
    if isinstance(value, BaseModel):
        return bool(getattr(value, "error", None))
    if isinstance(value, dict):
        return bool(value.get("error"))
    return False


def _rehydrate_section(key: str, value: object) -> object:
    """Validate a checkpointed dict into its typed model when a schema exists.

    Unknown sections (no entry in ``SECTION_MODELS``) and error stubs pass
    through untouched so consumers still see the legacy shape.
    """
    model_cls = SECTION_MODELS.get(key)
    if model_cls is None or not isinstance(value, dict):
        return value
    if _is_error_value(value):
        return value
    try:
        return model_cls.model_validate(value)
    except ValidationError as e:
        print(f"  Warning: could not validate checkpointed '{key}' against schema: {e}")
        return value


def _save_checkpoint(results: dict, output_dir: Path) -> None:
    """Save current results as checkpoint."""
    checkpoint_path = output_dir / "results.json"
    with open(checkpoint_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)


def _load_checkpoint(output_dir: Path) -> dict | None:
    """Load an existing checkpoint if present, reading older field spellings forward.

    A checkpoint can predate a schema rename, and a resume that dropped a section
    because its key moved would silently recompute it under a new name and leave the
    old one beside it. ``migrate_banked_results`` maps the names, so the resume sees
    one document.
    """
    checkpoint_path = output_dir / "results.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                return migrate_banked_results(json.load(f))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _detect_completed_sections(checkpoint: dict) -> set[int]:
    """Detect which sections have usable results in a checkpoint.

    ``always_rerun`` sections (scorecard) are never treated as completed;
    they reassemble their output from other sections on every run.
    """
    completed: set[int] = set()
    for spec in SECTIONS:
        if spec.always_rerun:
            continue
        if spec.key in checkpoint and checkpoint[spec.key] is not None:
            if _is_error_value(checkpoint[spec.key]):
                continue
            completed.add(spec.number)
    return completed


def run_full_analysis(
    signature_dir: Path | str,
    run_name: str,
    output_dir: Path | str | None = None,
    core_only: bool = True,
    skip_sections: set[int] | None = None,
    resume: bool = False,
    addon_dirs: list[Path | str] | None = None,
    mode_filter: list[str] | None = None,
) -> AnalysisResults:
    """Run the complete analysis gauntlet.

    Parameters
    ----------
    signature_dir : Path
        Directory containing gen_NNN.npz + gen_NNN.json files.
    run_name : str
        Label for this run (e.g. "8b_baseline").
    output_dir : Path, optional
        Where to save results. Defaults to outputs/analysis/{run_name}/.
    core_only : bool
        If True, use one rep per topic-mode pair.
    skip_sections : set of int, optional
        Section numbers to skip (1-11).
    resume : bool
        If True, load existing checkpoint and skip completed sections.
    addon_dirs : list[Path], optional
        Additional directories with feature arrays to merge.
    mode_filter : list[str], optional
        If provided, only include samples whose mode is in this list.

    Returns
    -------
    AnalysisResults
        Typed composite of run metadata + per-section typed results.
        The JSON written under ``{output_dir}/results.json`` is
        structurally identical to the pre-typed layout.
    """
    skip = set(skip_sections) if skip_sections else set()

    if output_dir is None:
        output_dir = outputs_root() / "analysis" / run_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print("=" * 60)
    print(f"ANALYSIS GAUNTLET: {run_name}")
    print("=" * 60)

    # Handle resume
    results: dict = {}
    if resume:
        checkpoint = _load_checkpoint(output_dir)
        if checkpoint is not None:
            completed = _detect_completed_sections(checkpoint)
            if completed:
                print(f"\nResuming from checkpoint. Completed sections: {sorted(completed)}")
                results = checkpoint
                # Rehydrate every schema-bearing key present in the checkpoint
                # (including always_rerun sections, so _print_summary sees typed
                # values if scorecard is explicitly --skip'd with a prior result).
                for spec in SECTIONS:
                    if spec.key in results:
                        results[spec.key] = _rehydrate_section(spec.key, results[spec.key])
                skip = skip | completed
            else:
                print("\nCheckpoint found but no completed sections. Starting fresh.")
        else:
            print("\nNo checkpoint found. Starting fresh.")

    # Base metadata (preserve from checkpoint if resuming, else set fresh)
    if "run_name" not in results:
        results["run_name"] = run_name
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results["core_only"] = core_only
    results["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Load data — only pull generated text if a requires_text section will run.
    print("\nLoading data...")
    load_text = any(
        spec.requires_text and spec.number not in skip for spec in SECTIONS
    )
    data = load_analysis_data(
        signature_dir=signature_dir,
        run_name=run_name,
        core_only=core_only,
        load_text=load_text,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )
    results["n_samples"] = data.n_samples
    print(f"  {data.n_samples} samples, {len(data.unique_modes)} modes, "
          f"{len(data.unique_topics)} topics")

    # Section timing (carried across resumes)
    section_times: dict[str, float] = results.get("section_times", {})

    # Context handed to each section runner. ``results`` is passed by
    # reference so scorecard sees every prior section's populated entry.
    ctx: dict[str, Any] = {
        "data": data,
        "results": results,
        "signature_dir": signature_dir,
        "addon_dirs": addon_dirs,
    }

    for spec in SECTIONS:
        if spec.number in skip:
            print(f"\n--- Section {spec.number}: {spec.name} --- SKIPPED")
            continue

        print(f"\n--- Section {spec.number}: {spec.name} ---")
        t0 = time.perf_counter()
        section_result = spec.runner(ctx)
        results[spec.key] = section_result

        if spec.always_rerun:
            # Scorecard evaluates accumulated results; not timed or checkpointed.
            continue

        section_times[spec.key] = time.perf_counter() - t0
        print(f"  Done ({section_times[spec.key]:.1f}s)")
        _save_checkpoint(results, output_dir)

        # Integrity surfaces NaN/Inf loudly so operators spot corruption immediately.
        if spec.key == "integrity" and not section_result.all_clean:
            print("  WARNING: NaN/Inf detected in features!")

    # Save final timing
    results["section_times"] = section_times

    # Final save (structurally identical wire format to pre-typed layout)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    _print_summary(results)

    # Validate the accumulated dict into the typed composite for return.
    # We model_validate the cleaned dict so any missing/Optional sections
    # land cleanly and any drift surfaces loudly here (extra="forbid").
    return AnalysisResults.model_validate(clean_for_json(results))


def _print_summary(results: dict) -> None:
    """Print key findings to stdout."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Classification
    clf = results.get("classification")
    clf_by_block: dict = {}
    if isinstance(clf, ClassificationResult):
        clf_by_block = clf.by_block
    # Show all blocks that have results
    reported_blocks = [ATTENTION_AND_CACHE, ALL_CORE, ALL_FAMILIES, EVERYTHING,
                      ATTENTION_AND_CACHE_WITH_FAMILIES]
    for block in reported_blocks:
        block_clf = clf_by_block.get(block)
        if block_clf is not None and block_clf.rf_5way.accuracy is not None:
            print(f"\n  5-way RF ({block}): {block_clf.rf_5way.accuracy:.1%}")

    # CV stability
    for block in [ATTENTION_AND_CACHE, EVERYTHING, ALL_CORE]:
        block_clf = clf_by_block.get(block)
        if block_clf is not None and block_clf.cv_stability is not None:
            stab = block_clf.cv_stability
            print(f"  CV stability ({block}): median={stab.median:.1%}, "
                  f"95% CI=[{stab.ci_lo:.1%}, {stab.ci_hi:.1%}]")

    # Permutation test
    for block in [ATTENTION_AND_CACHE, EVERYTHING, ALL_CORE]:
        block_clf = clf_by_block.get(block)
        if block_clf is not None and block_clf.permutation_test is not None:
            print(f"  Permutation p ({block}): {block_clf.permutation_test.p_value}")

    # The readout over the stored blocks
    ablation = results.get("legacy_bin_readout")
    if isinstance(ablation, LegacyBinReadoutResult) and ablation.block_ranking:
        rank_str = " > ".join(
            f"{entry.block}({entry.accuracy:.0%})" for entry in ablation.block_ranking
        )
        print(f"\n  Block ranking: {rank_str}")
        print(
            "  cache > attention > norms: "
            f"{ablation.cache_beats_attention_beats_norms}"
        )

    # ID
    id_data = results.get("intrinsic_dimension")
    if isinstance(id_data, IntrinsicDimensionResult) and id_data.global_:
        print("\n  Intrinsic dimension:")
        for block in [NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA, ATTENTION_AND_CACHE]:
            block_id = id_data.global_.get(block)
            if block_id is not None and isinstance(block_id.dadapy_id, (int, float)):
                print(f"    {block}: {block_id.dadapy_id:.1f}")
        if id_data.block_convergence is not None:
            print(f"  Block convergence (max diff): {id_data.block_convergence.max_pairwise_diff:.1f}")

    # CCGP
    ccgp = results.get("ccgp")
    if isinstance(ccgp, CCGPResult):
        summary = ccgp.summary
        print(f"\n  CCGP: min={summary.min_ccgp}, all_perfect={summary.all_perfect}")

    # Topology
    topo = results.get("topology")
    if isinstance(topo, TopologyResult):
        euc = topo.topology_summary.get("euclidean")
        if euc is not None:
            print(f"\n  Topology: nearest={euc.nearest_pair}, "
                  f"outgroup_ratio={euc.analogical_outgroup_ratio:.2f}")
        print(f"  Delta-hyperbolicity: delta_rel={topo.gromov_delta_euclidean.delta_relative:.3f}")

    # Semantic orthogonality
    semantic = results.get("semantic")
    if isinstance(semantic, SemanticResult) and semantic.per_block_semantic:
        per_block_sem = semantic.per_block_semantic
        print(f"\n  Semantic orthogonality ({len(per_block_sem)} blocks tested):")
        tfidf_bundle = semantic.tfidf_classification
        if tfidf_bundle is not None and tfidf_bundle.rf is not None:
            print(f"    TF-IDF surface baseline: {tfidf_bundle.rf.accuracy:.1%}")
        for block_name, block_data in per_block_sem.items():
            if block_data.error is not None:
                continue
            parts: list[str] = []
            if block_data.classification is not None and block_data.classification.rf is not None:
                parts.append(f"RF={block_data.classification.rf.accuracy:.1%}")
            if block_data.mantel_tfidf_cosine is not None:
                parts.append(f"Mantel r={block_data.mantel_tfidf_cosine.r:.3f}")
            if block_data.text_to_compute_r2 is not None:
                parts.append(f"R²={block_data.text_to_compute_r2.median_r2:.3f}")
            n_sub = (
                block_data.per_mode_surface_vs_compute.n_sub_semantic
                if block_data.per_mode_surface_vs_compute is not None else "?"
            )
            parts.append(f"sub-semantic modes={n_sub}")
            print(f"    {block_name}: {', '.join(parts)}")

    # Scorecard
    sc = results.get("scorecard")
    if isinstance(sc, ScorecardResult):
        summary = sc.summary
        print(f"\n  Prediction scorecard: "
              f"{summary.confirmed} confirmed, "
              f"{summary.partial} partial, "
              f"{summary.wrong} wrong")
        for pred in sc.predictions:
            print(f"    {pred.prediction}: {pred.outcome}")

    # Section times
    times = results.get("section_times", {})
    if times:
        total = sum(times.values())
        print(f"\n  Total analysis time: {total:.0f}s")
        for name, elapsed in sorted(times.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {elapsed:.1f}s")
