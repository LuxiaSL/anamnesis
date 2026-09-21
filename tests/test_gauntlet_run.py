"""The gauntlet as an orchestrator: dispatch, checkpoint, resume, summary.

This runs a real pass over a small synthetic corpus, with the sections that need
a GPU-scale corpus or an optional dependency skipped. What it is testing is not
the statistics — each section's arithmetic came over unchanged and its numbers
are the record's — but the machinery around them, which is what the rename to
``gauntlet`` and the schema split could have broken:

  * the section registry dispatches by string through ``importlib``, so a module
    renamed without its registry entry fails only when the section runs;
  * text is loaded only when a section that reads it will run;
  * a completed section is checkpointed, and a resumed run rehydrates each
    checkpointed section into its typed model rather than leaving a bare dict;
  * ``always_rerun`` means the scorecard is never treated as done;
  * the returned composite validates, which is the one place a section returning
    a shape its schema forbids would be caught.

Five sections stay out of the pass. Classification's key-block sweep is minutes of
CPU, so its parts are tested at small parameters in
``test_gauntlet_classification``; the intrinsic-dimension and CCGP sections want a
corpus with condition structure a toy corpus does not have; the contrastive and
semantic sections want optional dependencies. The other six run, including all
three geometry sections that a synthetic cloud can carry.

CPU only; no banked data, no model, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.gauntlet import (
    SECTION_KEYS,
    SECTION_MODELS,
    SECTION_NAMES,
    SECTIONS,
    run_full_analysis,
)
from anamnesis.analysis.gauntlet.integrity import run_integrity_checks
from anamnesis.analysis.gauntlet.schemas import (
    AnalysisResults,
    IntegrityResult,
    ScorecardResult,
)
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ATTENTION_AND_DELTAS,
    BLOCK_NPZ_KEYS,
    BLOCK_STORED_NAMES,
    CACHE_AND_KEYS,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    load_analysis_data,
)
from anamnesis.extraction.state_extractor import STORED_BLOCK_SLICES_KEY
from anamnesis.analysis.gauntlet.utils import (
    clean_for_json,
    get_available_blocks,
    remove_constant,
    standardize,
    timer,
)

MODES = ["linear", "socratic", "contrastive", "dialectical", "analogical"]
N_TOPICS = 8
#: Sections needing an optional dependency, a larger corpus, or minutes of CPU.
SKIPPED_SECTIONS = {2, 4, 5, 8, 9}


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Five modes × eight topics of separable signatures, with real metadata.

    Each mode gets its own offset so classification has something to find; the
    per-sample noise is seeded, so a fold accuracy that moves is a code change.
    """
    folder = tmp_path_factory.mktemp("synthetic_signatures")
    rng = np.random.default_rng(20260920)
    width = {
        NORMS_AND_OUTPUT_STATS: 6, ATTENTION_AND_DELTAS: 8,
        CACHE_AND_KEYS: 8, RESIDUAL_PCA: 4,
    }
    index = 0
    for mode_idx, mode in enumerate(MODES):
        for topic_idx in range(N_TOPICS):
            arrays: dict[str, np.ndarray] = {}
            names: list[str] = []
            slices: dict[str, list[int]] = {}
            cursor = 0
            for block, w in width.items():
                signal = float(mode_idx) + 0.15 * rng.standard_normal(w)
                arrays[BLOCK_NPZ_KEYS[block]] = signal.astype(np.float32)
                slices[BLOCK_STORED_NAMES[block]] = [cursor, cursor + w]
                names.extend(f"{block}_feat_{i}" for i in range(w))
                cursor += w
            np.savez(
                folder / f"gen_{index:03d}.npz",
                feature_names=np.array(names),
                **arrays,
            )
            (folder / f"gen_{index:03d}.json").write_text(json.dumps({
                "generation_id": index,
                "topic": f"topic_{topic_idx}",
                "topic_idx": topic_idx,
                "mode": mode,
                "mode_idx": mode_idx,
                "num_generated_tokens": 200 + 5 * mode_idx + topic_idx,
                "prompt_length": 40,
                "generated_text": f"{mode} text about topic {topic_idx} " * 8,
                "system_prompt": f"system prompt for {mode}",
                "user_prompt": f"user prompt for topic {topic_idx}",
                STORED_BLOCK_SLICES_KEY: slices,
            }))
            index += 1
    return folder


def test_the_registry_is_internally_consistent() -> None:
    numbers = [spec.number for spec in SECTIONS]
    assert numbers == sorted(numbers) == list(range(1, 12))
    assert len({spec.key for spec in SECTIONS}) == len(SECTIONS)
    assert SECTION_KEYS == {spec.number: spec.key for spec in SECTIONS}
    assert SECTION_NAMES == {spec.number: spec.name for spec in SECTIONS}
    assert [spec.number for spec in SECTIONS if spec.requires_text] == [9]
    assert [spec.number for spec in SECTIONS if spec.always_rerun] == [10]
    assert set(SECTION_MODELS) == {spec.key for spec in SECTIONS}


def test_integrity_reads_the_corpus_it_is_given(synthetic_run: Path) -> None:
    data = load_analysis_data(synthetic_run, run_name="synthetic", core_only=True)
    result = run_integrity_checks(data)
    assert isinstance(result, IntegrityResult)
    assert result.n_samples == len(MODES) * N_TOPICS
    assert result.n_modes == len(MODES)
    assert result.n_topics == N_TOPICS
    assert result.balanced is True
    assert result.all_clean is True
    assert result.total_features == 26
    assert result.length_by_mode is not None and set(result.length_by_mode) == set(MODES)


def test_a_full_pass_checkpoints_resumes_and_validates(synthetic_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    first = run_full_analysis(
        signature_dir=synthetic_run,
        run_name="synthetic",
        output_dir=out,
        core_only=True,
        skip_sections=SKIPPED_SECTIONS,
    )
    assert isinstance(first, AnalysisResults)
    assert first.run_name == "synthetic"
    assert first.n_samples == len(MODES) * N_TOPICS
    assert first.integrity is not None and first.legacy_bin_readout is not None
    assert first.clustering is not None and first.topology is not None
    assert first.manifold_geometry is not None
    assert isinstance(first.scorecard, ScorecardResult)
    assert first.classification is None, "a skipped section stays unpopulated"
    assert first.ccgp is None
    assert set(first.section_times) == {"integrity", "legacy_bin_readout", "clustering",
                                        "topology", "manifold_geometry"}

    checkpoint = json.loads((out / "results.json").read_text())
    assert checkpoint["run_name"] == "synthetic"
    assert "scorecard" in checkpoint
    assert (out / "figures").is_dir()

    second = run_full_analysis(
        signature_dir=synthetic_run,
        run_name="synthetic",
        output_dir=out,
        core_only=True,
        skip_sections=SKIPPED_SECTIONS,
        resume=True,
    )
    # Resumed sections come back from the checkpoint identical, and the
    # scorecard — always_rerun — is recomputed rather than trusted.
    assert second.integrity == first.integrity
    assert second.legacy_bin_readout == first.legacy_bin_readout
    assert second.scorecard is not None
    assert second.timestamp == first.timestamp, "resume keeps the original run's stamp"
    assert second.last_updated >= first.last_updated


def test_an_error_stub_in_a_checkpoint_is_not_a_completed_section(
    synthetic_run: Path, tmp_path: Path
) -> None:
    out = tmp_path / "analysis"
    out.mkdir()
    (out / "results.json").write_text(json.dumps({
        "run_name": "synthetic",
        "timestamp": "2020-01-01 00:00:00",
        "core_only": True,
        "integrity": {"error": "a previous run died here"},
    }))
    resumed = run_full_analysis(
        signature_dir=synthetic_run,
        run_name="synthetic",
        output_dir=out,
        core_only=True,
        skip_sections=SKIPPED_SECTIONS | {3, 6, 7, 11},
        resume=True,
    )
    assert isinstance(resumed.integrity, IntegrityResult), "the error stub was rerun"


def test_available_blocks_separates_the_expensive_composites(synthetic_run: Path) -> None:
    data = load_analysis_data(synthetic_run, run_name="synthetic", core_only=True,
                              load_text=False)
    all_blocks, key_blocks = get_available_blocks(data)
    assert set(all_blocks) >= {
        NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS,
        RESIDUAL_PCA, ALL_CORE,
    }
    assert set(key_blocks) <= set(all_blocks)
    assert NORMS_AND_OUTPUT_STATS not in key_blocks, (
        "a single block never pays for the expensive sweep"
    )
    assert ALL_CORE in key_blocks


def test_the_timer_reports_the_elapsed_time_it_measured() -> None:
    with timer("a label") as elapsed:
        pass
    assert elapsed["elapsed"] >= 0.0


def test_json_cleaning_handles_what_json_cannot(synthetic_run: Path) -> None:
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "array": np.arange(3),
        "scalar": np.float32(1.5),
        "nested": [{"int": np.int64(7)}],
    }
    cleaned = clean_for_json(payload)
    json.dumps(cleaned)  # raises if anything survived that json cannot write
    assert cleaned["array"] == [0, 1, 2]
    assert cleaned["nested"][0]["int"] == 7


def test_the_z_score_convention_is_column_wise_and_drops_frozen_columns() -> None:
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 6.0]])
    Z = standardize(X)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0.0)
    kept = remove_constant(X)
    assert kept.shape == (3, 2), "the constant column is the one that goes"
