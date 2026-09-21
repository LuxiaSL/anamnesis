"""A corpus narrower than the suite that defined the unions: every section says so.

Most of the gauntlet's headline readings are taken on the union of the attention block
and the cache-and-keys block. A corpus holding only the engineered families has no such
union — a union is built only when every member it names is present, because one built
short would report a narrower feature set under a label that names more.

So every section that reads a union by name has to answer the same way: state the
absence and return, rather than raise on a missing key and take the whole pass with it.
That is what these tests pin, one section at a time, for each entry point that asks for
a block or union by name.

The corpus is the four families the ``engineered`` union names, so that union *is*
present: the tests then distinguish "this section lost its union" from "this corpus has
nothing in it".

CPU only; the bank is synthetic and small.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anamnesis.analysis.gauntlet.clustering import run_clustering
from anamnesis.analysis.gauntlet.geometry import (
    run_ccgp,
    run_intrinsic_dimension,
    run_manifold_geometry,
    run_topology,
)
from anamnesis.analysis.gauntlet.integrity import run_integrity_checks
from anamnesis.analysis.gauntlet.legacy_bin_readout import run_legacy_bin_readout
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_FLOW,
    BLOCK_NPZ_KEYS,
    BLOCK_STORED_NAMES,
    GATE_FEATURES,
    RESIDUAL_TRAJECTORY,
    TEMPORAL_DYNAMICS,
    AnalysisData,
    load_analysis_data,
)
from anamnesis.analysis.gauntlet.utils import (
    absence_reason,
    get_available_blocks,
    is_error_stub,
)
from anamnesis.extraction.state_extractor import STORED_BLOCK_SLICES_KEY

FAMILY_WIDTHS = {
    RESIDUAL_TRAJECTORY: 4,
    ATTENTION_FLOW: 5,
    GATE_FEATURES: 3,
    TEMPORAL_DYNAMICS: 6,
}
MODES = ["linear", "socratic", "contrastive"]
N_TOPICS = 6


@pytest.fixture(scope="module")
def families_only(tmp_path_factory: pytest.TempPathFactory) -> AnalysisData:
    """Three modes × six topics, carrying the engineered families and nothing else."""
    folder = tmp_path_factory.mktemp("families_only")
    rng = np.random.default_rng(20260921)
    index = 0
    for mode_idx, mode in enumerate(MODES):
        for topic_idx in range(N_TOPICS):
            arrays: dict[str, np.ndarray] = {}
            names: list[str] = []
            slices: dict[str, list[int]] = {}
            cursor = 0
            for block, width in FAMILY_WIDTHS.items():
                signal = float(mode_idx) + 0.1 * rng.standard_normal(width)
                arrays[BLOCK_NPZ_KEYS[block]] = signal.astype(np.float32)
                slices[BLOCK_STORED_NAMES[block]] = [cursor, cursor + width]
                names.extend(f"{block}_feat_{i}" for i in range(width))
                cursor += width
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
                "num_generated_tokens": 150 + index,
                "generated_text": f"text for {mode} on topic {topic_idx}",
                "system_prompt": f"system for {mode}",
                "user_prompt": f"user for topic {topic_idx}",
                STORED_BLOCK_SLICES_KEY: slices,
            }))
            index += 1
    return load_analysis_data(folder, run_name="families_only", core_only=True,
                              load_text=False)


def test_the_corpus_holds_the_families_and_no_union_over_the_core_blocks(
    families_only: AnalysisData,
) -> None:
    """The premise every test below rests on, stated once."""
    assert set(families_only.run4.block_features) == set(FAMILY_WIDTHS)
    assert families_only.has_block(ALL_FAMILIES), "its four members are all present"
    assert not families_only.has_block(ATTENTION_AND_CACHE)
    all_blocks, _ = get_available_blocks(families_only)
    assert set(all_blocks) == set(FAMILY_WIDTHS) | {ALL_FAMILIES}
    # The engineered union is exactly its members' widths, not a subset of them.
    assert families_only.get_block(ALL_FAMILIES).shape[1] == sum(FAMILY_WIDTHS.values())


def test_the_absence_reason_names_the_block_and_what_is_present(
    families_only: AnalysisData,
) -> None:
    reason = absence_reason(families_only, ATTENTION_AND_CACHE)
    assert reason is not None
    assert ATTENTION_AND_CACHE in reason
    assert ATTENTION_FLOW in reason, "the reason says what the corpus does hold"
    assert absence_reason(families_only, ALL_FAMILIES) is None


def test_ccgp_reports_the_absence_instead_of_raising(families_only: AnalysisData) -> None:
    result = run_ccgp(families_only)
    assert result.error is not None and ATTENTION_AND_CACHE in result.error
    assert result.variants is None


def test_topology_reports_the_absence_instead_of_raising(families_only: AnalysisData) -> None:
    result = run_topology(families_only)
    assert result.error is not None and ATTENTION_AND_CACHE in result.error
    assert result.gromov_delta_euclidean is None


def test_manifold_geometry_reports_the_absence_instead_of_raising(
    families_only: AnalysisData,
) -> None:
    result = run_manifold_geometry(families_only)
    assert result.error is not None and ATTENTION_AND_CACHE in result.error
    assert result.curvature is None


def test_intrinsic_dimension_keeps_its_per_block_reading_and_names_what_it_lost(
    families_only: AnalysisData,
) -> None:
    """The per-block estimate does not need the union, so only the union's readings go."""
    result = run_intrinsic_dimension(families_only)
    if result.error is not None:
        pytest.skip(f"section 4 unavailable: {result.error}")
    assert result.per_mode is None
    assert result.gride is not None
    assert result.gride.error is not None and ATTENTION_AND_CACHE in result.gride.error
    assert result.global_ is not None and set(result.global_) >= set(FAMILY_WIDTHS)


def test_clustering_keeps_its_per_block_silhouettes_and_names_what_it_lost(
    families_only: AnalysisData,
) -> None:
    result = run_clustering(families_only)
    assert set(result.silhouette_by_block) >= set(FAMILY_WIDTHS)
    assert ATTENTION_AND_CACHE in str(result.kmeans_ari[ATTENTION_AND_CACHE])
    for embedding in result.embeddings.values():
        assert embedding.error is not None
        assert embedding.coords is None
    assert "error" in result.per_mode_silhouette_cosine


def test_the_stored_block_readout_keeps_the_blocks_it_has(
    families_only: AnalysisData,
) -> None:
    """Section 3 measures each present block; the two union-only readings say why not."""
    result = run_legacy_bin_readout(families_only)
    assert set(result.per_block_accuracy) >= set(FAMILY_WIDTHS)
    assert result.std_vs_mean.error is not None
    assert ATTENTION_AND_CACHE in result.std_vs_mean.error
    assert result.cohens_d_per_topic.error is not None
    assert result.cohens_d_per_topic.n_topics == 0


def test_integrity_reads_a_families_only_corpus_without_complaint(
    families_only: AnalysisData,
) -> None:
    result = run_integrity_checks(families_only)
    assert result.all_clean
    assert set(result.block_dims) >= set(FAMILY_WIDTHS)
    assert result.block_dims[ALL_FAMILIES] == sum(FAMILY_WIDTHS.values())


def test_the_contrastive_section_reports_the_absence(families_only: AnalysisData) -> None:
    from anamnesis.analysis.gauntlet.contrastive import HAS_TORCH, run_contrastive

    result = run_contrastive(families_only)
    assert result.error is not None
    if HAS_TORCH:
        assert ATTENTION_AND_CACHE in result.error


def test_the_semantic_section_reports_the_absence(families_only: AnalysisData) -> None:
    from anamnesis.analysis.gauntlet.semantic import run_semantic

    # No text was loaded for this corpus, which section 9 refuses on first; the
    # union it would then have read is the next thing it cannot have.
    result = run_semantic(families_only)
    assert result.error is not None


def test_the_scorecard_reads_this_corpus_own_stubs_without_dying(
    families_only: AnalysisData,
) -> None:
    """The consuming section, over the stubs the real sections actually produced.

    `tests/test_section_consumers.py` pins the rule against hand-built stubs; this
    is the same rule against the ones this corpus makes, which is what a pass over
    it hands to section 10.
    """
    from anamnesis.analysis.gauntlet.scorecard import run_scorecard

    produced = {
        "integrity": run_integrity_checks(families_only),
        "legacy_bin_readout": run_legacy_bin_readout(families_only),
        "intrinsic_dimension": run_intrinsic_dimension(families_only),
        "ccgp": run_ccgp(families_only),
        "topology": run_topology(families_only),
        "clustering": run_clustering(families_only),
        "manifold_geometry": run_manifold_geometry(families_only),
    }
    for key in ("ccgp", "topology", "manifold_geometry"):
        assert is_error_stub(produced[key]), f"{key} should have stubbed on this corpus"

    scorecard = run_scorecard(produced)
    assert len(scorecard.predictions) == 9
    for row in scorecard.predictions:
        assert row.outcome == "INSUFFICIENT_DATA", row.prediction
        assert row.unscorable_because is not None, row.prediction
    assert scorecard.summary.wrong == 0, "a missing measurement is not a failed prediction"
    assert scorecard.error is not None and "no prediction could be scored" in scorecard.error
