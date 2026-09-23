"""A section that reads another section's result treats the error stub as a value.

Most sections read the signature data. Two things read *sections*: the prediction
scorecard, which scores nine standing expectations off five of them, and the
summary the runner prints at the end of a pass. A section that could not run returns
its reason in place of its numbers, so for those two readers a stub is not an error
condition — it is one of the values their input can have, exactly as an absent block
is one of the states a corpus can be in.

Three properties, and the last two are the ones that matter:

  * a stubbed upstream produces a stubbed downstream reading, never an exception —
    otherwise one section that could not run ends the whole pass;
  * a prediction whose evidence is absent is **unscored**, never scored WRONG. A
    verdict against a standing expectation is a finding, and a finding that rests on
    a measurement nobody took is the same overstatement as a union reported at a
    width it does not have;
  * a verdict never travels without what bounds it. The row that orders three feature
    blocks carries the limit on what a block ordering can mean, and the scorecard
    carries what corpus its thresholds are fixed for — in the result and in the
    printed summary both, because a limit only one of those two surfaces states is a
    limit half the readers never meet.

The fixtures here are the smallest results documents that satisfy the section
schemas, so one upstream can be stubbed at a time and the rows that depend on it
identified exactly.

CPU only; no data, no model.
"""

from __future__ import annotations

from typing import Any

import pytest

from anamnesis.analysis.complementarity import load_results
from anamnesis.analysis.gauntlet import _print_summary, section_shortfall
from anamnesis.analysis.gauntlet.schemas import (
    AnalysisResults,
    CCGPResult,
    CCGPSummary,
    CCGPVariant,
    ClassificationResult,
    GlobalBlockIDResult,
    GromovDeltaResult,
    IntrinsicDimensionResult,
    LegacyBinReadoutResult,
    PerModeIDResult,
    ScorecardResult,
    TopologyMetricSummary,
    TopologyResult,
    BlockConvergenceResult,
)
from anamnesis.analysis.gauntlet.scorecard import REGISTERED_CORPUS, run_scorecard
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_DELTAS,
    CACHE_AND_KEYS,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
)
from anamnesis.analysis.gauntlet.utils import (
    BLOCK_READOUT_LIMIT,
    is_error_stub,
    section_reading,
)

MODES = ["linear", "contrastive", "dialectical", "socratic", "analogical"]

#: Which predictions each upstream section is the only source for.
ROWS_BY_UPSTREAM: dict[str, tuple[int, ...]] = {
    "ccgp": (1,),
    "topology": (2, 9),
    "intrinsic_dimension": (3, 4, 5),
    "legacy_bin_readout": (6,),
    "classification": (7, 8),
}
ALL_ROWS = tuple(sorted(row for rows in ROWS_BY_UPSTREAM.values() for row in rows))


def a_ccgp_result() -> CCGPResult:
    variant = CCGPVariant(
        multiclass_mean=0.8, multiclass_fold_accs=[0.8],
        per_mode_recall={m: 0.8 for m in MODES},
        n_decodable=1, n_dichotomies=1, ccgp_score=1.0, dichotomies=[],
    )
    return CCGPResult(
        variants={"knn3_seed42_5fold": variant},
        summary=CCGPSummary(min_ccgp=1.0, max_ccgp=1.0, all_perfect=True),
    )


def a_topology_result() -> TopologyResult:
    return TopologyResult(
        block=ATTENTION_AND_CACHE,
        euclidean_centroid_distances={"linear__socratic": 1.0},
        cosine_centroid_distances={"linear__socratic": 0.5},
        manhattan_centroid_distances={"linear__socratic": 2.0},
        hierarchical_clustering={"euclidean_average": "(linear,socratic)"},
        topology_summary={"euclidean": TopologyMetricSummary(
            nearest_pair="linear-socratic", nearest_dist=1.0,
            farthest_pair="analogical-linear", farthest_dist=4.0,
            analogical_outgroup_ratio=1.5,
        )},
        gromov_delta_euclidean=GromovDeltaResult(
            delta_max=1.0, delta_relative=0.1, delta_mean=0.5,
            delta_median=0.4, diameter=10.0, n_quadruples=100,
        ),
    )


def an_intrinsic_dimension_result() -> IntrinsicDimensionResult:
    def block(value: float) -> GlobalBlockIDResult:
        return GlobalBlockIDResult(
            n_features_clean=10, dadapy_id=value, bootstrap_by_seed={},
        )
    return IntrinsicDimensionResult(
        global_={
            NORMS_AND_OUTPUT_STATS: block(6.0),
            ATTENTION_AND_DELTAS: block(7.0),
            CACHE_AND_KEYS: block(8.0),
            RESIDUAL_PCA: block(20.0),
        },
        per_mode={
            mode: PerModeIDResult(n_samples=20, dadapy_id=float(i + 1))
            for i, mode in enumerate(MODES)
        },
        block_convergence=BlockConvergenceResult(
            max_pairwise_diff=2.0, converged_within_2=True,
            values={NORMS_AND_OUTPUT_STATS: 6.0, ATTENTION_AND_DELTAS: 7.0,
                    CACHE_AND_KEYS: 8.0},
        ),
    )


def a_legacy_bin_readout_result() -> LegacyBinReadoutResult:
    return LegacyBinReadoutResult.model_validate({
        "per_block_accuracy": {
            NORMS_AND_OUTPUT_STATS: 0.4, ATTENTION_AND_DELTAS: 0.5, CACHE_AND_KEYS: 0.6,
        },
        "pairwise_block_combinations": {},
        "leave_one_block_out": {},
        "block_ranking": [{"block": CACHE_AND_KEYS, "accuracy": 0.6}],
        "cache_beats_attention_beats_norms": True,
        "top_features_rf_attention_and_cache": [],
        "top_features_lr_attention_and_cache": [],
        "block_contribution_ratio": {},
        "std_vs_mean": {"n_std_features": 1, "n_mean_features": 1},
        "cohens_d_per_topic": {
            "per_topic": {}, "mean_d": None, "median_d": None, "std_d": None,
            "min_d": None, "max_d": None, "all_positive": None, "n_topics": 0,
        },
    })


def a_classification_result() -> ClassificationResult:
    block = {
        "rf_5way": {"accuracy": 0.7, "fold_accuracies": [0.7],
                    "confusion_matrix": [[1]], "labels": MODES},
        "topic_heldout": {"accuracy": 0.7, "fold_accuracies": [0.7], "n_groups": 20},
        "linear_probe": {"accuracy": 0.6, "fold_accuracies": [0.6]},
        "pairwise_binary": {
            "linear_vs_socratic": {"accuracy": 0.9, "fold_accuracies": [0.9]},
            "analogical_vs_linear": {"accuracy": 0.95, "fold_accuracies": [0.95]},
        },
        "rf_4way_no_analogical": {"error": "analogical absent"},
    }
    return ClassificationResult.model_validate(
        {ATTENTION_AND_CACHE: block, ALL_CORE: block}
    )


UPSTREAM_BUILDERS = {
    "ccgp": a_ccgp_result,
    "topology": a_topology_result,
    "intrinsic_dimension": an_intrinsic_dimension_result,
    "legacy_bin_readout": a_legacy_bin_readout_result,
    "classification": a_classification_result,
}

STUBS: dict[str, Any] = {
    "ccgp": CCGPResult(error="CCGP reads attention_and_cache not in this corpus"),
    "topology": TopologyResult(error="centroid topology reads attention_and_cache not in this corpus"),
    "intrinsic_dimension": IntrinsicDimensionResult(error="dadapy not installed"),
}
"""The three upstreams that have a stub path of their own. Sections 2 and 3 always
produce their numbers, so what the scorecard has to survive from them is their
absence, which the parametrized case below covers for all five."""

STUBBABLE = tuple(STUBS)


def a_full_results_document() -> dict[str, Any]:
    return {key: build() for key, build in UPSTREAM_BUILDERS.items()}


def rows_by_number(result: ScorecardResult) -> dict[int, Any]:
    return {int(row.prediction.split(".", 1)[0]): row for row in result.predictions}


def test_a_complete_document_scores_every_prediction() -> None:
    """The baseline the stubbed cases are read against."""
    result = run_scorecard(a_full_results_document())
    rows = rows_by_number(result)
    assert sorted(rows) == list(ALL_ROWS)
    assert result.error is None
    assert result.summary.insufficient == 0
    for number, row in rows.items():
        assert row.outcome != "INSUFFICIENT_DATA", number
        assert row.unscorable_because is None, number


@pytest.mark.parametrize("upstream", STUBBABLE)
def test_one_stubbed_upstream_unscores_its_rows_and_only_its_rows(upstream: str) -> None:
    """The rule: a stub reaches the consumer as a value, and costs exactly its rows."""
    document = a_full_results_document()
    document[upstream] = STUBS[upstream]
    assert is_error_stub(document[upstream]), "the fixture is a stub"

    result = run_scorecard(document)
    rows = rows_by_number(result)
    expected_unscored = set(ROWS_BY_UPSTREAM[upstream])

    unscored = {n for n, row in rows.items() if row.outcome == "INSUFFICIENT_DATA"}
    assert unscored == expected_unscored
    for number in expected_unscored:
        reason = rows[number].unscorable_because
        assert reason is not None, number
        assert upstream in reason, (number, reason)
    # The stub's own reason travels, so a reader learns why rather than only that.
    assert any(
        "not in this corpus" in (rows[n].unscorable_because or "")
        or "not installed" in (rows[n].unscorable_because or "")
        for n in expected_unscored
    )
    # Some rows still scored, so the section landed.
    assert result.error is None


@pytest.mark.parametrize("upstream", STUBBABLE)
def test_a_stubbed_upstream_never_scores_a_prediction_wrong(upstream: str) -> None:
    """A missing measurement is not a failed prediction."""
    document = a_full_results_document()
    document[upstream] = STUBS[upstream]
    rows = rows_by_number(run_scorecard(document))
    for number in ROWS_BY_UPSTREAM[upstream]:
        assert rows[number].outcome == "INSUFFICIENT_DATA"
        assert rows[number].outcome not in {"WRONG", "PARTIAL", "CONFIRMED"}


@pytest.mark.parametrize("upstream", sorted(ROWS_BY_UPSTREAM))
def test_an_upstream_absent_from_the_document_unscores_its_rows(upstream: str) -> None:
    """A section never run at all — named in ``--skip``, say — reads like a stub."""
    document = a_full_results_document()
    del document[upstream]
    rows = rows_by_number(run_scorecard(document))
    unscored = {n for n, row in rows.items() if row.outcome == "INSUFFICIENT_DATA"}
    assert unscored == set(ROWS_BY_UPSTREAM[upstream])
    for number in ROWS_BY_UPSTREAM[upstream]:
        assert rows[number].unscorable_because is not None, number


def test_no_upstream_at_all_scores_nothing_and_says_so() -> None:
    """The families-only shape: every section the scorecard reads is absent."""
    result = run_scorecard({})
    rows = rows_by_number(result)
    assert sorted(rows) == list(ALL_ROWS)
    assert result.summary.insufficient == result.summary.total == len(ALL_ROWS)
    assert result.summary.wrong == 0 and result.summary.partial == 0
    for number, row in rows.items():
        assert row.outcome == "INSUFFICIENT_DATA", number
        assert row.unscorable_because is not None, number
    assert result.error is not None
    assert "no prediction could be scored" in result.error


def test_a_scorecard_that_scored_nothing_is_a_shortfall(tmp_path: Any) -> None:
    """Its stub reaches the command layer, because that is what refuses on it."""
    results = AnalysisResults.model_validate({
        "run_name": "families_only",
        "timestamp": "2026-01-01T00:00:00",
        "core_only": True,
        "last_updated": "2026-01-01T00:00:00",
        "n_samples": 100,
        "scorecard": run_scorecard({}).model_dump(mode="json", exclude_none=True),
    })
    shortfall = section_shortfall(
        results, output_dir=tmp_path, command="pytest", skip_sections=None,
    )
    assert "scorecard" in shortfall.failures
    assert "no prediction could be scored" in shortfall.failures["scorecard"]
    assert "scorecard" not in shortfall.produced


def test_a_scorecard_that_scored_some_rows_is_produced(tmp_path: Any) -> None:
    document = a_full_results_document()
    document["ccgp"] = STUBS["ccgp"]
    scorecard = run_scorecard(document)
    assert scorecard.error is None
    results = AnalysisResults.model_validate({
        "run_name": "partial",
        "timestamp": "2026-01-01T00:00:00",
        "core_only": True,
        "last_updated": "2026-01-01T00:00:00",
        "n_samples": 100,
        "scorecard": scorecard.model_dump(mode="json", exclude_none=True),
    })
    shortfall = section_shortfall(
        results, output_dir=tmp_path, command="pytest", skip_sections=None,
    )
    assert "scorecard" in shortfall.produced


def test_the_printed_summary_reads_stubs_without_raising(capsys: Any) -> None:
    """The other consumer of a section's result, and the same rule."""
    document: dict[str, Any] = {
        "ccgp": STUBS["ccgp"],
        "topology": STUBS["topology"],
        "intrinsic_dimension": STUBS["intrinsic_dimension"],
        "section_times": {"ccgp": 0.1},
    }
    document["scorecard"] = run_scorecard(document)
    _print_summary(document)
    printed = capsys.readouterr().out
    assert "CCGP" in printed and "did not run" in printed
    assert "Topology" in printed
    assert "INSUFFICIENT_DATA" in printed


BLOCK_ORDERING_ROW = 6


def test_the_block_ordering_row_carries_its_limit_in_every_outcome() -> None:
    """Row 6 states what a block ordering cannot mean, scored or unscored.

    The limit is a property of the reading, not of the verdict, so it is on the row
    whether the row said CONFIRMED, WRONG or nothing at all.
    """
    for document in (a_full_results_document(), {}):
        row = rows_by_number(run_scorecard(document))[BLOCK_ORDERING_ROW]
        assert row.caveat == BLOCK_READOUT_LIMIT, row.outcome
        assert "localizes nothing" in row.caveat
        assert "run_subfamily_decomp" in row.caveat
    # The row's wording states the ordering it tests, not a substrate the limit denies.
    scored = rows_by_number(run_scorecard(a_full_results_document()))[BLOCK_ORDERING_ROW]
    assert scored.outcome == "CONFIRMED", "the fixture orders the three blocks"
    assert "load-bearing" not in scored.prediction


def test_the_printed_summary_shows_no_verdict_without_its_limit(capsys: Any) -> None:
    """The surface a first-time reader meets: the limit is printed beside the verdict."""
    document = a_full_results_document()
    document["run_name"] = "synthetic_demo"
    document["scorecard"] = run_scorecard(document, lane_id="synthetic-bank-v1")
    _print_summary(document)
    printed = capsys.readouterr().out

    assert "CONFIRMED" in printed
    # Twice: once under the block ranking the readout prints, once under row 6.
    assert printed.count("localizes nothing") == 2
    assert "drawn from a generator" in printed
    assert "synthetic_demo" in printed


def test_the_scorecard_states_what_corpus_its_thresholds_are_fixed_for() -> None:
    """A verdict is agreement with an expectation fixed elsewhere, and says so."""
    document = a_full_results_document()
    document["run_name"] = "somebody_elses_model"

    measured = run_scorecard(document, lane_id="cuda-h100-torch271")
    assert measured.corpus_caveat is not None
    assert REGISTERED_CORPUS in measured.corpus_caveat
    assert "somebody_elses_model" in measured.corpus_caveat
    assert "drawn from a generator" not in measured.corpus_caveat

    drawn = run_scorecard(document, lane_id="synthetic-bank-v1")
    assert drawn.corpus_caveat is not None
    assert "drawn from a generator" in drawn.corpus_caveat

    # A caller with no lane to hand still gets the registration stated.
    unknown = run_scorecard(document)
    assert unknown.corpus_caveat is not None
    assert REGISTERED_CORPUS in unknown.corpus_caveat
    assert "drawn from a generator" not in unknown.corpus_caveat


def test_section_reading_distinguishes_its_three_states() -> None:
    complete = a_ccgp_result()
    value, reason = section_reading({"ccgp": complete}, "ccgp", CCGPResult)
    assert value is complete and reason is None

    value, reason = section_reading({"ccgp": STUBS["ccgp"]}, "ccgp", CCGPResult)
    assert value is None and reason is not None
    assert "did not run" in reason and "not in this corpus" in reason

    value, reason = section_reading({}, "ccgp", CCGPResult)
    assert value is None and reason is not None and "not in these results" in reason

    value, reason = section_reading({"ccgp": {"variants": {}}}, "ccgp", CCGPResult)
    assert value is None and reason is not None and "did not validate" in reason

    value, reason = section_reading({"ccgp": {"error": "no"}}, "ccgp", CCGPResult)
    assert value is None and reason is not None and "did not run" in reason


def test_a_banked_document_with_a_stubbed_section_reads_as_incomplete_not_broken(
    tmp_path: Any, caplog: Any
) -> None:
    """``complementarity`` reads banked files, where a stub is equally a value."""
    import json
    import logging

    (tmp_path / "results.json").write_text(json.dumps({
        "run_name": "families_only",
        "n_samples": 100,
        "classification": {"error": "classification did not run"},
    }))
    with caplog.at_level(logging.INFO):
        results = load_results(tmp_path / "results.json")
    assert results is not None
    assert isinstance(results["classification"], dict), "left as the mapping it is"
    messages = [record.message for record in caplog.records]
    assert any("did not run" in message for message in messages)
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ], "an incomplete file is not a malformed one"
