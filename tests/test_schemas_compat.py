"""A banked ``results.json`` written under older field names still loads.

A gauntlet result is cited and re-read long after it is written, so a rename in a
section schema is only half a change: the other half is that every file already on
disk validates into the same numbers under the new names. These tests are that half.

Two levels of evidence, deliberately both:

  * a fixture spelled the way the older files are, which pins the mapping itself and
    runs on any machine;
  * the real banked documents under the outputs root's ``analysis/`` directory when
    they are reachable, which is the only check that the fixture resembles what was
    actually written. ``ANAMNESIS_OUTPUTS`` points at that root.

The first test walks the table itself, so an entry added to the map is exercised
without anyone remembering to extend a fixture; the hand-written document below it is
what checks that the renamed names then satisfy the strict schema.

CPU only; no model. The banked-document test skips when the data is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anamnesis.analysis.gauntlet.schemas import AnalysisResults
from anamnesis.analysis.gauntlet.schemas.compat import (
    FIELD_RENAMES,
    SECTION_RENAMES,
    migrate_banked_results,
)
from anamnesis.config.paths import outputs_root

BANKED_RUNS = ("8b_v2", "3b_v2", "8b_baseline", "3b_run4", "8b_v2_5way", "3b_v2_5way")
"""The runs a gauntlet pass banked before the rename. A machine holding none of them
skips, because the fixture above is what makes the mapping itself testable anywhere."""


def banked_analysis_dir() -> Path:
    """Where banked gauntlet runs live: the outputs root's ``analysis`` directory."""
    return outputs_root() / "analysis"


def test_the_table_maps_every_old_name_and_leaves_everything_else_alone() -> None:
    """Each entry fires, and a key the table does not name keeps its spelling."""
    for section, renames in FIELD_RENAMES.items():
        for old, new in renames.items():
            payload = {section: {old: 1, "untouched": 2}}
            migrated = migrate_banked_results(payload)
            assert migrated[section] == {new: 1, "untouched": 2}, (section, old)

    for old, new in SECTION_RENAMES.items():
        assert new in migrate_banked_results({old: {}})

    # A section with no entry is copied through, keys and all.
    passthrough = {"contrastive": {"tier_ablation": {"individual": {}}}}
    assert migrate_banked_results(passthrough) == passthrough


def test_renames_reach_nested_rows_not_only_the_section_top() -> None:
    """``tier`` names a column inside one row of section 3's ranking table."""
    migrated = migrate_banked_results(
        {"tier_ablation": {"tier_ranking": [{"tier": "T2.5", "accuracy": 0.6}]}}
    )
    assert migrated["legacy_bin_readout"]["block_ranking"] == [
        {"block": "T2.5", "accuracy": 0.6}
    ]


def test_a_document_already_on_the_current_names_is_unchanged() -> None:
    current: dict[str, Any] = {
        "legacy_bin_readout": {"per_block_accuracy": {"T2.5": 0.62}},
        "section_times": {"legacy_bin_readout": 12.5},
    }
    assert migrate_banked_results(current) == current


def test_a_document_in_the_old_spelling_validates_into_the_composite() -> None:
    """The mapping's point: a file's numbers arrive under the current field names."""
    old_spelling: dict[str, Any] = {
        "run_name": "fixture",
        "timestamp": "2026-01-01T00:00:00",
        "core_only": False,
        "last_updated": "2026-01-01T00:00:00",
        "n_samples": 100,
        "section_times": {"tier_ablation": 12.5},
        "tier_ablation": {
            "per_tier_accuracy": {"T1": 0.4, "T2": 0.5, "T2.5": 0.6},
            "pairwise_tier_combinations": {
                "T2+T2.5": {
                    "accuracy": 0.66,
                    "n_features": 442,
                    "individual_max": 0.6,
                    "gain_over_best_individual": 0.06,
                }
            },
            "leave_one_tier_out": {"T2.5": {"accuracy_without": 0.58, "cost_of_removal": 0.08}},
            "tier_ranking": [{"tier": "T2.5", "accuracy": 0.6}],
            "tier_inversion_t25_gt_t2_gt_t1": True,
            "top_features_rf_t2t25": [{"name": "cache_lookback_ratio_L16", "importance": 0.01}],
            "top_features_lr_t2t25": [],
            "tier_contribution_ratio": {"T2.5": 0.41},
            "std_vs_mean": {"n_std_features": 10, "n_mean_features": 10},
            "cohens_d_per_topic": {
                "per_topic": {},
                "mean_d": None,
                "median_d": None,
                "std_d": None,
                "min_d": None,
                "max_d": None,
                "all_positive": None,
                "n_topics": 0,
            },
        },
    }
    results = AnalysisResults.model_validate(migrate_banked_results(old_spelling))
    readout = results.legacy_bin_readout
    assert readout is not None
    assert readout.per_block_accuracy["T2.5"] == 0.6
    assert readout.cache_beats_attention_beats_norms is True
    assert readout.block_ranking[0].block == "T2.5"
    assert readout.leave_one_block_out["T2.5"].cost_of_removal == 0.08
    assert readout.top_features_rf_attention_and_cache[0].importance == 0.01
    assert results.section_times == {"legacy_bin_readout": 12.5}


@pytest.mark.parametrize("run", BANKED_RUNS)
def test_a_real_banked_run_validates_through_the_table(run: str) -> None:
    """The fixture above is a claim about real files; this is the claim checked."""
    path = banked_analysis_dir() / run / "results.json"
    if not path.exists():
        pytest.skip(f"banked analysis absent: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    results = AnalysisResults.model_validate(migrate_banked_results(document))
    assert results.run_name
    # Section 3 is the one the rename touched hardest, so its numbers are the ones
    # worth reading back: the accuracies must survive the key change unchanged.
    readout = results.legacy_bin_readout
    if readout is not None:
        original = document.get("tier_ablation", {}).get("per_tier_accuracy", {})
        assert readout.per_block_accuracy == original
