"""A banked ``results.json`` written under older names still loads.

A gauntlet result is cited and re-read long after it is written, so a rename in a
section schema — or in the label a block is reported under — is only half a change: the
other half is that every file already on disk validates into the same numbers under the
new names. These tests are that half.

Two levels of evidence, deliberately both:

  * fixtures spelled the way the older files are, which pin the mappings themselves and
    run on any machine;
  * the real banked documents under the outputs root's ``analysis/`` directory when
    they are reachable, which is the only check that the fixtures resemble what was
    actually written. ``ANAMNESIS_OUTPUTS`` points at that root.

The table-walking tests take their cases from the tables, so an entry added to a map is
exercised without anyone remembering to extend a fixture, and a label renamed without an
entry fails rather than passing quietly. The hand-written documents below them are what
check that the renamed names then satisfy the strict schema.

CPU only; no model. The banked-document test skips when the data is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anamnesis.analysis.gauntlet.schemas import AnalysisResults
from anamnesis.analysis.gauntlet.schemas.compat import (
    BLOCK_LABEL_RENAMES,
    COMPOUND_LABEL_FIELDS,
    FIELD_RENAMES,
    KNOWN_LABELS,
    LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES,
    LEGACY_EVERYTHING,
    LEGACY_FAMILY_UNION,
    LEGACY_UNION_COUNTERPARTS,
    LEGACY_UNION_LABELS,
    LEGACY_UNION_MEMBERS,
    READ_ONLY_LABELS,
    RETIRED_LABEL_SPELLINGS,
    SECTION_RENAMES,
    migrate_banked_results,
    union_counterpart,
)
from anamnesis.analysis.gauntlet.signature_io import (
    ALL_FAMILIES,
    ALL_LABELS,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    BLOCK_UNIONS,
    CACHE_AND_KEYS,
    EVERYTHING,
    NORMS_AND_OUTPUT_STATS,
)
from anamnesis.analysis.gauntlet.utils import clean_for_json
from anamnesis.config.paths import outputs_root

BANKED_RUNS = ("8b_v2", "3b_v2", "8b_baseline", "3b_run4", "8b_v2_5way", "3b_v2_5way")
"""The runs a gauntlet pass banked before the renames. A machine holding none of them
skips, because the fixtures above are what make the mappings testable anywhere."""


BASE_METADATA: dict[str, Any] = {
    "run_name": "fixture",
    "timestamp": "2026-01-01T00:00:00",
    "core_only": False,
    "last_updated": "2026-01-01T00:00:00",
    "n_samples": 10,
}
"""What every results document carries beside its sections."""


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
    passthrough = {"clustering": {"kmeans_ari": {"combined": 0.4}}}
    assert migrate_banked_results(passthrough) == passthrough


def test_every_label_says_whether_it_was_renamed() -> None:
    """The label table is total over the labels, so a rename cannot slip past.

    Derived from `anamnesis/analysis/gauntlet/signature_io.py` rather than listed: a
    block or union added there, or a label respelled there, has no entry here and fails
    this, which is the point.
    """
    assert set(RETIRED_LABEL_SPELLINGS) == set(ALL_LABELS)
    # A read-only label is not in that table, because it was never respelled — it is a
    # label the loader addresses no block under and a banked document still carries.
    assert READ_ONLY_LABELS & set(ALL_LABELS) == set()
    assert KNOWN_LABELS == set(ALL_LABELS) | READ_ONLY_LABELS | set(LEGACY_UNION_MEMBERS)
    # No retired spelling collides with a label in use, which would make the
    # translation ambiguous.
    assert set(BLOCK_LABEL_RENAMES) & set(ALL_LABELS) == set()
    # Every target is a label the loader can actually produce.
    assert set(BLOCK_LABEL_RENAMES.values()) <= set(ALL_LABELS)
    # No two retired spellings collapse onto one label.
    assert len(set(BLOCK_LABEL_RENAMES.values())) == len(BLOCK_LABEL_RENAMES)


def test_a_legacy_union_never_reads_as_a_current_label() -> None:
    """A banked union with other members lands on a label no current run carries.

    The legacy labels are disjoint from the live vocabulary and the read-only one, the
    banked spellings are disjoint from both label tables' other side, and each legacy
    label has a current counterpart that differs from it in membership — which is the
    whole reason it is not a rename.
    """
    legacy = set(LEGACY_UNION_MEMBERS)
    assert set(LEGACY_UNION_LABELS.values()) == legacy
    assert legacy & set(ALL_LABELS) == set()
    assert legacy & READ_ONLY_LABELS == set()
    assert set(LEGACY_UNION_LABELS) & set(ALL_LABELS) == set()
    assert set(LEGACY_UNION_LABELS) & set(BLOCK_LABEL_RENAMES) == set()
    assert set(LEGACY_UNION_COUNTERPARTS) == legacy
    for legacy_label, current in LEGACY_UNION_COUNTERPARTS.items():
        assert current in BLOCK_UNIONS
        assert set(LEGACY_UNION_MEMBERS[legacy_label]) != set(BLOCK_UNIONS[current])
        assert union_counterpart(legacy_label) == current
        assert union_counterpart(current) == legacy_label
    assert union_counterpart(ATTENTION_AND_CACHE) is None, "an unchanged union has none"


@pytest.mark.parametrize("banked,legacy", sorted(LEGACY_UNION_LABELS.items()))
def test_a_banked_union_resolves_to_its_legacy_label(banked: str, legacy: str) -> None:
    """One case per banked spelling, as a key and as a value naming a block."""
    migrated = migrate_banked_results(
        {"integrity": {"tier_dims": {banked: 7}}, "topology": {"tier": banked}}
    )
    assert migrated["integrity"]["block_dims"] == {legacy: 7}
    assert migrated["topology"]["block"] == legacy


BANKED_UNION_DOCUMENT: dict[str, Any] = {
    **BASE_METADATA,
    "n_samples": 160,
    "integrity": {
        "n_samples": 160, "n_modes": 8, "n_topics": 20,
        "modes": ["linear"], "topics": ["topic"],
        "samples_per_mode": {"linear": 20}, "samples_per_topic": {"topic": 8},
        "balanced": True,
        "tier_dims": {
            "T1": 249, "T2": 249, "T2.5": 145, "T3": 1250,
            "residual_trajectory": 215, "attention_flow": 343, "gate_features": 303,
            "temporal_dynamics": 630, "contrastive_projection": 800,
            "T2+T2.5": 394, "combined": 1893, "engineered": 1491, "combined_v2": 4184,
            "T2+T2.5+engineered": 1885,
        },
        "total_features": 4184,
        "nan_inf": {},
        "all_clean": True,
        "variance_report": {},
        "value_ranges": {},
    },
    "tier_ablation": {
        "per_tier_accuracy": {
            "T1": 0.50625, "T2": 0.46875, "T2.5": 0.475, "T3": 0.51875,
            "residual_trajectory": 0.33125, "attention_flow": 0.53125,
            "gate_features": 0.3625, "temporal_dynamics": 0.4875,
            "contrastive_projection": 1.0, "T2+T2.5": 0.51875, "combined": 0.5875,
            "engineered": 0.5125, "combined_v2": 1.0, "T2+T2.5+engineered": 0.59375,
        },
        "feature_importance_composite": "combined_v2",
        "pairwise_tier_combinations": {},
        "leave_one_tier_out": {},
        "tier_ranking": [{"tier": "engineered", "accuracy": 0.5125}],
        "tier_inversion_t25_gt_t2_gt_t1": False,
        "top_features_rf_t2t25": [],
        "top_features_lr_t2t25": [],
        "tier_contribution_ratio": {},
        "std_vs_mean": {"n_std_features": 10, "n_mean_features": 10},
        "cohens_d_per_topic": {
            "per_topic": {}, "mean_d": None, "median_d": None, "std_d": None,
            "min_d": None, "max_d": None, "all_positive": None, "n_topics": 0,
        },
    },
}
"""A banked eight-mode document, trimmed to the parts that carry union labels. The
widths and accuracies are a banked 8B run's own, so the membership test below checks
the legacy table against what was written rather than against a recollection of it."""


def test_a_banked_document_carries_the_legacy_unions_and_validates() -> None:
    """The fixture validates, holds each legacy union, and holds no current family union."""
    results = AnalysisResults.model_validate(migrate_banked_results(BANKED_UNION_DOCUMENT))
    assert results.integrity is not None and results.legacy_bin_readout is not None
    dims = results.integrity.block_dims
    accuracy = results.legacy_bin_readout.per_block_accuracy
    for labels in (set(dims), set(accuracy)):
        assert set(LEGACY_UNION_MEMBERS) <= labels
        assert labels <= KNOWN_LABELS
        assert labels & {ALL_FAMILIES, ATTENTION_AND_CACHE_WITH_FAMILIES, EVERYTHING} == set()
    assert accuracy[LEGACY_FAMILY_UNION] == 0.5125
    assert accuracy[LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES] == 0.59375
    assert results.legacy_bin_readout.feature_importance_composite == LEGACY_EVERYTHING
    assert results.legacy_bin_readout.block_ranking[0].block == LEGACY_FAMILY_UNION


def test_the_legacy_membership_is_what_the_banked_widths_add_up_to() -> None:
    """Each legacy union's recorded width is the sum of its members' recorded widths."""
    dims = migrate_banked_results(BANKED_UNION_DOCUMENT)["integrity"]["block_dims"]
    for legacy, members in LEGACY_UNION_MEMBERS.items():
        assert dims[legacy] == sum(dims[member] for member in members), legacy


@pytest.mark.parametrize("retired,current", sorted(BLOCK_LABEL_RENAMES.items()))
def test_a_retired_label_resolves_to_the_label_now_in_use(retired: str, current: str) -> None:
    """One case per retired spelling, as a key and as a value naming a block."""
    migrated = migrate_banked_results(
        {"integrity": {"tier_dims": {retired: 7}}, "topology": {"tier": retired}}
    )
    assert migrated["integrity"]["block_dims"] == {current: 7}
    assert migrated["topology"]["block"] == current


def test_a_combination_key_is_read_component_by_component() -> None:
    """A pair or triple of blocks is keyed by its members, not by a union label.

    ``T2+T2.5`` is the attention-and-cache union at the top of a section and the pair
    (attention, cache) inside a combinations table. The two must not come out with the
    same spelling, or one cell would answer to two names.
    """
    migrated = migrate_banked_results(
        {
            "tier_ablation": {
                "per_tier_accuracy": {"T2+T2.5": 0.66},
                "pairwise_tier_combinations": {"T2+T2.5": {"accuracy": 0.66}},
                "triple_tier_combinations": {"T1+T2+T2.5": {"accuracy": 0.7}},
                "cross_group_ablation": {"T2+T2.5+attention_flow": {"accuracy": 0.8}},
            }
        }
    )
    readout = migrated["legacy_bin_readout"]
    assert set(readout["per_block_accuracy"]) == {ATTENTION_AND_CACHE}
    assert set(readout["pairwise_block_combinations"]) == {
        f"{ATTENTION_AND_DELTAS}+{CACHE_AND_KEYS}"
    }
    assert set(readout["triple_block_combinations"]) == {
        f"{NORMS_AND_OUTPUT_STATS}+{ATTENTION_AND_DELTAS}+{CACHE_AND_KEYS}"
    }
    assert set(readout["cross_group_ablation"]) == {
        f"{ATTENTION_AND_CACHE}+{ATTENTION_FLOW}"
    }
    # Both readings are declared, so neither is an accident of the walk order.
    assert COMPOUND_LABEL_FIELDS["pairwise_block_combinations"] == "members"
    assert COMPOUND_LABEL_FIELDS["cross_group_ablation"] == "union_first"


def test_renames_reach_nested_rows_not_only_the_section_top() -> None:
    """``tier`` names a column inside one row of section 3's ranking table."""
    migrated = migrate_banked_results(
        {"tier_ablation": {"tier_ranking": [{"tier": "T2.5", "accuracy": 0.6}]}}
    )
    assert migrated["legacy_bin_readout"]["block_ranking"] == [
        {"block": CACHE_AND_KEYS, "accuracy": 0.6}
    ]


def test_a_document_already_on_the_current_names_is_unchanged() -> None:
    current: dict[str, Any] = {
        "legacy_bin_readout": {"per_block_accuracy": {CACHE_AND_KEYS: 0.62}},
        "section_times": {"legacy_bin_readout": 12.5},
    }
    assert migrate_banked_results(current) == current


def test_the_contrastive_section_reads_its_retired_key_vocabulary() -> None:
    """Section 8's keys were the ones no field name could spell, and now are field names."""
    migrated = migrate_banked_results(
        {
            "contrastive": {
                "T2+T2.5": {
                    "knn_accuracy_mean": 0.7, "knn_accuracy_std": 0.05,
                    "knn_fold_accs": [0.7], "silhouette_mean": 0.3,
                },
                "tier_ablation": {
                    "individual": {"T1": {
                        "knn_accuracy": 0.4, "knn_std": 0.01,
                        "silhouette": None, "n_features": 3,
                    }},
                    "pairwise": {"T1+T2": {
                        "knn_accuracy": 0.5, "knn_std": 0.01, "silhouette": None,
                        "n_features": 5, "best_individual_knn": 0.4,
                        "gain_over_best_individual": 0.1,
                    }},
                    "T2+T2.5": {
                        "knn_accuracy": 0.6, "knn_std": 0.01,
                        "silhouette": None, "n_features": 7,
                    },
                    "combined": {
                        "knn_accuracy": 0.55, "knn_std": 0.01,
                        "silhouette": None, "n_features": 9,
                    },
                    "super_additivity": {
                        "T2_alone": 0.5, "T2.5_alone": 0.45, "T2+T2.5_pair": 0.6,
                        "best_individual": 0.5, "gain": 0.1, "combined_knn": 0.55,
                        "T2+T2.5_beats_combined": True,
                    },
                },
            }
        }
    )
    section = migrated["contrastive"]
    assert set(section) == {ATTENTION_AND_CACHE, "block_ablation"}
    ablation = section["block_ablation"]
    assert set(ablation["individual"]) == {NORMS_AND_OUTPUT_STATS}
    assert set(ablation["pairwise"]) == {f"{NORMS_AND_OUTPUT_STATS}+{ATTENTION_AND_DELTAS}"}
    assert ablation["super_additivity"]["attention_and_cache_pair"] == 0.6
    assert ablation["super_additivity"]["attention_and_cache_beats_combined"] is True

    parsed = AnalysisResults.model_validate({**BASE_METADATA, **migrated})
    assert parsed.contrastive is not None
    assert parsed.contrastive.block_ablation is not None
    assert parsed.contrastive.block_ablation.attention_and_cache.n_features == 7


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
    assert readout.per_block_accuracy[CACHE_AND_KEYS] == 0.6
    assert readout.cache_beats_attention_beats_norms is True
    assert readout.block_ranking[0].block == CACHE_AND_KEYS
    assert readout.leave_one_block_out[CACHE_AND_KEYS].cost_of_removal == 0.08
    assert readout.top_features_rf_attention_and_cache[0].importance == 0.01
    assert results.section_times == {"legacy_bin_readout": 12.5}


def test_a_written_result_carries_only_labels_in_use() -> None:
    """Read accepts the retired spellings; write emits none of them.

    A document is round-tripped the way a pass writes one — through the typed models
    and ``clean_for_json`` — and then scanned for every retired spelling. One leaking
    out here would put the vocabulary back into the next banked file.
    """
    migrated = migrate_banked_results(
        {
            **BASE_METADATA,
            "topology": {"tier": "T2+T2.5"},
            "clustering": {
                "silhouette_by_tier": {"T2.5": {
                    "mode_silhouette_cosine": 0.1, "mode_silhouette_euclidean": 0.1,
                    "mode_silhouette": 0.1, "topic_silhouette": 0.0,
                }},
                "per_mode_silhouette": {},
                "per_mode_silhouette_cosine": {},
                "per_mode_silhouette_euclidean": {},
                "kmeans_ari": {"T2+T2.5": 0.4, "combined_v2": 0.5, EVERYTHING: 0.6},
                "embeddings": {"tsne_t2t25": {"error": "umap not installed"}},
            },
        }
    )
    written = clean_for_json(AnalysisResults.model_validate(migrated))
    text = json.dumps(written)
    for retired in (*BLOCK_LABEL_RENAMES, *LEGACY_UNION_LABELS):
        assert f'"{retired}"' not in text, retired
    for section_renames in FIELD_RENAMES.values():
        for retired_field in section_renames:
            assert f'"{retired_field}"' not in text, retired_field
    assert f'"{EVERYTHING}"' in text and f'"{CACHE_AND_KEYS}"' in text


@pytest.mark.parametrize("run", BANKED_RUNS)
def test_a_real_banked_run_validates_through_the_table(run: str) -> None:
    """The fixtures above are a claim about real files; this is the claim checked."""
    path = banked_analysis_dir() / run / "results.json"
    if not path.exists():
        pytest.skip(f"banked analysis absent: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    results = AnalysisResults.model_validate(migrate_banked_results(document))
    assert results.run_name
    # Section 3 is the one the renames touched hardest, so its numbers are the ones
    # worth reading back: the accuracies must survive the key change unchanged, and
    # every key must arrive on a label in use.
    readout = results.legacy_bin_readout
    if readout is not None:
        original = document.get("tier_ablation", {}).get("per_tier_accuracy", {})
        read_as = {**BLOCK_LABEL_RENAMES, **LEGACY_UNION_LABELS}
        assert readout.per_block_accuracy == {
            read_as.get(label, label): value for label, value in original.items()
        }
        assert set(readout.per_block_accuracy) <= KNOWN_LABELS
    if results.classification is not None:
        assert set(results.classification.by_block) <= KNOWN_LABELS
        assert ATTENTION_AND_CACHE in results.classification.by_block
        # A banked run holds its unions under the legacy labels only: no banked number
        # arrives under a current union whose membership differs.
        current_unions = {ALL_FAMILIES, ATTENTION_AND_CACHE_WITH_FAMILIES, EVERYTHING}
        assert set(results.classification.by_block) & current_unions == set()
