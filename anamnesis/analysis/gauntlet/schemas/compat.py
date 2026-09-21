"""Reading a banked ``results.json`` whose field names predate the current schemas.

A gauntlet result is a long-lived artifact: the numbers in
``outputs/analysis/<run>/results.json`` are cited, re-read across runs by
:mod:`anamnesis.analysis.complementarity`, and resumed from by the orchestrator's
checkpoint. Renaming a field in a schema therefore has two jobs, and this module is
the second one — every file already written must still load, and load into the same
numbers under the new names.

Why a migration table rather than validation aliases on the fields
-----------------------------------------------------------------
``extra="forbid"`` is the section schemas' stated policy: a key no model declares is a
validation error, because a file and the code disagreeing about what a number is
matters more than the file loading. An alias per field would widen every model's
accepted input permanently and silently, which spends that policy to buy back
compatibility. A table spends nothing: the mapping is enumerable in one read, it is
applied to the payload, and what comes out then goes through the unmodified strict
schema. The strictness that catches drift still runs, over renamed keys.

The table is also the only place a reader has to look to follow a citation from an
older file to the field that now holds it, which is what ``PORT-MAP.md`` points at.

Scope
-----
``migrate_banked_results`` is keyed by section, and each section's map is applied to
every nested dict inside that section — a key renamed at the top of section 3 and the
same key renamed inside one of its table rows need one entry, not two. The
``contrastive`` and ``classification`` sections are deliberately absent: their models
translate their own key spelling in their validators, because those labels were never
legal Python identifiers — a block label at the top of a classification result, a
``T2+T2.5`` union key — and so were already being translated before any of this.

A payload that already uses the current names passes through untouched, so the
function is safe to call on every read rather than only on old files.
"""

from __future__ import annotations

from typing import Any

# Renames at the top level of a results document: the section keys themselves.
SECTION_RENAMES: dict[str, str] = {
    "tier_ablation": "legacy_bin_readout",
}

# Renames applied to every dict key at any depth inside one section.
FIELD_RENAMES: dict[str, dict[str, str]] = {
    "integrity": {
        "tier_dims": "block_dims",
    },
    "legacy_bin_readout": {
        "per_tier_accuracy": "per_block_accuracy",
        "pairwise_tier_combinations": "pairwise_block_combinations",
        "triple_tier_combinations": "triple_block_combinations",
        "leave_one_tier_out": "leave_one_block_out",
        "tier_ranking": "block_ranking",
        "tier_inversion_t25_gt_t2_gt_t1": "cache_beats_attention_beats_norms",
        "tier_contribution_ratio": "block_contribution_ratio",
        "top_features_rf_t2t25": "top_features_rf_attention_and_cache",
        "top_features_lr_t2t25": "top_features_lr_attention_and_cache",
        # One row of the ranking table.
        "tier": "block",
    },
    "intrinsic_dimension": {
        "tier_convergence": "block_convergence",
    },
    "topology": {
        "tier": "block",
    },
    "clustering": {
        "silhouette_by_tier": "silhouette_by_block",
        "tsne_t2t25": "tsne_attention_and_cache",
        "umap_t2t25": "umap_attention_and_cache",
    },
    "semantic": {
        "per_tier_semantic": "per_block_semantic",
        "per_tier": "per_block",
        "compute_t2t25": "compute_attention_and_cache",
    },
    # The per-section timing map is keyed by section name, so it moves with them.
    "section_times": dict(SECTION_RENAMES),
    "scorecard": {
        "tier_inversion_holds": "cache_beats_attention_beats_norms",
        "per_tier_accuracy": "per_block_accuracy",
        "t3_id": "residual_pca_id",
        "mean_t1_t2": "mean_norms_and_attention_id",
        "t2t25_accuracy": "attention_and_cache_accuracy",
    },
}


def _rename_keys(value: Any, renames: dict[str, str]) -> Any:
    """``value`` with every dict key at any depth replaced through ``renames``.

    Lists and scalars are walked but not otherwise touched. A key absent from
    ``renames`` keeps its spelling, so a payload already on the current names comes
    back equal to what went in.
    """
    if isinstance(value, dict):
        return {
            renames.get(str(key), str(key)): _rename_keys(item, renames)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_keys(item, renames) for item in value]
    return value


def migrate_banked_results(payload: Any) -> Any:
    """A results document with older field spellings mapped onto the current ones.

    Takes the parsed JSON of a ``results.json`` and returns a new document; the input
    is not modified. Anything that is not a dict comes back unchanged, so a caller
    can hand over whatever it parsed without checking first.

    Section keys are renamed through ``SECTION_RENAMES``, and each section's body is
    then walked with that section's entry in ``FIELD_RENAMES``. A section with no
    entry — ``contrastive``, or anything a later version adds — is copied through
    untouched.
    """
    if not isinstance(payload, dict):
        return payload
    out: dict[str, Any] = {}
    for key, value in payload.items():
        section = SECTION_RENAMES.get(str(key), str(key))
        renames = FIELD_RENAMES.get(section)
        out[section] = _rename_keys(value, renames) if renames else value
    return out
