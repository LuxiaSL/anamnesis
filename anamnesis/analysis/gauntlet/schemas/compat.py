"""Reading a banked ``results.json`` whose field and block names predate the current ones.

A gauntlet result is a long-lived artifact: the numbers in
``outputs/analysis/<run>/results.json`` are cited, re-read across runs by
:mod:`anamnesis.analysis.complementarity`, and resumed from by the orchestrator's
checkpoint. Renaming a field or a block label therefore has two jobs, and this module is
the second one — every file already written must still load, and load into the same
numbers under the new names.

Two things get renamed here, and they are renamed in this order:

``SECTION_RENAMES`` / ``FIELD_RENAMES``
    Section keys, and the field names inside one section.
``BLOCK_LABEL_RENAMES``
    The label a block or a union of blocks is reported under, wherever it appears in
    the document — as a dict key, as a component of a ``a+b`` combination key, or as a
    string value naming which block a number came from. Labels are not confined to one
    section, so this pass walks the whole document, and it runs second so that it sees
    field names already on their current spelling.

Why migration tables rather than validation aliases on the fields
-----------------------------------------------------------------
``extra="forbid"`` is the section schemas' stated policy: a key no model declares is a
validation error, because a file and the code disagreeing about what a number is
matters more than the file loading. An alias per field would widen every model's
accepted input permanently and silently, which spends that policy to buy back
compatibility. A table spends nothing: the mapping is enumerable in one read, it is
applied to the payload, and what comes out then goes through the unmodified strict
schema. The strictness that catches drift still runs, over renamed keys.

The tables are also the only place a reader has to look to follow a citation from an
older file to the field that now holds it, which is what ``PORT-MAP.md`` points at.

Scope
-----
``migrate_banked_results`` reads; nothing here writes. A result is written under the
current names only, so the retired spellings live in this module and nowhere else.

``FIELD_RENAMES`` is keyed by section, and each section's map is applied to every
nested dict inside that section — a key renamed at the top of section 3 and the same
key renamed inside one of its table rows need one entry, not two.

A payload that already uses the current names passes through untouched, so the function
is safe to call on every read rather than only on old files.
"""

from __future__ import annotations

from typing import Any

from anamnesis.analysis.gauntlet.signature_io import (
    ALL_CORE,
    ALL_FAMILIES,
    ATTENTION_AND_CACHE,
    ATTENTION_AND_CACHE_WITH_FAMILIES,
    ATTENTION_AND_DELTAS,
    ATTENTION_FLOW,
    CACHE_AND_KEYS,
    CONTRASTIVE_PROJECTION,
    EVERYTHING,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
    TEMPORAL_DYNAMICS,
)

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
    "contrastive": {
        "tier_ablation": "block_ablation",
        "T2_alone": "attention_alone",
        "T2.5_alone": "cache_alone",
        "T2+T2.5_pair": "attention_and_cache_pair",
        "T2+T2.5_beats_combined": "attention_and_cache_beats_combined",
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


# WIRE VOCABULARY, read-side only: the right-hand strings are label spellings that
# banked files carry, matched against what comes off disk and never written. They are
# not a taxonomy, and editing one to look better makes an existing file unreadable.
#
# Current label → the retired spelling, or None where the current label is the only
# one a file has ever carried. Keyed by the current label and total over the label
# set, so a block or union added to `anamnesis/analysis/gauntlet/signature_io.py` has
# to say which it is, and a rename cannot land without an entry here.
RETIRED_LABEL_SPELLINGS: dict[str, str | None] = {
    NORMS_AND_OUTPUT_STATS: "T1",
    ATTENTION_AND_DELTAS: "T2",
    CACHE_AND_KEYS: "T2.5",
    RESIDUAL_PCA: "T3",
    ATTENTION_AND_CACHE: "T2+T2.5",
    ATTENTION_AND_CACHE_WITH_FAMILIES: "T2+T2.5+engineered",
    EVERYTHING: "combined_v2",
    ALL_CORE: None,
    ALL_FAMILIES: None,
    RESIDUAL_TRAJECTORY: None,
    ATTENTION_FLOW: None,
    GATE_FEATURES: None,
    TEMPORAL_DYNAMICS: None,
    CONTRASTIVE_PROJECTION: None,
}

BLOCK_LABEL_RENAMES: dict[str, str] = {
    retired: current
    for current, retired in RETIRED_LABEL_SPELLINGS.items()
    if retired is not None
}

# Fields whose keys are several labels joined with ``+``, and which reading applies:
# ``members`` translates each component on its own, ``union_first`` takes the longest
# leading run that is a label. The distinction is load-bearing — ``T2+T2.5`` is the
# pair (attention, cache) in a combinations table and the attention-and-cache union
# everywhere else, so one reading for both would give one cell two spellings.
COMPOUND_LABEL_FIELDS: dict[str, str] = {
    "pairwise_block_combinations": "members",
    "triple_block_combinations": "members",
    "pairwise": "members",
    "cross_group_ablation": "union_first",
}

_WHOLE = "whole"


def _translate_whole(label: str) -> str:
    return BLOCK_LABEL_RENAMES.get(label, label)


def _translate_members(key: str) -> str:
    return "+".join(_translate_whole(part) for part in key.split("+"))


def _translate_union_first(key: str) -> str:
    """``key`` with its longest leading label runs translated, left to right.

    ``T2+T2.5+attention_flow`` is the attention-and-cache union plus one family, so the
    two-component run is taken before its components are.
    """
    parts = key.split("+")
    out: list[str] = []
    start = 0
    while start < len(parts):
        for end in range(len(parts), start, -1):
            candidate = "+".join(parts[start:end])
            if candidate in BLOCK_LABEL_RENAMES:
                out.append(BLOCK_LABEL_RENAMES[candidate])
                start = end
                break
        else:
            out.append(parts[start])
            start += 1
    return "+".join(out)


_KEY_TRANSLATORS = {
    _WHOLE: _translate_whole,
    "members": _translate_members,
    "union_first": _translate_union_first,
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


def _relabel(value: Any, key_style: str = _WHOLE) -> Any:
    """``value`` with every retired block label replaced by the current one.

    ``key_style`` says how this dict's own keys are read — a plain label, or one of the
    combination spellings in ``COMPOUND_LABEL_FIELDS``. It applies to one level only:
    the entries under a combination key are ordinary fields again. String values equal
    to a retired label are translated too, which is how ``block_ranking`` rows and the
    ``cross_group_baseline`` pointer say which block they mean.
    """
    if isinstance(value, dict):
        translate = _KEY_TRANSLATORS[key_style]
        return {
            translate(str(key)): _relabel(item, COMPOUND_LABEL_FIELDS.get(str(key), _WHOLE))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relabel(item, _WHOLE) for item in value]
    if isinstance(value, str):
        return _translate_whole(value)
    return value


def migrate_banked_results(payload: Any) -> Any:
    """A results document with older spellings mapped onto the current ones.

    Takes the parsed JSON of a ``results.json`` and returns a new document; the input
    is not modified. Anything that is not a dict comes back unchanged, so a caller
    can hand over whatever it parsed without checking first.

    Section keys are renamed through ``SECTION_RENAMES``, each section's body is then
    walked with that section's entry in ``FIELD_RENAMES``, and the whole document is
    walked once more for block labels. A section with no ``FIELD_RENAMES`` entry — or
    anything a later version adds — is carried through the first two passes untouched
    and still gets its labels read forward.
    """
    if not isinstance(payload, dict):
        return payload
    out: dict[str, Any] = {}
    for key, value in payload.items():
        section = SECTION_RENAMES.get(str(key), str(key))
        renames = FIELD_RENAMES.get(section)
        out[section] = _rename_keys(value, renames) if renames else value
    return _relabel(out)
