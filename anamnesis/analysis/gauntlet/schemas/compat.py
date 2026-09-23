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
``BLOCK_LABEL_RENAMES`` / ``LEGACY_UNION_LABELS``
    The label a block or a union of blocks is reported under, wherever it appears in
    the document — as a dict key, as a component of a ``a+b`` combination key, or as a
    string value naming which block a number came from. Labels are not confined to one
    section, so this pass walks the whole document, and it runs second so that it sees
    field names already on their current spelling.

The two label tables answer different questions. A rename says the banked label and
the current one address the same columns under two spellings, so the number reads
forward onto the current label. A legacy union label says the banked union held a
different set of blocks than any union the loader builds now, so its number reads
onto a label no current run carries, and a comparison that meets it beside a current
union sees two names rather than one.

Why migration tables rather than validation aliases on the fields
-----------------------------------------------------------------
``extra="forbid"`` is the section schemas' stated policy: a key no model declares is a
validation error, because a file and the code disagreeing about what a number is
matters more than the file loading. An alias per field would widen every model's
accepted input permanently and silently, which spends that policy to buy back
compatibility. A table spends nothing: the mapping is enumerable in one read, it is
applied to the payload, and what comes out then goes through the unmodified strict
schema. The strictness that catches drift still runs, over renamed keys.

The tables are also the only place a reader has to look to follow a name in an older
banked file to the field that now holds it.

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
    ALL_LABELS,
    CACHE_AND_KEYS,
    EVERYTHING,
    GATE_FEATURES,
    NORMS_AND_OUTPUT_STATS,
    RESIDUAL_PCA,
    RESIDUAL_TRAJECTORY,
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
# Current label → the retired spelling, or None where no banked file carries these
# same columns under another spelling. Keyed by the current label and total over the
# label set, so a block or union added to
# `anamnesis/analysis/gauntlet/signature_io.py` has to say which it is, and a rename
# cannot land without an entry here. A banked union whose membership differs from the
# current one is not a rename, and is read through ``LEGACY_UNION_LABELS`` below.
RETIRED_LABEL_SPELLINGS: dict[str, str | None] = {
    NORMS_AND_OUTPUT_STATS: "T1",
    ATTENTION_AND_DELTAS: "T2",
    CACHE_AND_KEYS: "T2.5",
    RESIDUAL_PCA: "T3",
    ATTENTION_AND_CACHE: "T2+T2.5",
    ATTENTION_AND_CACHE_WITH_FAMILIES: None,
    EVERYTHING: None,
    ALL_CORE: None,
    ALL_FAMILIES: None,
    RESIDUAL_TRAJECTORY: None,
    ATTENTION_FLOW: None,
    GATE_FEATURES: None,
}

BLOCK_LABEL_RENAMES: dict[str, str] = {
    retired: current
    for current, retired in RETIRED_LABEL_SPELLINGS.items()
    if retired is not None
}

# Block labels a banked file carries that no corpus is written with, so the loader
# addresses no block under them and they are absent from the table above. They are
# here because a reader that checks a banked document's labels against the live
# vocabulary alone would reject six results files over blocks they legitimately hold.
READ_ONLY_LABELS: frozenset[str] = frozenset({"temporal_dynamics", "contrastive_projection"})

# The unions a banked file reports whose membership no current union matches. Each is
# read under a label of its own with a ``legacy_`` prefix, so the banked number keeps
# its identity and cannot be looked up under a current label by accident. The labels
# are literals rather than derived from the current ones: respelling a current union
# must not move the label a banked number is read under.
LEGACY_FAMILY_UNION = "legacy_engineered"
LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES = "attention_and_cache+legacy_engineered"
LEGACY_EVERYTHING = "legacy_every_block"

LEGACY_UNION_MEMBERS: dict[str, tuple[str, ...]] = {
    LEGACY_FAMILY_UNION: (
        RESIDUAL_TRAJECTORY, ATTENTION_FLOW, GATE_FEATURES, "temporal_dynamics",
    ),
    LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES: (
        ATTENTION_AND_DELTAS, CACHE_AND_KEYS,
        RESIDUAL_TRAJECTORY, ATTENTION_FLOW, GATE_FEATURES, "temporal_dynamics",
    ),
    LEGACY_EVERYTHING: (
        NORMS_AND_OUTPUT_STATS, ATTENTION_AND_DELTAS, CACHE_AND_KEYS, RESIDUAL_PCA,
        RESIDUAL_TRAJECTORY, ATTENTION_FLOW, GATE_FEATURES,
        "temporal_dynamics", "contrastive_projection",
    ),
}
"""Legacy union label → the blocks the banked union concatenated. The widths a banked
``integrity`` section records for each union are the sum of these members' widths,
which is how the membership is known rather than recalled."""

LEGACY_UNION_COUNTERPARTS: dict[str, str] = {
    LEGACY_FAMILY_UNION: ALL_FAMILIES,
    LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES: ATTENTION_AND_CACHE_WITH_FAMILIES,
    LEGACY_EVERYTHING: EVERYTHING,
}
"""Legacy union label → the current union it resembles and does not equal. What a
cross-run comparison consults to say that two runs report the same role under two
different memberships, rather than silently finding the label absent on one side."""

# WIRE VOCABULARY, read-side only, like the retired spellings above. Banked spelling
# → the legacy label its number reads as. More than one spelling may land on one
# legacy label: the attention-and-cache-plus-families union is spelled with the bin
# labels in some banked files and with the block labels in others, and it is one
# membership either way.
LEGACY_UNION_LABELS: dict[str, str] = {
    "engineered": LEGACY_FAMILY_UNION,
    "T2+T2.5+engineered": LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES,
    "attention_and_cache+engineered": LEGACY_ATTENTION_AND_CACHE_WITH_FAMILIES,
    "combined_v2": LEGACY_EVERYTHING,
}

KNOWN_LABELS: frozenset[str] = ALL_LABELS | READ_ONLY_LABELS | frozenset(LEGACY_UNION_MEMBERS)
"""Every block label a results document may carry once read: the live vocabulary, the
read-only spellings beside it, and the legacy union labels. What a reader of a banked
file checks against."""


def union_counterpart(label: str) -> str | None:
    """The other-membership label for a union whose membership changed, else None.

    A legacy union label answers with the current union it resembles, and a current
    union label with its legacy one. Two runs that report a union under a label and
    its counterpart hold the same role over different blocks, so their numbers are not
    one measurement.
    """
    if label in LEGACY_UNION_COUNTERPARTS:
        return LEGACY_UNION_COUNTERPARTS[label]
    for legacy, current in LEGACY_UNION_COUNTERPARTS.items():
        if current == label:
            return legacy
    return None


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

_READ_FORWARD: dict[str, str] = {**BLOCK_LABEL_RENAMES, **LEGACY_UNION_LABELS}
"""Every banked label spelling the read side translates, renames and legacy unions
together, so each translator consults one table."""


def _translate_whole(label: str) -> str:
    return _READ_FORWARD.get(label, label)


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
            if candidate in _READ_FORWARD:
                out.append(_READ_FORWARD[candidate])
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
    """``value`` with every banked block label replaced by the label it now reads as.

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
