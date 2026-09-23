"""Sub-family decomposition: which part of a family carries the signal.

`feature_map` classifies a feature by what it reads and how — its source, its
method, its depth. That is the taxonomy, and it is deliberately coarse: it answers
"is this attention or residual" across the whole vector. This module asks the
other question, inside one family: within ``attention_flow``, is the signal
system-prompt mass or recency bias? Within ``gate_features``, is it sparsity or
drift? The two are not competing classifications — one cuts across families by
substrate, this one cuts a single family into the signals its own feature names
spell out.

The method is one classifier per family, reading the family's naming convention,
plus a random forest per resulting sub-family under stratified cross-validation.
A sub-family's accuracy beside the full family's is the readout: a part that
matches the whole says the rest is redundant, and a part far below it says the
signal is distributed rather than located.

Two callers need the naming rules and they need them differently, so both are here
rather than in two modules. A decomposition holds a block and cuts its names with that
block's classifier; a reader of a ranked importance list holds a bare name from any
family and calls :func:`classify_signal`, which dispatches on the spelling. One set of
rules answers both, because a sub-family label keyed two ways is two vocabularies in
tables printed side by side.

Feature names come from the loader rather than from a second read of the bank. A
block's names are the slice of the vector's name list its own metadata assigns to
it, and a decomposition whose names and columns disagree is silently mapping
indices onto the wrong features — so a length mismatch is refused here rather than
producing a plausible table.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from anamnesis.analysis.gauntlet.signature_io import (
    ATTENTION_FLOW,
    GATE_FEATURES,
    Run4Data,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

N_ESTIMATORS = 200
N_SPLITS = 5
SEED = 42

UNKNOWN = "unknown"


class SignalRule(BaseModel):
    """One naming rule: the sub-family a feature belongs to when a mark is in its name.

    A mark is a substring rather than a prefix, because a signal's name sits between a
    layer marker and an operator suffix and neither is in a fixed place: one bank writes
    ``attn_flow_L16_recency_bias_mean`` and another ``af_recency_bias_L16``. The marks
    are the signal's own words, so a match on one is a match on the signal wherever the
    layer went.

    A rule may carry two marks for one signal, which is how a family renamed between
    banks stays one sub-family. Rules are tried in order and the first match wins, so a
    mark contained in another rule's mark comes first.
    """

    model_config = ConfigDict(frozen=True)
    subfamily: str
    marks: tuple[str, ...]

    def matches(self, name: str) -> bool:
        return any(mark in name for mark in self.marks)


def _first_match(name: str, rules: tuple[SignalRule, ...], unplaced: str) -> str:
    for rule in rules:
        if rule.matches(name):
            return rule.subfamily
    return unplaced


_AF_RULES: tuple[SignalRule, ...] = (
    SignalRule(subfamily="af_sysprompt_decay", marks=("sysprompt_decay", "prompt_decay")),
    SignalRule(subfamily="af_head_diversity_recency", marks=("head_diversity_recency",)),
    SignalRule(
        subfamily="af_head_diversity_sysprompt",
        marks=("head_diversity_sysprompt", "head_diversity_prompt"),
    ),
    SignalRule(subfamily="af_region_sysprompt", marks=("region_sysprompt", "region_prompt")),
    SignalRule(subfamily="af_region_early_gen", marks=("region_early",)),
    SignalRule(subfamily="af_region_mid_gen", marks=("region_mid",)),
    SignalRule(subfamily="af_region_recent", marks=("region_recent",)),
    SignalRule(subfamily="af_recency_bias", marks=("recency_bias",)),
    SignalRule(subfamily="af_sysprompt_mass", marks=("sysprompt_mass", "prompt_mass")),
)
"""The nine allocation statistics, under both spellings the banks carry: the region a
name calls ``sysprompt`` a later one calls ``prompt``, and it is one signal either way."""


def classify_attention_flow(name: str) -> str:
    """An ``attention_flow`` feature by which allocation statistic it is."""
    return _first_match(name, _AF_RULES, "af_unknown")


_GF_RULES: tuple[SignalRule, ...] = (
    # First, because a cross-layer feature has no single layer to belong to and one of
    # its marks contains a signal name that would otherwise file it under a layer's.
    SignalRule(
        subfamily="gf_cross_layer",
        marks=("cross_layer", "layer_agreement", "layer_sparsity_diversity"),
    ),
    SignalRule(subfamily="gf_sparsity", marks=("sparsity",)),
    SignalRule(subfamily="gf_drift", marks=("drift",)),
    SignalRule(subfamily="gf_eff_dim", marks=("eff_dim",)),
    SignalRule(subfamily="gf_topk_overlap", marks=("topk",)),
)


def classify_gate_features(name: str) -> str:
    """A ``gate_features`` feature by its signal, with cross-layer features apart."""
    return _first_match(name, _GF_RULES, "gf_unknown")


_RT_RULES: tuple[SignalRule, ...] = (
    SignalRule(subfamily="rt_acceleration", marks=("acceleration",)),
    SignalRule(subfamily="rt_direction_change", marks=("direction_change",)),
    SignalRule(subfamily="rt_directness", marks=("directness",)),
    SignalRule(subfamily="rt_velocity", marks=("velocity",)),
)


def classify_residual_trajectory(name: str) -> str:
    """A residual-trajectory feature by which property of the path it reads."""
    return _first_match(name, _RT_RULES, "rt_unknown")


_TD_RULES: tuple[SignalRule, ...] = (
    SignalRule(subfamily="td_attn_entropy", marks=("attn_entropy",)),
    SignalRule(subfamily="td_head_agreement", marks=("head_agreement",)),
    SignalRule(subfamily="td_key_drift", marks=("key_drift",)),
    SignalRule(subfamily="td_key_novelty", marks=("key_novelty",)),
    SignalRule(subfamily="td_lookback_ratio", marks=("lookback_ratio",)),
)


def classify_temporal_dynamics(name: str) -> str:
    """A windowed feature by the signal the window is taken over.

    The windows and the frequency-band reads of one signal are one sub-family: they are
    operators over the same series, and an importance summed over one window of one
    signal is a number over three features.
    """
    return _first_match(name, _TD_RULES, "td_unknown")


_PH_RULES: tuple[SignalRule, ...] = (
    SignalRule(subfamily="ph_head_entropy", marks=("head_entropy",)),
    SignalRule(subfamily="ph_head_role", marks=("head_role",)),
    SignalRule(subfamily="ph_key_spread", marks=("key_spread",)),
    SignalRule(subfamily="ph_sink_head", marks=("sink_head",)),
)


def classify_per_head(name: str) -> str:
    """A per-head feature by which head statistic it spreads over the heads."""
    return _first_match(name, _PH_RULES, "ph_unknown")


_BASE_RULES: tuple[SignalRule, ...] = (
    SignalRule(subfamily="activation_norm", marks=("activation_norm",)),
    SignalRule(subfamily="attn_entropy", marks=("attn_entropy",)),
    SignalRule(subfamily="head_agreement", marks=("head_agreement",)),
    SignalRule(subfamily="delta_cosine", marks=("delta_cosine",)),
    SignalRule(subfamily="delta_norm", marks=("delta_norm",)),
    SignalRule(subfamily="cache_anchor_strength", marks=("anchor_strength",)),
    SignalRule(subfamily="cache_attn_decay_rate", marks=("attn_decay",)),
    SignalRule(subfamily="cache_coverage", marks=("cache_coverage",)),
    SignalRule(subfamily="cache_lookback_ratio", marks=("lookback_ratio",)),
    SignalRule(subfamily="cache_recency", marks=("cache_recency",)),
    SignalRule(subfamily="cache_sink_mass", marks=("sink_mass",)),
    SignalRule(subfamily="kv_key_drift", marks=("key_drift",)),
    SignalRule(subfamily="kv_key_eff_dim", marks=("key_eff_dim",)),
    SignalRule(subfamily="kv_key_novelty", marks=("key_novelty",)),
    SignalRule(subfamily="kv_key_spread", marks=("key_spread",)),
    SignalRule(subfamily="epoch_max_transition", marks=("max_transition",)),
    SignalRule(subfamily="epoch_n_transitions", marks=("n_transitions",)),
    SignalRule(subfamily="epoch_regularity", marks=("epoch_regularity",)),
    SignalRule(subfamily="spectral_fiedler", marks=("fiedler",)),
    SignalRule(subfamily="spectral_hfer", marks=("hfer",)),
    SignalRule(subfamily="spectral_smoothness", marks=("smoothness",)),
    SignalRule(subfamily="spectral_entropy", marks=("spectral_entropy",)),
    SignalRule(subfamily="cross_layer_keys", marks=("cross_layer",)),
    SignalRule(subfamily="logit_entropy", marks=("logit_entropy",)),
    SignalRule(subfamily="top1_prob", marks=("top1_prob",)),
    SignalRule(subfamily="top5_mass", marks=("top5_mass",)),
    SignalRule(subfamily="chosen_rank", marks=("chosen_rank",)),
    # Before the bare surprise rule, which its name contains.
    SignalRule(subfamily="surprise_boundary", marks=("surprise_boundary",)),
    SignalRule(subfamily="surprise", marks=("surprise",)),
)
"""The signals of the four core blocks, whose names carry no family prefix. A mean and a
standard deviation of one signal are one sub-family, and so are its five trajectory
samples: the grain is the signal, and the operator over it is what a sub-family reading
sums across."""

_TEMPORAL_SAMPLE = re.compile(r"_t(\d+)(_|$)")


def _by_temporal_sample(prefix: str, name: str) -> str:
    """A projection family's sub-family: the generation-time sample it was taken at.

    A coordinate index in a fitted or learned basis means nothing on its own, and there
    are fifty of them per sample, so the sample is the only grouping in these names that
    a summed importance is a number over.
    """
    match = _TEMPORAL_SAMPLE.search(name)
    return f"{prefix}_t{match.group(1)}" if match else f"{prefix}_unknown"


_BY_PREFIX: tuple[tuple[tuple[str, ...], Callable[[str], str]], ...] = (
    (("attn_flow_", "af_"), classify_attention_flow),
    (("gate_", "gf_"), classify_gate_features),
    (("res_traj", "rt_"), classify_residual_trajectory),
    (("td_",), classify_temporal_dynamics),
    (("ph_",), classify_per_head),
    (("cp_",), lambda name: _by_temporal_sample("cp", name)),
    (("pca_",), lambda name: _by_temporal_sample("pca", name)),
)
"""Which classifier reads which spelling, tried before the prefix-free core names. A
family with two spellings maps both to one classifier, so the two banks produce one set
of sub-family labels rather than two."""


def classify_signal(name: str) -> str:
    """The signal one banked feature name reads, with the layer dropped.

    This is the classifier of record for the question "which part of a family", and the
    grain a ranked importance list supports: a layer is dropped, because an importance
    summed over one layer of one signal is a number over two or three features. The
    other question — which substrate a feature reads — is
    :mod:`anamnesis.feature_map`'s, at the grain of the whole family, and the two are
    not substitutes.

    A name no rule places is returned as ``other(<name>)``: the bucket is the honest
    answer, and naming the feature in it is what lets a reader find the column. Many of
    them together say the naming convention has moved and these rules have not.
    """
    for prefixes, classifier in _BY_PREFIX:
        if name.startswith(prefixes):
            return classifier(name)
    return _first_match(name, _BASE_RULES, f"other({name})")


SUBFAMILY_CLASSIFIERS: dict[str, Callable[[str], str]] = {
    ATTENTION_FLOW: classify_attention_flow,
    GATE_FEATURES: classify_gate_features,
}
"""Which classifier a whole-family decomposition reads a block's names with. A block
here is cut into sub-families and each part scored; a family absent from this table has
no such cut, which is a fact about its naming rather than a gap here.
:func:`classify_signal` answers for a bare name from any family, which is the other
caller — it has a name and no block, so it dispatches on the spelling instead."""

FULL_FAMILY = "_full_family"
"""The row every decomposition carries: the whole family, as the comparison every
sub-family reading is against."""


class SubsetAccuracy(BaseModel):
    """One feature subset's cross-validated accuracy, and how wide it was."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float
    n_features: int = Field(ge=0)
    std: float | None = None
    fold_accs: list[float] = Field(default_factory=list)
    error: str | None = None


def feature_names_for(data: Run4Data, block: str) -> list[str]:
    """A block's feature names, checked against the width of its matrix.

    The loader takes the names from the slice table the bank's own metadata carries.
    Where that table is absent the names cannot be assigned to a block at all, and
    where they disagree with the matrix's width the mapping from name to column is
    wrong — both refuse here, because the decomposition's whole output is that
    mapping.
    """
    names = data.block_feature_names.get(block)
    if names is None or len(names) == 0:
        raise KeyError(
            f"no feature names for block {block!r}: the bank's metadata carries no slice "
            f"table for it, so its columns cannot be named"
        )
    width = int(data.get_block(block).shape[1])
    if len(names) != width:
        raise ValueError(
            f"block {block!r} has {width} columns but {len(names)} names — a sub-family "
            f"cut would map names onto the wrong columns"
        )
    return [str(n) for n in names]


def accuracy_on_subset(
    X: F32,
    y: NDArray[Any],
    mask: NDArray[np.bool_],
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> SubsetAccuracy:
    """Cross-validated random-forest accuracy over the columns the mask selects.

    The scaler is refitted per fold on the training half, so a fold's test rows do
    not inform their own standardization.
    """
    n_features = int(mask.sum())
    if n_features == 0:
        return SubsetAccuracy(accuracy=0.0, n_features=0, error="no features")

    X_subset = X[:, mask]
    scaler = StandardScaler()
    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=1)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accuracies: list[float] = []
    for train_index, test_index in folds.split(X_subset, y):
        X_train = scaler.fit_transform(X_subset[train_index])
        X_test = scaler.transform(X_subset[test_index])
        classifier.fit(X_train, y[train_index])
        accuracies.append(float(classifier.score(X_test, y[test_index])))
    return SubsetAccuracy(
        accuracy=float(np.mean(accuracies)),
        std=float(np.std(accuracies)),
        fold_accs=accuracies,
        n_features=n_features,
    )


def _mask_for(width: int, indices: Sequence[int]) -> NDArray[np.bool_]:
    mask = np.zeros(width, dtype=bool)
    mask[list(indices)] = True
    return mask


def decompose_family(
    data: Run4Data,
    block: str,
    feature_names: Sequence[str],
    classifier: Callable[[str], str],
) -> dict[str, SubsetAccuracy]:
    """Cut one family into sub-families and score each, plus the whole family."""
    X = data.get_block(block)
    y = data.modes
    subfamilies: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        subfamilies.setdefault(classifier(name), []).append(index)

    results: dict[str, SubsetAccuracy] = {}
    for subfamily, indices in sorted(subfamilies.items()):
        results[subfamily] = accuracy_on_subset(X, y, _mask_for(X.shape[1], indices))
        logger.info(
            f"    {subfamily:<30} {results[subfamily].accuracy:.1%} "
            f"({results[subfamily].n_features} features)"
        )
    results[FULL_FAMILY] = accuracy_on_subset(X, y, np.ones(X.shape[1], dtype=bool))
    logger.info(
        f"    {FULL_FAMILY:<30} {results[FULL_FAMILY].accuracy:.1%} "
        f"({results[FULL_FAMILY].n_features} features)"
    )
    return results


def decompose_run(data: Run4Data) -> dict[str, dict[str, SubsetAccuracy]]:
    """Every family this run carries a naming convention for, cut and scored.

    A family absent from the run, or one whose names the bank cannot assign to it,
    is skipped with the reason logged: a decomposition that quietly omits a family
    reads as a family with no signal.
    """
    out: dict[str, dict[str, SubsetAccuracy]] = {}
    for block, classifier in SUBFAMILY_CLASSIFIERS.items():
        if block not in data.block_features:
            continue
        try:
            names = feature_names_for(data, block)
        except (KeyError, ValueError) as exc:
            logger.warning(f"{block}: not decomposed ({exc})")
            continue
        logger.info(f"  --- {block} ({len(names)} features) ---")
        out[f"{block}_by_signal"] = decompose_family(data, block, names, classifier)
    return out


def decomposition_document(
    results: Mapping[str, Mapping[str, Mapping[str, SubsetAccuracy]]]
) -> dict[str, Any]:
    """Several runs' decompositions as the banked document."""
    return {
        run: {cut: {name: row.model_dump() for name, row in rows.items()} for cut, rows in cuts.items()}
        for run, cuts in results.items()
    }


def default_output_path(analysis_dir: Path, *, mode_filter: Sequence[str] | None) -> Path:
    """Where a decomposition lands: a filtered pass writes beside the full one.

    A pass over five modes and a pass over eight are different measurements, so
    they do not share a file.
    """
    suffix = f"_{len(mode_filter)}way" if mode_filter else ""
    return Path(analysis_dir) / f"subfamily_decomp{suffix}.json"


__all__ = [
    "FULL_FAMILY",
    "SUBFAMILY_CLASSIFIERS",
    "SignalRule",
    "SubsetAccuracy",
    "accuracy_on_subset",
    "classify_attention_flow",
    "classify_gate_features",
    "classify_per_head",
    "classify_residual_trajectory",
    "classify_signal",
    "classify_temporal_dynamics",
    "decompose_family",
    "decompose_run",
    "decomposition_document",
    "default_output_path",
    "feature_names_for",
]
