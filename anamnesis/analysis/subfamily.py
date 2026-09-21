"""Sub-family decomposition: which part of a family carries the signal.

`feature_map` classifies a feature by what it reads and how — its source, its
method, its depth. That is the taxonomy, and it is deliberately coarse: it answers
"is this attention or residual" across the whole vector. This module asks the
other question, inside one family: within ``temporal_dynamics``, is the signal in
the attention-entropy series or in key drift? Within ``attention_flow``, is it
system-prompt mass or recency bias? The two are not competing classifications —
one cuts across families by substrate, this one cuts a single family into the
signals its own feature names spell out.

The method is one classifier per family, reading the family's naming convention,
plus a random forest per resulting sub-family under stratified cross-validation.
A sub-family's accuracy beside the full family's is the readout: a part that
matches the whole says the rest is redundant, and a part far below it says the
signal is distributed rather than located.

``temporal_dynamics`` gets two further cuts that the others do not, because it is
the only family whose features carry a *temporal operator* as well as a signal:
the coarse cut groups its signals by which surface they came from, and the
operator cut groups by window and spectral treatment, which is what says whether
windowing bought anything over the plain series.

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

from anamnesis.analysis.gauntlet.signature_io import Run4Data

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

N_ESTIMATORS = 200
N_SPLITS = 5
SEED = 42

UNKNOWN = "unknown"


def classify_temporal_dynamics(name: str) -> str:
    """A ``temporal_dynamics`` feature by the signal it is a series of.

    Names run ``td_L<layer>_<signal>_<operator>``, so the signal is the third
    field and the two key-derived signals are told apart by the fourth.
    """
    parts = name.split("_")
    if len(parts) < 3:
        return UNKNOWN
    signal = parts[2]
    if signal == "attn":
        return "td_T2_attn_entropy"
    if signal == "head":
        return "td_T2_head_agreement"
    if signal == "key":
        if len(parts) >= 4 and parts[3] == "drift":
            return "td_T2.5_key_drift"
        if len(parts) >= 4 and parts[3] == "novelty":
            return "td_T2.5_key_novelty"
        return "td_T2.5_key"
    if signal == "lookback":
        return "td_T2.5_lookback_ratio"
    return UNKNOWN


def classify_temporal_operator(name: str) -> str:
    """A ``temporal_dynamics`` feature by the operator applied to its series.

    The four windows are named in the feature; everything computed on the spectrum
    reads as one operator, because the question the cut answers is whether the
    spectral treatment bought anything, not which spectral statistic did.
    """
    for window in ("w0", "w1", "w2", "w3"):
        if f"_{window}_" in name:
            return window
    for spectral in ("_dominant_freq", "_spectral_centroid", "_bandwidth", "_band_energy"):
        if spectral in name:
            return "stft"
    return "other"


_AF_SIGNALS: tuple[tuple[str, str], ...] = (
    ("sysprompt_mass", "af_sysprompt_mass"),
    ("sysprompt_decay", "af_sysprompt_decay"),
    ("recency_bias", "af_recency_bias"),
    ("region_sysprompt", "af_region_sysprompt"),
    ("region_early", "af_region_early_gen"),
    ("region_mid", "af_region_mid_gen"),
    ("region_recent", "af_region_recent"),
    ("head_diversity_recency", "af_head_diversity_recency"),
    ("head_diversity_sysprompt", "af_head_diversity_sysprompt"),
)

_AF_NAME_RE = re.compile(
    r"attn_flow_L\d+_(.+?)_"
    r"(mean|std|w\d+_|dominant_|spectral_|bandwidth|low_|mid_|high_|decay)"
)


def classify_attention_flow(name: str) -> str:
    """An ``attention_flow`` feature by which allocation statistic it is.

    Names run ``attn_flow_L<layer>_<signal>_<operator>``. The signal is read out of
    the name where the operator suffix makes the boundary unambiguous, and by
    substring otherwise — the substrings are the same nine signals either way, so
    the two paths agree and the second exists for names the pattern does not span.
    """
    match = _AF_NAME_RE.match(name)
    haystack = match.group(1) if match else name
    for needle, subfamily in _AF_SIGNALS:
        if needle in haystack:
            return subfamily
    return "af_unknown"


_GF_NAME_RE = re.compile(r"gate_L\d+_(\w+?)_")


def classify_gate_features(name: str) -> str:
    """A ``gate_features`` feature by its signal, with cross-layer features apart.

    A cross-layer feature has no single layer to belong to, so it is its own
    sub-family rather than being filed under whichever layer it mentions first.
    """
    if "cross_layer" in name or "layer_agreement" in name or "layer_sparsity_diversity" in name:
        return "gf_cross_layer"
    match = _GF_NAME_RE.match(name)
    if match:
        return f"gf_{match.group(1)}"
    for needle, subfamily in (
        ("sparsity", "gf_sparsity"),
        ("drift", "gf_drift"),
        ("eff_dim", "gf_eff_dim"),
        ("topk", "gf_topk_overlap"),
    ):
        if needle in name:
            return subfamily
    return "gf_unknown"


_CP_NAME_RE = re.compile(r"cp_L\d+_(t\d+)_d\d+")


def classify_contrastive_projection(name: str) -> str:
    """A ``contrastive_projection`` feature by the temporal position it projects.

    The learned dimensions are not interpretable individually, so the only cut that
    means anything is *when* in the generation the state was taken.
    """
    match = _CP_NAME_RE.match(name)
    return f"cp_{match.group(1)}" if match else "cp_unknown"


SUBFAMILY_CLASSIFIERS: dict[str, Callable[[str], str]] = {
    "temporal_dynamics": classify_temporal_dynamics,
    "attention_flow": classify_attention_flow,
    "gate_features": classify_gate_features,
    "contrastive_projection": classify_contrastive_projection,
}
"""Which classifier reads which family's names. A family absent from this table has
no sub-family convention to read, which is a fact about its naming rather than a
gap here."""

TD_COARSE_GROUPS: dict[str, list[str]] = {
    "td_T2": ["td_T2_attn_entropy", "td_T2_head_agreement"],
    "td_T2.5": ["td_T2.5_key_drift", "td_T2.5_key_novelty", "td_T2.5_lookback_ratio"],
}
"""The coarse cut: the signals grouped by the substrate they are series of. The
group names are the record's, and the taxonomy sweep is what will rename them."""

TD_OPERATOR_GROUPS: dict[str, list[str]] = {
    "td_w0_only": ["w0"],
    "td_w0_w1": ["w0", "w1"],
    "td_windowed": ["w0", "w1", "w2", "w3"],
    "td_stft_only": ["stft"],
}
"""The operator cut, nested on purpose: each group is the previous one plus a
window, so the readout is what the next window added."""

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


def decompose_by_groups(
    data: Run4Data,
    block: str,
    feature_names: Sequence[str],
    classifier: Callable[[str], str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, SubsetAccuracy]:
    """Score named unions of sub-families — the coarse and operator cuts."""
    X = data.get_block(block)
    y = data.modes
    labels = [classifier(name) for name in feature_names]
    results: dict[str, SubsetAccuracy] = {}
    for group, members in groups.items():
        indices = [i for i, label in enumerate(labels) if label in members]
        results[group] = accuracy_on_subset(X, y, _mask_for(X.shape[1], indices))
        logger.info(
            f"    {group:<30} {results[group].accuracy:.1%} "
            f"({results[group].n_features} features)"
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
        if block == "temporal_dynamics":
            logger.info("  coarse cut, by the substrate each signal is a series of:")
            out["td_coarse"] = decompose_by_groups(
                data, block, names, classify_temporal_dynamics, TD_COARSE_GROUPS
            )
            logger.info("  operator cut, by window and spectral treatment:")
            out["td_by_operator"] = decompose_by_groups(
                data, block, names, classify_temporal_operator, TD_OPERATOR_GROUPS
            )
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
    "SubsetAccuracy",
    "TD_COARSE_GROUPS",
    "TD_OPERATOR_GROUPS",
    "accuracy_on_subset",
    "classify_attention_flow",
    "classify_contrastive_projection",
    "classify_gate_features",
    "classify_temporal_dynamics",
    "classify_temporal_operator",
    "decompose_by_groups",
    "decompose_family",
    "decompose_run",
    "decomposition_document",
    "default_output_path",
    "feature_names_for",
]
