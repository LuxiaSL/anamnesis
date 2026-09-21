"""The leak gate: does a feature set carry the signal, or the topic?

Some features are similarity statistics — a cross-layer representational
similarity, a covariance alignment, anything whose value is "how alike are these
two things". A number like that can track **what** was being processed rather than
**how**, and it will do so quietly: the accuracy looks real, and it is real, but
what it decodes is the topic the two conditions happened to differ on.

The gate is two accuracies and a null:

* **grouped** — GroupKFold by topic, so the topics a fold is tested on were never
  trained on. This is the only accuracy allowed to support a claim.
* **naive** — plain stratified folds, where the same topic appears on both sides.
  It is computed *only* so the gap between the two can be reported: the gap is the
  leak magnitude, and a leak nobody measured is a leak nobody can size.
* **the permutation null** — labels shuffled, the grouped accuracy recomputed,
  respecting the same folds. A feature set clears the gate when its grouped
  accuracy exceeds the null's 95th percentile, not when it exceeds chance:
  small feature sets under grouped folds have null distributions well above
  chance, which is exactly the case the bare-chance reading gets wrong.

A topic-decode readout rides beside them — the same features asked to predict the
topic — because it says in one number how much content the set carries on its own.

Three verdicts, and the middle one is the finding: a set that clears the null is a
**leak-safe carrier**, a set whose naive accuracy is well above its grouped one and
whose grouped accuracy sits inside the null is **leak-dominated**, and a set that
is at the null either way is **inert**. Inert and leak-dominated are different
statements about the same low grouped number, which is why the naive leg is not
optional.

The corpus is the caller's: a cell is a directory of signatures and its own
metadata, and this module reads whatever set of them it is handed. The arm
protocols that chose which cells to build stay in the frozen record — a gate that
knew their directory names would only work for them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, StratifiedKFold

from anamnesis.analysis.audit_lib import residualize_all
from anamnesis.analysis.battery.floors import load_signature_matrix

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

N_SPLITS = 5
N_PERMUTATIONS = 1000
BAND_PERCENTILE = 95.0
LEAK_GAP_BAR = 0.05
"""How far the naive accuracy has to sit above the grouped one before the pair is
read as leak-dominated rather than merely inert."""

LEAK_SAFE = "LEAK-SAFE CARRIER"
LEAK_DOMINATED = "LEAK-DOMINATED (the naive signal is topic leak; no leak-safe carry)"
INERT = "INERT (no carry above the topic-grouped null band)"


class SignatureCell(BaseModel):
    """One condition's banked signatures, with the metadata that keys them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, description="The condition this cell is")
    run_dir: Path = Field(description="Directory holding metadata.json")
    signatures_subdir: str = Field(default="signatures_v3")

    @property
    def signature_dir(self) -> Path:
        return self.run_dir / self.signatures_subdir


class GateCorpus(BaseModel):
    """Several cells, joined: rows, labels, topics and generation lengths."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    X: Any = Field(description="[n, d] float32 signature rows, cells stacked in order")
    y: Any = Field(description="[n] int label, the cell's index in reading order")
    topics: Any = Field(description="[n] int topic index — the grouping unit")
    lengths: Any = Field(description="[n] float generated-token counts")
    feature_names: list[str]
    labels: list[str]

    @property
    def n_topics(self) -> int:
        return int(len(set(np.asarray(self.topics).tolist())))

    @property
    def chance(self) -> float:
        return 1.0 / len(self.labels)


def load_cell(cell: SignatureCell) -> tuple[F32, list[str], list[int], list[float]]:
    """One cell's rows, names, topic indices and generated lengths.

    Topic and length come from the run's own metadata, keyed by generation id, so a
    cell whose signatures were filtered (a short generation dropped for a
    non-standard vector) still lines up with its own records.
    """
    X, names, gen_ids = load_signature_matrix(cell.signature_dir)
    document = json.loads((cell.run_dir / "metadata.json").read_text(encoding="utf-8"))
    generations = (
        document["generations"]
        if isinstance(document, dict) and "generations" in document
        else document
    )
    records = {int(g["generation_id"]): g for g in generations}
    missing = [gid for gid in gen_ids if gid not in records]
    if missing:
        raise KeyError(f"{cell.label}: metadata has no record for generations {missing}")
    topics = [int(records[gid]["topic_idx"]) for gid in gen_ids]
    lengths = [float(records[gid].get("num_generated_tokens", 0)) for gid in gen_ids]
    return X, names, topics, lengths


def load_corpus(cells: Sequence[SignatureCell]) -> GateCorpus:
    """Join cells into one corpus, refusing a feature-name disagreement.

    Two cells whose feature names differ are two different vectors, and stacking
    them would put unlike columns under one index — so the join refuses rather than
    aligning by position.
    """
    if len(cells) < 2:
        raise ValueError("a leak gate needs at least two cells to discriminate")
    rows: list[F32] = []
    labels: list[int] = []
    topics: list[int] = []
    lengths: list[float] = []
    names: list[str] | None = None
    for index, cell in enumerate(cells):
        X, cell_names, cell_topics, cell_lengths = load_cell(cell)
        if names is None:
            names = cell_names
        elif cell_names != names:
            raise ValueError(
                f"{cell.label}: feature names differ from {cells[0].label} — "
                f"two vectors cannot be stacked into one corpus"
            )
        rows.append(X)
        labels += [index] * len(X)
        topics += cell_topics
        lengths += cell_lengths
    assert names is not None
    return GateCorpus(
        X=np.vstack(rows),
        y=np.asarray(labels, dtype=np.int64),
        topics=np.asarray(topics, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.float64),
        feature_names=names,
        labels=[cell.label for cell in cells],
    )


def residualize_length(X: NDArray[Any], lengths: NDArray[Any]) -> F64:
    """Remove the part of every feature that is a linear function of length.

    Length is the confound this instrument keeps rediscovering: a condition that
    writes longer answers moves every windowed statistic, and a classifier reading
    that is reading the length. The regression itself is
    :func:`~anamnesis.analysis.audit_lib.residualize_all` — the whole-matrix control
    the audit library already holds — and this names length as the covariate. It is
    fitted over the whole corpus on purpose: it is a nuisance projection applied
    identically to every row, not a model whose generalization is measured.
    """
    return residualize_all(
        np.asarray(X, dtype=np.float64),
        np.asarray(lengths, dtype=np.float64).reshape(-1, 1),
    )


def grouped_accuracy(
    X: NDArray[Any],
    y: NDArray[Any],
    groups: NDArray[Any],
    columns: Sequence[int] | None = None,
    *,
    n_splits: int = N_SPLITS,
) -> float:
    """Out-of-fold accuracy with whole topics held out — the honest accuracy."""
    rows = X if columns is None else X[:, list(columns)]
    accuracies = [
        float((LinearDiscriminantAnalysis().fit(rows[train], y[train]).predict(rows[test]) == y[test]).mean())
        for train, test in GroupKFold(n_splits=n_splits).split(rows, y, groups)
    ]
    return float(np.mean(accuracies))


def naive_accuracy(
    X: NDArray[Any],
    y: NDArray[Any],
    columns: Sequence[int] | None = None,
    *,
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> float:
    """Out-of-fold accuracy with topics free to appear on both sides.

    Reported only as the other end of the leak gap. It is never the accuracy of
    record, which is why it says so in its own name.
    """
    rows = X if columns is None else X[:, list(columns)]
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accuracies = [
        float((LinearDiscriminantAnalysis().fit(rows[train], y[train]).predict(rows[test]) == y[test]).mean())
        for train, test in folds.split(rows, y)
    ]
    return float(np.mean(accuracies))


class NullBand(BaseModel):
    """The permutation null for a grouped accuracy, and where the observation sits."""

    model_config = ConfigDict(extra="forbid")

    obs: float
    null_p50: float
    null_p95_bar: float
    null_p975: float
    perm_p: float
    nperm: int
    clears_band: bool


def permutation_band(
    X: NDArray[Any],
    y: NDArray[Any],
    groups: NDArray[Any],
    columns: Sequence[int] | None = None,
    *,
    nperm: int = N_PERMUTATIONS,
    seed: int = 0,
) -> NullBand:
    """Shuffle the labels, keep the folds, and see what the folds alone can do.

    The p-value is ``(hits + 1) / (nperm + 1)``, which cannot report zero: the
    permutation resolution is a floor on what a permutation test can say.
    """
    rng = np.random.default_rng(seed)
    observed = grouped_accuracy(X, y, groups, columns)
    null = np.array(
        [grouped_accuracy(X, rng.permutation(y), groups, columns) for _ in range(nperm)]
    )
    bar = float(np.percentile(null, BAND_PERCENTILE))
    return NullBand(
        obs=round(observed, 4),
        null_p50=round(float(np.percentile(null, 50)), 4),
        null_p95_bar=round(bar, 4),
        null_p975=round(float(np.percentile(null, 97.5)), 4),
        perm_p=round(float((np.sum(null >= observed) + 1) / (nperm + 1)), 4),
        nperm=nperm,
        clears_band=bool(observed > bar),
    )


def verdict(band: NullBand, naive: float) -> str:
    """Leak-safe, leak-dominated or inert, from the two accuracies and the null."""
    if band.clears_band:
        return LEAK_SAFE
    if naive - band.obs >= LEAK_GAP_BAR and band.obs <= band.null_p95_bar:
        return LEAK_DOMINATED
    return INERT


class FeatureSetGate(BaseModel):
    """One feature set's gate: both accuracies, the null, the topic read, the verdict."""

    model_config = ConfigDict(extra="forbid")

    features: list[str]
    n_features: int = Field(gt=0)
    grouped: float
    naive: float
    leak_gap: float
    null_band: NullBand
    topic_decode_naive: float
    chance: float
    chance_topic: float
    verdict: str

    @property
    def quotable(self) -> bool:
        """Whether a claim may rest on this feature set at all."""
        return self.verdict == LEAK_SAFE


def gate_feature_set(
    corpus: GateCorpus,
    columns: Sequence[int],
    *,
    residualized: NDArray[Any] | None = None,
    nperm: int = N_PERMUTATIONS,
    seed: int = 0,
) -> FeatureSetGate:
    """Run the gate over the named columns of a corpus.

    ``residualized`` lets one length regression be shared across several feature
    sets of the same corpus, which is what keeps two sets' numbers comparable: a
    per-set regression would residualize each against a different fit.
    """
    if not columns:
        raise ValueError("a gate over zero features has nothing to decide")
    X = residualize_length(corpus.X, corpus.lengths) if residualized is None else residualized
    y = np.asarray(corpus.y)
    groups = np.asarray(corpus.topics)
    band = permutation_band(X, y, groups, columns, nperm=nperm, seed=seed)
    naive = naive_accuracy(X, y, columns, seed=seed)
    topic_decode = naive_accuracy(X, groups, columns, seed=seed)
    return FeatureSetGate(
        features=[corpus.feature_names[i] for i in columns],
        n_features=len(columns),
        grouped=band.obs,
        naive=round(naive, 4),
        leak_gap=round(naive - band.obs, 4),
        null_band=band,
        topic_decode_naive=round(topic_decode, 4),
        chance=round(corpus.chance, 4),
        chance_topic=round(1.0 / corpus.n_topics, 4),
        verdict=verdict(band, naive),
    )


def select_features(
    feature_names: Sequence[str], *, prefix: str | None = None, contains: str | None = None
) -> list[int]:
    """Column indices by name, either by prefix or by substring.

    Both are offered because they answer different questions: a prefix names one
    family's features exactly, and a substring names every feature computed the same
    way wherever it lives.
    """
    if (prefix is None) == (contains is None):
        raise ValueError("select_features takes exactly one of prefix or contains")
    if prefix is not None:
        return [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
    return [i for i, name in enumerate(feature_names) if contains in name]


def run_leak_gate(
    cells: Sequence[SignatureCell],
    feature_sets: Mapping[str, Sequence[int]],
    *,
    nperm: int = N_PERMUTATIONS,
    seed: int = 0,
    corpus: GateCorpus | None = None,
) -> dict[str, Any]:
    """Gate several named feature sets of one corpus, as the banked document.

    Every set is gated against the same length residualization and the same folds,
    and the document records the corpus shape beside the verdicts — a gate read
    without its n and its topic count is not a gate.
    """
    corpus = corpus if corpus is not None else load_corpus(cells)
    residualized = residualize_length(corpus.X, corpus.lengths)
    gates = {
        name: gate_feature_set(
            corpus, columns, residualized=residualized, nperm=nperm, seed=seed
        )
        for name, columns in feature_sets.items()
    }
    for name, gate in gates.items():
        logger.info(
            f"  {name}: grouped {gate.grouped:.3f} vs null p95 {gate.null_band.null_p95_bar:.3f}"
            f" (naive {gate.naive:.3f}, leak gap {gate.leak_gap:+.3f}) -> {gate.verdict}"
        )
    return {
        "gate": "leak_gate",
        "cells": [cell.label for cell in cells] if cells else corpus.labels,
        "n_rows": int(np.asarray(corpus.y).shape[0]),
        "n_topics": corpus.n_topics,
        "law": (
            "a feature set carries leak-safe signal iff its GroupKFold-by-topic accuracy "
            "exceeds the topic-grouped permutation null's p95 band; leak_gap = naive - "
            "grouped; topic_decode = the same features asked to predict the topic. Features "
            "are length-residualized once for every set, and the classifier is LDA."
        ),
        "feature_sets": {name: gate.model_dump() for name, gate in gates.items()},
        "quotable": {name: gate.quotable for name, gate in gates.items()},
    }


__all__ = [
    "BAND_PERCENTILE",
    "GateCorpus",
    "SignatureCell",
    "FeatureSetGate",
    "INERT",
    "LEAK_DOMINATED",
    "LEAK_SAFE",
    "NullBand",
    "gate_feature_set",
    "grouped_accuracy",
    "load_cell",
    "load_corpus",
    "naive_accuracy",
    "permutation_band",
    "residualize_length",
    "run_leak_gate",
    "select_features",
    "verdict",
]
