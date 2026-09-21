"""The leak gate: three planted corpora, three verdicts.

The gate's whole value is telling apart two low grouped accuracies that mean opposite
things, so the tests plant each case and check the verdict by name:

  * a **leak-safe carrier** — a feature that shifts with the condition and not with the
    topic — clears its own permutation null;
  * an **inert** set — pure noise — sits at its null, which the gate reports as inert
    rather than as a leak;
  * the **verdict law** itself, on the three shapes of evidence it is defined over,
    including the bar between leak-dominated and inert — which corpus produces which
    shape depends on the classifier, and what must not drift is how evidence in hand is
    read.

Also tested: the null band is above chance for a narrow feature set under grouped folds
(which is why clearing chance is not the bar), the permutation p-value cannot be zero,
the length residualization is shared across feature sets so two sets stay comparable,
the join refuses cells whose feature names disagree, and the command's own refusals —
a selector it does not understand, a set that selects nothing, and a feature count that
does not match what the caller expected.

CPU only; no model, no device.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.leak_gate import (
    INERT,
    LEAK_DOMINATED,
    LEAK_SAFE,
    NullBand,
    SignatureCell,
    gate_feature_set,
    grouped_accuracy,
    load_corpus,
    naive_accuracy,
    permutation_band,
    residualize_length,
    run_leak_gate,
    select_features,
    verdict,
)
from anamnesis.scripts.leak_gate import main

FEATURE_NAMES = ["xrt_cka_L5_L11", "xrt_cka_global_mean", "kv_key_cka_L8", "gate_sparsity_L8"]
N_TOPICS = 10
N_SEEDS = 4


def write_cell(
    root: Path,
    label: str,
    *,
    condition: float,
    topic_scale: float,
    noise: float,
    seed: int,
    subdir: str = "signatures_v3",
) -> SignatureCell:
    """A cell whose features are a chosen mixture of condition, topic and noise.

    Column 0 carries the condition, column 1 carries the topic, and the last two are
    noise, so a feature selection is what decides which planted structure is under test.
    """
    sig_dir = root / label / subdir
    sig_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    generations = []
    gid = 0
    for topic in range(N_TOPICS):
        for _ in range(N_SEEDS):
            vector = noise * rng.standard_normal(len(FEATURE_NAMES))
            vector[0] += condition
            vector[1] += topic_scale * topic
            np.savez(
                sig_dir / f"gen_{gid:03d}.npz",
                feature_names=np.array(FEATURE_NAMES),
                features=vector.astype(np.float32),
            )
            generations.append(
                {
                    "generation_id": gid,
                    "mode": label,
                    "topic_idx": topic,
                    "mode_idx": 0,
                    "num_generated_tokens": 60 + gid % 7,
                }
            )
            gid += 1
    (root / label / "metadata.json").write_text(json.dumps({"generations": generations}))
    return SignatureCell(label=label, run_dir=root / label, signatures_subdir=subdir)


def leak_safe_corpus(root: Path) -> list[SignatureCell]:
    return [
        write_cell(root, "linear", condition=0.0, topic_scale=0.0, noise=0.2, seed=1),
        write_cell(root, "socratic", condition=3.0, topic_scale=0.0, noise=0.2, seed=2),
    ]


def test_a_condition_carrying_feature_clears_its_own_null(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    gate = gate_feature_set(corpus, select_features(FEATURE_NAMES, prefix="xrt_cka_L5"), nperm=60)
    assert gate.verdict == LEAK_SAFE
    assert gate.quotable
    assert gate.grouped > gate.null_band.null_p95_bar
    assert gate.leak_gap == pytest.approx(gate.naive - gate.grouped)


def test_pure_noise_reads_as_inert_rather_than_as_a_leak(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    gate = gate_feature_set(corpus, select_features(FEATURE_NAMES, prefix="gate_"), nperm=60)
    assert gate.verdict == INERT
    assert not gate.quotable


def test_the_verdict_law_separates_a_leak_from_an_inert_set() -> None:
    """The decision itself, on the three shapes of evidence it is defined over.

    Which planted corpus produces which shape is a property of the classifier and the
    corpus; what must not drift is the reading of the evidence once it is in hand.
    """
    cleared = NullBand(
        obs=0.80, null_p50=0.50, null_p95_bar=0.62, null_p975=0.65, perm_p=0.01,
        nperm=100, clears_band=True,
    )
    inside = NullBand(
        obs=0.55, null_p50=0.50, null_p95_bar=0.62, null_p975=0.65, perm_p=0.40,
        nperm=100, clears_band=False,
    )
    assert verdict(cleared, naive=0.99) == LEAK_SAFE, (
        "a set that clears its own null carries signal whatever its naive number says"
    )
    assert verdict(inside, naive=0.90) == LEAK_DOMINATED
    assert verdict(inside, naive=0.56) == INERT, (
        "no gap and no carry is inert, which is a different finding from a leak"
    )
    assert verdict(inside, naive=inside.obs + 0.049) == INERT, "the gap bar is 0.05"


def test_the_gate_reports_the_leak_gap_whichever_way_it_falls(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    gate = gate_feature_set(
        corpus, select_features(FEATURE_NAMES, contains="cka"), nperm=40
    )
    assert gate.leak_gap == pytest.approx(gate.naive - gate.grouped)
    assert gate.chance == pytest.approx(0.5)
    assert gate.chance_topic == pytest.approx(1 / N_TOPICS)
    assert 0.0 <= gate.topic_decode_naive <= 1.0


def test_the_null_band_sits_above_chance_for_a_narrow_feature_set(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    band = permutation_band(
        residualize_length(corpus.X, corpus.lengths),
        np.asarray(corpus.y),
        np.asarray(corpus.topics),
        select_features(FEATURE_NAMES, prefix="gate_"),
        nperm=60,
    )
    assert band.null_p95_bar > corpus.chance, (
        "clearing chance is not the bar: a narrow set's grouped null sits above it"
    )
    assert band.perm_p >= 1 / (band.nperm + 1), "a permutation p cannot be zero"


def test_the_two_accuracies_differ_only_in_how_they_fold(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    X = residualize_length(corpus.X, corpus.lengths)
    columns = select_features(FEATURE_NAMES, prefix="xrt_cka_L5")
    assert grouped_accuracy(X, np.asarray(corpus.y), np.asarray(corpus.topics), columns) > 0.9
    assert naive_accuracy(X, np.asarray(corpus.y), columns) > 0.9


def test_one_residualization_is_shared_across_feature_sets(tmp_path: Path) -> None:
    corpus = load_corpus(leak_safe_corpus(tmp_path))
    shared = residualize_length(corpus.X, corpus.lengths)
    first = gate_feature_set(
        corpus, select_features(FEATURE_NAMES, prefix="xrt_cka_L5"), residualized=shared, nperm=40
    )
    again = gate_feature_set(
        corpus, select_features(FEATURE_NAMES, prefix="xrt_cka_L5"), residualized=shared, nperm=40
    )
    assert first.grouped == again.grouped and first.naive == again.naive


def test_cells_whose_feature_names_disagree_are_not_joined(tmp_path: Path) -> None:
    cells = leak_safe_corpus(tmp_path)
    for path in sorted(cells[1].signature_dir.glob("gen_*.npz")):
        np.savez(
            path,
            feature_names=np.array(["something", "else"]),
            features=np.zeros(2, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="feature names differ"):
        load_corpus(cells)


def test_a_gate_needs_two_cells_and_at_least_one_feature(tmp_path: Path) -> None:
    cells = leak_safe_corpus(tmp_path)
    with pytest.raises(ValueError, match="at least two cells"):
        load_corpus(cells[:1])
    corpus = load_corpus(cells)
    with pytest.raises(ValueError, match="nothing to decide"):
        gate_feature_set(corpus, [])
    with pytest.raises(ValueError, match="exactly one of prefix or contains"):
        select_features(FEATURE_NAMES)


def test_the_document_carries_the_law_the_verdicts_were_taken_under(tmp_path: Path) -> None:
    cells = leak_safe_corpus(tmp_path)
    document = run_leak_gate(
        cells,
        {
            "routing_cka": select_features(FEATURE_NAMES, prefix="xrt_cka_"),
            "all_cka": select_features(FEATURE_NAMES, contains="cka"),
        },
        nperm=40,
    )
    assert document["n_topics"] == N_TOPICS
    assert document["n_rows"] == 2 * N_TOPICS * N_SEEDS
    assert "GroupKFold-by-topic" in document["law"] and "leak_gap" in document["law"]
    assert set(document["feature_sets"]) == {"routing_cka", "all_cka"}
    assert document["feature_sets"]["all_cka"]["n_features"] == 3


def test_the_command_refuses_a_selector_it_cannot_read(tmp_path: Path) -> None:
    cells = leak_safe_corpus(tmp_path)
    argv = [
        "--cell", f"linear={cells[0].run_dir}",
        "--cell", f"socratic={cells[1].run_dir}",
        "--out", str(tmp_path / "gate.json"),
        "--nperm", "20",
    ]
    with pytest.raises(SystemExit, match="prefix:"):
        main([*argv, "--feature-set", "cka=startswith:xrt"])
    with pytest.raises(SystemExit, match="selected no feature"):
        main([*argv, "--feature-set", "cka=prefix:nothing_"])
    with pytest.raises(SystemExit, match="expected 6"):
        main([*argv, "--feature-set", "cka=prefix:xrt_cka_", "--expect", "6"])


def test_the_command_exits_on_the_first_feature_sets_verdict(tmp_path: Path) -> None:
    cells = leak_safe_corpus(tmp_path)
    out = tmp_path / "gate.json"
    argv = [
        "--cell", f"linear={cells[0].run_dir}",
        "--cell", f"socratic={cells[1].run_dir}",
        "--out", str(out), "--nperm", "20",
    ]
    assert main([*argv, "--feature-set", "carrier=prefix:xrt_cka_L5", "--expect", "1"]) == 0
    assert json.loads(out.read_text())["quotable"]["carrier"] is True
    assert main([*argv, "--feature-set", "noise=prefix:gate_"]) == 1
