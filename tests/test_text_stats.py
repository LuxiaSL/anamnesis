"""Reading the generated text: the coherence, lexicon and null-ratio readouts.

The text channel's job is to catch the failure the signature cannot see — a
degenerate generation that looks like a change — and to keep the lexical readout
from being over-read. So the tests are about the shapes of those judgments:

  * a repetitive generation scores low type-token ratio and high trigram
    repetition, and a generation with no text does not enter the average at all;
  * the marker lexicons are word-boundary matched and case-insensitive, so
    "maybes" is not a hedge and "Perhaps" is;
  * the group rates pool the k resamples of a prompt into ONE observation, and
    report the census a reader needs to know two cells are comparable;
  * the placebo floor splits a cell against itself, which is what says what zero
    looks like;
  * the null-ratio guard suppresses a ratio to a near-zero denominator and always
    emits the band readout instead.

The entropy/NLL readout needs a model, so it is covered by a stub that records
the shapes and the index arithmetic — which is where an off-by-one would put the
surprisal of the wrong token against the wrong position.

CPU only; no real model, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.text_stats import (
    DEF_RE,
    DEFINITIVE_MARKERS,
    HEDGE_MARKERS,
    HEDGE_RE,
    add_null_ratios,
    entropy_and_nll_over_generation,
    group_marker_rates,
    marker_rate,
    placebo_marker_floor,
    text_stats,
    texts_by_prompt_group,
)

torch = pytest.importorskip("torch", reason="the entropy/NLL readout is a torch path")


def write_metadata(path: Path, generations: list[dict], *, wrapped: bool = True) -> Path:
    payload = {"generations": generations} if wrapped else generations
    path.write_text(json.dumps(payload))
    return path


# ── Coherence ─────────────────────────────────────────────────────────────────
def test_coherence_catches_a_degenerate_generation(tmp_path: Path) -> None:
    varied = write_metadata(tmp_path / "varied.json", [
        {"generated_text": "the quick brown fox jumps over a lazy dog today",
         "num_generated_tokens": 10},
    ])
    looping = write_metadata(tmp_path / "loop.json", [
        {"generated_text": "and so on " * 10, "num_generated_tokens": 30},
    ])
    good = text_stats(varied)
    bad = text_stats(looping)
    assert good["mean_ttr"] > bad["mean_ttr"], "a loop has fewer distinct tokens"
    assert bad["mean_trigram_rep"] > good["mean_trigram_rep"]
    assert good["mean_trigram_rep"] == 0.0
    assert bad["mean_trigram_rep"] > 0.8, "a loop is almost all repeated trigrams"


def test_empty_generations_do_not_enter_the_average(tmp_path: Path) -> None:
    path = write_metadata(tmp_path / "meta.json", [
        {"generated_text": "one two three four", "num_generated_tokens": 4},
        {"generated_text": "", "num_generated_tokens": 0},
        {"generated_text": "   ", "num_generated_tokens": 0},
    ])
    stats = text_stats(path)
    assert stats["n"] == 1, "n counts the generations the statistics rest on"
    assert stats["mean_len"] == 4.0


def test_a_bank_with_no_usable_text_reports_zeros_rather_than_failing(tmp_path: Path) -> None:
    path = write_metadata(tmp_path / "meta.json", [{"generated_text": ""}])
    stats = text_stats(path)
    assert stats == {"n": 0, "mean_len": 0.0, "mean_ttr": 0.0, "mean_trigram_rep": 0.0}


def test_both_metadata_shapes_are_read(tmp_path: Path) -> None:
    rows = [{"generated_text": "alpha beta gamma", "num_generated_tokens": 3}]
    wrapped = text_stats(write_metadata(tmp_path / "w.json", rows, wrapped=True))
    bare = text_stats(write_metadata(tmp_path / "b.json", rows, wrapped=False))
    assert wrapped == bare


def test_the_token_count_from_metadata_wins_over_the_word_count(tmp_path: Path) -> None:
    path = write_metadata(tmp_path / "m.json", [
        {"generated_text": "three words here", "num_generated_tokens": 99},
    ])
    assert text_stats(path)["mean_len"] == 99.0, "the tokenizer's count, not a split()"


# ── Lexicon ───────────────────────────────────────────────────────────────────
def test_markers_are_word_bounded_and_case_insensitive() -> None:
    hits, words = marker_rate("Perhaps this might work", HEDGE_RE)
    assert hits == 2 and words == 4
    assert marker_rate("maybes are not maybe", HEDGE_RE)[0] == 1, "word boundaries hold"
    assert marker_rate("This is definitely certainly true", DEF_RE)[0] == 2
    assert marker_rate("", HEDGE_RE) == (0, 1), "the denominator never reaches zero"
    assert len(HEDGE_MARKERS) > 20 and len(DEFINITIVE_MARKERS) > 20
    assert not set(HEDGE_MARKERS) & set(DEFINITIVE_MARKERS), "the lexicons are disjoint"


def write_cell(root: Path, cell: str, texts_by_group: dict[tuple[int, int], list[str]]) -> None:
    (root / cell).mkdir(parents=True)
    gens = []
    for (topic_idx, mode_idx), texts in texts_by_group.items():
        for text in texts:
            gens.append({"topic_idx": topic_idx, "mode_idx": mode_idx,
                         "generated_text": text})
    (root / cell / "metadata.json").write_text(json.dumps({"generations": gens}))


def test_the_prompt_group_is_the_unit_not_the_generation(tmp_path: Path) -> None:
    write_cell(tmp_path, "V7_L16_a0.3", {
        (0, 0): ["maybe this", "perhaps that", "possibly the other"],
        (1, 0): ["definitely this", "certainly that"],
    })
    groups = texts_by_prompt_group(tmp_path, "V7_L16_a0.3")
    assert groups is not None
    assert set(groups) == {(0, 0), (1, 0)}
    assert len(groups[(0, 0)]) == 3, "the k resamples stay together in their group"

    rates = group_marker_rates(groups)
    assert rates["n_groups"] == 2
    assert len(rates["hedge_pg"]) == 2, "one number per group, not per generation"
    assert rates["hedge_pg"][0] > rates["hedge_pg"][1]
    assert rates["def_pg"][1] > rates["def_pg"][0]
    assert rates["net_hedge_per_1k"] == pytest.approx(
        np.mean(rates["hedge_pg"]) - np.mean(rates["def_pg"])
    )
    assert rates["census_mode_hist"] == {0: 2}, "the census is what makes cells comparable"


def test_a_cell_that_was_never_generated_is_a_skip(tmp_path: Path) -> None:
    assert texts_by_prompt_group(tmp_path, "V7_L16_a9.9") is None


def test_the_placebo_floor_splits_a_cell_against_itself(tmp_path: Path) -> None:
    write_cell(tmp_path, "baseline", {
        (t, 0): ["maybe perhaps possibly", "definitely certainly clearly",
                 "maybe again", "certainly again"]
        for t in range(4)
    })
    groups = texts_by_prompt_group(tmp_path, "baseline")
    assert groups is not None
    floor = placebo_marker_floor(groups, seed=1)
    assert floor["n_placebo_groups"] == 4
    assert floor["placebo_hedge_abs_mean"] >= 0.0
    assert floor["placebo_net_abs_mean"] >= 0.0
    # The same seed gives the same floor: a noise floor that moved between runs
    # could not be compared against anything.
    assert placebo_marker_floor(groups, seed=1) == floor


def test_a_cell_whose_groups_cannot_be_split_reports_no_floor() -> None:
    """One generation per group leaves nothing to split against, so the floor is
    n=0 and its statistics are NaN — unreadable rather than zero, which is the
    honest answer and is what the ``n_placebo_groups`` column is there to say."""
    with pytest.warns(RuntimeWarning):
        floor = placebo_marker_floor({(0, 0): ["only one"], (1, 0): ["also one"]})
    assert floor["n_placebo_groups"] == 0
    assert np.isnan(floor["placebo_hedge_abs_mean"])


# ── Null ratios ───────────────────────────────────────────────────────────────
def rows_with(steered: float, nulls: list[float], key: str = "entropy_rise") -> list[dict]:
    rows = [{"site": 16, "alpha_frac": 0.3, key: steered}]
    rows += [{"site": 16, "alpha_frac": 0.3, key: v, "is_null": True} for v in nulls]
    return rows


def test_a_ratio_to_a_stable_denominator_is_reported() -> None:
    rows = rows_with(2.0, [1.0, 1.02, 0.98, 1.0])
    add_null_ratios(rows, null_prefixes=("RBAND",), keys=("entropy_rise",))
    row = rows[0]
    assert row["entropy_rise_over_Rc"] == pytest.approx(2.0, abs=0.01)
    band = row["entropy_rise_vs_Rc_band"]
    assert band["ratio_suppressed_zero_denom"] is False
    assert band["outside_null_band"] is True
    assert band["z_vs_null"] is not None and band["z_vs_null"] > 0


def test_a_ratio_to_a_near_zero_denominator_is_suppressed_and_banded() -> None:
    rows = rows_with(0.3, [0.001, -0.002, 0.0015, -0.0005])
    add_null_ratios(rows, null_prefixes=("RBAND",), keys=("entropy_rise",))
    row = rows[0]
    assert row["entropy_rise_over_Rc"] is None, "the exploding ratio is refused"
    band = row["entropy_rise_vs_Rc_band"]
    assert band["ratio_suppressed_zero_denom"] is True
    assert band["outside_null_band"] is True, "the band readout still carries the finding"
    assert band["null_min"] <= band["null_mean"] <= band["null_max"]


def test_a_row_with_no_matching_nulls_gets_no_ratio() -> None:
    rows = [{"site": 16, "alpha_frac": 0.3, "entropy_rise": 1.0},
            {"site": 22, "alpha_frac": 0.3, "entropy_rise": 0.5, "is_null": True}]
    add_null_ratios(rows, null_prefixes=(), keys=("entropy_rise",))
    assert rows[0]["entropy_rise_over_Rc"] is None
    assert "entropy_rise_vs_Rc_band" not in rows[0], "no nulls, no band"
    assert "entropy_rise_over_Rc" not in rows[1], "a null is not read against itself"


def test_nulls_are_matched_by_site_and_dose() -> None:
    rows = [
        {"site": 16, "alpha_frac": 0.3, "entropy_rise": 2.0},
        {"site": 16, "alpha_frac": 0.3, "entropy_rise": 1.0, "is_null": True},
        {"site": 16, "alpha_frac": 0.3, "entropy_rise": 1.0, "is_null": True},
        {"site": 16, "alpha_frac": 0.9, "entropy_rise": 99.0, "is_null": True},
    ]
    add_null_ratios(rows, null_prefixes=("RBAND",), keys=("entropy_rise",))
    assert rows[0]["entropy_rise_vs_Rc_band"]["null_mean"] == pytest.approx(1.0)


# ── Entropy and NLL ───────────────────────────────────────────────────────────
class StubOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class StubModel:
    """Returns fixed logits, so the test is about the index arithmetic."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.calls: list[bool] = []

    def __call__(self, ids: torch.Tensor, use_cache: bool = True) -> StubOutput:
        self.calls.append(use_cache)
        return StubOutput(self._logits)


def test_entropy_and_nll_read_the_generated_span_only() -> None:
    vocab, T, prompt_length = 4, 7, 3
    logits = torch.zeros((1, T, vocab))
    logits[0, :, 0] = 10.0                      # every position: near-certain token 0
    ids = torch.zeros((1, T), dtype=torch.long)
    ids[0, prompt_length:] = 1                  # the generated tokens are NOT token 0

    model = StubModel(logits)
    entropy, nll = entropy_and_nll_over_generation(model, ids, prompt_length, None)
    assert model.calls == [False], "one forward pass, no cache"
    n_generated = T - prompt_length
    assert entropy.shape == (n_generated,)
    assert nll.shape == (n_generated,)
    assert np.all(entropy > 0.0) and np.all(entropy < np.log(vocab))
    # The model was certain of token 0 and got token 1, so surprisal is high.
    assert np.all(nll > 5.0)

    # A confident and correct model: low surprisal on the same span.
    ids_correct = torch.zeros((1, T), dtype=torch.long)
    _entropy, nll_correct = entropy_and_nll_over_generation(
        StubModel(logits), ids_correct, prompt_length, None
    )
    assert np.all(nll_correct < 0.01)
    assert nll_correct.shape == nll.shape
