"""The 2AFC harness, tested as the discipline it is rather than as plumbing.

Each case pins a control against the failure it exists for: blinding against the
key leaking into a rendered prompt, the ceiling against a null being read off a
ruler with no range, the non-run flag against a silent authentication failure
being read as a judge that saw nothing, anti-circularity against a battery
grading its own homework, and the pre-draw discipline against a concurrent pass
producing a different contrast than a sequential one.

No case reaches a provider. The backends here are stubs that record what they
were asked and answer from a script, which is also the only way to assert that a
key never entered a judge's context: the stub keeps every prompt it saw.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.judging import harness as H
from anamnesis.judging.prompts import FORMALITY, MODE, RenderedPrompt, SOCRATIC


# ── Stubs ─────────────────────────────────────────────────────────────────────
class ScriptedBackend:
    """A judge that answers from a rule and remembers everything it was shown."""

    family = "stub"

    def __init__(self, answers: list[str | None] | None = None, *, always: str | None = None,
                 refuse: bool = False, error: str | None = None,
                 json_reply: bool = False) -> None:
        self.answers = list(answers or [])
        self.always = always
        self.refuse = refuse
        self.error = error
        self.json_reply = json_reply
        self.seen: list[RenderedPrompt] = []
        self.models: list[str] = []

    def complete(self, *, model, system, user, schema=None) -> H.RawReply:
        self.seen.append(RenderedPrompt(system=system, user=user))
        self.models.append(model)
        if self.error is not None:
            return H.RawReply(model=model, error=self.error, usage=H.Usage(errors=1))
        answer = None if self.refuse else (
            self.always if self.always is not None
            else (self.answers.pop(0) if self.answers else "A")
        )
        if answer is None:
            text = "I would rather not say."
        elif self.json_reply:
            text = json.dumps({"choice": answer, "confidence": 4, "tell": "the questions"})
        else:
            text = f"Answer: {answer}"
        return H.RawReply(
            text=text, model=model, usage=H.Usage(calls=1, input_tokens=10, output_tokens=2)
        )


class TruthfulBackend:
    """A perfect judge: it reads the key it is not supposed to have.

    It is handed the key explicitly, which is what makes it a fixture rather than
    a leak — the harness never gives it one.
    """

    family = "oracle"

    def __init__(self, packet: H.BlindPacket, *, accuracy: float = 1.0) -> None:
        self.packet = packet
        self.accuracy = accuracy
        self.by_text = {
            (p.first, p.second): packet.keys[p.pair_id].target_side for p in packet.pairs
        }
        self.calls = 0

    def complete(self, *, model, system, user, schema=None) -> H.RawReply:
        side = next(
            (s for (first, second), s in self.by_text.items() if first[:40] in user and second[:40] in user),
            "A",
        )
        self.calls += 1
        wrong = self.calls > int(round(self.accuracy * len(self.packet.pairs)))
        if wrong:
            side = "B" if side == "A" else "A"
        return H.RawReply(
            text=side, model=model, usage=H.Usage(calls=1, input_tokens=5, output_tokens=1)
        )


def words(tag: str, n: int = 30) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def corpus(tag: str, groups: int = 10, per_group: int = 3) -> H.Corpus:
    return H.Corpus(
        by_group={str(g): [words(f"{tag}-{g}-{i}") for i in range(per_group)] for g in range(groups)},
        labels={str(g): f"topic {g}" for g in range(groups)},
    )


def packet(n_pairs: int = 20, seed: int = 7) -> H.BlindPacket:
    return H.draw_pairs(
        contrast="cell",
        target=corpus("steered"),
        distractors={"rider": corpus("rider")},
        n_pairs=n_pairs,
        seed=seed,
    )


# ── Blinding ──────────────────────────────────────────────────────────────────
def test_the_judge_facing_type_has_no_field_that_holds_the_answer() -> None:
    """The guarantee is structural: rendering has no key in scope to leak."""
    forbidden = {"target_side", "steered_is", "correct", "key", "labels", "answer"}
    assert set(H.BlindPair.model_fields) & forbidden == set()
    assert "target_side" in H.PairKey.model_fields


def test_swapping_every_key_changes_no_rendered_prompt() -> None:
    """The strongest statement of the blinding: the prompts are a function of the
    pairs alone, so the key cannot have reached them."""
    original = packet()
    flipped = original.model_copy(
        update={
            "keys": {
                pid: key.model_copy(update={"target_side": "B" if key.target_side == "A" else "A"})
                for pid, key in original.keys.items()
            }
        }
    )
    first = ScriptedBackend()
    second = ScriptedBackend()
    H.run_contrast(first, original, FORMALITY, models=["m"], workers=1)
    H.run_contrast(second, flipped, FORMALITY, models=["m"], workers=1)
    assert [p.user for p in first.seen] == [p.user for p in second.seen]


def test_the_banked_packet_holds_no_answer_and_the_key_goes_elsewhere(tmp_path: Path) -> None:
    drawn = packet()
    drawn.write(packet_path=tmp_path / "packet.json", key_path=tmp_path / "key.json")
    packet_text = (tmp_path / "packet.json").read_text()
    assert "target_side" not in packet_text
    assert "target_side" in (tmp_path / "key.json").read_text()
    restored = H.BlindPacket.read(packet_path=tmp_path / "packet.json", key_path=tmp_path / "key.json")
    assert restored.pairs == drawn.pairs
    assert restored.keys == drawn.keys


def test_one_path_for_packet_and_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(H.JudgingError, match="apart from the packet"):
        packet().write(packet_path=tmp_path / "both.json", key_path=tmp_path / "both.json")


def test_a_key_from_another_contrast_is_refused_on_read(tmp_path: Path) -> None:
    packet(seed=1).write(packet_path=tmp_path / "p1.json", key_path=tmp_path / "k1.json")
    packet(seed=2).write(packet_path=tmp_path / "p2.json", key_path=tmp_path / "k2.json")
    with pytest.raises(H.ReconstructionMismatch, match="different pairs"):
        H.BlindPacket.read(packet_path=tmp_path / "p1.json", key_path=tmp_path / "k2.json")


# ── Drawing ───────────────────────────────────────────────────────────────────
def test_the_same_seed_draws_the_same_contrast() -> None:
    assert packet(seed=11).model_dump() == packet(seed=11).model_dump()
    assert packet(seed=11).digest() != packet(seed=12).digest()


def test_pairs_are_drawn_within_a_group() -> None:
    drawn = packet()
    by_text = {}
    for tag, source in (("steered", corpus("steered")), ("rider", corpus("rider"))):
        for group, texts in source.by_group.items():
            for text in texts:
                by_text[text] = (tag, group)
    for pair in drawn.pairs:
        group = drawn.keys[pair.pair_id].group
        assert {by_text[pair.first][1], by_text[pair.second][1]} == {group}
        assert {by_text[pair.first][0], by_text[pair.second][0]} == {"steered", "rider"}


def test_position_is_randomized_rather_than_fixed() -> None:
    sides = [key.target_side for key in packet(n_pairs=40).keys.values()]
    assert 10 <= sides.count("A") <= 30, sides.count("A")


def test_several_distractor_sources_are_cycled_and_recorded() -> None:
    drawn = H.draw_pairs(
        contrast="hardening",
        target=corpus("target"),
        distractors={m: corpus(m) for m in ("analogical", "dialectical", "socratic")},
        n_pairs=30,
        seed=3,
    )
    sources = {key.labels["distractor"] for key in drawn.keys.values()}
    assert sources == {"analogical", "dialectical", "socratic"}


def test_per_group_defaults_so_a_bigger_bank_yields_more_pairs() -> None:
    small = H.draw_pairs(contrast="c", target=corpus("t", per_group=2),
                         distractors={"r": corpus("r")}, n_pairs=40, seed=1)
    large = H.draw_pairs(contrast="c", target=corpus("t", per_group=8),
                         distractors={"r": corpus("r")}, n_pairs=40, seed=1)
    assert len(large.pairs) > len(small.pairs)


def test_a_contrast_with_no_shared_group_is_refused() -> None:
    other = H.Corpus(by_group={"99": [words("x")]})
    with pytest.raises(H.JudgingError, match="share no group"):
        H.draw_pairs(contrast="c", target=corpus("t"), distractors={"r": other}, n_pairs=10, seed=1)


def test_a_contrast_below_the_pair_floor_is_refused() -> None:
    with pytest.raises(H.JudgingError, match="usable pairs"):
        H.draw_pairs(
            contrast="c",
            target=corpus("t", groups=2, per_group=1),
            distractors={"r": corpus("r", groups=2, per_group=1)},
            n_pairs=2,
            seed=1,
        )


def test_min_chars_drops_the_texts_whose_length_is_the_tell() -> None:
    target = corpus("t")
    target.by_group["0"] = ["tiny"]
    drawn = H.draw_pairs(contrast="c", target=target, distractors={"r": corpus("r")},
                         n_pairs=40, seed=1, min_chars=60)
    assert all("tiny" not in (p.first, p.second) for p in drawn.pairs)


# ── Concurrency does not move anything ────────────────────────────────────────
def test_one_worker_and_sixteen_produce_the_same_rows() -> None:
    drawn = packet(n_pairs=24)
    sequential = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    concurrent = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=16)
    assert [r.pair_id for r in sequential.rows] == [r.pair_id for r in concurrent.rows]
    assert sequential.win_rate == concurrent.win_rate


# ── Statistics ────────────────────────────────────────────────────────────────
def test_wilson_is_the_interval_and_not_the_normal_approximation() -> None:
    low, high = H.wilson(40, 40)
    assert high <= 1.0 and low > 0.9
    assert H.wilson(0, 0) == (0.0, 1.0)
    low, high = H.wilson(20, 40)
    assert low < 0.5 < high


def test_the_binomial_p_is_one_sided_against_chance() -> None:
    assert H.binom_p_ge(20, 40) == pytest.approx(0.56269, abs=1e-4)
    assert H.binom_p_ge(40, 40) == pytest.approx(0.5**40, abs=1e-12)
    assert H.binom_p_ge(0, 0) == 1.0


def test_a_perfect_contrast_separates_and_qualifies() -> None:
    drawn = packet(n_pairs=20)
    result = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    assert result.win_rate == 1.0
    assert result.separates and result.qualifies
    assert result.n_scored == 20 and result.n_failed == 0
    assert result.binom_p_ge_chance < 0.001


def test_a_chance_contrast_neither_separates_nor_qualifies() -> None:
    drawn = packet(n_pairs=20)
    result = H.run_contrast(
        ScriptedBackend(always="A"), drawn, FORMALITY, models=["m"], workers=1
    )
    assert 0.2 <= (result.win_rate or 0) <= 0.8
    assert not result.separates


def test_the_result_carries_the_prompt_and_packet_it_rests_on() -> None:
    drawn = packet()
    result = H.run_contrast(ScriptedBackend(), drawn, SOCRATIC, models=["m"],
                            variant="socratic", workers=1)
    assert result.prompt_set == "socratic"
    assert result.prompt_digest == SOCRATIC.digest()
    assert result.packet_digest == drawn.digest()
    assert result.variant == "socratic"
    assert result.law == H.BLINDING_LAW


def test_by_distractor_decomposes_without_the_judge_having_seen_a_source() -> None:
    drawn = H.draw_pairs(
        contrast="hardening", target=corpus("t"),
        distractors={"a": corpus("a"), "b": corpus("b")}, n_pairs=20, seed=5,
    )
    result = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    assert set(result.by_distractor) == {"a", "b"}
    # the source name lives in the key and is never rendered into a prompt
    recorder = ScriptedBackend()
    H.run_contrast(recorder, drawn, FORMALITY, models=["m"], workers=1)
    assert all("distractor" not in prompt.user for prompt in recorder.seen)


# ── The non-run rule ──────────────────────────────────────────────────────────
def test_zero_scored_pairs_is_a_non_run_rather_than_a_null() -> None:
    drawn = packet()
    result = H.run_contrast(
        ScriptedBackend(error="AuthenticationError: no key"), drawn, FORMALITY,
        models=["m"], workers=1,
    )
    assert result.is_non_run
    assert result.win_rate is None
    assert not result.separates and not result.qualifies
    assert result.usage.calls == 0 and result.usage.errors == len(drawn.pairs)
    verdict = H.interpret_with_ceiling(result, None)
    assert verdict.verdict == "non-run"
    assert "absence of" in verdict.reason


# ── The model ladder ──────────────────────────────────────────────────────────
def test_an_unparseable_reply_escalates_exactly_as_an_error_does() -> None:
    backend = ScriptedBackend(answers=[None, "A"])
    reply = H.ask_with_ladder(
        backend, RenderedPrompt(user="q"), models=["primary", "fallback"], answer="letter"
    )
    assert reply.answer == "A"
    assert reply.model == "fallback"
    assert reply.usage.fallbacks == 1
    assert backend.models == ["primary", "fallback"]


def test_the_ladder_reports_the_reason_when_every_grade_fails() -> None:
    reply = H.ask_with_ladder(
        ScriptedBackend(refuse=True), RenderedPrompt(user="q"),
        models=["a", "b"], answer="letter",
    )
    assert reply.answer is None
    assert "unparseable" in (reply.error or "")


def test_a_ladder_with_no_model_is_refused() -> None:
    with pytest.raises(H.JudgingError, match="at least one model"):
        H.ask_with_ladder(ScriptedBackend(), RenderedPrompt(user="q"), models=[], answer="letter")


def test_a_reasoning_leading_reply_is_still_read() -> None:
    assert H.read_letter("Looking at both, I think the answer is B.") == "B"
    assert H.read_letter("neither") is None


def test_a_confidence_outside_the_scale_is_clamped_on_read() -> None:
    choice, confidence, tell = H.read_json_choice('```json\n{"choice": "A", "confidence": 9}\n```')
    assert (choice, confidence) == ("A", 5)
    assert tell is None
    assert H.read_json_choice("not json") == (None, None, None)
    assert H.read_json_choice('{"choice": "C"}')[0] is None


def test_the_json_paradigm_carries_confidence_into_the_result() -> None:
    drawn = packet(n_pairs=10)
    result = H.run_contrast(
        ScriptedBackend(always="A", json_reply=True), drawn, MODE,
        models=["m"], variant="linear", workers=1,
    )
    assert result.mean_confidence == 4.0


# ── The reader-grade ladder ───────────────────────────────────────────────────
def test_the_ladder_reports_the_grade_a_pass_passes_at() -> None:
    drawn = packet(n_pairs=20)

    class GradedBackend:
        family = "graded"

        def complete(self, *, model, system, user, schema=None) -> H.RawReply:
            oracle = TruthfulBackend(drawn)
            if model == "strong":
                return oracle.complete(model=model, system=system, user=user)
            return H.RawReply(text="A", model=model, usage=H.Usage(calls=1))

    result = H.run_reader_ladder(
        GradedBackend(), drawn, FORMALITY, grades=["strong", "weak"], workers=1
    )
    assert result.passes_at == ("strong",)
    assert result.weakest_pass == "strong"
    assert result.results["weak"].separates is False


def test_a_grade_is_read_alone_so_no_fallback_answers_for_it() -> None:
    drawn = packet(n_pairs=10)
    backend = ScriptedBackend()
    H.run_reader_ladder(backend, drawn, FORMALITY, grades=["strong", "weak"], workers=1)
    assert set(backend.models) == {"strong", "weak"}


def test_an_empty_ladder_is_refused() -> None:
    with pytest.raises(H.JudgingError, match="at least one grade"):
        H.run_reader_ladder(ScriptedBackend(), packet(), FORMALITY, grades=[])


# ── The ceiling control ───────────────────────────────────────────────────────
def separating(name: str = "target") -> H.ContrastResult:
    drawn = packet(n_pairs=20)
    result = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    return result.model_copy(update={"contrast": name})


def at_chance(name: str = "target") -> H.ContrastResult:
    drawn = packet(n_pairs=20)
    result = H.run_contrast(ScriptedBackend(always="A"), drawn, FORMALITY, models=["m"], workers=1)
    return result.model_copy(update={"contrast": name, "wins": 10, "win_rate": 0.5,
                                     "wilson95": H.wilson(10, 20)})


def test_a_positive_stands_without_a_ceiling() -> None:
    verdict = H.interpret_with_ceiling(separating(), None)
    assert verdict.verdict == "positive"
    assert "could not have manufactured" in verdict.reason


def test_a_null_without_a_ceiling_is_uninterpretable() -> None:
    verdict = H.interpret_with_ceiling(at_chance(), None)
    assert verdict.verdict == "uninterpretable"
    assert "no ceiling contrast" in verdict.reason


def test_a_null_beside_a_ceiling_that_does_not_separate_is_uninterpretable() -> None:
    verdict = H.interpret_with_ceiling(at_chance(), at_chance("CEIL"))
    assert verdict.verdict == "uninterpretable"
    assert "dynamic range" in verdict.reason


def test_a_null_beside_a_ceiling_with_range_is_a_null() -> None:
    verdict = H.interpret_with_ceiling(at_chance(), separating("CEIL"))
    assert verdict.verdict == "null"
    assert verdict.ceiling_contrast == "CEIL"


def test_a_ceiling_that_scored_nothing_cannot_license_a_null() -> None:
    non_run = at_chance("CEIL").model_copy(update={"n_scored": 0})
    verdict = H.interpret_with_ceiling(at_chance(), non_run)
    assert verdict.verdict == "uninterpretable"
    assert "scored no pairs" in verdict.reason


# ── Anti-circularity ──────────────────────────────────────────────────────────
def test_a_criterion_drawn_from_the_scoring_instrument_is_refused() -> None:
    with pytest.raises(H.CircularityError, match="two hats"):
        H.assert_not_circular(FORMALITY, scoring_instrument=FORMALITY.criterion_source)


def test_a_criterion_that_names_the_scorer_is_refused() -> None:
    battery = FORMALITY.model_copy(update={"criterion_source": "the frozen formality marker battery"})
    with pytest.raises(H.CircularityError):
        H.assert_not_circular(battery, scoring_instrument="formality marker battery")


def test_an_independent_criterion_passes() -> None:
    H.assert_not_circular(FORMALITY, scoring_instrument="attention-allocation signature classifier")
    H.assert_not_circular(MODE, scoring_instrument="five-way LDA over mid-layer features")


def test_a_study_that_names_no_instrument_cannot_be_checked() -> None:
    with pytest.raises(H.CircularityError, match="name the instrument"):
        H.assert_not_circular(FORMALITY, scoring_instrument="  ")
    undeclared = FORMALITY.model_copy(update={"criterion_source": ""})
    with pytest.raises(H.CircularityError, match="no criterion source"):
        H.assert_not_circular(undeclared, scoring_instrument="anything")


# ── The coherence gate ────────────────────────────────────────────────────────
def test_the_gate_reads_the_window_it_says_it_reads() -> None:
    texts = ["HEAD " + "x" * 3000 + " TAIL"]
    tail = ScriptedBackend(always="5")
    H.run_coherence_gate(tail, texts, models=["m"], window="tail", chars=20, workers=1)
    assert "TAIL" in tail.seen[0].user and "HEAD" not in tail.seen[0].user
    head = ScriptedBackend(always="5")
    H.run_coherence_gate(head, texts, models=["m"], window="head", chars=20, workers=1)
    assert "HEAD" in head.seen[0].user and "TAIL" not in head.seen[0].user


def test_the_gate_separates_a_shift_from_a_collapse() -> None:
    fluent = H.run_coherence_gate(
        ScriptedBackend(always="5"), [words("t")] * 8, models=["m"], workers=1, floor=3.0
    )
    collapsed = H.run_coherence_gate(
        ScriptedBackend(always="1"), [words("t")] * 8, models=["m"], workers=1, floor=3.0
    )
    assert fluent.mean == 5.0 and fluent.passes is True
    assert collapsed.mean == 1.0 and collapsed.passes is False


def test_the_gate_states_no_verdict_where_no_floor_was_stated() -> None:
    result = H.run_coherence_gate(ScriptedBackend(always="4"), [words("t")], models=["m"], workers=1)
    assert result.passes is None and result.mean == 4.0


def test_an_unreadable_rating_is_missing_rather_than_zero() -> None:
    result = H.run_coherence_gate(
        ScriptedBackend(refuse=True), [words("t")] * 4, models=["m"], workers=1
    )
    assert result.n == 0 and result.mean is None


# ── A second judge family ─────────────────────────────────────────────────────
def test_two_families_over_one_packet_report_agreement_and_no_re_score() -> None:
    drawn = packet(n_pairs=20)
    first = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    second = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m2"], workers=1)
    agreement = H.compare_families(first, second)
    assert agreement.agreement == 1.0
    assert agreement.n_compared == 20
    assert not agreement.verdict_flipped
    assert "never an automatic re-score" in agreement.reading


def test_a_flipped_verdict_is_stop_and_surface() -> None:
    drawn = packet(n_pairs=20)
    first = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    second = H.run_contrast(ScriptedBackend(always="A"), drawn, FORMALITY, models=["m"], workers=1)
    agreement = H.compare_families(first, second)
    assert agreement.verdict_flipped and agreement.surface


def test_two_passes_over_different_packets_are_not_comparable() -> None:
    a, b = packet(n_pairs=20, seed=1), packet(n_pairs=20, seed=2)
    first = H.run_contrast(TruthfulBackend(a), a, FORMALITY, models=["m"], workers=1)
    second = H.run_contrast(TruthfulBackend(b), b, FORMALITY, models=["m"], workers=1)
    with pytest.raises(H.ReconstructionMismatch, match="digests differ"):
        H.compare_families(first, second)


# ── Reconstruction ────────────────────────────────────────────────────────────
def test_a_reconstruction_that_matches_the_banked_key_is_allowed() -> None:
    drawn = packet()
    banked = {pid: {"target_side": key.target_side} for pid, key in drawn.keys.items()}
    H.verify_against_key(drawn, banked)
    H.verify_against_key(drawn, {pid: {"steered_is": k.target_side} for pid, k in drawn.keys.items()})


def test_a_reconstruction_that_differs_stops_before_any_call() -> None:
    drawn = packet()
    banked = {pid: {"target_side": "A"} for pid in drawn.keys}
    with pytest.raises(H.ReconstructionMismatch, match="not judging"):
        H.verify_against_key(drawn, banked)


def test_a_banked_key_row_with_no_side_is_refused() -> None:
    drawn = packet()
    with pytest.raises(H.ReconstructionMismatch, match="states no side"):
        H.verify_against_key(drawn, {pid: {"topic": 1} for pid in drawn.keys})


# ── The banked sheet form ─────────────────────────────────────────────────────
def test_a_packet_round_trips_through_the_key_free_sheet(tmp_path: Path) -> None:
    drawn = packet(n_pairs=10)
    H.write_pairs_md(drawn, tmp_path / "pairs.md", question="Which text is A/B more formal?")
    sheet = (tmp_path / "pairs.md").read_text()
    assert "target_side" not in sheet
    key = {p.pair_id: {"steered": drawn.keys[p.pair_id].target_side} for p in drawn.pairs}
    reread = H.packet_from_pairs_md(sheet, contrast="cell", key=key)
    assert reread.question == "Which text is A/B more formal?"
    assert [p.first for p in reread.pairs] == [p.first for p in drawn.pairs]
    assert {pid: k.target_side for pid, k in reread.keys.items()} == {
        pid: k.target_side for pid, k in drawn.keys.items()
    }


def test_a_sheet_without_a_question_heading_is_refused() -> None:
    with pytest.raises(H.JudgingError, match="parse failure"):
        H.parse_pairs_md("## PAIR 1\n### A\nx\n### B\ny\n")


def test_a_sheet_row_with_no_key_is_judged_but_cannot_win() -> None:
    sheet_packet = H.packet_from_pairs_md(
        "# Which is A/B more formal?\n\n## PAIR 1\n\n### A\n" + words("a") + "\n\n### B\n" + words("b"),
        contrast="unkeyed",
    )
    result = H.run_contrast(ScriptedBackend(), sheet_packet, FORMALITY, models=["m"], workers=1)
    assert result.n_scored == 0 and result.rows[0].error == "no key for this pair"


def test_the_annex_carrier_renders_the_banked_question() -> None:
    pair = H.BlindPair(pair_id="1", first="aaa", second="bbb")
    rendered = H.annex_prompt("Which text is A/B more socratic?", pair)
    assert "Which text is A/B more socratic?" in rendered.user
    assert rendered.system is None


# ── Corpora ───────────────────────────────────────────────────────────────────
def test_a_bank_loads_grouped_by_topic_with_the_short_ones_dropped(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"generations": [
        {"generation_id": 0, "topic_idx": 0, "topic": "rivers", "generated_text": words("a")},
        {"generation_id": 1, "topic_idx": 0, "topic": "rivers", "generated_text": "too short"},
        {"generation_id": 2, "topic_idx": 1, "topic": "clocks", "generated_text": words("b")},
    ]}))
    loaded = H.texts_by_topic(meta)
    assert loaded.groups() == ("0", "1")
    assert len(loaded.by_group["0"]) == 1
    assert loaded.label("1") == "clocks"


def test_a_bare_generation_list_loads_as_well(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps([{"generation_id": 0, "topic_idx": 3, "generated_text": words("a")}]))
    assert H.texts_by_topic(meta).groups() == ("3",)


def test_a_flat_prompt_bank_refuses_a_count_it_cannot_group(tmp_path: Path) -> None:
    flat = tmp_path / "raws.json"
    flat.write_text(json.dumps([words("a"), words("b"), words("c")]))
    with pytest.raises(H.JudgingError, match="not a multiple"):
        H.texts_by_prompt(flat, ["p1", "p2"])
    assert H.texts_by_prompt(flat, ["p1", "p2", "p3"]).groups() == ("p1", "p2", "p3")


def test_a_jsonl_prompt_bank_groups_by_prompt_id(tmp_path: Path) -> None:
    rows = tmp_path / "floors.jsonl"
    rows.write_text("\n".join(
        json.dumps({"prompt_id": pid, "text": words(pid)}) for pid in ("p1", "p1", "p2")
    ))
    loaded = H.texts_by_prompt(rows, [])
    assert loaded.groups() == ("p1", "p2") and len(loaded.by_group["p1"]) == 2


def test_the_single_text_sample_is_redrawable_from_its_seed() -> None:
    source = corpus("t")
    assert H.sample_texts(source, 5, seed=4) == H.sample_texts(source, 5, seed=4)
    assert H.sample_texts(source, 5, seed=4) != H.sample_texts(source, 5, seed=5)


# ── Keys come from the environment, and the suite never calls out ─────────────
def test_the_first_family_refuses_to_build_a_client_without_an_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    backend = H.AnthropicBackend()
    assert backend.api_key_present() is False
    with pytest.raises(H.JudgingError, match="from nowhere else"):
        backend.client()


def test_the_second_family_refuses_without_its_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    backend = H.OpenRouterBackend()
    assert backend.api_key_present() is False
    with pytest.raises(H.JudgingError, match="from nowhere else"):
        backend.complete(model="m", system=None, user="q")


def test_no_backend_takes_a_key_as_an_argument() -> None:
    """A key that can be passed in is a key that can be written to a receipt."""
    import inspect

    for backend in (H.AnthropicBackend, H.OpenRouterBackend):
        parameters = inspect.signature(backend.__init__).parameters
        assert not any("key" in name for name in parameters), backend


def test_the_effort_knob_is_gated_on_the_models_that_reject_it() -> None:
    """A whole rater arm was lost to sending this knob where it errors."""
    assert "claude-haiku-4-5".startswith(H.NO_EFFORT_MODELS)
    assert not "claude-opus-5".startswith(H.NO_EFFORT_MODELS)


def test_a_receipt_carries_every_rate_beside_its_law() -> None:
    drawn = packet(n_pairs=20)
    result = H.run_contrast(TruthfulBackend(drawn), drawn, FORMALITY, models=["m"], workers=1)
    out = H.receipt(study="s", results=[result],
                    interpretations=[H.interpret_with_ceiling(result, None)])
    assert out["law"] == H.BLINDING_LAW
    assert out["usage"]["calls"] == 20
    assert tuple(out["contrasts"][0]["wilson95"]) == result.wilson95
    assert out["interpretations"][0]["verdict"] == "positive"
    json.dumps(out)  # the receipt is serialisable as banked
