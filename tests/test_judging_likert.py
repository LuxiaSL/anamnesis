"""The Likert paradigm: what it yields that a forced choice cannot, and its limits.

Purity is the reason this paradigm is kept, so most of these cases are about
purity: that it is a difference rather than a rating, that it can go negative,
that an unreadable judgement is missing rather than zero, and that the
cross-channel correlation it enables refuses to report a number over too few
points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.judging import likert as L
from anamnesis.judging.harness import RawReply, Usage
from anamnesis.judging.prompts import VALID_MODES


class StubJudge:
    """A Likert judge that answers from a list, with no network anywhere."""

    family = "stub"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.systems: list[str | None] = []

    def complete(self, *, model, system, user, schema=None) -> RawReply:
        self.calls += 1
        self.systems.append(system)
        if not self.replies:
            return RawReply(model=model, error="exhausted", usage=Usage(errors=1))
        return RawReply(text=self.replies.pop(0), model=model, usage=Usage(calls=1))


def rating(primary: str = "socratic", **ratings: int) -> str:
    table = {m: 2 for m in VALID_MODES}
    table.update(ratings)
    return json.dumps(
        {"ratings": table, "primary_mode": primary, "confidence": 4, "reasoning": "questions"}
    )


# ── Parsing ───────────────────────────────────────────────────────────────────
def test_a_well_formed_judgement_parses() -> None:
    parsed = L.parse_likert_reply(rating(socratic=5))
    assert parsed is not None
    assert parsed.primary_mode == "socratic"
    assert parsed.ratings["socratic"] == 5
    assert parsed.confidence == 4


def test_markdown_fencing_is_stripped() -> None:
    assert L.parse_likert_reply("```json\n" + rating() + "\n```") is not None


def test_a_reply_truncated_inside_its_reasoning_keeps_its_ratings() -> None:
    """The scored part is already complete by the time the prose is cut."""
    table = json.dumps({m: 3 for m in VALID_MODES})
    truncated = (
        '{"ratings": ' + table + ', "primary_mode": "linear", "confidence": 3, "reasoning": "the te'
    )
    parsed = L.parse_likert_reply(truncated)
    assert parsed is not None and parsed.primary_mode == "linear"
    assert parsed.reasoning == "(truncated)"


def test_a_rating_outside_the_scale_is_a_missing_datum() -> None:
    assert L.parse_likert_reply(rating(socratic=9)) is None
    assert L.parse_likert_reply(rating(primary="ironic")) is None
    assert L.parse_likert_reply("not json at all") is None
    assert L.parse_likert_reply(json.dumps({"ratings": "five"})) is None
    assert L.parse_likert_reply(json.dumps([1, 2, 3])) is None


def test_a_missing_confidence_reads_as_the_midpoint() -> None:
    payload = json.loads(rating())
    del payload["confidence"]
    parsed = L.parse_likert_reply(json.dumps(payload))
    assert parsed is not None and parsed.confidence == 3


def test_a_mode_missing_from_the_ratings_is_a_missing_datum() -> None:
    payload = json.loads(rating())
    del payload["ratings"]["dialectical"]
    assert L.parse_likert_reply(json.dumps(payload)) is None


# ── Purity ────────────────────────────────────────────────────────────────────
def test_purity_is_a_difference_and_not_a_rating() -> None:
    flat = {m: 5 for m in VALID_MODES}
    assert L.purity(flat, "socratic") == 0.0
    peaked = {m: 1 for m in VALID_MODES} | {"socratic": 5}
    assert L.purity(peaked, "socratic") == 4.0


def test_purity_goes_negative_where_the_judge_read_something_else() -> None:
    misread = {m: 5 for m in VALID_MODES} | {"socratic": 1}
    assert L.purity(misread, "socratic") == -4.0


def test_purity_refuses_a_mode_it_has_no_rating_for() -> None:
    with pytest.raises(L.JudgingError, match="intended mode"):
        L.purity({m: 3 for m in VALID_MODES if m != "linear"}, "linear")


# ── Scoring a bank ────────────────────────────────────────────────────────────
def test_the_judge_never_sees_the_mode_instruction() -> None:
    judge = StubJudge([rating()])
    L.score_text(judge, model="m", topic="rivers", text="a text")
    assert judge.systems[0] is not None
    assert "which you do not know" in judge.systems[0]


def test_a_bad_read_retries_the_same_model_rather_than_another() -> None:
    """A Likert number is a reading on one judge's scale; two scales do not mix."""
    judge = StubJudge(["garbage", "still garbage", rating()])
    parsed, usage = L.score_text(judge, model="m", topic="t", text="x", retries=3)
    assert parsed is not None
    assert judge.calls == 3 and usage.calls == 3


def test_a_judgement_that_never_reads_is_a_failure_rather_than_a_zero() -> None:
    parsed, usage = L.score_text(StubJudge(["garbage"] * 3), model="m", topic="t", text="x")
    assert parsed is None and usage.calls == 3


def generations(n: int = 4) -> list[dict[str, object]]:
    modes = list(VALID_MODES)
    return [
        {"generation_id": i, "mode": modes[i % len(modes)], "topic": f"topic {i}",
         "generated_text": "a text"}
        for i in range(n)
    ]


def test_a_bank_scores_and_names_what_it_could_not_read() -> None:
    gens = generations(3)
    gens[1]["generated_text"] = ""
    judge = StubJudge([rating(primary="linear", linear=5), rating(primary="socratic", socratic=5)])
    scores, failed, usage = L.score_bank(judge, gens, model="m")
    assert [s.generation_id for s in scores] == [0, 2]
    assert failed == [1]
    assert usage.calls == 2


def test_resume_skips_what_is_already_scored() -> None:
    gens = generations(2)
    first = StubJudge([rating(primary="linear", linear=5)] * 2)
    scores, _, _ = L.score_bank(first, gens, model="m")
    second = StubJudge([rating(primary="linear", linear=5)])
    resumed, _, usage = L.score_bank(second, gens, model="m", already_scored=scores)
    assert len(resumed) == len(scores)
    assert usage.calls == 0


def test_a_receipt_round_trips_its_scores(tmp_path: Path) -> None:
    judge = StubJudge([rating(primary="linear", linear=5)] * 2)
    scores, failed, usage = L.score_bank(judge, generations(2), model="judge-1")
    receipt = L.likert_receipt(
        model="judge-1", scores=scores, failed=failed, summary=L.summarize(scores), usage=usage
    )
    path = tmp_path / "judge_scores.json"
    path.write_text(json.dumps(receipt))
    assert L.read_scores(path) == scores
    assert receipt["rubric_donor"].endswith("run_judge_scoring.py")


# ── Summaries ─────────────────────────────────────────────────────────────────
def scored(primary_by_mode: dict[str, str]) -> list[L.LikertScore]:
    out = []
    for i, (intended, primary) in enumerate(primary_by_mode.items()):
        ratings = {m: 2 for m in VALID_MODES} | {primary: 5}
        out.append(
            L.LikertScore(
                generation_id=i, mode_intended=intended, topic=f"t{i}", ratings=ratings,
                primary_mode=primary, judge_confidence=4, reasoning="",
                mode_purity=L.purity(ratings, intended), correct=primary == intended,
            )
        )
    return out


def test_the_summary_diagonal_is_the_accuracy() -> None:
    summary = L.summarize(scored({m: m for m in VALID_MODES}))
    assert summary.overall_accuracy == 1.0
    matrix = np.array(summary.confusion_matrix)
    assert matrix.trace() == len(VALID_MODES)
    assert summary.confusion_labels == list(VALID_MODES)


def test_purity_is_higher_where_the_judge_was_right() -> None:
    mixed = scored({"socratic": "socratic", "linear": "analogical"})
    summary = L.summarize(mixed)
    assert summary.overall_accuracy == 0.5
    assert summary.mean_purity_when_correct > summary.mean_purity_when_incorrect


def test_a_summary_of_nothing_is_refused() -> None:
    with pytest.raises(L.JudgingError, match="no scores"):
        L.summarize([])


# ── The cross-channel readout ─────────────────────────────────────────────────
def write_signatures(sig_dir: Path, scores: list[L.LikertScore], *, aligned: bool) -> None:
    """One vector per score: aligned puts pure texts near their mode's centre."""
    sig_dir.mkdir(parents=True, exist_ok=True)
    centres = {m: np.eye(len(VALID_MODES))[i] * 4 for i, m in enumerate(VALID_MODES)}
    rng = np.random.default_rng(0)
    for score in scores:
        offset = (5 - score.mode_purity) if aligned else rng.normal(scale=3)
        vector = centres[score.mode_intended] + offset * rng.normal(size=len(VALID_MODES)) * 0.4
        np.savez(sig_dir / f"gen_{score.generation_id:03d}.npz", features=vector)


def many_scores(n_per_mode: int = 6) -> list[L.LikertScore]:
    out: list[L.LikertScore] = []
    gid = 0
    for mode in VALID_MODES:
        for k in range(n_per_mode):
            ratings = {m: 2 for m in VALID_MODES} | {mode: 2 + (k % 4)}
            out.append(
                L.LikertScore(
                    generation_id=gid, mode_intended=mode, topic=f"t{gid}", ratings=ratings,
                    primary_mode=mode, judge_confidence=4, reasoning="",
                    mode_purity=L.purity(ratings, mode), correct=True,
                )
            )
            gid += 1
    return out


def test_the_correlation_reads_purity_against_centroid_distance(tmp_path: Path) -> None:
    scores = many_scores()
    write_signatures(tmp_path, scores, aligned=True)
    result = L.purity_signature_correlation(scores, tmp_path)
    assert result is not None
    assert result.n_samples == len(scores)
    assert -1.0 <= result.purity_distance_correlation <= 1.0
    assert result.median_purity_split == pytest.approx(np.median([s.mode_purity for s in scores]))


def test_the_correlation_refuses_a_handful_of_points(tmp_path: Path) -> None:
    scores = many_scores(1)
    write_signatures(tmp_path, scores, aligned=True)
    assert L.purity_signature_correlation(scores, tmp_path, min_samples=50) is None
    assert L.purity_signature_correlation(scores, tmp_path / "empty") is None


def test_the_correlation_refuses_a_set_missing_a_mode(tmp_path: Path) -> None:
    scores = [s for s in many_scores() if s.mode_intended != "dialectical"]
    write_signatures(tmp_path, scores, aligned=True)
    assert L.purity_signature_correlation(scores, tmp_path) is None


def test_an_unreadable_signature_file_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    scores = many_scores()
    write_signatures(tmp_path, scores, aligned=True)
    (tmp_path / f"gen_{scores[0].generation_id:03d}.npz").write_bytes(b"not an npz")
    result = L.purity_signature_correlation(scores, tmp_path)
    assert result is not None and result.n_samples == len(scores) - 1


# ── Loading a bank ────────────────────────────────────────────────────────────
def test_core_only_keeps_one_generation_per_mode_topic_cell(tmp_path: Path) -> None:
    for gid, (mode, topic) in enumerate(
        [("linear", "rivers"), ("linear", "rivers"), ("socratic", "rivers")]
    ):
        (tmp_path / f"gen_{gid:03d}.json").write_text(
            json.dumps({"generation_id": gid, "mode": mode, "topic": topic,
                        "generated_text": "text"})
        )
    assert len(L.load_generations(tmp_path)) == 2
    assert len(L.load_generations(tmp_path, core_only=False)) == 3


def test_a_directory_with_no_generations_is_refused(tmp_path: Path) -> None:
    with pytest.raises(L.JudgingError, match="no gen_"):
        L.load_generations(tmp_path)
