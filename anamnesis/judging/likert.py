"""The Likert paradigm — rating a text on every mode at once, without a contrast.

The 2AFC harness asks which of two texts did something. This asks one text how
much of each of five modes it exhibits, and which one it is most. They are
different measurements and the difference is not a matter of taste:

* A forced choice measures whether a **difference between two texts** is visible.
  It cannot say how strongly either text expresses anything, because both sides
  of a pair could be faint and one still wins.
* A Likert rating measures how strongly **one text** reads as a category, on a
  scale the judge holds in its own head. It cannot control for the topic, the
  length or the judge's calibration drifting between texts, because there is no
  second text holding any of that constant.

The forced choice is the paradigm of record for hardening a claim, and this is
kept for what it uniquely yields: **purity**, the intended mode's rating minus
the mean of the other four, which is a graded per-text readout rather than a
per-contrast rate. Purity is what makes
:func:`purity_signature_correlation` sayable — whether the texts a judge reads as
purely in-mode are the texts sitting nearest their mode's centroid in signature
space. That is a cross-channel question, and a rate over pairs cannot ask it.

The judge is blind in the sense this paradigm allows: it sees the writing prompt
and the text and never the mode instruction. That is weaker than the 2AFC's
blinding, where the answer is not in the judge's context at all, and the
difference is why a Likert accuracy is reported as a rating rather than quoted as
a hardened result.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from anamnesis.judging.harness import JudgeBackend, JudgingError, Usage
from anamnesis.judging.prompts import (
    LIKERT_DONOR,
    VALID_MODES,
    likert_system_prompt,
    likert_user_message,
)

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3


class LikertRating(BaseModel):
    """One parsed judgement: five ratings, a classification, a confidence, a reason."""

    model_config = ConfigDict(extra="forbid")

    ratings: dict[str, int]
    primary_mode: str
    confidence: int
    reasoning: str = ""


class LikertScore(BaseModel):
    """One generation's judgement, joined back to what it was generated under."""

    model_config = ConfigDict(extra="forbid")

    generation_id: int
    mode_intended: str
    topic: str
    ratings: dict[str, int]
    primary_mode: str
    judge_confidence: int
    reasoning: str
    mode_purity: float
    correct: bool


class PerModeAccuracy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: float
    n: int
    n_correct: int


class LikertSummary(BaseModel):
    """What a set of ratings says overall, with the confusion it rests on."""

    model_config = ConfigDict(extra="forbid")

    n: int
    overall_accuracy: float
    per_mode_accuracy: dict[str, PerModeAccuracy] = Field(default_factory=dict)
    mean_purity: float
    std_purity: float
    mean_confidence: float
    confusion_matrix: list[list[int]]
    confusion_labels: list[str]
    mean_purity_when_correct: float
    mean_purity_when_incorrect: float


class PurityCorrelation(BaseModel):
    """Whether judged purity tracks distance from the mode's centroid in signature space."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int
    purity_distance_correlation: float
    high_purity_mean_distance: float
    low_purity_mean_distance: float
    median_purity_split: float


def parse_likert_reply(raw: str) -> LikertRating | None:
    """Read a judgement out of a reply, repairing the one truncation that recurs.

    Markdown fencing is stripped. A reply cut off inside its ``reasoning`` field
    is repaired by closing the string, because the ratings — the part that is
    scored — are already complete by then and discarding them would lose a good
    judgement over a token budget. Everything else that does not parse, or parses
    into a rating outside one to five, or into a mode outside the five, returns
    None: an unreadable judgement is a missing datum, not a zero.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[cleaned.index("\n") + 1:] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        marker = '"reasoning"'
        index = cleaned.find(marker)
        if index < 0:
            return None
        head = cleaned[:index + len(marker)]
        try:
            data = json.loads(head + ': "(truncated)"}')
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    ratings = data.get("ratings")
    if not isinstance(ratings, dict):
        return None
    for mode in VALID_MODES:
        value = ratings.get(mode)
        if not isinstance(value, (int, float)) or not (1 <= value <= 5):
            return None
    primary = data.get("primary_mode")
    if primary not in VALID_MODES:
        return None
    confidence = data.get("confidence")
    if confidence is None:
        confidence = 3
    if not isinstance(confidence, (int, float)) or not (1 <= confidence <= 5):
        return None
    reasoning = data.get("reasoning", "")
    return LikertRating(
        ratings={m: int(ratings[m]) for m in VALID_MODES},
        primary_mode=str(primary),
        confidence=int(confidence),
        reasoning=reasoning if isinstance(reasoning, str) else str(reasoning),
    )


def score_text(
    backend: JudgeBackend,
    *,
    model: str,
    topic: str,
    text: str,
    retries: int = DEFAULT_RETRIES,
) -> tuple[LikertRating | None, Usage]:
    """Ask one judge for one text's ratings, retrying the same model on a bad read.

    The retry stays on the same model rather than falling to another, which is the
    opposite of the 2AFC ladder's rule and is right for the opposite reason: a
    Likert number is a reading on one judge's internal scale, and mixing two
    judges' scales inside one table would make the ratings incomparable. A 2AFC
    answer is a choice between two texts and carries no scale, so a fallback there
    costs nothing.
    """
    system = likert_system_prompt()
    user = likert_user_message(topic, text)
    usage = Usage()
    for attempt in range(retries):
        raw = backend.complete(model=model, system=system, user=user, schema=None)
        usage.add(raw.usage)
        if raw.error is not None or raw.text is None:
            logger.warning("judge call failed (attempt %d/%d): %s", attempt + 1, retries, raw.error)
            continue
        parsed = parse_likert_reply(raw.text)
        if parsed is not None:
            return parsed, usage
        logger.warning(
            "unreadable judgement (attempt %d/%d): %s", attempt + 1, retries, raw.text[:200]
        )
    return None, usage


def purity(ratings: Mapping[str, int], intended: str) -> float:
    """The intended mode's rating minus the mean of the other four.

    A text rated 5 on its own mode and 5 on every other mode is not a pure
    example of anything, and a raw rating cannot tell that from a 5 that stands
    alone. The difference can be negative, which is the case where the judge read
    the text as more of something else than of what it was asked for.
    """
    if intended not in ratings:
        raise JudgingError(f"no rating for the intended mode {intended!r}")
    others = [ratings[m] for m in VALID_MODES if m != intended]
    return round(ratings[intended] - float(np.mean(others)), 3)


def load_generations(sig_dir: Path | str, *, core_only: bool = True) -> list[dict[str, Any]]:
    """Read a run's per-generation metadata, optionally one per mode-topic cell.

    ``core_only`` keeps the first generation of each mode-topic pair. That makes
    the judged set a design rather than a sample: every cell contributes once, so
    a mode with more generations banked does not weight the accuracy.
    """
    sig_dir = Path(sig_dir)
    files = sorted(sig_dir.glob("gen_*.json"))
    if not files:
        raise JudgingError(f"no gen_*.json files in {sig_dir}")
    generations: list[dict[str, Any]] = []
    for path in files:
        meta = json.loads(path.read_text(encoding="utf-8"))
        generations.append(
            {
                "generation_id": int(meta["generation_id"]),
                "mode": meta["mode"],
                "topic": meta["topic"],
                "generated_text": meta.get("generated_text", ""),
                "file_stem": path.stem,
            }
        )
    if core_only:
        seen: set[tuple[str, str]] = set()
        kept = []
        for gen in generations:
            cell = (gen["mode"], gen["topic"])
            if cell not in seen:
                seen.add(cell)
                kept.append(gen)
        generations = kept
    return generations


def score_bank(
    backend: JudgeBackend,
    generations: Sequence[Mapping[str, Any]],
    *,
    model: str,
    already_scored: Sequence[LikertScore] = (),
    retries: int = DEFAULT_RETRIES,
    on_progress: Callable[[Sequence[LikertScore], Sequence[int]], None] | None = None,
) -> tuple[list[LikertScore], list[int], Usage]:
    """Score a bank, skipping what is already scored and naming what failed.

    Resume is by generation id against ``already_scored``, which is read from a
    partial receipt. Failures are returned as ids rather than dropped, so a rerun
    knows what to retry and a reader knows the denominator.
    """
    scores = list(already_scored)
    done = {s.generation_id for s in scores}
    failed: list[int] = []
    usage = Usage()
    for gen in generations:
        gid = int(gen["generation_id"])
        if gid in done:
            continue
        text = gen.get("generated_text", "")
        if not text:
            failed.append(gid)
            continue
        rating, call_usage = score_text(
            backend, model=model, topic=gen["topic"], text=text, retries=retries
        )
        usage.add(call_usage)
        if rating is None:
            failed.append(gid)
            continue
        mode = str(gen["mode"])
        scores.append(
            LikertScore(
                generation_id=gid,
                mode_intended=mode,
                topic=str(gen["topic"]),
                ratings=rating.ratings,
                primary_mode=rating.primary_mode,
                judge_confidence=rating.confidence,
                reasoning=rating.reasoning,
                mode_purity=purity(rating.ratings, mode),
                correct=rating.primary_mode == mode,
            )
        )
        if on_progress is not None:
            on_progress(scores, failed)
    return scores, failed, usage


def summarize(scores: Sequence[LikertScore]) -> LikertSummary:
    """Accuracy, purity and the confusion matrix the accuracy is a diagonal of."""
    if not scores:
        raise JudgingError("no scores to summarize")
    correct = sum(1 for s in scores if s.correct)
    per_mode: dict[str, PerModeAccuracy] = {}
    for mode in VALID_MODES:
        subset = [s for s in scores if s.mode_intended == mode]
        if subset:
            hits = sum(1 for s in subset if s.correct)
            per_mode[mode] = PerModeAccuracy(
                accuracy=hits / len(subset), n=len(subset), n_correct=hits
            )
    confusion = np.zeros((len(VALID_MODES), len(VALID_MODES)), dtype=int)
    for s in scores:
        confusion[VALID_MODES.index(s.mode_intended), VALID_MODES.index(s.primary_mode)] += 1
    purities = [s.mode_purity for s in scores]
    right = [s.mode_purity for s in scores if s.correct]
    wrong = [s.mode_purity for s in scores if not s.correct]
    return LikertSummary(
        n=len(scores),
        overall_accuracy=correct / len(scores),
        per_mode_accuracy=per_mode,
        mean_purity=float(np.mean(purities)),
        std_purity=float(np.std(purities)),
        mean_confidence=float(np.mean([s.judge_confidence for s in scores])),
        confusion_matrix=confusion.tolist(),
        confusion_labels=list(VALID_MODES),
        mean_purity_when_correct=float(np.mean(right)) if right else 0.0,
        mean_purity_when_incorrect=float(np.mean(wrong)) if wrong else 0.0,
    )


def purity_signature_correlation(
    scores: Sequence[LikertScore],
    sig_dir: Path | str,
    *,
    min_samples: int = 10,
) -> PurityCorrelation | None:
    """Correlate judged purity with cosine distance to the intended mode's centroid.

    This is the cross-channel readout: the text channel says how purely a
    generation reads, the signature channel says how far it sits from its mode's
    centre, and the correlation asks whether they are looking at the same thing.
    The centroids are computed over the judged set itself after standardizing, so
    the distances are within-set and mean nothing outside it.

    Returns None rather than a number where fewer than ``min_samples`` generations
    have signatures on disk, or where a mode is missing entirely: a correlation
    over a handful of points is noise with a decimal point.
    """
    sig_dir = Path(sig_dir)
    features: dict[int, np.ndarray] = {}
    for score in scores:
        path = sig_dir / f"gen_{score.generation_id:03d}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path) as data:
                features[score.generation_id] = data["features"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("could not read %s: %s", path, exc)
    usable = [s for s in scores if s.generation_id in features]
    if len(usable) < min_samples:
        return None

    matrix = np.stack([features[s.generation_id] for s in usable])
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(matrix)
    modes = [s.mode_intended for s in usable]
    purities = np.array([s.mode_purity for s in usable])

    centroids: dict[str, np.ndarray] = {}
    for mode in VALID_MODES:
        mask = np.array([m == mode for m in modes])
        if mask.any():
            centroids[mode] = scaled[mask].mean(axis=0)
    if len(centroids) < len(VALID_MODES):
        return None

    distances = np.array(
        [
            float(cdist(scaled[i].reshape(1, -1), centroids[s.mode_intended].reshape(1, -1),
                        metric="cosine")[0, 0])
            for i, s in enumerate(usable)
        ]
    )
    if np.std(purities) < 1e-10 or np.std(distances) < 1e-10:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(purities, distances)[0, 1])
    median = float(np.median(purities))
    high = purities >= median
    low = purities < median
    return PurityCorrelation(
        n_samples=len(usable),
        purity_distance_correlation=correlation,
        high_purity_mean_distance=float(distances[high].mean()) if high.any() else 0.0,
        low_purity_mean_distance=float(distances[low].mean()) if low.any() else 0.0,
        median_purity_split=median,
    )


def likert_receipt(
    *,
    model: str,
    scores: Sequence[LikertScore],
    failed: Sequence[int] = (),
    summary: LikertSummary | None = None,
    correlation: PurityCorrelation | None = None,
    usage: Usage | None = None,
) -> dict[str, Any]:
    """The banked record of a Likert pass, including what it could not read."""
    return {
        "paradigm": "likert (five dimensions rated 1-5 plus a primary classification)",
        "rubric_donor": LIKERT_DONOR,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_scored": len(scores),
        "n_failed": len(failed),
        "failed_generation_ids": list(failed),
        "scores": [s.model_dump() for s in scores],
        "summary": summary.model_dump() if summary else None,
        "purity_signature_correlation": correlation.model_dump() if correlation else None,
        "usage": (usage or Usage()).model_dump(),
    }


def read_scores(path: Path | str) -> list[LikertScore]:
    """Read the scores out of a partial receipt, for a resumed pass."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LikertScore(**row) for row in payload.get("scores", [])]


__all__ = [
    "DEFAULT_RETRIES",
    "LikertRating",
    "LikertScore",
    "LikertSummary",
    "PerModeAccuracy",
    "PurityCorrelation",
    "likert_receipt",
    "load_generations",
    "parse_likert_reply",
    "purity",
    "purity_signature_correlation",
    "read_scores",
    "score_bank",
    "score_text",
    "summarize",
]
