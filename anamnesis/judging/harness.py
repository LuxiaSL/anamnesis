"""The blind 2AFC judge harness — the behavioural channel, contrast-hardened.

A signature says how a forward pass ran. A judge says what a reader sees. The
two are different channels and the second one is only worth reporting if it was
taken under a discipline, because a judge is a measuring instrument with a
dynamic range, a grade, and a way of being fooled.

**Two-alternative forced choice is the paradigm of record.** Asking a judge to
name a text's mode without a comparison measures how nameable the category is;
asking which of two same-topic texts followed a procedure measures whether the
difference is there to be seen. A blind k-way judge-blindness result turned out
to be an artifact of the k-way task rather than a fact about the texts, and the
answer to it is not a better k-way prompt: it is the contrast. So the pair is the
unit here, and a claim that is not contrast-hardened is not quoted.

Five controls are structural rather than optional, and the harness is built so
that skipping one takes deliberate effort:

**The key never reaches the judge's context.** :class:`BlindPair` — the only type
a prompt is rendered from — has no field that says which side is which. The
answer lives in :class:`PairKey`, in a separate collection, written to a separate
file, and :meth:`BlindPacket.write` refuses to put both at one path. This is the
strongest of the donors' three guarantees: a separate key file makes the mistake
recoverable, a type with no key field makes it unavailable.

**Dynamic range is measured, not assumed** (:func:`interpret_with_ceiling`). A
null is uninterpretable until the ruler has shown it can separate something. A
weak judge can hide an effect but cannot manufacture a dose-ordered positive, so
a positive stands on its own and a null does not: it stands on a ceiling contrast
run beside it and read first.

**Grade is a measured axis** (:func:`run_reader_ladder`). Expression tracks reader
grade — an effect caught by every strong rater and missed at exact chance by a
small one is a real effect with a stated audience. Running a ladder turns that
from an assumption into a number, and a pass reports the grade it passes at.

**The criterion is not the scorer** (:func:`assert_not_circular`). A judge told to
look for the same features a marker battery counts is the battery wearing a
second hat, and the two columns of the resulting table are one column. Every
prompt set declares where its criterion came from, and a study declares what
instrument scores it; the harness refuses the pairing rather than trusting the
operator to notice.

**A run with zero successful calls is a non-run** (:attr:`ContrastResult.is_non_run`).
Silent authentication failures have produced finished jobs holding no evidence.
Zero scored pairs is not a null result; it is an absence of result, and it is
reported under its own name.

Around those: Wilson intervals on every win rate (a rate without an interval at
n=40 is a rumour), a one-sided binomial p against chance, the coherence gate that
separates an in-window shift from a collapse, and a model fallback ladder in
which an unparseable reply escalates exactly as a transport error does.

Concurrency does not touch any of it. Every random draw happens in the main
thread before a single call is dispatched, so a packet, its key and its blinding
are identical at one worker and at sixteen; the pool only fetches answers, and
``map`` returns them in the order the pairs were drawn.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.analysis.battery.text_decode import maybe_decode
from anamnesis.judging.prompts import (
    ANNEX_TEMPLATE,
    COHERENCE_PROMPT,
    DIGIT_1_5_PATTERN,
    LETTER_PATTERN,
    PromptSet,
    RenderedPrompt,
)

Side = Literal["A", "B"]
Verdict = Literal["positive", "null", "uninterpretable", "non-run"]

DEFAULT_MAX_CHARS = 2200
DEFAULT_QUALIFY_THRESHOLD = 0.70
DEFAULT_MIN_PAIRS = 8
DEFAULT_WORKERS = 8

# `effort` is not accepted on every model: it errors rather than being ignored,
# and sending it anyway has cost a whole rater arm — every call failed and
# nothing was scored. Structured outputs are fine on all of them; only the knob
# is gated. (lesson, banked with the c5 install-lever panel)
NO_EFFORT_MODELS: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-4-5")

# Status codes worth waiting out rather than falling to a weaker model for.
TRANSIENT_STATUS: frozenset[int] = frozenset({429, 500, 529})

BLINDING_LAW = (
    "blind 2AFC: position randomized per pair by a deterministic RNG drawn in the "
    "main thread before dispatch; the key is held in a separate file and never "
    "enters judge context, and the judge never sees the arm labels."
)


class JudgingError(RuntimeError):
    """A condition that makes a judged number meaningless rather than small."""


class CircularityError(JudgingError):
    """The judge's criterion and the instrument scoring it are the same source."""


class ReconstructionMismatch(JudgingError):
    """A rebuilt packet does not match the banked key it claims to reproduce."""


# ── Corpora ───────────────────────────────────────────────────────────────────
class Corpus(BaseModel):
    """Texts grouped by the thing a pair must hold constant, with display labels.

    The group is almost always the topic: pairing across topics measures what was
    written about rather than how, which is the confound the whole instrument
    exists to avoid. Keys are strings so a corpus round-trips through JSON
    unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    by_group: dict[str, list[str]] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_group))

    def label(self, group: str) -> str:
        return self.labels.get(group, group)

    def extend(self, other: Corpus) -> None:
        """Merge another corpus in, as several baseline cells pool into one rider set."""
        for group, texts in other.by_group.items():
            self.by_group.setdefault(group, []).extend(texts)
        self.labels.update(other.labels)


def _generations(meta_path: Path | str) -> list[dict[str, Any]]:
    """The generation list of a bank's metadata, wrapped or bare."""
    md = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return md["generations"] if isinstance(md, dict) and "generations" in md else md


def texts_by_topic(
    meta_path: Path | str,
    *,
    min_words: int = 20,
    decode: bool = True,
) -> Corpus:
    """Load a generation bank grouped by ``topic_idx``.

    ``min_words`` drops the near-empty generations, because a floor-yield beside a
    paragraph is a length tell rather than a style tell. ``decode`` runs the
    byte-BPE repair a tokenizer-level bank needs; without it a judge reads
    re-tokenized garbage and answers about that.
    """
    corpus = Corpus()
    for gen in _generations(meta_path):
        text = gen.get("generated_text", "")
        if decode:
            text = maybe_decode(text)
        text = text.strip()
        if len(text.split()) < min_words:
            continue
        group = str(gen["topic_idx"])
        corpus.by_group.setdefault(group, []).append(text)
        topic = gen.get("topic")
        if isinstance(topic, str):
            corpus.labels.setdefault(group, topic)
    return corpus


def texts_by_prompt(
    path: Path | str,
    prompt_ids: Sequence[str],
    *,
    min_chars: int = 60,
) -> Corpus:
    """Load a prompt-keyed bank: a ``.jsonl`` with a ``prompt_id`` per row, or a
    flat JSON list of texts written in prompt order.

    A flat list carries its grouping only in its length, so a count that is not a
    multiple of the prompt count is refused rather than split on a guess.
    """
    path = Path(path)
    corpus = Corpus()
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            corpus.by_group.setdefault(str(row["prompt_id"]), []).append(text)
        return corpus
    texts = json.loads(path.read_text(encoding="utf-8"))
    if not prompt_ids:
        raise JudgingError(f"{path}: a flat text list needs the prompt ids it was written in order of")
    if len(texts) % len(prompt_ids) != 0:
        raise JudgingError(
            f"{path}: {len(texts)} texts is not a multiple of {len(prompt_ids)} prompts — "
            "the prompt grouping of a flat file cannot be inferred"
        )
    per = len(texts) // len(prompt_ids)
    for i, pid in enumerate(prompt_ids):
        kept = [t.strip() for t in texts[i * per:(i + 1) * per] if len(t.strip()) >= min_chars]
        if kept:
            corpus.by_group[str(pid)] = kept
    return corpus


# ── Blinding ──────────────────────────────────────────────────────────────────
class BlindPair(BaseModel):
    """What a judge sees: two texts and, where the prompt asks for one, a topic.

    There is no field here that says which side is which. That is the blinding
    guarantee, and it is a property of the type rather than of the loop that
    builds it: :meth:`PromptSet.render` is reached from this object, so the
    side labels are not in scope at the point the text is assembled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    first: str
    second: str
    topic: str | None = None


class PairKey(BaseModel):
    """The answer to one pair, held apart from it.

    ``labels`` carries the arm bookkeeping — which cell, which distractor source —
    so a result can be decomposed afterwards without any of it having been in the
    judge's context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    target_side: Side
    group: str
    labels: dict[str, str] = Field(default_factory=dict)


class BlindPacket(BaseModel):
    """A drawn contrast: the judge-facing pairs, their keys, and how they were drawn."""

    model_config = ConfigDict(extra="forbid")

    contrast: str
    pairs: list[BlindPair]
    keys: dict[str, PairKey]
    seed: int | None = None
    question: str | None = None
    law: str = BLINDING_LAW

    def key_of(self, pair_id: str) -> PairKey:
        try:
            return self.keys[pair_id]
        except KeyError:
            raise JudgingError(f"no key for pair {pair_id!r} in contrast {self.contrast!r}") from None

    def digest(self) -> str:
        """A hash over the judge-facing content, so two passes can be shown to
        have read the same items without comparing the texts by eye."""
        payload = "\x1f".join(
            f"{p.pair_id}\x00{p.topic or ''}\x00{p.first}\x00{p.second}" for p in self.pairs
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write(self, *, packet_path: Path | str, key_path: Path | str) -> None:
        """Bank the packet and its key at two paths, refusing one path for both."""
        packet_path, key_path = Path(packet_path), Path(key_path)
        if packet_path.resolve() == key_path.resolve():
            raise JudgingError(
                "the key must be written apart from the packet; one path was given for both"
            )
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(
            json.dumps(
                {
                    "contrast": self.contrast,
                    "question": self.question,
                    "law": self.law,
                    "digest": self.digest(),
                    "pairs": [p.model_dump() for p in self.pairs],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        key_path.write_text(
            json.dumps(
                {
                    "contrast": self.contrast,
                    "seed": self.seed,
                    "digest": self.digest(),
                    "keys": {k: v.model_dump() for k, v in self.keys.items()},
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    @classmethod
    def read(cls, *, packet_path: Path | str, key_path: Path | str) -> BlindPacket:
        """Read back a banked packet, refusing a key that belongs to other pairs."""
        packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
        key = json.loads(Path(key_path).read_text(encoding="utf-8"))
        if packet.get("digest") != key.get("digest"):
            raise ReconstructionMismatch(
                f"key digest {key.get('digest')!r} does not match packet digest "
                f"{packet.get('digest')!r} — the key belongs to different pairs"
            )
        return cls(
            contrast=str(packet["contrast"]),
            question=packet.get("question"),
            law=str(packet.get("law", BLINDING_LAW)),
            seed=key.get("seed"),
            pairs=[BlindPair(**p) for p in packet["pairs"]],
            keys={k: PairKey(**v) for k, v in key["keys"].items()},
        )


def draw_pairs(
    *,
    contrast: str,
    target: Corpus,
    distractors: Mapping[str, Corpus],
    n_pairs: int,
    seed: int,
    per_group: int | None = None,
    prefix: str | None = None,
    min_chars: int = 0,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    labels: Mapping[str, str] | None = None,
) -> BlindPacket:
    """Draw a blinded contrast: target texts against same-group distractors.

    The construction is the one the steering and census tables were banked under:
    walk the groups both sides share, take up to ``per_group`` target texts from
    each, sample one distractor from the same group for each of them, shuffle,
    cut to ``n_pairs``, and randomize each pair's position. ``per_group`` defaults
    to the ceiling of ``n_pairs`` over the shared groups, floored at two, so a
    bank with more generations per topic actually yields more pairs.

    ``distractors`` is a mapping of *named sources* cycled across the candidate
    list. One source is the pole-versus-pole or baseline-rider case; several is
    target-versus-any-other, and the source each pair drew from is recorded in its
    key so the result decomposes by distractor without the judge having seen a
    source name.

    Every draw happens here, in order, from one seeded generator. Nothing after
    this function is random, which is what makes the key and the packet
    reproducible from the seed and independent of the worker count.
    """
    if not distractors:
        raise JudgingError(f"{contrast}: no distractor source given")
    if n_pairs < 1:
        raise JudgingError(f"{contrast}: n_pairs must be at least 1")
    source_names = sorted(distractors)
    shared = sorted(
        set(target.by_group)
        & {g for name in source_names for g in distractors[name].by_group}
    )
    if not shared:
        raise JudgingError(
            f"{contrast}: target and distractors share no group — pairing across groups "
            "would measure the topic rather than the contrast"
        )
    cap = per_group if per_group is not None else max(2, -(-n_pairs // len(shared)))

    rng = random.Random(seed)
    candidates: list[tuple[str, str, str, str]] = []  # group, target text, distractor text, source
    cycle = 0
    for group in shared:
        available = [t for t in target.by_group.get(group, []) if len(t) >= min_chars]
        for text in available[:cap]:
            source = source_names[cycle % len(source_names)]
            cycle += 1
            pool = [t for t in distractors[source].by_group.get(group, []) if len(t) >= min_chars]
            if not pool:
                continue
            candidates.append((group, text, rng.choice(pool), source))
    rng.shuffle(candidates)
    candidates = candidates[:n_pairs]
    if len(candidates) < min_pairs:
        raise JudgingError(
            f"{contrast}: only {len(candidates)} usable pairs (floor {min_pairs}) — "
            "a contrast this thin has no interval worth reading"
        )

    stem = prefix if prefix is not None else contrast
    pairs: list[BlindPair] = []
    keys: dict[str, PairKey] = {}
    for i, (group, target_text, distractor_text, source) in enumerate(candidates):
        pair_id = f"{stem}-{i:03d}"
        target_is_a = rng.random() < 0.5
        first, second = (
            (target_text, distractor_text) if target_is_a else (distractor_text, target_text)
        )
        pairs.append(
            BlindPair(
                pair_id=pair_id,
                first=first,
                second=second,
                topic=target.label(group),
            )
        )
        keys[pair_id] = PairKey(
            pair_id=pair_id,
            target_side="A" if target_is_a else "B",
            group=group,
            labels={"distractor": source, **(dict(labels) if labels else {})},
        )
    return BlindPacket(contrast=contrast, pairs=pairs, keys=keys, seed=seed)


def sample_texts(corpus: Corpus, n: int, *, seed: int) -> list[str]:
    """Draw texts for the single-text gate: flattened across groups, then shuffled.

    The gate is a property of a cell rather than of a pair, so its sample is not
    grouped. It is still drawn from a seeded generator in the main thread, for the
    same reason everything else here is: a gate that cannot be redrawn is a number
    nobody can check.
    """
    flat = [text for group in sorted(corpus.by_group) for text in corpus.by_group[group]]
    random.Random(seed).shuffle(flat)
    return flat[:n]


def verify_against_key(packet: BlindPacket, banked_key: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a reconstruction that does not match a banked key, before any call.

    A second pass over an existing table is only a second pass if it reads the
    same items under the same blinding. Where the first pass banked its packet,
    re-reading it is enough; where only a key survives, a rebuilt packet is
    checked against that key here — pair for pair, side for side — and a mismatch
    stops the run rather than producing a table that looks comparable.
    """
    mine = {pid: key.target_side for pid, key in packet.keys.items()}
    theirs: dict[str, str] = {}
    for pid, row in banked_key.items():
        side = row.get("target_side") or row.get("steered_is")
        if side is None:
            raise ReconstructionMismatch(f"banked key row {pid!r} states no side")
        theirs[str(pid)] = str(side)
    if mine == theirs:
        return
    only_mine = sorted(set(mine) - set(theirs))
    only_theirs = sorted(set(theirs) - set(mine))
    differing = sorted(p for p in set(mine) & set(theirs) if mine[p] != theirs[p])
    raise ReconstructionMismatch(
        f"reconstruction does not reproduce the banked pass: +{len(only_mine)} extra, "
        f"-{len(only_theirs)} missing, {len(differing)} sides differ "
        f"(first differing: {differing[:3]}) — not judging"
    )


_PAIR_HEADING = re.compile(r"^## PAIR (\S+?)\s*(?:\(topic: (.*?)\))?\s*$")


def _pair_order(pair_id: str) -> tuple[int, int | str]:
    """Numeric ids sort numerically; anything else sorts after them, as text.

    The record's sheets number their pairs, and a numeric sort is what reproduces
    their order. A packet banked by this harness carries its contrast in the id,
    which sorts as text without changing what a numeric sheet does.
    """
    return (0, int(pair_id)) if pair_id.isdigit() else (1, pair_id)


def parse_pairs_md(markdown: str) -> tuple[str, dict[str, dict[str, str]]]:
    """Read a banked key-free pair sheet: its question, and its pairs by id.

    This is the on-disk form the record's judge sessions read, and reading it is
    how a second judge family re-judges those tables without reconstructing
    anything: the items are the items the first family saw.
    """
    lines = markdown.splitlines()
    question = ""
    for line in lines:
        if line.startswith("# ") and "A/B" in line:
            question = line.lstrip("# ").strip()
            break
    pairs: dict[str, dict[str, Any]] = {}
    current: str | None = None
    side: str | None = None
    for line in lines:
        heading = _PAIR_HEADING.match(line)
        if heading:
            current = heading.group(1)
            pairs[current] = {"topic": heading.group(2), "A": [], "B": []}
            side = None
            continue
        if line.strip() in ("### A", "### B"):
            side = line.strip()[-1]
            continue
        if current and side and not line.startswith("## "):
            pairs[current][side].append(line)
    for pair in pairs.values():
        pair["A"] = "\n".join(pair["A"]).strip()
        pair["B"] = "\n".join(pair["B"]).strip()
    if not question or not pairs:
        raise JudgingError("pair sheet parse failure: no A/B question heading, or no pairs")
    return question, pairs


def packet_from_pairs_md(
    markdown: str,
    *,
    contrast: str,
    key: Mapping[str, Mapping[str, Any]] | None = None,
) -> BlindPacket:
    """Build a packet from a banked pair sheet, carrying its own question.

    Rows whose key states no side are carried as pairs without keys: they are
    judged and reported, and they contribute to agreement between families, but
    they cannot contribute to a win rate, which needs an answer.
    """
    question, parsed = parse_pairs_md(markdown)
    pairs: list[BlindPair] = []
    keys: dict[str, PairKey] = {}
    for pid in sorted(parsed, key=_pair_order):
        pairs.append(
            BlindPair(
                pair_id=pid,
                first=parsed[pid]["A"],
                second=parsed[pid]["B"],
                topic=parsed[pid].get("topic"),
            )
        )
        row = (key or {}).get(pid, {})
        side = row.get("target_side") or row.get("steered")
        if side in ("A", "B"):
            keys[pid] = PairKey(
                pair_id=pid,
                target_side=side,
                group=str(parsed[pid].get("topic") or pid),
                labels={k: str(v) for k, v in row.items() if k not in ("steered", "target_side")},
            )
    return BlindPacket(contrast=contrast, pairs=pairs, keys=keys, question=question)


def write_pairs_md(packet: BlindPacket, path: Path | str, *, question: str) -> None:
    """Bank a packet in the key-free sheet form, so a later pass can re-read it."""
    out = [f"# {question}", ""]
    for pair in packet.pairs:
        topic = f" (topic: {pair.topic})" if pair.topic else ""
        out += [f"## PAIR {pair.pair_id}{topic}", "", "### A", pair.first, "", "### B", pair.second, ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


# ── Backends ──────────────────────────────────────────────────────────────────
class Usage(BaseModel):
    """What a pass cost, and how often it fell down the ladder."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fallbacks: int = 0
    errors: int = 0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.fallbacks += other.fallbacks
        self.errors += other.errors


class RawReply(BaseModel):
    """One completion, before anyone tries to read an answer out of it."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    model: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)


def _client_library(module: str) -> Any:
    """Import a provider client on first use, naming the extra that installs it.

    Judging is the one layer that talks to a vendor, so its two libraries are an
    extra rather than a dependency. A process that never judges never imports
    them, and a process that tries to judge without them is told what to install
    rather than shown an import error from three frames down.
    """
    import importlib  # noqa: PLC0415 — deferred with the library it loads

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise JudgingError(
            f"the judging layer needs {module!r}, which ships in the 'judge' extra: "
            "install anamnesis[judge]"
        ) from exc


class JudgeBackend(Protocol):
    """A transport that completes one turn on one named model.

    The ladder does not live here. A backend retries what is worth waiting out and
    reports everything else as an error; which model to try next, and whether an
    unparseable reply counts as a failure, are the harness's decisions because
    they are scientific ones.
    """

    family: str

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        schema: Mapping[str, Any] | None = None,
    ) -> RawReply: ...


class AnthropicBackend:
    """The first judge family, over the Anthropic SDK.

    The client is constructed on first use, from the environment and never from an
    argument, so a key cannot reach a receipt through this object. The SDK import
    is deferred for the same reason the donors deferred it: judging is an extra,
    and the rest of the package imports without it.
    """

    family = "anthropic"

    def __init__(self, *, retries: int = 3, backoff_seconds: float = 2.0) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._client: Any | None = None

    @staticmethod
    def api_key_present() -> bool:
        """Whether an API key is in the environment, without reading its value."""
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def client(self) -> Any:
        if self._client is None:
            if not self.api_key_present():
                raise JudgingError(
                    "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment; "
                    "the judge reads its key from the environment and from nowhere else"
                )
            self._client = _client_library("anthropic").Anthropic()
        return self._client

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        schema: Mapping[str, Any] | None = None,
    ) -> RawReply:
        client = self.client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": user}],
        }
        if system is not None:
            kwargs["system"] = system
        if schema is not None:
            config: dict[str, Any] = {"format": {"type": "json_schema", "schema": dict(schema)}}
            if not model.startswith(NO_EFFORT_MODELS):
                config["effort"] = "low"
            kwargs["output_config"] = config
        last = "no attempt made"
        for attempt in range(self.retries):
            try:
                response = client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 — one bad call must not kill the pass
                last = f"{type(exc).__name__}: {exc}"
                status = getattr(exc, "status_code", None)
                if status in TRANSIENT_STATUS and attempt < self.retries - 1:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                return RawReply(model=model, error=last, usage=Usage(errors=1))
            usage = Usage(
                calls=1,
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
            )
            stop = getattr(response, "stop_reason", None)
            # Check the stop reason before touching content: a refusal yields empty
            # or partial content, and indexing it would raise.
            if stop == "refusal":
                return RawReply(model=model, error="refusal", stop_reason=stop, usage=usage)
            text = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"), None
            )
            if text is None:
                return RawReply(
                    model=model, error=f"no text block (stop_reason={stop})",
                    stop_reason=stop, usage=usage,
                )
            return RawReply(text=text, model=model, stop_reason=stop, usage=usage)
        return RawReply(model=model, error=last, usage=Usage(errors=1))


class OpenRouterBackend:
    """The second judge family, over OpenRouter's chat completions.

    A second family is how a judged table is checked for being an artifact of one
    model's taste. It is a different transport and a different vendor on purpose;
    what it shares with the first family is the ladder contract and nothing else.
    """

    family = "openrouter"
    url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, retries: int = 2, backoff_seconds: float = 2.0, timeout: float = 180.0) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout

    @staticmethod
    def api_key_present() -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def _key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise JudgingError(
                "no OPENROUTER_API_KEY in the environment; the second judge family reads "
                "its key from the environment and from nowhere else"
            )
        return key

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        schema: Mapping[str, Any] | None = None,
    ) -> RawReply:
        key = self._key()
        requests = _client_library("requests")
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        last = "no attempt made"
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.url,
                    timeout=self.timeout,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "max_tokens": 2000, "messages": messages},
                )
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise RuntimeError(str(payload["error"])[:120])
            except Exception as exc:  # noqa: BLE001 — same contract as the first family
                last = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                return RawReply(model=model, error=last, usage=Usage(errors=1))
            counts = payload.get("usage", {})
            usage = Usage(
                calls=1,
                input_tokens=int(counts.get("prompt_tokens", 0)),
                output_tokens=int(counts.get("completion_tokens", 0)),
            )
            text = payload["choices"][0]["message"].get("content")
            return RawReply(text=text, model=model, usage=usage)
        return RawReply(model=model, error=last, usage=Usage(errors=1))


# ── Reading a reply ───────────────────────────────────────────────────────────
class JudgeAnswer(BaseModel):
    """What one turn yielded: an answer, or a reason there is none."""

    model_config = ConfigDict(extra="forbid")

    answer: str | None = None
    model: str | None = None
    confidence: int | None = None
    tell: str | None = None
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)


def read_letter(text: str, pattern: str = LETTER_PATTERN) -> str | None:
    """Scan the reply for a bare letter, upper-cased first.

    The scan rather than an equality test is deliberate: a reasoning-leading model
    answers with a paragraph and its letter inside it, and rejecting that would
    throw away a good judgement over a formatting habit.
    """
    match = re.search(pattern, text.strip().upper())
    return match.group(1) if match else None


def read_json_choice(text: str) -> tuple[str | None, int | None, str | None]:
    """Parse a one-object JSON reply into choice, confidence and tell.

    Markdown fencing is stripped the way the donors stripped it. Numeric bounds
    stated in a schema are not enforced by the provider, so a confidence outside
    one to five is clamped on read rather than trusted.
    """
    cleaned = text.strip().strip("`").lstrip("json").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(payload, dict):
        return None, None, None
    choice = payload.get("choice")
    choice = choice if choice in ("A", "B") else None
    raw_confidence = payload.get("confidence")
    confidence: int | None = None
    if isinstance(raw_confidence, (int, float)):
        confidence = max(1, min(5, int(raw_confidence)))
    tell = payload.get("tell")
    return choice, confidence, (tell if isinstance(tell, str) else None)


def ask_with_ladder(
    backend: JudgeBackend,
    prompt: RenderedPrompt,
    *,
    models: Sequence[str],
    answer: Literal["letter", "json_choice", "digit"],
    schema: Mapping[str, Any] | None = None,
) -> JudgeAnswer:
    """Ask each model in turn until one answers, and say which one did.

    An unparseable reply escalates exactly as a transport error does. That is the
    donors' shared convention and it is the right one: a judge that cannot be read
    has not judged, and treating its silence as a failed call rather than as a
    missing datum keeps the fallback count honest.
    """
    if not models:
        raise JudgingError("a judge ladder needs at least one model")
    usage = Usage()
    last_error = "no attempt made"
    for index, model in enumerate(models):
        raw = backend.complete(model=model, system=prompt.system, user=prompt.user, schema=schema)
        usage.add(raw.usage)
        if index > 0:
            usage.fallbacks += 1
        if raw.error is not None or raw.text is None:
            last_error = raw.error or "empty reply"
            continue
        if answer == "json_choice":
            choice, confidence, tell = read_json_choice(raw.text)
            if choice is not None:
                return JudgeAnswer(
                    answer=choice, model=model, confidence=confidence, tell=tell, usage=usage
                )
        else:
            pattern = DIGIT_1_5_PATTERN if answer == "digit" else LETTER_PATTERN
            letter = read_letter(raw.text, pattern)
            if letter is not None:
                return JudgeAnswer(answer=letter, model=model, usage=usage)
        last_error = f"unparseable reply from {model}"
    return JudgeAnswer(error=last_error, usage=usage)


def _dispatch(work: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """Fetch answers concurrently, in the order the items were drawn.

    One worker and sixteen give the same list: the draws already happened, and
    ``map`` preserves order, so the only thing concurrency changes is how long the
    pass takes.
    """
    if workers <= 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))


# ── Statistics ────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials.

    Wilson rather than the normal approximation because these n are forty, not
    four hundred, and the normal interval at forty runs off the end of the scale.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((centre - half) / denominator, 4), round((centre + half) / denominator, 4))


def binom_p_ge(k: int, n: int, p0: float = 0.5) -> float:
    """One-sided P(X >= k) under Binom(n, p0) — the probability chance did this."""
    if n == 0:
        return 1.0
    return round(sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1)), 5)


# ── Contrasts ─────────────────────────────────────────────────────────────────
class PairOutcome(BaseModel):
    """One pair's result, with the key's labels folded back in after the fact."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    answer: str | None = None
    correct: bool | None = None
    confidence: int | None = None
    judge_model: str | None = None
    distractor: str | None = None
    error: str | None = None


class ContrastResult(BaseModel):
    """A judged contrast: the rate, its interval, and everything needed to doubt it."""

    model_config = ConfigDict(extra="forbid")

    contrast: str
    prompt_set: str
    prompt_version: str
    prompt_digest: str
    variant: str | None = None
    judge_family: str
    judge_ladder: tuple[str, ...]
    packet_digest: str
    n_pairs: int
    n_scored: int
    n_failed: int
    wins: int
    win_rate: float | None
    wilson95: tuple[float, float]
    binom_p_ge_chance: float
    threshold: float = DEFAULT_QUALIFY_THRESHOLD
    mean_confidence: float | None = None
    by_distractor: dict[str, float] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    rows: list[PairOutcome] = Field(default_factory=list)
    law: str = BLINDING_LAW

    @property
    def is_non_run(self) -> bool:
        """Zero scored pairs is an absence of evidence, not a null result.

        Authentication can fail silently enough that a job finishes, writes a
        file and reports a rate of nothing. Such a pass is reported under its own
        name so it cannot be read as a judge that saw no difference.
        """
        return self.n_scored == 0

    @property
    def separates(self) -> bool:
        """Whether the interval clears chance — the dynamic-range question."""
        return not self.is_non_run and self.wilson95[0] > 0.5

    @property
    def qualifies(self) -> bool:
        """Whether the rate clears the stated selection threshold."""
        return not self.is_non_run and (self.win_rate or 0.0) >= self.threshold


def run_contrast(
    backend: JudgeBackend,
    packet: BlindPacket,
    prompt: PromptSet,
    *,
    models: Sequence[str],
    variant: str | None = None,
    workers: int = DEFAULT_WORKERS,
    max_chars: int | None = DEFAULT_MAX_CHARS,
    threshold: float = DEFAULT_QUALIFY_THRESHOLD,
    schema: Mapping[str, Any] | None = None,
) -> ContrastResult:
    """Judge a drawn packet and report the rate with its interval.

    Rendering happens first and in full, from :class:`BlindPair` alone; dispatch
    happens second; scoring against the key happens third, when every answer is
    already in. The three phases are separate so that the only thing the judge
    ever receives is a rendered pair, and the only thing the key ever touches is
    an answer that has already been given.
    """
    rendered = [
        prompt.render(
            a=pair.first, b=pair.second, variant=variant, topic=pair.topic, max_chars=max_chars
        )
        for pair in packet.pairs
    ]
    answers = _dispatch(
        lambda item: ask_with_ladder(
            backend, item, models=models, answer=prompt.answer, schema=schema
        ),
        rendered,
        workers,
    )

    usage = Usage()
    rows: list[PairOutcome] = []
    wins = scored = 0
    confidences: list[int] = []
    per_source: dict[str, list[bool]] = {}
    for pair, reply in zip(packet.pairs, answers):
        usage.add(reply.usage)
        key = packet.keys.get(pair.pair_id)
        source = key.labels.get("distractor") if key else None
        if reply.answer is None or key is None:
            rows.append(
                PairOutcome(
                    pair_id=pair.pair_id,
                    answer=reply.answer,
                    judge_model=reply.model,
                    distractor=source,
                    error=reply.error or (None if key else "no key for this pair"),
                )
            )
            continue
        correct = reply.answer == key.target_side
        scored += 1
        wins += int(correct)
        if reply.confidence is not None:
            confidences.append(reply.confidence)
        per_source.setdefault(source or "unnamed", []).append(correct)
        rows.append(
            PairOutcome(
                pair_id=pair.pair_id,
                answer=reply.answer,
                correct=correct,
                confidence=reply.confidence,
                judge_model=reply.model,
                distractor=source,
            )
        )

    return ContrastResult(
        contrast=packet.contrast,
        prompt_set=prompt.name,
        prompt_version=prompt.version,
        prompt_digest=prompt.digest(),
        variant=variant,
        judge_family=backend.family,
        judge_ladder=tuple(models),
        packet_digest=packet.digest(),
        n_pairs=len(packet.pairs),
        n_scored=scored,
        n_failed=len(packet.pairs) - scored,
        wins=wins,
        win_rate=round(wins / scored, 4) if scored else None,
        wilson95=wilson(wins, scored),
        binom_p_ge_chance=binom_p_ge(wins, scored),
        threshold=threshold,
        mean_confidence=round(sum(confidences) / len(confidences), 2) if confidences else None,
        by_distractor={
            name: round(sum(hits) / len(hits), 3) for name, hits in sorted(per_source.items())
        },
        usage=usage,
        rows=rows,
        law=packet.law,
    )


# ── The coherence gate ────────────────────────────────────────────────────────
class CoherenceResult(BaseModel):
    """What the single-text gate found, and where in the text it looked."""

    model_config = ConfigDict(extra="forbid")

    n: int
    mean: float | None
    scores: list[int] = Field(default_factory=list)
    window: Literal["head", "tail"]
    chars: int
    floor: float | None = None
    usage: Usage = Field(default_factory=Usage)

    @property
    def passes(self) -> bool | None:
        """Whether the mean clears a stated floor, or None where none was stated."""
        if self.floor is None or self.mean is None:
            return None
        return self.mean >= self.floor


def run_coherence_gate(
    backend: JudgeBackend,
    texts: Sequence[str],
    *,
    models: Sequence[str],
    window: Literal["head", "tail"] = "tail",
    chars: int = 1200,
    workers: int = DEFAULT_WORKERS,
    floor: float | None = None,
) -> CoherenceResult:
    """Rate single texts one to five, blind, and report the mean.

    This is what separates an in-window shift from a collapse. An intervention
    whose behavioural number moves because the text fell apart has moved nothing
    worth naming, and the 2AFC alone cannot tell the two apart: a degenerate text
    is easy to pick out of a pair.

    The window is where degeneration is read. A tail window reads the end of the
    document, which is where a generation loses its thread; a head window reads
    the opening, which is where the donors' earlier gates looked. Both are
    available and the choice is recorded, because two cells compared across
    different windows are not comparable.
    """
    windowed = [text[-chars:] if window == "tail" else text[:chars] for text in texts]
    replies = _dispatch(
        lambda text: ask_with_ladder(
            backend,
            RenderedPrompt(user=COHERENCE_PROMPT.format(t=text)),
            models=models,
            answer="digit",
        ),
        windowed,
        workers,
    )
    usage = Usage()
    scores: list[int] = []
    for reply in replies:
        usage.add(reply.usage)
        if reply.answer is not None:
            scores.append(int(reply.answer))
    return CoherenceResult(
        n=len(scores),
        mean=round(sum(scores) / len(scores), 3) if scores else None,
        scores=scores,
        window=window,
        chars=chars,
        floor=floor,
        usage=usage,
    )


# ── The reader-grade ladder ───────────────────────────────────────────────────
class ReaderLadderResult(BaseModel):
    """One contrast read by several grades of reader, strongest first."""

    model_config = ConfigDict(extra="forbid")

    contrast: str
    grades: list[str]
    results: dict[str, ContrastResult]

    @property
    def passes_at(self) -> tuple[str, ...]:
        """The grades whose interval clears chance, in the order they were run."""
        return tuple(g for g in self.grades if g in self.results and self.results[g].separates)

    @property
    def weakest_pass(self) -> str | None:
        """The weakest reader that still separates — the grade a pass passes at."""
        passing = self.passes_at
        return passing[-1] if passing else None


def run_reader_ladder(
    backend: JudgeBackend,
    packet: BlindPacket,
    prompt: PromptSet,
    *,
    grades: Sequence[str],
    variant: str | None = None,
    workers: int = DEFAULT_WORKERS,
    max_chars: int | None = DEFAULT_MAX_CHARS,
    threshold: float = DEFAULT_QUALIFY_THRESHOLD,
    schema: Mapping[str, Any] | None = None,
) -> ReaderLadderResult:
    """Run the same packet past several readers and report the grade it passes at.

    ``grades`` is ordered strongest to weakest, and each grade is a single-model
    ladder rather than a fallback chain — a fallback would silently promote a
    weak reader's answer into a strong reader's row, which is the one thing this
    measurement cannot survive.
    """
    if not grades:
        raise JudgingError("a reader ladder needs at least one grade")
    results = {
        grade: run_contrast(
            backend,
            packet,
            prompt,
            models=[grade],
            variant=variant,
            workers=workers,
            max_chars=max_chars,
            threshold=threshold,
            schema=schema,
        )
        for grade in grades
    }
    return ReaderLadderResult(contrast=packet.contrast, grades=list(grades), results=results)


# ── The ceiling control ───────────────────────────────────────────────────────
class Interpretation(BaseModel):
    """A contrast read against the ruler's own dynamic range."""

    model_config = ConfigDict(extra="forbid")

    contrast: str
    verdict: Verdict
    reason: str
    win_rate: float | None = None
    wilson95: tuple[float, float] | None = None
    ceiling_contrast: str | None = None
    ceiling_win_rate: float | None = None


def interpret_with_ceiling(
    target: ContrastResult,
    ceiling: ContrastResult | None,
) -> Interpretation:
    """Read a contrast, refusing to call a null where the ruler has no range.

    The asymmetry is the whole rule. A weak judge can hide an effect but cannot
    manufacture a sign-controlled positive, so a separating target needs no
    ceiling to be believed. A target that does not separate is a statement about
    two things at once — the effect and the instrument — and it is only a
    statement about the effect once the ceiling contrast has shown the instrument
    can separate anything at all. Run the ceiling, read it first.
    """
    common = {
        "contrast": target.contrast,
        "win_rate": target.win_rate,
        "wilson95": target.wilson95,
        "ceiling_contrast": ceiling.contrast if ceiling else None,
        "ceiling_win_rate": ceiling.win_rate if ceiling else None,
    }
    if target.is_non_run:
        return Interpretation(
            verdict="non-run",
            reason=(
                f"{target.n_pairs} pairs drawn and none scored; this is an absence of "
                "evidence rather than a null, and the usage record says why"
            ),
            **common,
        )
    if target.separates:
        return Interpretation(
            verdict="positive",
            reason=(
                f"win rate {target.win_rate} with interval {target.wilson95} clearing chance; "
                "a weak ruler could have hidden this and could not have manufactured it, so it "
                "stands without the ceiling"
            ),
            **common,
        )
    if ceiling is None:
        return Interpretation(
            verdict="uninterpretable",
            reason=(
                "the target does not separate and no ceiling contrast was run, so the result is "
                "a statement about the effect and the instrument at once"
            ),
            **common,
        )
    if ceiling.is_non_run:
        return Interpretation(
            verdict="uninterpretable",
            reason=f"the ceiling contrast {ceiling.contrast!r} scored no pairs, so it measures nothing",
            **common,
        )
    if not ceiling.separates:
        return Interpretation(
            verdict="uninterpretable",
            reason=(
                f"the ceiling contrast {ceiling.contrast!r} does not separate either "
                f"(rate {ceiling.win_rate}, interval {ceiling.wilson95}); this ruler has no "
                "demonstrated dynamic range, so the target's null is about the ruler"
            ),
            **common,
        )
    return Interpretation(
        verdict="null",
        reason=(
            f"the target does not separate (rate {target.win_rate}, interval {target.wilson95}) "
            f"while the ceiling {ceiling.contrast!r} does (rate {ceiling.win_rate}); the ruler "
            "has range and read nothing here"
        ),
        **common,
    )


# ── Anti-circularity ──────────────────────────────────────────────────────────
def _normalize(source: str) -> str:
    return " ".join(source.lower().split())


def assert_not_circular(prompt: PromptSet, *, scoring_instrument: str) -> None:
    """Refuse a study whose judge criterion comes from the instrument scoring it.

    Where the description a judge is given is drawn from the same battery whose
    counts the judge's answer is compared against, the judge is not a second
    channel: it is a slower, more expensive restatement of the first. The rule is
    mechanical here because it is easy to violate by accident — the most natural
    source for a good description of a target *is* the thing already built to
    detect it.
    """
    criterion = _normalize(prompt.criterion_source)
    instrument = _normalize(scoring_instrument)
    if not criterion:
        raise CircularityError(
            f"prompt set {prompt.name!r} declares no criterion source, so circularity "
            "cannot be ruled out"
        )
    if not instrument:
        raise CircularityError(
            "a study must name the instrument that scores it, or the rule has nothing to check"
        )
    if criterion == instrument or criterion in instrument or instrument in criterion:
        raise CircularityError(
            f"prompt set {prompt.name!r} draws its criterion from {prompt.criterion_source!r}, "
            f"which is the instrument scoring the result ({scoring_instrument!r}); the two "
            "columns of that table would be one column wearing two hats"
        )


# ── A second judge family ─────────────────────────────────────────────────────
class AgreementResult(BaseModel):
    """How two judge families read the same items, and whether that changes a verdict."""

    model_config = ConfigDict(extra="forbid")

    contrast: str
    n_compared: int
    n_agree: int
    agreement: float | None
    first_family: str
    second_family: str
    first_rate: float | None
    second_rate: float | None
    verdict_flipped: bool
    surface: bool
    reading: str = (
        "divergence between judge families is an instrument finding about the families, "
        "scoped to this contrast, and never an automatic re-score; a flipped scored verdict "
        "is stop-and-surface."
    )


def compare_families(first: ContrastResult, second: ContrastResult) -> AgreementResult:
    """Compare two passes over the same packet, pair by pair.

    A second family is only a check if it read the same items, so the packet
    digests are compared before anything else and a mismatch is refused. What the
    comparison produces is a reading, not a correction: the second family's rate
    does not replace the first's.
    """
    if first.packet_digest != second.packet_digest:
        raise ReconstructionMismatch(
            "the two passes did not read the same packet; their digests differ, so their "
            "rates are not comparable"
        )
    first_by_id = {row.pair_id: row.answer for row in first.rows}
    compared = agree = 0
    for row in second.rows:
        other = first_by_id.get(row.pair_id)
        if row.answer is None or other is None:
            continue
        compared += 1
        agree += int(row.answer == other)
    flipped = first.qualifies != second.qualifies or first.separates != second.separates
    return AgreementResult(
        contrast=first.contrast,
        n_compared=compared,
        n_agree=agree,
        agreement=round(agree / compared, 3) if compared else None,
        first_family=first.judge_family,
        second_family=second.judge_family,
        first_rate=first.win_rate,
        second_rate=second.win_rate,
        verdict_flipped=flipped,
        surface=flipped,
    )


def annex_prompt(question: str, pair: BlindPair, *, max_chars: int | None = DEFAULT_MAX_CHARS) -> RenderedPrompt:
    """Render a banked packet's own question over one of its pairs."""
    left = pair.first[:max_chars] if max_chars is not None else pair.first
    right = pair.second[:max_chars] if max_chars is not None else pair.second
    return RenderedPrompt(user=ANNEX_TEMPLATE.format(question=question, a=left, b=right))


def receipt(
    *,
    study: str,
    results: Iterable[ContrastResult],
    interpretations: Iterable[Interpretation] = (),
    coherence: Mapping[str, CoherenceResult] | None = None,
) -> dict[str, Any]:
    """Assemble the banked record of a study: every rate beside its own law."""
    rows = list(results)
    total = Usage()
    for row in rows:
        total.add(row.usage)
    return {
        "study": study,
        "law": BLINDING_LAW,
        "contrasts": [row.model_dump() for row in rows],
        "interpretations": [i.model_dump() for i in interpretations],
        "coherence": {k: v.model_dump() for k, v in (coherence or {}).items()},
        "usage": total.model_dump(),
    }


__all__ = [
    "AgreementResult",
    "AnthropicBackend",
    "BLINDING_LAW",
    "BlindPacket",
    "BlindPair",
    "CircularityError",
    "CoherenceResult",
    "ContrastResult",
    "Corpus",
    "Interpretation",
    "JudgeAnswer",
    "JudgeBackend",
    "JudgingError",
    "NO_EFFORT_MODELS",
    "OpenRouterBackend",
    "PairKey",
    "PairOutcome",
    "RawReply",
    "ReaderLadderResult",
    "ReconstructionMismatch",
    "Usage",
    "annex_prompt",
    "ask_with_ladder",
    "assert_not_circular",
    "binom_p_ge",
    "compare_families",
    "draw_pairs",
    "interpret_with_ceiling",
    "packet_from_pairs_md",
    "parse_pairs_md",
    "read_json_choice",
    "read_letter",
    "receipt",
    "run_coherence_gate",
    "run_contrast",
    "run_reader_ladder",
    "sample_texts",
    "texts_by_prompt",
    "texts_by_topic",
    "verify_against_key",
    "wilson",
    "write_pairs_md",
]
