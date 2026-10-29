"""The measurement behind G3's public-voice check: how hard a page of prose is to process.

The features are the processing-difficulty features of Clark, Martin & Mahowald,
"Why is AI-generated language so hard to understand?"
(https://arxiv.org/abs/2610.29162). Each makes prose measurably harder to read on
the first pass, and machine-written prose over-uses all of them. They are read off
a spaCy ``en_core_web_trf`` dependency parse, as in the paper; the animacy and
denominal judgements here are lexicon heuristics where the paper used a classifier.

  ``dep_len``      mean dependency arc length, directly standardized over the
                   sentence-length bins in ``BINS`` with the reference's bin weights
                   (punctuation arcs and the root excluded; a sentence over 30 words
                   is dropped from the mean and counted as ``long`` instead)
  ``orc``          object relative clauses with an overt relativizer
                   ("the memo that the committee wrote")
  ``orc_reduced``  object relatives without one ("the memo the committee wrote")
  ``orc_all``      the two together: every object relative clause
  ``stacked``      chains of three or more common nouns in compound relations
                   ("cache invalidation policy harness"; names excluded)
  ``inanimate``    an active mind or agency verb whose subject is a thing
                   ("the pipeline decided", "it wants")
  ``denominal``    a verb whose lemma the reference uses almost only as a noun
                   ("to task", "to version")
  ``long``         sentences over 30 words

Rates are per 1,000 words. A page's ratio to the reference is its rate divided by
the reference's. The reference is ``legibility_reference.json`` beside this module:
the bin weights, the per-bin arc lengths, the four feature rates and the denominal
lemma list, measured over the parsed slice of a fixed sample of English Wikipedia
articles. It is a few kilobytes of statistics, not a corpus, which is what lets the
check run anywhere the parser runs.

The text extraction and the counting functions are the ones the reference
statistics were computed with, unchanged: a ratio means something only when the
page and the reference are counted by the same code, so an edit to any of them is
an edit to the instrument and needs a rebuilt reference. Two further measures are
left out: the rate of word pairs unseen in the reference needs the reference's
whole bigram table, and the rare-word rate needs a word-frequency package. Neither
has a reference rate to be a ratio of, so neither could be held to a threshold.

This module imports spaCy only inside ``load_parser``, so the gate's own logic, and
its tests, run on a machine without the parser installed.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REFERENCE = Path(__file__).resolve().parent / "legibility_reference.json"

#: The model the reference was parsed with. A different model parses differently,
#: and every ratio would then compare two instruments rather than two texts.
PARSER_MODEL = "en_core_web_trf"

BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
LONG_SENTENCE_WORDS = 30
ANIMATE_VERBS = {
    "want", "decide", "think", "believe", "know", "feel", "try", "choose", "refuse", "hope", "expect", "prefer",
    "intend", "realize", "realise", "remember", "forget", "understand", "learn", "insist", "agree", "admit", "claim",
    "argue", "wonder", "worry", "care", "like", "love", "hate", "notice", "recognize", "recognise", "assume", "imagine",
    "plan", "attempt", "pretend", "promise", "wish", "desire", "fear", "trust", "doubt", "guess", "dare", "bother",
    "mind", "struggle", "manage", "seek", "aim", "strive", "tend", "ask", "say", "tell", "speak", "listen", "hear", "see",
    "watch", "look", "smell", "taste", "consider", "judge", "reckon", "suppose", "suspect", "accept", "reject",
    "complain", "apologize", "thank", "praise", "blame", "forgive", "obey", "ignore", "pay", "lie", "cheat", "win", "lose",
}
PERSON_NOUNS = {
    "person", "people", "man", "woman", "men", "women", "child", "children", "human", "humans", "user", "users", "reader",
    "readers", "author", "authors", "writer", "speaker", "listener", "researcher", "researchers", "team", "student",
    "students", "teacher", "friend", "friends", "someone", "anyone", "everyone", "nobody", "somebody", "everybody",
    "participant", "participants", "annotator", "annotators", "judge", "judges", "rater", "raters", "customer", "client",
    "operator", "developer", "developers", "engineer", "engineers", "scientist", "scientists", "player", "character",
    "characters", "agent", "agents", "assistant", "group", "committee", "audience", "colleague", "partner", "owner",
    "holder", "addressee", "narrator", "protagonist", "boy", "girl", "mother", "father", "parent", "parents",
}
ANIMATE_PRON = {"i", "you", "we", "he", "she", "they", "me", "us", "him", "her", "them", "who", "whoever", "one"}
INANIMATE_PRON = {"it", "this", "that", "which", "these", "those", "something", "nothing", "everything", "anything"}
RELATIVIZERS = {"that", "which", "who", "whom", "whose"}
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

#: The counted features, in report order. ``long`` is counted per sentence, and
#: ``orc_all`` is the sum of the two object-relative counts.
COUNTED = ("orc", "orc_reduced", "orc_all", "stacked", "inanimate", "denominal", "long")
#: The features the reference holds a rate for, so a ratio is defined.
REFERENCED = ("orc", "orc_reduced", "orc_all", "stacked", "inanimate")


class LegibilityError(RuntimeError):
    """A condition that makes the measurement unrunnable rather than a page illegible."""


# ---------------------------------------------------------------------------- text
def md_text(s: str) -> str:
    """The prose of a Markdown page: code fences, tables, images and link targets dropped.

    An inline code span becomes the word ``CODE``, so a sentence keeps its shape
    while the identifier inside it stays out of the counts.
    """
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " CODE ", s)
    s = re.sub(r"^\s*\|.*$", " ", s, flags=re.M)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"^#+\s*", "", s, flags=re.M)
    s = re.sub(r"[*_]{1,3}", "", s)
    return s


def paragraphs(text: str, min_words: int) -> list[str]:
    """Blank-line paragraphs, and lines that open like a heading or a list item."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9#*\-•])", text)]
    return [p for p in paras if len(WORD.findall(p)) >= min_words]


def docstring_markdown(source: str, filename: str = "<module>") -> str:
    """A module docstring, with its reStructuredText roles written as Markdown code spans.

    ``:mod:`x``` and ````x```` both become ```x```, which ``md_text`` then reads as
    one code word, exactly as it reads an inline span in a Markdown page.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise LegibilityError(f"{filename}: syntax error at line {exc.lineno} ({exc.msg})") from exc
    doc = ast.get_docstring(tree, clean=True) or ""
    doc = re.sub(r":[a-z]+:`([^`]*)`", r"`\1`", doc)
    return doc.replace("``", "`")


# ---------------------------------------------------------------------------- features
def compound_chain(tok: Any) -> list[Any]:
    """The maximal compound-noun chain whose head is tok (tok included), in text order."""
    out = [tok]
    for c in tok.children:
        if c.dep_ == "compound" and c.pos_ in ("NOUN", "PROPN"):
            out = compound_chain(c) + out
    return sorted(out, key=lambda t: t.i)


def subject_of(verb: Any) -> Any:
    for c in verb.children:
        if c.dep_ == "nsubj":
            return c
    return None


def animacy(subj: Any) -> str:
    """animate / inanimate / unknown, by lexicon; a relativizer subject takes its antecedent's animacy."""
    t = subj
    if t.lower_ in RELATIVIZERS and t.head.dep_ == "relcl":
        t = t.head.head
    lw = t.lower_
    if t.pos_ == "PRON":
        return "animate" if lw in ANIMATE_PRON else "inanimate" if lw in INANIMATE_PRON else "unknown"
    if t.ent_type_ in ("PERSON", "NORP"):
        return "animate"
    if t.lemma_.lower() in PERSON_NOUNS or lw in PERSON_NOUNS:
        return "animate"
    if t.text == "CODE":
        return "inanimate"
    if t.pos_ in ("NOUN", "PROPN"):
        return "unknown" if t.ent_type_ == "ORG" else "inanimate"
    return "unknown"


def sentence_features(sent: Any, denom: set[str] | None) -> dict[str, Any]:
    """One parsed sentence's word count, mean arc length and flagged spans per feature."""
    words = [t for t in sent if not t.is_punct and not t.is_space]
    n = len(words)
    arcs = [abs(t.i - t.head.i) for t in words if t.dep_ != "ROOT" and t.head is not t]
    f: dict[str, Any] = {"n": n, "dep": (sum(arcs) / len(arcs)) if arcs else 0.0, "orc": [], "orc_reduced": [],
                         "stacked": [], "inanimate": [], "denominal": []}
    for t in sent:
        if t.dep_ == "relcl" and t.pos_ in ("VERB", "AUX"):
            kids = list(t.children)
            relz = [c for c in kids if c.lower_ in RELATIVIZERS or c.tag_ in ("WDT", "WP", "WP$")]
            marked = any(x.i < t.i and (x.lower_ in RELATIVIZERS or x.tag_ in ("WDT", "WP", "WP$")) for x in t.subtree)
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass") and c not in relz for c in kids)
            obj_rel = [c for c in relz if c.dep_ in ("dobj", "pobj", "obj")]
            if obj_rel and has_subj:
                f["orc"].append(f"{t.head.text} … {t.text}")
            elif not relz and not marked and has_subj and not any(c.dep_ in ("dobj", "obj") for c in kids) \
                    and not any(c.dep_ == "advmod" and c.tag_ == "WRB" for c in kids):
                f["orc_reduced"].append(f"{t.head.text} … {t.text}")
        if t.pos_ == "NOUN" and t.dep_ != "compound":
            ch = compound_chain(t)
            nouns = [x for x in ch if x.pos_ == "NOUN"]
            if len(nouns) >= 3:
                f["stacked"].append(" ".join(x.text for x in ch))
        if t.pos_ == "VERB" and t.lemma_.lower() in ANIMATE_VERBS and t.tag_ != "VBN" \
                and not (t.lemma_.lower() == "look" and any(c.lower_ == "like" for c in t.children)):
            s = subject_of(t)
            if s is not None and animacy(s) == "inanimate":
                f["inanimate"].append(f"{s.text} {t.text}")
        if denom is not None and t.pos_ == "VERB" and t.lemma_.lower() in denom:
            f["denominal"].append(t.text)
    return f


# ---------------------------------------------------------------------------- reference and parser
@dataclass(frozen=True)
class Reference:
    """The reference statistics every ratio is taken against."""

    source: str
    bin_weights: list[float]
    dep_by_bin: list[float]
    dep_len: float
    rates: dict[str, float]
    denominal_lemmas: frozenset[str]


def load_reference(path: Path = DEFAULT_REFERENCE) -> Reference:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegibilityError(f"reference unreadable: {path} ({exc})") from exc
    try:
        ref = Reference(
            source=str(raw["source"]),
            bin_weights=[float(w) for w in raw["bin_weights"]],
            dep_by_bin=[float(d) for d in raw["dep_by_bin"]],
            dep_len=float(raw["dep_len"]),
            rates={str(k): float(v) for k, v in raw["rates"].items()},
            denominal_lemmas=frozenset(str(x) for x in raw["denominal_lemmas"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LegibilityError(f"reference malformed: {path} ({exc})") from exc
    if "orc" in ref.rates and "orc_reduced" in ref.rates:
        ref.rates["orc_all"] = ref.rates["orc"] + ref.rates["orc_reduced"]
    if len(ref.bin_weights) != len(BINS) or len(ref.dep_by_bin) != len(BINS):
        raise LegibilityError(f"reference has {len(ref.bin_weights)} length bins; this module uses {len(BINS)}")
    missing = [k for k in REFERENCED if not ref.rates.get(k)]
    if missing:
        raise LegibilityError(f"reference holds no rate for {missing}")
    return ref


def load_parser() -> Any:
    """The spaCy pipeline the reference was parsed with, or a refusal that says how to get it."""
    try:
        import spacy
    except ImportError as exc:
        raise LegibilityError(
            "spaCy is not installed; `pip install -r tools/legibility_requirements.txt` installs "
            "the parser at the versions the reference was built with"
        ) from exc
    try:
        return spacy.load(PARSER_MODEL)
    except OSError as exc:
        raise LegibilityError(
            f"the {PARSER_MODEL} model is not installed; "
            "`pip install -r tools/legibility_requirements.txt` installs it"
        ) from exc


# ---------------------------------------------------------------------------- one page
@dataclass
class PageMetrics:
    """One page's measurement: word and sentence counts, arc length, and per-feature counts."""

    words: int
    sentences: int
    dep_len: float
    counts: dict[str, int]
    flags: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def rate(self, feature: str) -> float:
        """Occurrences per 1,000 words."""
        return self.counts[feature] * 1000.0 / self.words if self.words else 0.0

    def rates(self) -> dict[str, float]:
        return {feature: self.rate(feature) for feature in COUNTED}

    def ratios(self, ref: Reference) -> dict[str, float]:
        """Each referenced feature, and ``dep_len``, as a multiple of the reference."""
        out = {"dep_len": self.dep_len / ref.dep_len}
        for feature in REFERENCED:
            out[feature] = self.rate(feature) / ref.rates[feature]
        return out


def measure(text: str, nlp: Any, ref: Reference, min_words: int = 4) -> PageMetrics:
    """Parse already-extracted prose and count every feature.

    ``text`` is prose as ``md_text`` returns it. A page with no countable words
    measures as zero words, which the caller decides how to treat.
    """
    paras = paragraphs(text, min_words)
    words = 0
    sentences = 0
    dep_bins: dict[int, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    flags: dict[str, list[dict[str, str]]] = defaultdict(list)
    denom = set(ref.denominal_lemmas)
    for doc in nlp.pipe(paras, batch_size=8):
        for sent in doc.sents:
            f = sentence_features(sent, denom)
            if f["n"] == 0:
                continue
            words += f["n"]
            sentences += 1
            b = next((i for i, (lo, hi) in enumerate(BINS) if lo <= f["n"] <= hi), None)
            if b is not None:
                dep_bins[b].append(f["dep"])
            if f["n"] > LONG_SENTENCE_WORDS:
                counts["long"] += 1
                flags["long"].append({"match": f"{f['n']} words", "sentence": sent.text.strip()})
            for key in ("orc", "orc_reduced", "stacked", "inanimate", "denominal"):
                counts[key] += len(f[key])
                for item in f[key]:
                    flags[key].append({"match": item, "sentence": sent.text.strip()})
    counts["orc_all"] = counts["orc"] + counts["orc_reduced"]
    w = ref.bin_weights
    dep = sum(
        w[i] * (sum(dep_bins[i]) / len(dep_bins[i]) if dep_bins[i] else ref.dep_by_bin[i]) for i in range(len(BINS))
    )
    return PageMetrics(
        words=words,
        sentences=sentences,
        dep_len=dep,
        counts={feature: counts[feature] for feature in COUNTED},
        flags=dict(flags),
    )


def describe(metrics: PageMetrics, ref: Reference) -> Iterable[str]:
    """Report lines for one page: arc length, then each feature's rate and ratio."""
    ratios = metrics.ratios(ref)
    yield f"   dep_len {metrics.dep_len:.2f} (ref {ref.dep_len:.2f}, ×{ratios['dep_len']:.2f})"
    for feature in COUNTED:
        line = f"   {feature:13s} {metrics.rate(feature):7.2f}/1k"
        if feature in ratios:
            line += f"  ref {ref.rates[feature]:6.2f}  ×{ratios[feature]:.2f}"
        yield line
