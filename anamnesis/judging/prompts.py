"""The judge prompt table — five versioned 2AFC sets, and the two shared rubrics.

A prompt is **data**, not code. What a judge was asked is part of what a judged
number means, so the text is stored verbatim, carries the path it was banked at,
and is pinned by hash in the test suite. A reader who wants to know what the
formality number rests on reads :data:`FORMALITY`'s ``user_template`` and has the
whole of it; a reader who suspects the text drifted runs the suite, which fails
on a changed byte.

Each set carries four provenance fields beyond its text:

``version``
    ``v1`` is the text as banked in the frozen record. A prompt that is reworded
    is a new version with its own row, never an edit of an existing one — two
    numbers taken under different wordings are two measurements.
``donor``
    The frozen-record path the text comes from, so the banked tables that used it
    can be found.
``criterion_source``
    Where the *description of the target* comes from. This is the anti-circularity
    field and :func:`anamnesis.judging.harness.assert_not_circular` refuses a
    study whose scoring instrument is also the source of its judge's criterion. A
    judge told to look for the same markers a regex battery counts is not a second
    channel; it is the first channel wearing a second hat.
``answer``
    How the reply is read: ``letter`` scans the first text block for a bare A or
    B, ``json_choice`` parses a one-object JSON reply with ``choice`` and
    ``confidence``. The two shapes come from different donors and both are kept,
    because a parse convention is part of a banked table's identity.

**The five are not interchangeable.** :data:`MODE` and :data:`SOCRATIC` both ask
about the same five processing modes and ask differently: MODE asks which text
*followed a mode instruction* and puts the criterion in a system prompt (the
census hardening shape), SOCRATIC asks which text is *more* that mode in a single
user turn (the steering-shift shape). Their mode descriptions are different prose.
A win rate from one is not comparable with a win rate from the other, which is
why they are two rows rather than one row with a flag.

Two rubrics are shared rather than per-axis, and are also stamped:
:data:`COHERENCE_PROMPT` (the single-text 1-5 gate that separates an in-window
shift from a collapse) and the Likert rubric behind
:func:`likert_system_prompt`, which is the non-2AFC paradigm's whole instrument.
:data:`ANNEX_TEMPLATE` is the generic two-text form whose question is supplied by
the banked packet it re-judges; it is a carrier, not a criterion, so it is not one
of the five.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerFormat = Literal["letter", "json_choice"]

# The reply patterns the donors scanned for, on the upper-cased first text block.
LETTER_PATTERN = r"\b([AB])\b"
DIGIT_1_5_PATTERN = r"\b([1-5])\b"


class PromptSetError(ValueError):
    """A prompt set asked for a variant it does not carry, or a name not in the table."""


class RenderedPrompt(BaseModel):
    """One judge-facing turn: what goes in the system slot, and what goes in the user slot.

    The pair's key is not a field here and cannot be. Rendering is reached from
    :class:`anamnesis.judging.harness.BlindPair`, which does not carry the answer,
    so there is no call path along which the side labels could be formatted into
    the text a judge reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str | None = None
    user: str


class PromptSet(BaseModel):
    """A versioned, provenance-stamped 2AFC instrument.

    Substitution is done with :meth:`str.replace` rather than :meth:`str.format`
    for the criterion placeholders, because two of the five system prompts contain
    a literal JSON example. Doubling those braces to satisfy ``format`` would make
    the stored text differ from the donor's, and the stored text being byte-equal
    to the donor's is the property this module exists for. The
    two-text placeholders ``{a}`` and ``{b}`` are substituted with ``format``,
    exactly as every donor did; braces inside the texts themselves are arguments
    and are not reinterpreted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str = "v1"
    axis: str
    donor: str
    criterion_source: str
    answer: AnswerFormat
    system_template: str | None = None
    user_template: str
    descriptions: dict[str, str] = Field(default_factory=dict)

    @property
    def pattern(self) -> str:
        """The regex a ``letter`` reply is scanned with; unused for ``json_choice``."""
        return LETTER_PATTERN

    def variants(self) -> tuple[str, ...]:
        """The criteria this set can be rendered for.

        A set whose criterion is written inline in the prompt has one variant,
        named after the set itself; a set that carries a description table has one
        variant per description.
        """
        return tuple(sorted(self.descriptions)) if self.descriptions else (self.name,)

    def description_for(self, variant: str | None) -> str | None:
        """The criterion prose for a variant, or None when it is inline."""
        if not self.descriptions:
            if variant is not None and variant != self.name:
                raise PromptSetError(
                    f"{self.name} carries its criterion inline and has no variant "
                    f"{variant!r}; its only variant is {self.name!r}"
                )
            return None
        if variant is None:
            raise PromptSetError(
                f"{self.name} needs a variant, one of {self.variants()}"
            )
        try:
            return self.descriptions[variant]
        except KeyError:
            raise PromptSetError(
                f"{self.name} has no variant {variant!r}; it carries {self.variants()}"
            ) from None

    def render(
        self,
        *,
        a: str,
        b: str,
        variant: str | None = None,
        topic: str | None = None,
        max_chars: int | None = None,
    ) -> RenderedPrompt:
        """Render one pair into the turn a judge is sent.

        ``max_chars`` truncates each side, which the donors did at 2200 and which
        is a control rather than a formatting convenience: an unbounded pair lets
        one long text and one short one become a length tell.
        """
        description = self.description_for(variant)
        name = variant if description is not None else self.name
        left = a[:max_chars] if max_chars is not None else a
        right = b[:max_chars] if max_chars is not None else b

        def fill(template: str) -> str:
            filled = template
            if description is not None:
                filled = (
                    filled.replace("{variant_title}", name.capitalize())
                    .replace("{variant_upper}", name.upper())
                    .replace("{variant}", name)
                    .replace("{description}", description)
                )
            return filled

        user = fill(self.user_template)
        if "{topic}" in user:
            if topic is None:
                raise PromptSetError(f"{self.name} renders a topic line and none was given")
            user = user.replace("{topic}", topic)
        user = user.format(a=left, b=right)
        system = fill(self.system_template) if self.system_template is not None else None
        return RenderedPrompt(system=system, user=user)

    def digest(self) -> str:
        """A stable hash over everything a judge could read from this set.

        Name, version, both templates and the description table in sorted order.
        The provenance fields are excluded: a corrected donor path is bookkeeping,
        a changed word is a different instrument.
        """
        parts = [self.name, self.version, self.system_template or "", self.user_template]
        parts += [f"{k}\x00{self.descriptions[k]}" for k in sorted(self.descriptions)]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# ── 1. MODE — census contrast-hardening and pole-vs-pole qualification ────────
# Verbatim from run_2afc_mode_hardening.py. Its mode descriptions are paraphrased
# from the mode definitions themselves (anamnesis/modes/run4_modes.py) with the
# format constraint dropped, which is what keeps them independent of any marker
# battery that scores the same texts.

MODE_DESCRIPTIONS: dict[str, str] = {
    "linear": ("A clear forward sequence, each idea building on the last, laid out step by step "
               "from beginning to end with no backtracking or reconsidering of previous points."),
    "analogical": ("Explanation built primarily through extended analogies and parallels to other "
                   "domains — each key concept illuminated by a comparison from everyday life or another field."),
    "socratic": ("Exploration through a chain of questions and provisional answers: pose a question, "
                 "offer a tentative answer, use it to generate the next question; inquiry drives it forward."),
    "contrastive": ("Comparing and contrasting multiple perspectives — for each major point, at least "
                    "two viewpoints presented and their relative strengths and weaknesses weighed."),
    "dialectical": ("Propose a position, challenge it with the strongest counterarguments, then work "
                    "toward a revised understanding accounting for both the original position and its critiques."),
}

_TWO_TEXT_USER = "Topic: {topic}\n\n=== TEXT A ===\n{a}\n\n=== TEXT B ===\n{b}"

MODE = PromptSet(
    name="mode",
    axis="which of two same-topic texts followed a named processing mode's procedure",
    donor="anamnesis/scripts/run_2afc_mode_hardening.py",
    criterion_source="anamnesis/modes/run4_modes.py (the mode definitions, format constraint dropped)",
    answer="json_choice",
    system_template=(
        "You are an expert text analyst. You will be shown TWO texts (A and B) generated by a "
        "language model on the same topic. Exactly ONE was produced under an instruction to use "
        "this processing mode:\n\n**{variant_title}**: {description}\n\n"
        "Decide which text follows that mode. Respond with ONLY a JSON object (no markdown "
        "fencing): {\"choice\": \"A\" or \"B\", \"confidence\": <1-5>}"
    ),
    user_template=_TWO_TEXT_USER,
    descriptions=MODE_DESCRIPTIONS,
)

# ── 2. SOCRATIC — the steering-shift mode axis ────────────────────────────────
# Verbatim from vmb_a5_judge_socratic.py: one user turn, "which is MORE X", and
# its own description table, written for a shift readout rather than for an
# instruction-following readout.

SHIFT_MODE_DESCRIPTIONS: dict[str, str] = {
    "socratic": ("drives inquiry through probing questions, draws the reader forward by interrogation "
                 "rather than assertion, surfaces assumptions and asks the reader to examine them, and "
                 "reasons by successive questioning instead of flat exposition"),
    "analogical": ("explains primarily through extended analogies and parallels to other domains or "
                   "everyday life, building understanding by mapping the topic onto familiar comparisons"),
    "linear": ("presents ideas in a clear forward sequence, each point building on the last, moving step "
               "by step from beginning to end without backtracking or reconsidering"),
    "contrastive": ("compares and contrasts multiple perspectives or approaches, presenting differing "
                    "viewpoints and weighing their relative strengths and weaknesses"),
    "dialectical": ("proposes a position, challenges it with the strongest counterarguments, and works "
                    "toward a revised synthesis that accounts for both the position and its critiques"),
}

SOCRATIC = PromptSet(
    name="socratic",
    axis="which of two texts is MORE a named mode — the steering-shift metric",
    donor="anamnesis/scripts/vmb_a5_judge_socratic.py",
    criterion_source="hand-written mode prose in the donor, independent of the marker batteries",
    answer="letter",
    user_template="""You will see two AI-generated texts, A and B, responding to similar prompts.

Which text is written in a MORE {variant_upper} style? A {variant} text {description}.

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B.""",
    descriptions=SHIFT_MODE_DESCRIPTIONS,
)

# ── 3. ANALOGICAL — the off-genre behavioural leg ─────────────────────────────
# Verbatim from vmb_d4_judge_analogical.py. Its criterion is inline and longer
# than the shift table's analogical entry, written for a narrative corpus where
# the marker instrument is genre-noisy and the judge is the leg of record.

ANALOGICAL = PromptSet(
    name="analogical",
    axis="which of two texts develops its subject more through analogies",
    donor="anamnesis/scripts/vmb_d4_judge_analogical.py",
    criterion_source="hand-written analogical criterion in the donor, not the marker battery",
    answer="letter",
    user_template="""You will see two AI-generated texts, A and B, responding to similar prompts.

Which text explains or develops its subject MORE THROUGH ANALOGIES? An analogical text \
builds understanding by comparing its subject to other domains — extended comparisons, \
"like/as if" constructions, metaphors carried through the passage, parallels drawn to \
everyday life or other fields — rather than direct literal description.

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B.""",
)

# ── 4. TEMPERATURE — the sampling-temperature axis ────────────────────────────
# Verbatim from run_2afc_temperature_hardening.py. The temperature axis has no
# pure-mode corpus, so the contrast is hot-versus-same-topic-cold.

TEMPERATURE_DESCRIPTION = (
    "produced at a HIGHER sampling temperature — its word choices are more random, "
    "surprising, and less predictable; a lower-temperature text is more deterministic, "
    "repetitive, and 'safe' in its phrasing."
)

TEMPERATURE = PromptSet(
    name="temperature",
    axis="which of two same-topic texts was sampled at the higher temperature",
    donor="anamnesis/scripts/run_2afc_temperature_hardening.py",
    criterion_source="the sampler's own definition of temperature, stated in plain words",
    answer="json_choice",
    system_template=(
        "You are an expert text analyst. You will be shown TWO texts (A and B) generated by "
        "the same language model on the same topic, at DIFFERENT sampling temperatures. "
        "Exactly ONE was {description}\n\nDecide which text (A or B) was produced at the "
        "higher temperature. Respond with ONLY a JSON object (no markdown fencing): "
        "{\"choice\": \"A\" or \"B\", \"confidence\": <1-5>}"
    ),
    user_template=_TWO_TEXT_USER,
    descriptions={"temperature": TEMPERATURE_DESCRIPTION},
)

# ── 5. FORMALITY — the register axis ─────────────────────────────────────────
# Verbatim from vmb_a5_judge_formality.py, criterion inline.

FORMALITY = PromptSet(
    name="formality",
    axis="which of two texts is written in the more formal register",
    donor="anamnesis/scripts/vmb_a5_judge_formality.py",
    criterion_source="hand-written register criterion in the donor, not a marker battery",
    answer="letter",
    user_template="""You will see two AI-generated texts, A and B, responding to similar prompts.

Which text is written in a MORE FORMAL register (precise, professional, ceremonious, \
no contractions or colloquialisms)?

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B.""",
)

PROMPT_SETS: dict[str, PromptSet] = {
    p.name: p for p in (MODE, SOCRATIC, ANALOGICAL, TEMPERATURE, FORMALITY)
}


def prompt_set(name: str) -> PromptSet:
    """Look up one of the five by name, naming the table on a miss."""
    try:
        return PROMPT_SETS[name]
    except KeyError:
        raise PromptSetError(
            f"no prompt set {name!r}; the table holds {tuple(sorted(PROMPT_SETS))}"
        ) from None


# ── The coherence gate ───────────────────────────────────────────────────────
# Verbatim from vmb_a5_judge_socratic.py. A single text, blind, 1-5. This is what
# separates an in-window shift (the effect moves, the text stays readable) from a
# collapse (the effect moves because the text fell apart), and a shift number
# reported without it is not interpretable.

COHERENCE_PROMPT = """You will see one AI-generated text. Rate its COHERENCE on a 1-5 scale:

5 = fully coherent, fluent, on-topic, well-formed sentences throughout
4 = mostly coherent, minor lapses
3 = understandable but noticeably degraded (some repetition, drift, or awkwardness)
2 = substantially degraded (heavy repetition, broken syntax, loss of thread)
1 = incoherent (word salad, collapse, near-unreadable)

Text:
{t}

Answer with exactly one digit: 1, 2, 3, 4, or 5."""

COHERENCE_DONOR = "anamnesis/scripts/vmb_a5_judge_socratic.py"

# ── The generic carrier for a banked packet's own question ───────────────────
# Verbatim from vmb_judge_family2_annex.py. A banked packet states the question
# it was judged under; a second judge family re-reads the same items under the
# same words, which is what makes the two passes comparable.

ANNEX_TEMPLATE = """You will see two AI-generated texts, A and B.

{question}

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B."""

ANNEX_DONOR = "anamnesis/scripts/vmb_judge_family2_annex.py"


# ── The Likert rubric — the non-2AFC paradigm ────────────────────────────────
# Verbatim from run_judge_scoring.py: five dimensions rated 1-5 plus a primary
# classification, from which purity is the intended rating minus the mean of the
# other four.

VALID_MODES: tuple[str, ...] = ("linear", "analogical", "socratic", "contrastive", "dialectical")

JUDGE_MODE_DESCRIPTIONS: dict[str, str] = {
    "linear": (
        "Sequential, forward-moving exposition. Ideas presented one after another, "
        "each building on the last. No backtracking, no reconsideration of previous "
        "points. Straightforward progression from start to finish."
    ),
    "analogical": (
        "Explanation driven by analogies and parallels to other domains. Key concepts "
        "are illuminated through comparisons to everyday life or other fields. "
        "Frequent use of 'it's like...', 'think of it as...', 'similarly in...' "
        "constructions. Understanding built through connections."
    ),
    "socratic": (
        "Exploration through questions and provisional answers. The text poses "
        "questions, offers tentative responses, then uses those answers to generate "
        "new questions. An inquiry-driven structure where questions guide the "
        "explanation forward."
    ),
    "contrastive": (
        "Explores topics by comparing and contrasting multiple perspectives or "
        "approaches. Presents different viewpoints side by side, evaluating their "
        "relative strengths and weaknesses. Frequent use of 'on the other hand', "
        "'whereas', 'in contrast' language."
    ),
    "dialectical": (
        "Proposes a clear position, then challenges it with counterarguments, "
        "then works toward a synthesis or revised understanding. Shows a "
        "thesis-challenge-revision structure. The text argues with itself, "
        "building through productive disagreement."
    ),
}

LIKERT_DONOR = "anamnesis/scripts/run_judge_scoring.py"


def likert_system_prompt() -> str:
    """Build the Likert judge's system prompt from the rubric, as the donor did.

    The judge sees the text and the writing prompt and not the mode instruction,
    which is what makes it blind; the response schema is stated in the prompt
    because the reply is parsed as text.
    """
    mode_descriptions = [
        f"{i}. **{mode.title()}**: {desc}"
        for i, (mode, desc) in enumerate(JUDGE_MODE_DESCRIPTIONS.items(), 1)
    ]
    ratings_json = ",\n    ".join(f'"{m}": <1-5>' for m in VALID_MODES)
    mode_list = ", ".join(VALID_MODES)
    return (
        "You are an expert text analyst. You will be shown a text generated by a "
        "language model in response to a writing prompt. The model was given a "
        "specific processing mode instruction (which you do not know), and your job "
        "is to assess which processing mode(s) the text exhibits.\n\n"
        "Rate the text on each of these dimensions from 1 (not at all present) "
        "to 5 (strongly and consistently present throughout):\n\n"
        + "\n\n".join(mode_descriptions)
        + "\n\n"
        "After rating, classify which single mode BEST describes the overall text.\n\n"
        "Respond with ONLY a JSON object (no markdown fencing, no commentary):\n"
        "{\n"
        '  "ratings": {\n'
        f"    {ratings_json}\n"
        "  },\n"
        f'  "primary_mode": "<one of: {mode_list}>",\n'
        '  "confidence": <1-5>,\n'
        '  "reasoning": "<1 sentence explanation>"\n'
        "}"
    )


def likert_user_message(topic: str, text: str) -> str:
    """The Likert judge's user turn: the writing prompt and the text, nothing else."""
    return f"**Writing prompt:** {topic}\n\n**Generated text:**\n\n{text}"


__all__ = [
    "ANALOGICAL",
    "ANNEX_DONOR",
    "ANNEX_TEMPLATE",
    "AnswerFormat",
    "COHERENCE_DONOR",
    "COHERENCE_PROMPT",
    "DIGIT_1_5_PATTERN",
    "FORMALITY",
    "JUDGE_MODE_DESCRIPTIONS",
    "LETTER_PATTERN",
    "LIKERT_DONOR",
    "MODE",
    "MODE_DESCRIPTIONS",
    "PROMPT_SETS",
    "PromptSet",
    "PromptSetError",
    "RenderedPrompt",
    "SHIFT_MODE_DESCRIPTIONS",
    "SOCRATIC",
    "TEMPERATURE",
    "TEMPERATURE_DESCRIPTION",
    "VALID_MODES",
    "likert_system_prompt",
    "likert_user_message",
    "prompt_set",
]
