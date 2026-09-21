"""The five format-controlled processing modes.

Each mode is a system prompt that asks for a different way of working through a
topic, and every one of them carries :data:`FORMAT_CONSTRAINT`: flowing prose,
no lists, no headers. That clause is what makes the set an instrument rather
than a style sampler — without it a classifier can separate the modes from
surface layout alone, and the question of whether the *computation* differs
never gets asked.

Four of the five — linear, socratic, contrastive, dialectical — are the hard
set: they produce prose a reader struggles to tell apart. Analogical is the easy
one, and the gap between the two is a calibration for any detector run over this
corpus.
"""

from __future__ import annotations

FORMAT_CONSTRAINT = (
    " Write in flowing paragraphs. Do not use bullet points, numbered lists, "
    "headers, or any visual formatting structure."
)
"""The clause appended to every mode prompt, which removes surface layout as a cue."""

RUN4_MODES: dict[str, str] = {
    "linear": (
        "Present your ideas in a clear sequence, each building on the last. "
        "Move forward without backtracking or reconsidering previous points. "
        "Lay out the topic step by step from beginning to end."
        + FORMAT_CONSTRAINT
    ),
    "analogical": (
        "Explain this primarily through extended analogies and parallels to "
        "other domains. For each key concept, find a comparison from everyday "
        "life or another field that illuminates it. Build understanding "
        "through these connections."
        + FORMAT_CONSTRAINT
    ),
    "socratic": (
        "Develop your exploration through a sequence of questions and "
        "provisional answers. Pose a question, offer a tentative answer, "
        "then use that answer to generate the next question. Let the chain "
        "of inquiry drive the explanation forward."
        + FORMAT_CONSTRAINT
    ),
    "contrastive": (
        "Explore this by comparing and contrasting multiple perspectives or "
        "approaches. For each major point, present at least two different "
        "viewpoints and evaluate their relative strengths and weaknesses."
        + FORMAT_CONSTRAINT
    ),
    "dialectical": (
        "Begin by proposing a clear position on the topic. Then challenge "
        "that position with the strongest counterarguments you can find. "
        "Work toward a revised understanding that accounts for both the "
        "original position and its critiques."
        + FORMAT_CONSTRAINT
    ),
}
"""Mode name to system prompt, for the five modes of the core protocol."""

RUN4_MODE_INDEX: dict[str, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
}
"""Mode name to label index; the order every stored label in this corpus uses."""
