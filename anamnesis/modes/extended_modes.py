"""The eight-mode set: the core five plus three format-controlled additions.

Structured, compressed and associative ask for ways of working that announce
themselves in layout — an outline, a dense summary, a drifting chain. Here they
carry the same format constraint as the core five, so what remains of their
distinctiveness is in the computation rather than in the shape of the page. That
is what makes them usable for feature engineering: surface signal is gone before
a feature family ever sees the text.

The eight span the range the corpus is built to cover. Analogical, structured,
compressed and associative are computationally distinctive; linear, socratic,
contrastive and dialectical are the hard set a detector is judged on.
"""

from __future__ import annotations

from anamnesis.modes.run4_modes import FORMAT_CONSTRAINT, RUN4_MODES

_FORMAT_CONTROLLED_ADDITIONS: dict[str, str] = {
    "structured": (
        "Organize your response as a systematic analysis. Identify the key "
        "components or dimensions of the topic, address each one methodically, "
        "and show how they relate to each other. Be thorough and organized "
        "in your coverage."
        + FORMAT_CONSTRAINT
    ),
    "compressed": (
        "Express your ideas as densely and concisely as possible. Pack "
        "maximum information into minimum words. Every sentence should "
        "carry significant meaning — eliminate all filler, hedging, and "
        "unnecessary elaboration. Be precise and information-dense."
        + FORMAT_CONSTRAINT
    ),
    "associative": (
        "Let your thinking flow freely through associations and connections. "
        "When one idea reminds you of another, follow that thread. Explore "
        "the topic through a stream of related concepts, tangents, and "
        "unexpected connections rather than a predetermined structure."
        + FORMAT_CONSTRAINT
    ),
}

EXTENDED_MODES: dict[str, str] = {
    **RUN4_MODES,
    **_FORMAT_CONTROLLED_ADDITIONS,
}
"""Mode name to system prompt for all eight modes."""

EXTENDED_MODE_INDEX: dict[str, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
    "structured": 5,
    "compressed": 6,
    "associative": 7,
}
"""Mode name to label index; the core five keep their indices from the five-mode set."""
