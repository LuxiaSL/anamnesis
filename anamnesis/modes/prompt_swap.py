"""Prompt-swap pairs: the confound test that rides along with every batch.

A prompt-swap generation puts one mode's system prompt in the context window and
directs the model, in the user turn, to execute a different mode. The two cues
disagree, so a feature that classifies these samples by system prompt is reading
the text in the context window, and a feature that classifies them by what the
model actually did is reading the execution.

That distinction is the difference between a result and an artifact, which is
why swap generations belong in every extraction batch rather than in a separate
validation run: a new feature family is answered on the same pass that
introduces it.
"""

from __future__ import annotations

from dataclasses import dataclass

from anamnesis.modes.registry import CORE_MODE_SET, mode_set

DEFAULT_USER_TEMPLATE = "Write about: {topic}"
"""The user turn a non-swapped generation uses, and the base a swap prepends to."""


@dataclass(frozen=True)
class PromptSwapPair:
    """A system prompt from one mode against an execution directive from another.

    Attributes
    ----------
    system_mode : str
        The mode whose system prompt sits in the context window.
    execution_mode : str
        The mode whose behaviour the user directive asks for.
    user_directive : str
        The instruction that overrides the system prompt.
    label : str
        Short name for the pair, as stored beside the generation.
    """

    system_mode: str
    execution_mode: str
    user_directive: str
    label: str

    def get_system_prompt(self) -> str:
        """The system prompt of :attr:`system_mode`, from the core mode set.

        Raises
        ------
        anamnesis.modes.registry.UnknownModeSetError
            When the pair names a mode the core set does not hold.
        """
        return mode_set(CORE_MODE_SET).prompt(self.system_mode)

    def format_user_prompt(self, topic: str, template: str = DEFAULT_USER_TEMPLATE) -> str:
        """The user turn: the override directive, then the topic as usual."""
        base = template.format(topic=topic)
        return f"{self.user_directive}\n\n{base}"


PROMPT_SWAP_PAIRS: list[PromptSwapPair] = [
    # The canonical confound test, carried since the original supplementary:
    # a socratic system prompt against linear execution.
    PromptSwapPair(
        system_mode="socratic",
        execution_mode="linear",
        user_directive=(
            "Write your response as straightforward sequential exposition. "
            "Do not ask questions or use Socratic devices."
        ),
        label="socratic→linear",
    ),

    # The nearest-neighbour pair: dialectical and contrastive sit closest in both
    # the 3B and the 8B topology. A detector that cannot separate this swap is
    # reporting that the two modes really are computationally alike.
    PromptSwapPair(
        system_mode="dialectical",
        execution_mode="contrastive",
        user_directive=(
            "Explore this by comparing and contrasting multiple perspectives. "
            "Do not argue for or against a position — present balanced comparisons."
        ),
        label="dialectical→contrastive",
    ),

    # Easy against hard: analogical carries the strongest signal in the set, and
    # this pair asks whether that signal is the prompt text or the computation.
    PromptSwapPair(
        system_mode="analogical",
        execution_mode="linear",
        user_directive=(
            "Write your response as straightforward sequential exposition. "
            "Do not use analogies, metaphors, or comparisons to other domains."
        ),
        label="analogical→linear",
    ),
]
"""The swap pairs every extraction batch includes."""
