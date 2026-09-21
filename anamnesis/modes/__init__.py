"""Processing modes: the system prompts a run's labels refer to.

Three sets, and the confound test that travels with them:

* :mod:`anamnesis.modes.run4_modes` — the five format-controlled modes of the
  core protocol.
* :mod:`anamnesis.modes.extended_modes` — those five plus structured, compressed
  and associative, under the same format constraint.
* :mod:`anamnesis.modes.prompt_swap` — pairs that put one mode's system prompt
  against another mode's execution, which is how a feature is shown to read the
  computation rather than the context window.

The prompt strings are the protocol: a label in banked data means the text in
this package, so the strings are read rather than edited. Modes from earlier
protocols are part of the frozen record and are not re-exported here — a mode set
this package names is one an extraction can still run.
"""

from __future__ import annotations

from anamnesis.modes.extended_modes import EXTENDED_MODE_INDEX, EXTENDED_MODES
from anamnesis.modes.prompt_swap import (
    DEFAULT_USER_TEMPLATE,
    PROMPT_SWAP_PAIRS,
    PromptSwapPair,
)
from anamnesis.modes.run4_modes import FORMAT_CONSTRAINT, RUN4_MODE_INDEX, RUN4_MODES

__all__ = [
    "DEFAULT_USER_TEMPLATE",
    "EXTENDED_MODES",
    "EXTENDED_MODE_INDEX",
    "FORMAT_CONSTRAINT",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
    "RUN4_MODES",
    "RUN4_MODE_INDEX",
]
