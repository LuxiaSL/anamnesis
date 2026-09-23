"""Processing modes: the system prompts a run's labels refer to.

Two modules, and the confound test that travels with them:

* :mod:`anamnesis.modes.registry` — the registry. Mode sets with their prompts and
  label indices, label vocabularies a banked corpus carries that this package
  cannot run, and the predicted maps between two vocabularies. The rows are data,
  in ``mode_sets.json`` beside the module, and ``ANAMNESIS_MODE_SETS`` names further
  files of the same shape — so a run over a researcher's own modes needs no code.
* :mod:`anamnesis.modes.prompt_swap` — pairs that put one mode's system prompt
  against another mode's execution, which is how a feature is shown to read the
  computation rather than the context window.

Two sets ship. :data:`~anamnesis.modes.registry.CORE_MODE_SET` is the five
format-controlled modes of the core protocol, and
:data:`~anamnesis.modes.registry.EXTENDED_MODE_SET` extends it with structured,
compressed and associative under the same constraint. The five are the hard subset:
a pair drawn from them is prose a reader struggles to tell apart, and the three
additions are computationally more distinctive and therefore easier.

The prompt strings are the protocol: a label in banked data means the exact text
the registry holds, so the strings are read rather than edited, and their bytes are
pinned by the test suite. Modes from earlier protocols are part of the frozen
record and ship no prompts — a mode *set* this package names is one an extraction
can still run, and a vocabulary it can only read is filed as one.
"""

from __future__ import annotations

from anamnesis.modes.prompt_swap import (
    DEFAULT_USER_TEMPLATE,
    PROMPT_SWAP_PAIRS,
    PromptSwapPair,
)
from anamnesis.modes.registry import (
    CORE_MODE_SET,
    DEFAULT_MODE_MAPPING,
    EXTENDED_MODE_SET,
    MODE_SETS_ENV,
    MODE_SETS_FILE,
    Label,
    LabelVocabulary,
    MappingReference,
    Mode,
    ModeMapping,
    ModeRegistry,
    ModeRegistryError,
    ModeRegistryFile,
    ModeSet,
    ModeSetSpec,
    UnknownModeSetError,
    easy_modes,
    format_constraint,
    hard_modes,
    label_vocabulary,
    load_mode_registry,
    mapping_wildcards,
    mode_indices,
    mode_mapping,
    mode_prompts,
    mode_set,
    mode_set_names,
)

__all__ = [
    "CORE_MODE_SET",
    "DEFAULT_MODE_MAPPING",
    "DEFAULT_USER_TEMPLATE",
    "EXTENDED_MODE_SET",
    "MODE_SETS_ENV",
    "MODE_SETS_FILE",
    "Label",
    "LabelVocabulary",
    "MappingReference",
    "Mode",
    "ModeMapping",
    "ModeRegistry",
    "ModeRegistryError",
    "ModeRegistryFile",
    "ModeSet",
    "ModeSetSpec",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
    "UnknownModeSetError",
    "easy_modes",
    "format_constraint",
    "hard_modes",
    "label_vocabulary",
    "load_mode_registry",
    "mapping_wildcards",
    "mode_indices",
    "mode_mapping",
    "mode_prompts",
    "mode_set",
    "mode_set_names",
]
