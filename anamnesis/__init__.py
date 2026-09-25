"""Anamnesis: the instrument for computational state signatures.

A signature is a lossy compression of the causal history of a forward pass over
a span of text — what the computation did, not what the text said.

Importing this package does nothing but make its submodules addressable. Each
one is imported for what it holds:

* :mod:`anamnesis.config` — per-model presets, one pass's settings, named runs.
  The presets and the runs are registry files, and further model files are named by
  ``ANAMNESIS_MODELS``, so a checkpoint is addressable by name without a code change.
* :mod:`anamnesis.modes` — the processing-mode registry: sets with their prompts and
  label indices, vocabularies a banked corpus carries that no prompt here produces,
  and the prompt-swap pairs the confound test is built from. ``ANAMNESIS_MODE_SETS``
  names further files, so a run over another researcher's modes needs no code either.
* :mod:`anamnesis.feature_map` — the executable ``source × method × depth``
  taxonomy: what a feature name means, read by extraction and analysis alike.
* :mod:`anamnesis.provenance` — the digests a bank is stamped with and read back
  through, which is how two banks are known to be joinable.
* :mod:`anamnesis.optimize` — black-box search in high dimensions, and the probe
  that says whether a budget is worth spending. Domain-free numerics: a caller
  with a fitness function and a dimension is its whole audience.
* :mod:`anamnesis.shortfall` — expected-versus-produced accounting and the
  refusal a command makes when it produced fewer units than it was asked for,
  which is how a partial corpus stops being reported as a complete one.

The subpackages each open with a docstring that routes their own contents:

* :mod:`anamnesis.extraction` — the write side: what a forward pass is read into,
  the hook path and the fast lane that read it, and the replay that makes a
  signature a fact about a span of text rather than about a generation.
* :mod:`anamnesis.analysis` — the read side: the gauntlet's eleven standing
  analyses, the battery's metrology, and the analyses that each answer one question.
* :mod:`anamnesis.steering` — whether a direction in the residual stream can make a
  pass run a given way: construction, screens, gates and readouts.
* :mod:`anamnesis.judging` — the behavioural channel: blind forced choices and
  ratings from a model provider, the one part that talks to one.
* :mod:`anamnesis.orchestration` — which device a worker gets and how work is
  partitioned; nothing in it computes a feature.
* :mod:`anamnesis.scripts` — the command-line entry points, in the order a run moves
  through them.

One more module sits beside them. :mod:`anamnesis.synthetic_bank` writes a bank of
the right shape drawn from a seed, so the read side runs before there is a model to
run it on.

The package directory also ships :mod:`anamnesis.config`'s run registry and the
prompt-set data, so a checkout is enough to describe a run without reaching for
a data store.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
