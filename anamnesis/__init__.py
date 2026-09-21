"""Anamnesis: the instrument for computational state signatures.

A signature is a lossy compression of the causal history of a forward pass over
a span of text — what the computation did, not what the text said.

Importing this package does nothing but make its submodules addressable. Each
one is imported for what it holds:

* :mod:`anamnesis.config` — per-model presets, one pass's settings, named runs.
* :mod:`anamnesis.modes` — the processing-mode prompt sets and the prompt-swap
  pairs the confound test is built from.
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

The package directory also ships :mod:`anamnesis.config`'s run registry and the
prompt-set data, so a checkout is enough to describe a run without reaching for
a data store.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
