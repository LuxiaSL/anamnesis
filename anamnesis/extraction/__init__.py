"""Extraction: what a forward pass is read into, and the code that reads it.

Three layers, kept apart on purpose:

* :mod:`anamnesis.extraction.state_extractor` — the numeric anchor. Raw arrays in,
  a named feature vector out, in pure numpy. It carries no model awareness, so the
  numbers it produces can be reproduced on a machine with no GPU and no weights,
  and every faster path is defined as agreeing with it.
* :mod:`anamnesis.extraction.state_extractor_reference` — the golden master the
  anchor's optimisations are checked against. The two are read together: one is
  the claim, the other is the proof, and ``tests/test_extraction_equivalence.py``
  is where they meet.
* :mod:`anamnesis.extraction.feature_families` — the engineered families, each a
  self-contained extractor over banked tensors, orchestrated by
  :mod:`anamnesis.extraction.feature_pipeline`.

:mod:`anamnesis.extraction.raw_saver` is the on-disk form of a capture, and the
reason the families can run without a model: what a generation was is banked once
and re-read as often as the feature set changes.

This module imports none of them. A submodule is addressed by name, which keeps
importing the package free of the heaviest dependency any one member happens to
need, and keeps the anchor's purity a property a test can assert.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
