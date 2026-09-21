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

The model-facing side is three more modules and a subpackage:

* :mod:`anamnesis.extraction.model_loader` — the checkpoint on a device with hooks
  on it. Keys, values and queries are captured pre-RoPE from the projection
  modules rather than from the cache, because a post-RoPE key has its position
  baked in and the geometric features would be reading position. It also hosts the
  activation-write path and the optional-hook pattern a new architecture extends.
* :mod:`anamnesis.extraction.streaming_generate` — the autoregressive loop, run so
  that collecting states costs one transfer per step instead of one tensor object
  per step per layer.
* :mod:`anamnesis.extraction.generation_runner` — one pass end to end: prompt,
  seed, generate, convert, extract, save, and the replay manifest that makes the
  run reproducible.
* :mod:`anamnesis.extraction.replay` — the determinism core. Teacher-forcing a
  realized token sequence reproduces the states that produced it, which is what
  makes a signature an object about a span of text rather than about a generation.

This module imports none of them. A submodule is addressed by name, which keeps
importing the package free of the heaviest dependency any one member happens to
need, and keeps the anchor's purity a property a test can assert.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
