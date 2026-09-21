"""Replay: reading a forward pass the model has already taken.

A decoder forward is a deterministic function of its input tokens, so the states
an autoregressive generation produced can be recovered by teacher-forcing the
realized sequence through one instrumented forward. That is what makes a
signature an object about a *span of text* rather than about a generation: replay
collects signatures over text the model never chose, and it reproduces the states
of text it did choose bit-for-bit on the same box.

Four modules are the capability itself, in the order a replay uses them:

* :mod:`~anamnesis.extraction.replay.manifest` — what a run can be replayed over:
  the realized token sequences and the prompt boundary that splits each one.
  Written by a generation pass; reconstructed, with validation, for a bank that
  predates manifests.
* :mod:`~anamnesis.extraction.replay.extract` — the plain teacher-forced pass.
  One forward with ``use_cache=False``, sliced to the same per-step alignment the
  generate path banks, which is what keeps features from the two paths
  comparable.
* :mod:`~anamnesis.extraction.replay.cache_surgery` — exact edits to a key/value
  cache: eviction, re-rotation to new positions, turn-aligned dialogue
  eviction. Pure tensor math, so the geometry is testable without a model.
* :mod:`~anamnesis.extraction.replay.cached` — a teacher-forced continuation
  against an injected, possibly surgered cache. The prompt boundary is the cache
  length, so the extractor's prompt/generated split lands exactly on the
  cache/continuation seam.

Two more are how a replay is actually run over a corpus:

* :mod:`~anamnesis.extraction.replay.cell` — the production loop and the capture
  surface it runs against. Three callers must run *the same* loop, which is the
  only basis on which the faster two are allowed to exist.
* :mod:`~anamnesis.extraction.replay.checkpoint_series` — that loop walked through
  a series of adapter checkpoints on one model load, with the pristine restore
  that keeps merge arithmetic from drifting along the series. What it varies is
  which weights the replay runs against, not how the replay works.

This module imports none of them: the manifest is readable with no model runtime
present, and addressing it should not pull one in.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
