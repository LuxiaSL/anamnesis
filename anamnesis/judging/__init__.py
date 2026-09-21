"""Judging: the behavioural channel, and the discipline that makes it citable.

A judge is a measuring instrument. It has a dynamic range, a grade, and a way of
being fooled, and a judged number reported without those is a rumour with a
decimal point. The three modules here are what keeps one from being that:

* :mod:`~anamnesis.judging.harness` — the blind two-alternative forced choice.
  Pair construction, the key held in a type that no prompt is rendered from, the
  ceiling control that decides whether a null may be called, the reader-grade
  ladder, the anti-circularity rule, Wilson intervals, the coherence gate and
  the model fallback ladder.
* :mod:`~anamnesis.judging.prompts` — the five versioned prompt sets and the two
  shared rubrics, verbatim and provenance-stamped. Prompts are data: what a judge
  was asked is part of what its number means, so the text is pinned by hash.
* :mod:`~anamnesis.judging.likert` — the non-2AFC paradigm, kept for the one
  thing a forced choice cannot yield: a graded per-text purity that can be
  correlated against signature-space distance.

Judging talks to a model provider, which the rest of the instrument does not, so
the provider SDKs are an optional extra and their imports are deferred to first
use. This module imports nothing for the same reason: naming the judging package
must not pull a vendor client, a scoring stack or an HTTP library into a process
that only wanted to read a docstring.

API keys are read from the environment and from nowhere else. No key is ever an
argument, a field, or anything a receipt could carry.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
