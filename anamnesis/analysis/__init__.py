"""Analysis: what is read off a bank of signatures, and the gates on reading it.

Extraction produces signatures; this package consumes them. The boundary matters
for one reason in particular: a signature carries the identity of the arithmetic
that produced it, and two signatures produced by different arithmetic are not
two measurements of the same quantity.

* :mod:`~anamnesis.analysis.lane_guard` — the read-side gate on that identity.
  Historical banks predate lane tagging and stay readable, but an untagged bank
  cannot be certified as one known backend, so it cannot be combined with a
  tagged one. Every loader that assembles a scientific input from more than one
  file passes its metadata through here first.

What reads the signatures, grouped by what it is asking:

* :mod:`~anamnesis.analysis.gauntlet` — the eleven standing analyses over one
  corpus, and the loader every other reading here borrows.
* :mod:`~anamnesis.analysis.battery` — the metrology prior to every arm: how large
  a difference has to be, how many samples it takes to see one
  (:mod:`~anamnesis.analysis.battery.stage0`), and which rows the internals see
  that the cheap readers miss (:mod:`~anamnesis.analysis.battery.census`).
* :mod:`~anamnesis.analysis.audit_lib` — the controls an audit runs under: the
  length control, the leak-free folds, the surface sampling and the readout pair.
* :mod:`~anamnesis.analysis.prompt_swap` — whether the signal is the instruction
  or the execution, which is the confound the whole programme rests on.
* :mod:`~anamnesis.analysis.subfamily` — which part of a family carries a family's
  signal.
* :mod:`~anamnesis.analysis.cross_run` and
  :mod:`~anamnesis.analysis.contrastive_mlp` — whether two corpora's mode
  vocabularies name the same thing, and the learned projection that asks.
* :mod:`~anamnesis.analysis.complementarity` — the readings that exist only
  *between* banked results, so they cost a re-read rather than a re-run.
* :mod:`~anamnesis.analysis.leak_gate` — whether a feature set carries the signal
  or the topic it was measured on.
* :mod:`~anamnesis.analysis.encoder_ladder` — whether a hand-built projection or the
  raw state is the limitation, which is the only way to read a null from features.
* :mod:`~anamnesis.analysis.text_stats` — the text channel beside the signature,
  which is what catches a degenerate generation reading as a change.

This module imports none of them: a loader names the guard, the guard names
nothing back, and addressing one reading does not pull in another's dependencies.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
