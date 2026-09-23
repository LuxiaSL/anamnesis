"""Equivalence: the proof that the specification and the fast lane agree.

A signature is only comparable to another signature computed the same way. Two
extraction paths exist — the numeric anchor in
:mod:`anamnesis.extraction.state_extractor` and the device lane in
:mod:`anamnesis.extraction.fast` — so "they agree" is a claim that has to be
measured, and measured on the machine the numbers will be produced on.

**Agreement is a property of a box, not of this code.** Floating-point reduction
order differs across BLAS builds, GPU architectures and thread counts, so a
different machine gives different numbers in the last few digits. That is
expected rather than wrong. What is wrong is joining outputs from two boxes, or
two lanes, inside one contrast: the difference between them is not signal, and
:mod:`anamnesis.analysis.lane_guard` is the read-side gate that refuses the join.
`anamnesis/scripts/qualify_box.py` is how a user qualifies their own box.

Two modules:

* :mod:`~anamnesis.extraction.equivalence.fidelity` — the evidence checks. Given
  a reference batch, a candidate batch, an independent repeat of the candidate,
  and a ruler the caller supplies, it renders the gate verdicts. It fits nothing:
  a threshold is a fixed module constant, an unexecuted gate never reads as a pass, and
  an insufficient bound is reported as needing more work rather than as failure.
* :mod:`~anamnesis.extraction.equivalence.path_floor` — cheap sufficient
  certificates for the per-row path bound. Because the L2 norm of a coordinate
  vector is at least the largest single coordinate, a bound computed from the
  first incremental step alone can already exceed a candidate's full-vector
  distance, and the remaining steps need not run.

This module imports neither: the certificates need a model runtime, the checks
do not, and reading one should not require the other.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
