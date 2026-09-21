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

This module imports nothing: a loader names the guard, the guard names nothing
back.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
