"""Orchestration: getting a pass onto the machine it has to run on.

Nothing here computes a feature. What lives here is the arithmetic and the
process bookkeeping between a pass and the hardware — and the reason that can be
a separate layer at all is a property of the science: a generation is seeded from
its own coordinates and a replay is teacher-forced from its own banked tokens, so
no output depends on which worker produced it, in what order, or in what process.
Scheduling is therefore free to be a cost decision.

* :mod:`anamnesis.orchestration.gpu` — which device a worker gets, what
  environment it is handed, and the guard that refuses a roster walked one cell
  per model load.
* :mod:`anamnesis.orchestration.launch` — partition the work, spawn the
  subprocesses, wait, report; and assemble a generation pass's banked records
  into a run.
* :mod:`anamnesis.orchestration.workers` — workers that stay loaded, fed by a
  crash-safe file queue, and the byte-level parity check that licenses them.

Importing this package imports none of the three: a launcher plan is arithmetic
and reading it should not require torch.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
