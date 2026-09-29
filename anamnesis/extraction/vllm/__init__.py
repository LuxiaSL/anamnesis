"""The vLLM lane: the fast lane's features, computed from a vLLM engine.

The fast lane (:mod:`anamnesis.extraction.fast`) replays each generation through
a hooked eager forward. This package computes the same named feature vector from
a vLLM engine instead: an instrumented attention backend gathers the attention
statistics inside the engine's own kernel, the model runner is hooked for the
rest of the substrate, and the vector is reduced afterwards with the fast lane's
own reducers. A lane exists per model in
:data:`anamnesis.extraction.vllm.envelope.LANE_MODELS`, only inside the envelope
declared there, and on a host only after that host's install check.

* :mod:`~anamnesis.extraction.vllm.envelope` — the engine settings, environment,
  package pins, models, execution conditions and lane identity, and the startup
  guard that refuses anything else.
* :mod:`~anamnesis.extraction.vllm.stats_kernel`,
  :mod:`~anamnesis.extraction.vllm.second_pass`,
  :mod:`~anamnesis.extraction.vllm.step_products`,
  :mod:`~anamnesis.extraction.vllm.rows` and
  :mod:`~anamnesis.extraction.vllm.backend` — the instrumented attention: an
  online-statistics copy of the engine's Triton kernel, a bounded second pass for
  the rows whose normalized probabilities are needed, their reduction, which rows
  those are, and the backend that routes them.
* :mod:`~anamnesis.extraction.vllm.capture`,
  :mod:`~anamnesis.extraction.vllm.runner` and
  :mod:`~anamnesis.extraction.vllm.receipts` — the verified capture of every
  request's substrate, group by group, with content receipts.
* :mod:`~anamnesis.extraction.vllm.tensor_parallel` and
  :mod:`~anamnesis.extraction.vllm.tp_worker` — the same capture for a model split
  over several GPUs: run inside every worker, gathered to the single-GPU layouts
  in rank order, and checked for agreement across ranks.
* :mod:`~anamnesis.extraction.vllm.readout` and
  :mod:`~anamnesis.extraction.vllm.adapter` — the reduction to the feature vector,
  in a process that never imports the engine.
* :mod:`~anamnesis.extraction.vllm.conformance` — the install check's decision:
  identical, own-lane (a lane of this host's own) or refused.
* :mod:`~anamnesis.extraction.vllm.hub` — the calibrations the fixtures were
  reduced with, pinned by digest and fetched on first use.
* :mod:`~anamnesis.extraction.vllm.transfer` — the lane-agreement audit: how far
  two lanes sit apart on matched tokens, read against a base lane's measured
  regime.
* :mod:`~anamnesis.extraction.vllm.extensions` — extension lanes: the guard that
  admits a declared fine-tune on its identity, recording an audit when it names
  one.
* :mod:`~anamnesis.extraction.vllm.runtime` and
  :mod:`~anamnesis.extraction.vllm.steps` — the two processes a pass runs as, the
  host fingerprint, and the check and replay flows the commands call.

This module imports none of them: the engine is an optional dependency, and
nothing here should require it to be importable.
"""
