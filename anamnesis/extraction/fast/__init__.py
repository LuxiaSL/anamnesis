"""The fast lane: the primary extraction path, computed on the device.

:mod:`anamnesis.extraction.state_extractor` is the specification — pure numpy,
raw arrays in, a named vector out, reproducible on a machine with no GPU. This
package is the implementation of that specification that runs at the speed the
program needs, and :mod:`anamnesis.extraction.equivalence` is the harness that
says the two agree on a given box.

The lane is not a fork of the feature definitions. It imports the families'
naming and slicing directly, and derives its whole output schema from the
canonical pipeline rather than restating it, so a family that gains a feature
gains it on both paths or the schema check fails:

* :mod:`~anamnesis.extraction.fast.features` — the entry point. One eager forward
  per span (or one batched prefill for several), with every reduction done before
  the tensors leave the device.
* :mod:`~anamnesis.extraction.fast.ops` — the scalar and time-series operators,
  in torch, holding the anchor's float32/float64 boundaries and its population
  standard deviation. The collector they write into refuses a duplicate name and
  refuses to emit a vector whose names are not exactly the declared schema.
* :mod:`~anamnesis.extraction.fast.attention` — the per-layer attention
  reduction, run inside the hook so the all-layer weights never coexist.
* :mod:`~anamnesis.extraction.fast.families` — residual, key/value/query, gate
  and cross-layer reductions.
* :mod:`~anamnesis.extraction.fast.schema` — the feature names and family
  slices, resolved by running the canonical extractor over tiny synthetic
  tensors. Names come out; numbers never do.
* :mod:`~anamnesis.extraction.fast.batch_layout` — ragged batch packing for
  cached replay. Prefixes left-padded, continuations right-padded, per-row
  absolute positions preserved, padding removed before any feature sees it.
* :mod:`~anamnesis.extraction.fast.runtime` — the arithmetic, schema, calibration
  and capture surface a lane pass runs against, resolved once for every entry
  point that runs one. A lane is only worth qualifying if the configuration
  measured is the configuration banked.
* :mod:`~anamnesis.extraction.fast.harvest` — one span against a model the
  caller keeps resident, in process: the feature vector, the lane's receipt and,
  when asked, the per-position logit series, with nothing written to disk.

This module imports none of them: addressing the layout arithmetic should not
pull in torch.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
