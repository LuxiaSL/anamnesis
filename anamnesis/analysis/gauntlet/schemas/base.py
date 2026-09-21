"""The one configuration every result model shares.

``extra="forbid"`` is the whole policy: a checkpointed ``results.json`` that
carries a key no model declares fails validation loudly instead of being read
with the unknown field dropped. Schema drift in an analysis result is not a
cosmetic problem — it means the file and the code disagree about what a number
is — so it surfaces at the boundary rather than downstream.

Two rules follow from it and hold across every section module:

- Success-path fields stay required; error-path fields are ``Optional`` with
  default ``None``, so the ``{"error": "..."}`` stub a failed section writes
  round-trips through ``model_dump(exclude_none=True)`` unchanged.
- Numpy arrays are never stored on a model. Section runners call ``.tolist()``,
  ``float()`` or ``int()`` before handing data to a schema, which is what makes
  the models JSON-shaped by construction rather than by a serializer hook.
"""

from __future__ import annotations

from pydantic import ConfigDict

_FORBID = ConfigDict(extra="forbid")
"""Shared ``model_config``: unknown keys are a validation error, not a shrug."""
