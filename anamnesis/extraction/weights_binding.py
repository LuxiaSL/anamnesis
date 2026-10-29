"""A lane binds its weights: the check, and the record a bound row carries.

A registry row (:class:`~anamnesis.config.ModelPreset`) may declare
``checkpoint_sha256``, the :func:`~anamnesis.extraction.weights_digest.weights_digest`
of the model it describes. Every place this package loads or receives a model for such
a row checks the loaded weights against it and refuses a mismatch by name:
:func:`~anamnesis.extraction.model_loader.load_model` for a configuration built from the
row, and :func:`~anamnesis.extraction.fast.runtime.prepare_fast_lane` and
:func:`~anamnesis.extraction.fast.runtime.resolve_fast_lane` for the fast lane. A caller
that builds a :class:`~anamnesis.extraction.fast.features.GpuFeatureLane` or an
:class:`~anamnesis.extraction.fast.external.ExternalSpan` over a model of its own calls
:func:`require_weights` itself, once per model it signs with, and stamps the record it
returns beside each row's ``lane_id``.

A row that declares no digest binds nothing: no digest is computed, nothing is
recorded, and its lanes are exactly what they were before the field existed.

The record (:func:`weights_record`) is what
:func:`~anamnesis.analysis.lane_guard.require_declared_contrast` reads to tell which
weights a row was computed under. A bank stamps it as ``weights`` inside the row's
``extraction_lane`` metadata (or at the top level of the row's metadata).
"""

from __future__ import annotations

from typing import Any

import torch

from anamnesis.config import ModelPreset, resolve_preset
from anamnesis.extraction.weights_digest import WEIGHTS_DIGEST_VERSION, weights_digest

WEIGHTS_RECORD_KEY = "weights"
"""The metadata key a row's weights record sits under."""


class WeightsMismatchError(ValueError):
    """A model whose weights are not the ones its registry row binds."""


def weights_record(preset: str | ModelPreset) -> dict[str, Any] | None:
    """The record a row computed under ``preset``'s bound weights carries, or None.

    The record names the digest definition, the registry key, the weights digest, and
    the adapter's file digest when the row declares one. A row that binds no weights
    has no record.
    """
    row = resolve_preset(preset)
    if row.checkpoint_sha256 is None:
        return None
    record: dict[str, Any] = dict(
        digest=WEIGHTS_DIGEST_VERSION, preset=row.name, checkpoint_sha256=row.checkpoint_sha256
    )
    if row.adapter_sha256 is not None:
        record["adapter_sha256"] = row.adapter_sha256
    return record


def check_weights_digest(digest: str, expected: str, *, preset_name: str) -> None:
    """Refuse a weights digest other than the one ``preset_name`` binds.

    Raises
    ------
    WeightsMismatchError
        Naming the preset and both digests.
    """
    if digest != expected:
        raise WeightsMismatchError(
            f"preset {preset_name!r} binds weights {expected}, and the loaded model's "
            f"{WEIGHTS_DIGEST_VERSION} digest is {digest}; a different checkpoint needs its "
            "own preset key and its own calibration"
        )


def require_weights(
    model: Any, preset: str | ModelPreset, *, digest: str | None = None
) -> dict[str, Any] | None:
    """Refuse ``model`` unless its weights are the ones ``preset`` binds.

    Parameters
    ----------
    model
        A ``torch.nn.Module``, or anything with a ``model`` attribute holding one (a
        :class:`~anamnesis.extraction.model_loader.LoadedModel`).
    preset
        A registry name or a row.
    digest
        The model's :func:`weights_digest`, when the caller has already computed it
        for these exact weights; otherwise it is computed here, at the cost of one
        host copy of every tensor.

    Returns
    -------
    The row's :func:`weights_record`, to stamp beside each row's lane id; None when the
    row binds no weights, in which case nothing is computed.

    Raises
    ------
    WeightsMismatchError
        When the digest is not the row's ``checkpoint_sha256``.
    """
    row = resolve_preset(preset)
    if row.checkpoint_sha256 is None:
        return None
    if digest is None:
        # A Hugging Face causal LM is itself a module with a ``model`` attribute (its
        # decoder), so only a non-module is unwrapped.
        module = model if isinstance(model, torch.nn.Module) else model.model
        digest = weights_digest(module)
    check_weights_digest(digest, row.checkpoint_sha256, preset_name=row.name)
    return weights_record(row)
