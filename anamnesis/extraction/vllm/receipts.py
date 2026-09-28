"""The substrate a capture holds, and content receipts for its tensors.

A capture is refused unless its fields are exactly :data:`SUBSTRATE_FIELDS`:
the seven projections and outputs the fast lane reduces, and the attention
observables the instrumented backend produces. Engine-returned logprobs are
sampler outputs, not substrate, and are refused by name, so no feature path can
read them.

Receipts hash each tensor's native bytes without converting its precision, so
a receipt is a statement about exactly what was captured.
"""
from __future__ import annotations

import hashlib
import json

import torch

SUBSTRATE_FIELDS = frozenset({
    'hidden', 'keys', 'values', 'queries', 'gates', 'logits', 'chosen',
    'attn_stats', 'attn_coverage', 'attn_h_mean', 'attn_h_heads',
    'attn_spectral_rows', 'attn_decay_rows', 'attn_span_rows',
    'attn_entropy_rows', 'attn_head_ent', 'attn_head_sink', 'attn_head_prompt',
    'attn_head_recency'})
"""Every field a capture holds: the hidden states, the sampled layers' queries,
keys, values and gate pre-activations, the prompt logits and chosen tokens, the
first-pass attention statistics of every layer, and the second pass's reduced
products."""


def assert_substrate_fields(fields, *, context: str) -> None:
    """Refuse a capture whose fields are not exactly :data:`SUBSTRATE_FIELDS`.

    Raises
    ------
    ValueError
        Naming ``context`` and any logprob, extra or missing field.
    """
    names = {str(field) for field in fields}
    returned = sorted(name for name in names if 'logprob' in name.lower())
    if returned:
        raise ValueError(
            f'{context}: engine-returned logprob fields are never feature '
            f'substrate: {returned}')
    extra = sorted(names - SUBSTRATE_FIELDS)
    missing = sorted(SUBSTRATE_FIELDS - names)
    if extra or missing:
        raise ValueError(
            f'{context}: capture fields must be exactly the substrate; '
            f'extra {extra}, missing {missing}')


def tensor_receipt(value: torch.Tensor) -> dict:
    """Hash native bytes without converting floating-point precision."""
    tensor = value.detach().cpu().contiguous()
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def capture_receipt(capture: dict) -> dict:
    """Return stable named hashes for every observation and their joint digest."""
    entries = {}
    for name, value in sorted(capture.items()):
        if isinstance(value, dict):
            for layer, tensor in sorted(value.items()):
                entries[f'{name}/{layer}'] = tensor_receipt(tensor)
        else:
            entries[name] = tensor_receipt(value)
    encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return {'tensors': entries, 'sha256': hashlib.sha256(encoded).hexdigest()}
