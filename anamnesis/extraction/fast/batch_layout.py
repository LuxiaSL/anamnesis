"""Explicit cached-replay packing, without altering any row's absolute positions.

Prefixes are left padded, continuations right padded. Physical KV-cache indices
are shared across the batch; RoPE/feature positions remain per-row absolute
positions. Padding is removed before any attention feature reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplaySpan:
    tokens: tuple[int, ...]
    start: int
    end: int

    def validate(self, vocab_size: int) -> None:
        if (
            not 0 < self.start < self.end <= len(self.tokens)
            or self.end - self.start < 2
        ):
            raise ValueError("nonempty prefix and continuation >=2 required")
        if any(
            not isinstance(t, (int, np.integer)) or not 0 <= t < vocab_size
            for t in self.tokens[: self.end]
        ):
            raise ValueError("invalid token ID")


@dataclass(frozen=True)
class BatchLayout:
    spans: tuple[ReplaySpan, ...]
    prefix_ids: NDArray[np.int64]
    prefix_mask: NDArray[np.int64]
    prefix_positions: NDArray[np.int64]
    continuation_ids: NDArray[np.int64]
    attention_mask: NDArray[np.int64]
    continuation_positions: NDArray[np.int64]
    composition_sha256: str

    @property
    def prefix_width(self) -> int:
        return self.prefix_ids.shape[1]

    @property
    def continuation_width(self) -> int:
        return self.continuation_ids.shape[1]

    def attention_slices(self, row: int) -> tuple[slice, slice]:
        """Query/key slices removing padding; causal future zeros stay intact."""
        span = self.spans[row]
        length = span.end - span.start
        return slice(0, length), slice(
            self.prefix_width - span.start, self.prefix_width + length
        )


def pack_spans(
    spans: tuple[ReplaySpan, ...], vocab_size: int, pad_token: int = 0
) -> BatchLayout:
    if not spans or vocab_size < 1 or not 0 <= pad_token < vocab_size:
        raise ValueError("invalid batch/vocabulary/padding token")
    for span in spans:
        span.validate(vocab_size)
    batch = len(spans)
    prefix = max(s.start for s in spans)
    continuation = max(s.end - s.start for s in spans)
    prefix_ids = np.full((batch, prefix), pad_token, dtype=np.int64)
    prefix_mask = np.zeros_like(prefix_ids)
    prefix_positions = np.zeros_like(prefix_ids)
    continuation_ids = np.full((batch, continuation), pad_token, dtype=np.int64)
    continuation_mask = np.zeros_like(continuation_ids)
    continuation_positions = np.zeros_like(continuation_ids)
    for i, span in enumerate(spans):
        left = prefix - span.start
        length = span.end - span.start
        prefix_ids[i, left:] = span.tokens[: span.start]
        prefix_mask[i, left:] = 1
        prefix_positions[i, left:] = np.arange(span.start)
        continuation_ids[i, :length] = span.tokens[span.start : span.end]
        continuation_mask[i, :length] = 1
        continuation_positions[i, :length] = np.arange(span.start, span.end)
    identity = dict(
        policy="left-prefix/right-continuation-v1",
        pad_token=pad_token,
        rows=[
            dict(tokens=list(s.tokens[: s.end]), start=s.start, end=s.end)
            for s in spans
        ],
    )
    return BatchLayout(
        spans,
        prefix_ids,
        prefix_mask,
        prefix_positions,
        continuation_ids,
        np.concatenate((prefix_mask, continuation_mask), axis=1),
        continuation_positions,
        hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    )
