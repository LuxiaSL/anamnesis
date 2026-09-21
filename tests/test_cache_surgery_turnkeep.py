"""CPU smoke for the A4b turn-keep port (P3) — cache_surgery dialogue eviction geometry.

Validates the utilities ported from kv-rotation chat.py: turn_token_spans (prefix-stable span
recovery), oldest_turns_to_evict (protections + oldest-first + unreachable-not-clipped), and
turn_keep_indices (sinks + span removal). No model/GPU — a mock prefix-stable tokenizer.
"""
from __future__ import annotations

import torch

from anamnesis.extraction.replay.cache_surgery import (
    oldest_turns_to_evict,
    turn_keep_indices,
    turn_token_spans,
)


class MockTok:
    """Prefix-stable chat template: each role renders to a fixed-length id block."""

    _N = {"system": 3, "user": 5, "assistant": 7}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids: list[int] = []
        for m in messages:
            ids += [ord(m["role"][0])] * self._N[m["role"]]
        if add_generation_prompt:
            ids += [99, 99]                      # generation header, a pure suffix
        return ids


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"}, {"role": "user", "content": "U3"}]


def test_spans_recovered_and_gen_prompt_excluded():
    ctx, spans = turn_token_spans(MockTok(), MSGS, add_generation_prompt=True)
    assert [(s.role, s.n_tokens) for s in spans] == [
        ("system", 3), ("user", 5), ("assistant", 7), ("user", 5), ("assistant", 7), ("user", 5)]
    assert spans[0].start == 0 and spans[1].start == 3          # contiguous
    assert ctx.shape[1] == 34 and sum(s.n_tokens for s in spans) == 32   # gen prompt (2) excluded


def test_oldest_first_with_protections():
    _, spans = turn_token_spans(MockTok(), MSGS, add_generation_prompt=True)
    ev = oldest_turns_to_evict(spans, target_tokens=6, protect_roles=("system",), protect_last=2)
    assert ev == [1, 2]                          # oldest evictable (user U1 + assistant A1), ≥6 freed
    assert 0 not in ev and 4 not in ev and 5 not in ev          # system + last-2 protected


def test_unreachable_target_not_clipped():
    _, spans = turn_token_spans(MockTok(), MSGS, add_generation_prompt=True)
    ev = oldest_turns_to_evict(spans, target_tokens=10 ** 6, protect_last=2)
    assert ev == [1, 2, 3]                        # all evictable turns, not a silent clip


def test_keep_indices_protect_sinks_and_system():
    ctx, spans = turn_token_spans(MockTok(), MSGS, add_generation_prompt=True)
    keep = turn_keep_indices(spans, [1, 2], ctx.shape[1], num_sink_tokens=4)
    assert torch.all(keep[:4] == torch.arange(4))              # sinks kept
    kept = set(keep.tolist())
    assert all(i in kept for i in range(spans[0].start, spans[0].end))   # system fully kept
    # evicted turns 1,2 gone EXCEPT the first 4 tokens held as sinks (sinks override eviction)
    assert not any(i in kept for i in range(4, spans[2].end))
    assert all(i in kept for i in range(4))                   # sink tokens 0-3 kept
    assert ctx.shape[1] - 1 in kept                           # trailing gen-prompt tokens kept


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok {name}")
    print("A4b turn-keep smoke PASS")
