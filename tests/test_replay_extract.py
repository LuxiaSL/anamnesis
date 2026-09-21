"""Replay alignment: the determinism core, checked against a running forward.

Replay is the claim that teacher-forcing a realized token sequence through one
instrumented forward reproduces the per-position states that incremental generation
produced. Everything in that claim is alignment arithmetic — which positions of the
single forward correspond to which banked step, and how many banked steps an
N-token generation has — and alignment arithmetic is what a tiny model with random
weights can check completely.

The contract, stated once: for a prompt at positions 0..P-1 and generated tokens
g_0..g_{N-1} at P..P+N-1, the generate path banks T = N-1 entries, entry i being the
state AT g_i whose logits predict g_{i+1}. Replay reproduces it by slicing positions
P..P+N-2. A slice that is off by one here corrupts every feature in the vector while
raising nothing, which is why the offsets are asserted by value rather than by shape.

Runs on a CPU with no checkpoint: see `synthetic_runtime`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.replay.cache_surgery import (
    KVSnapshot,
    evict,
    middle_region_keep,
)
from anamnesis.extraction.replay.cached import (
    cache_length,
    remap_positional_means,
    replay_extract_cached,
)
from anamnesis.extraction.replay.extract import replay_extract
from synthetic_runtime import HookPlan, loaded_tiny_model

FULL_PLAN = HookPlan(
    key_layers=[0, 1, 2], value_layers=[0, 1, 2], query_layers=[0, 1, 2],
    attn_output_layers=[0, 1, 2], gate_layers=[0, 1, 2],
)


# ── the plain teacher-forced pass ─────────────────────────────────────────────


def test_a_replay_banks_one_fewer_entry_than_the_generation_has_tokens() -> None:
    loaded, model = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))                 # L = 12
    prompt_length = 5                        # N = 7, so T = 6
    raw = replay_extract(loaded, ids, prompt_length=prompt_length)

    assert len(raw.hidden_states) == 6
    assert len(raw.attentions) == 6
    assert len(raw.logits) == 6
    assert raw.chosen_token_ids.shape == (6,)
    assert raw.prompt_length == prompt_length


def test_the_banked_tokens_are_the_realized_ones_shifted_by_one() -> None:
    """Entry i's logits predict g_{i+1}, so `chosen_token_ids` starts at g_1."""
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))
    raw = replay_extract(loaded, ids, prompt_length=5)
    assert raw.chosen_token_ids.tolist() == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_the_attention_row_of_each_step_covers_exactly_its_causal_prefix() -> None:
    """Step i attends over positions 0..P+i, so its row has P+i+1 columns. A row one
    column too wide is a slice that reached into the future."""
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    prompt_length = 5
    raw = replay_extract(loaded, list(range(1, 13)), prompt_length=prompt_length)
    widths = [row.shape[-1] for row in raw.attentions]
    assert widths == [prompt_length + i + 1 for i in range(len(widths))]


def test_hidden_states_carry_the_embedding_output_at_index_zero() -> None:
    """The layer axis is num_layers + 1, with the embedding output first. An off-by-one
    here shifts every layer-indexed feature to its neighbour."""
    loaded, model = loaded_tiny_model(FULL_PLAN)
    raw = replay_extract(loaded, list(range(1, 13)), prompt_length=5)
    n_layers = model.config.num_hidden_layers
    for step in raw.hidden_states:
        assert step.shape == (n_layers + 1, model.config.hidden_size)


def test_keys_are_captured_per_key_value_head_not_per_query_head() -> None:
    """Under grouped-query attention the two counts differ: attention weights index by
    query head, the cache by key/value head. A capture that used the query count would
    reshape four heads' worth of numbers into two heads' shape."""
    loaded, model = loaded_tiny_model(FULL_PLAN)
    raw = replay_extract(loaded, list(range(1, 13)), prompt_length=5)
    cfg = model.config
    assert cfg.num_attention_heads != cfg.num_key_value_heads
    assert raw.pre_rope_keys[0][0].shape == (cfg.num_key_value_heads, cfg.head_dim)
    assert raw.queries is not None
    assert raw.queries[0][0].shape == (cfg.num_attention_heads, cfg.head_dim)
    assert raw.attentions[0].shape[1] == cfg.num_attention_heads


def test_every_registered_hook_surface_arrives_and_no_other() -> None:
    loaded, model = loaded_tiny_model(
        HookPlan(key_layers=[0, 2], value_layers=[1], query_layers=[2], gate_layers=[0])
    )
    raw = replay_extract(loaded, list(range(1, 13)), prompt_length=5)
    assert sorted(raw.pre_rope_keys) == [0, 2]
    assert raw.v_proj_values is not None and sorted(raw.v_proj_values) == [1]
    assert raw.queries is not None and sorted(raw.queries) == [2]
    assert raw.gate_activations is not None and sorted(raw.gate_activations) == [0]
    assert raw.attn_outputs is None            # no o_proj hook registered


def test_hook_state_is_clear_after_a_replay() -> None:
    """A capture left behind would be re-sliced into the next generation's features."""
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    replay_extract(loaded, list(range(1, 13)), prompt_length=5)
    assert loaded.hook_state.pre_rope_keys == {}
    assert loaded.hook_state.gate_activations == {}


def test_two_replays_of_one_sequence_agree_exactly() -> None:
    """Same code, same input, same box: the floor for a replay is exactly zero."""
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))
    first = replay_extract(loaded, ids, prompt_length=5)
    second = replay_extract(loaded, ids, prompt_length=5)
    for a, b in zip(first.hidden_states, second.hidden_states):
        assert np.array_equal(a, b)
    for a, b in zip(first.attentions, second.attentions):
        assert np.array_equal(a, b)
    for a, b in zip(first.logits, second.logits):
        assert np.array_equal(a, b)


def test_a_tensor_or_array_sequence_is_accepted_like_a_list() -> None:
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    ids = list(range(1, 13))
    from_list = replay_extract(loaded, ids, prompt_length=5)
    from_array = replay_extract(loaded, np.asarray(ids), prompt_length=5)
    from_tensor = replay_extract(loaded, torch.tensor(ids), prompt_length=5)
    assert np.array_equal(from_list.logits[0], from_array.logits[0])
    assert np.array_equal(from_list.logits[0], from_tensor.logits[0])


def test_positional_means_ride_along_rather_than_being_banked_per_generation() -> None:
    loaded, model = loaded_tiny_model(FULL_PLAN)
    means = np.zeros((model.config.num_hidden_layers + 1, 64, model.config.hidden_size),
                     dtype=np.float32)
    raw = replay_extract(loaded, list(range(1, 13)), prompt_length=5, positional_means=means)
    assert raw.positional_means is means


@pytest.mark.parametrize(
    ("prompt_length", "length", "match"),
    [
        (0, 12, "out of range"),
        (12, 12, "out of range"),
        (11, 12, "need >=2 generated tokens"),
    ],
)
def test_a_split_replay_cannot_serve_is_refused(prompt_length: int, length: int, match: str) -> None:
    """One generated token has no transition to bank, and a prompt that is the whole
    sequence has nothing after it. Both raise rather than banking an empty vector."""
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    with pytest.raises(ValueError, match=match):
        replay_extract(loaded, list(range(1, length + 1)), prompt_length=prompt_length)


def test_a_batch_is_refused_because_the_alignment_is_per_sequence() -> None:
    loaded, _ = loaded_tiny_model(FULL_PLAN)
    with pytest.raises(ValueError, match="single sequence"):
        replay_extract(loaded, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), prompt_length=2)


# ── the cached-bridge pass ────────────────────────────────────────────────────


def test_a_cached_replay_splits_at_the_cache_boundary() -> None:
    """The prompt/generated boundary is the cache length, which is what puts the
    extractor's split exactly on the cache/continuation seam."""
    loaded, model = loaded_tiny_model(FULL_PLAN)
    context = torch.tensor([list(range(1, 8))])           # cache length 7
    cache = model.prefill_cache(context)
    continuation = list(range(8, 13))                     # N = 5, so T = 4

    raw = replay_extract_cached(loaded, cache, continuation, position_offset=7)
    assert raw.prompt_length == 7
    assert len(raw.hidden_states) == 4
    assert raw.chosen_token_ids.tolist() == [9.0, 10.0, 11.0, 12.0]
    widths = [row.shape[-1] for row in raw.attentions]
    assert widths == [7 + i + 1 for i in range(4)]


def test_cache_length_reads_the_forms_a_cache_arrives_in() -> None:
    loaded, model = loaded_tiny_model(FULL_PLAN)
    cache = model.prefill_cache(torch.tensor([list(range(1, 8))]))
    assert cache_length(cache) == 7
    assert cache_length(tuple(cache)) == 7


def test_a_cached_replay_refuses_a_position_offset_behind_the_cache() -> None:
    """An offset below the cache length would place a continuation token at a position
    the cache already occupies."""
    loaded, model = loaded_tiny_model(FULL_PLAN)
    cache = model.prefill_cache(torch.tensor([list(range(1, 8))]))
    with pytest.raises(ValueError, match="position_offset"):
        replay_extract_cached(loaded, cache, [8, 9, 10], position_offset=3)


def test_a_cached_replay_refuses_a_one_token_continuation() -> None:
    loaded, model = loaded_tiny_model(FULL_PLAN)
    cache = model.prefill_cache(torch.tensor([list(range(1, 8))]))
    with pytest.raises(ValueError, match="need >= 2 continuation tokens"):
        replay_extract_cached(loaded, cache, [8], position_offset=7)


def test_a_cached_replay_over_an_evicted_cache_sees_the_shorter_context() -> None:
    """Eviction is the point of the cached path: the attention columns after surgery
    count the survivors, so a readout over a surgered cache is a readout over what the
    model could actually attend to."""
    loaded, model = loaded_tiny_model(FULL_PLAN)
    context = torch.tensor([list(range(1, 21))])          # cache length 20
    cache = model.prefill_cache(context)
    keys = [pair[0] for pair in cache]
    values = [pair[1] for pair in cache]
    positions = torch.arange(20)
    snapshot = KVSnapshot(keys=keys, values=values, positions=[positions] * len(keys))

    keep = middle_region_keep(20, evict_frac=0.25, num_sinks=2, recent_protect=4)
    trimmed = evict(snapshot, keep)
    assert trimmed.seq_len() == 15

    trimmed_cache = [(trimmed.keys[i], trimmed.values[i]) for i in range(trimmed.num_layers)]
    raw = replay_extract_cached(loaded, trimmed_cache, [21, 22, 23], position_offset=15)
    assert raw.prompt_length == 15
    assert [row.shape[-1] for row in raw.attentions] == [16, 17]


def test_positional_means_are_remapped_only_when_the_offset_moved() -> None:
    """A cache whose survivors keep their original positions puts the continuation past
    the cache length, so a lookup at `cache_len + t` has to find the means of the TRUE
    position. For the offsets where the two coincide, the table is handed back untouched."""
    means = np.arange(2 * 32 * 3, dtype=np.float32).reshape(2, 32, 3)

    identical = remap_positional_means(means, cache_len=10, position_offset=10, n_steps=4)
    assert identical is means

    remapped = remap_positional_means(means, cache_len=10, position_offset=20, n_steps=3)
    assert remapped is not means
    for step in range(3):
        assert np.array_equal(remapped[:, 10 + step, :], means[:, 20 + step, :])
    assert np.array_equal(means, np.arange(2 * 32 * 3, dtype=np.float32).reshape(2, 32, 3))


def test_remapping_refuses_an_offset_behind_the_cache() -> None:
    means = np.zeros((2, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="position_offset 4 < cache_len 10"):
        remap_positional_means(means, cache_len=10, position_offset=4, n_steps=2)


def test_remapping_is_a_no_op_without_a_table() -> None:
    assert remap_positional_means(None, cache_len=10, position_offset=20, n_steps=2) is None
