"""Cache surgery: the snapshot, the eviction geometry, and the two RoPE gates.

Surgery is what makes an intervention on a cache a measurement rather than a
corruption. Everything it rests on is exact tensor math, so it is checked exactly:

* **Re-rotation is exact only because RoPE is a rotation homomorphism.** Moving a
  token from position p to p' left-multiplies its key by R(p' - p), and that
  composes only while the frequencies are position-independent. The homomorphism
  check is what refuses a dynamically-rescaled scheme instead of producing keys that
  are almost right.
* **The frequencies have to be the ones the runtime uses.** A table that is a
  homomorphism can still be the wrong table — any static table is — so the value
  gate compares the reconstruction against the model's own live buffer and aborts on
  a mismatch rather than rotating with frequencies the forward pass does not use.
* **Values are position-free and are never touched.** A surgery that rotated them
  would be editing content while claiming to edit position.
* **The keep geometry protects the sinks and the recent tail**, because those are
  what a decoder leans on, and evicting them would confound "less context" with
  "no anchor".

All CPU, all synthetic: the geometry is integers and the rotation is a closed form.
"""

from __future__ import annotations

import math

import pytest
import torch

from anamnesis.extraction.replay.cache_surgery import (
    KVSnapshot,
    assert_rotation_homomorphism,
    default_inv_freq,
    evict,
    from_hf_cache,
    inv_freq_from_config,
    llama3_inv_freq,
    middle_region_keep,
    operative_inv_freq,
    reindex,
    reindex_keys,
    rotate_half,
    to_hf_dynamic_cache,
)
from synthetic_runtime import TinyCausalLM, TinyRopeConfig

HEAD_DIM = 8
LAYERS = 3
KV_HEADS = 2


def snapshot(seq_len: int = 10, seed: int = 0) -> KVSnapshot:
    torch.manual_seed(seed)
    keys = [torch.randn(1, KV_HEADS, seq_len, HEAD_DIM) for _ in range(LAYERS)]
    values = [torch.randn(1, KV_HEADS, seq_len, HEAD_DIM) for _ in range(LAYERS)]
    positions = [torch.arange(seq_len) for _ in range(LAYERS)]
    return KVSnapshot(keys=keys, values=values, positions=positions)


# ── the rotation ──────────────────────────────────────────────────────────────


def test_rotating_by_zero_is_the_identity() -> None:
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    key = torch.randn(1, KV_HEADS, 4, HEAD_DIM)
    rotated = reindex_keys(key, torch.zeros(4), inv)
    assert torch.allclose(rotated, key, atol=1e-6)


def test_rotating_forward_then_back_returns_the_key() -> None:
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    key = torch.randn(1, KV_HEADS, 4, HEAD_DIM)
    delta = torch.tensor([3.0, -2.0, 7.0, 0.0])
    there = reindex_keys(key, delta, inv)
    back = reindex_keys(there, -delta, inv)
    assert torch.allclose(back, key, atol=1e-5)


def test_the_rotation_composes_which_is_what_makes_re_rotation_exact() -> None:
    assert_rotation_homomorphism(default_inv_freq(HEAD_DIM, 500_000.0))
    assert_rotation_homomorphism(llama3_inv_freq(
        128, 500_000.0, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ))


def test_a_reindex_that_does_not_compose_is_refused(monkeypatch) -> None:
    """The check exists for dynamically-rescaled schemes, where the frequencies depend on
    the position and R(a)R(b) is not R(a+b). Standing in for one: a reindex that scales
    by the delta rather than rotating by it, which is additive in no sense. Without the
    check, every re-rotated key would be subtly wrong and nothing would raise."""
    from anamnesis.extraction.replay import cache_surgery

    def not_composable(key, delta_pos, inv_freq):
        return key * (1.0 + delta_pos.reshape(1, 1, -1, 1))

    monkeypatch.setattr(cache_surgery, "reindex_keys", not_composable)
    with pytest.raises(AssertionError, match="not a rotation homomorphism"):
        cache_surgery.assert_rotation_homomorphism(default_inv_freq(HEAD_DIM, 10_000.0))


def test_an_unsupported_rope_scheme_is_refused_before_any_surgery() -> None:
    """Exactness is established per scheme, so an unrecognised one raises at the gate
    rather than after a confusing result."""
    config = TinyRopeConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=HEAD_DIM,
        num_hidden_layers=2, vocab_size=32, rope_theta=10_000.0,
    )
    config.rope_scaling = {"rope_type": "dynamic", "factor": 4.0}  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="unsupported rope_scaling"):
        inv_freq_from_config(config)


def test_rotate_half_swaps_the_two_halves_with_a_sign() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_an_odd_head_dimension_has_no_rope_table() -> None:
    with pytest.raises(ValueError, match="must be even"):
        default_inv_freq(7, 10_000.0)


def test_a_delta_of_the_wrong_length_is_refused() -> None:
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    with pytest.raises(ValueError, match="delta_pos length"):
        reindex_keys(torch.randn(1, KV_HEADS, 4, HEAD_DIM), torch.zeros(3), inv)


def test_the_standard_table_is_the_closed_form() -> None:
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    expected = [1.0 / (10_000.0 ** (2 * i / HEAD_DIM)) for i in range(HEAD_DIM // 2)]
    assert inv.tolist() == pytest.approx(expected)
    # The longest wavelength is the first frequency's, which is what the llama3 rescaling
    # is defined against.
    assert 2 * math.pi / inv[0].item() == pytest.approx(2 * math.pi)


# ── the snapshot ──────────────────────────────────────────────────────────────


def test_a_snapshot_requires_a_position_per_token() -> None:
    keys = [torch.randn(1, KV_HEADS, 5, HEAD_DIM)]
    with pytest.raises(ValueError, match="positions length"):
        KVSnapshot(keys=keys, values=list(keys), positions=[torch.arange(4)])


def test_a_snapshot_requires_the_same_number_of_layers_everywhere() -> None:
    keys = [torch.randn(1, KV_HEADS, 5, HEAD_DIM)] * 2
    with pytest.raises(ValueError, match="per-layer list lengths"):
        KVSnapshot(keys=keys, values=keys[:1], positions=[torch.arange(5)] * 2)


def test_the_next_position_is_one_past_the_furthest_token() -> None:
    snap = snapshot(seq_len=10)
    assert snap.seq_len() == 10
    assert snap.next_position == 10
    assert snap.num_layers == LAYERS


def test_eviction_drops_tokens_and_carries_their_positions_unchanged() -> None:
    """After eviction the survivors still claim their original positions, which is the
    naive case: the continuation then starts past the cache length, not at it."""
    snap = snapshot(seq_len=10)
    keep = torch.tensor([0, 1, 2, 7, 8, 9])
    trimmed = evict(snap, keep)
    assert trimmed.seq_len() == 6
    assert trimmed.positions[0].tolist() == [0, 1, 2, 7, 8, 9]
    assert trimmed.next_position == 10
    assert torch.equal(trimmed.keys[0][0, :, 3], snap.keys[0][0, :, 7])
    assert torch.equal(trimmed.values[0][0, :, 3], snap.values[0][0, :, 7])


def test_reindexing_moves_the_keys_and_leaves_the_values_alone() -> None:
    """Values are position-free. A surgery that rotated them would be editing content."""
    snap = snapshot(seq_len=6)
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    moved = reindex(snap, torch.arange(6) + 4, inv)
    assert moved.positions[0].tolist() == [4, 5, 6, 7, 8, 9]
    for layer in range(LAYERS):
        assert torch.equal(moved.values[layer], snap.values[layer])
        assert not torch.allclose(moved.keys[layer], snap.keys[layer])


def test_reindexing_to_the_same_positions_is_the_identity() -> None:
    snap = snapshot(seq_len=6)
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    same = reindex(snap, torch.arange(6), inv)
    for layer in range(LAYERS):
        assert torch.allclose(same.keys[layer], snap.keys[layer], atol=1e-6)


def test_evicting_then_reindexing_closes_the_gap_the_eviction_opened() -> None:
    """The rotated case: survivors are packed to contiguous positions, so the model sees
    a shorter context rather than one with a hole in its position sequence."""
    snap = snapshot(seq_len=10)
    inv = default_inv_freq(HEAD_DIM, 10_000.0)
    trimmed = evict(snap, torch.tensor([0, 1, 2, 7, 8, 9]))
    packed = reindex(trimmed, torch.arange(6), inv)
    assert packed.positions[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert packed.next_position == 6
    # The tokens that did not move are bit-identical, since their delta is zero.
    assert torch.allclose(packed.keys[0][0, :, :3], trimmed.keys[0][0, :, :3], atol=1e-6)


def test_reindexing_refuses_a_position_list_of_the_wrong_length() -> None:
    snap = snapshot(seq_len=6)
    with pytest.raises(ValueError, match="new_positions length"):
        reindex(snap, torch.arange(4), default_inv_freq(HEAD_DIM, 10_000.0))


def test_a_snapshot_round_trips_through_a_framework_cache() -> None:
    """Surgery is pure tensor math on a snapshot, and the result has to go back into a
    cache the forward will accept."""
    snap = snapshot(seq_len=7)
    cache = to_hf_dynamic_cache(snap)
    recovered = from_hf_cache(cache, positions=torch.arange(7))
    assert recovered.num_layers == LAYERS
    assert recovered.seq_len() == 7
    for layer in range(LAYERS):
        assert torch.equal(recovered.keys[layer], snap.keys[layer])
        assert torch.equal(recovered.values[layer], snap.values[layer])


def test_an_unrecognised_cache_type_is_refused() -> None:
    from anamnesis.extraction.replay.cache_surgery import _extract_kv

    with pytest.raises(TypeError, match="unsupported cache type"):
        _extract_kv(object())


# ── the keep geometry ─────────────────────────────────────────────────────────


def test_the_evicted_block_is_contiguous_and_spares_the_sinks_and_the_tail() -> None:
    keep = middle_region_keep(100, evict_frac=0.2, num_sinks=4, recent_protect=32)
    kept = set(keep.tolist())
    assert len(kept) == 80
    assert all(i in kept for i in range(4))                  # sinks
    assert all(i in kept for i in range(68, 100))            # recent tail
    evicted = sorted(set(range(100)) - kept)
    assert evicted == list(range(evicted[0], evicted[0] + 20))


def test_an_eviction_of_nothing_keeps_everything() -> None:
    keep = middle_region_keep(50, evict_frac=0.0)
    assert keep.tolist() == list(range(50))


def test_an_eviction_larger_than_the_evictable_region_is_refused() -> None:
    """Better to raise than to reach into the protected sinks or the recent tail, which
    would confound less context with no anchor."""
    with pytest.raises(ValueError, match="cannot evict"):
        middle_region_keep(40, evict_frac=0.9, num_sinks=4, recent_protect=32)


def test_the_kept_indices_are_a_sorted_long_tensor() -> None:
    keep = middle_region_keep(60, evict_frac=0.25)
    assert keep.dtype == torch.long
    assert keep.tolist() == sorted(keep.tolist())


# ── the value gate on the frequency table ─────────────────────────────────────


def with_live_inv_freq(model: TinyCausalLM, table: torch.Tensor) -> TinyCausalLM:
    """Attach a live rotary buffer, as a real decoder materialises one."""
    model.register_buffer("rotary_emb_inv_freq", table.clone(), persistent=False)
    return model


def tiny_rope_model(theta: float = 10_000.0, head_dim: int = HEAD_DIM) -> TinyCausalLM:
    model = TinyCausalLM(num_layers=2, hidden_size=16, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=head_dim)
    model.config = TinyRopeConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
        num_hidden_layers=2, vocab_size=32, rope_theta=theta,
    )
    return model


def test_the_operative_table_is_the_live_buffer_when_it_matches_the_config() -> None:
    model = tiny_rope_model(theta=500_000.0)
    expected = default_inv_freq(HEAD_DIM, 500_000.0)
    with_live_inv_freq(model, expected)
    operative = operative_inv_freq(model)
    assert torch.allclose(operative, expected, rtol=1e-6)


def test_a_live_buffer_that_disagrees_with_the_config_aborts_the_surgery() -> None:
    """Any static table composes, so the homomorphism check cannot catch a wrong theta.
    This gate can: it compares against what the runtime actually rotates with."""
    model = tiny_rope_model(theta=500_000.0)
    with_live_inv_freq(model, default_inv_freq(HEAD_DIM, 10_000.0))
    with pytest.raises(ValueError, match="RoPE VALUE gate FAILED"):
        operative_inv_freq(model)


def test_without_a_live_buffer_the_reconstruction_stands_in() -> None:
    """The gate cannot run, so it says so and falls back rather than refusing outright."""
    model = tiny_rope_model(theta=500_000.0)
    operative = operative_inv_freq(model)
    assert torch.allclose(operative, default_inv_freq(HEAD_DIM, 500_000.0), rtol=1e-6)


def test_a_dual_rope_config_is_surfaced_rather_than_silently_halved(caplog) -> None:
    """Interleaved-attention architectures rotate their local layers with a different
    base, and one table cannot be right for both — so the scope of the table is stated
    instead of assumed."""
    config = TinyRopeConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=HEAD_DIM,
        num_hidden_layers=2, vocab_size=32, rope_theta=1_000_000.0,
    )
    config.rope_local_base_freq = 10_000.0                   # type: ignore[attr-defined]
    with caplog.at_level("WARNING"):
        inv_freq_from_config(config)
    assert any("dual-RoPE" in record.message for record in caplog.records)
