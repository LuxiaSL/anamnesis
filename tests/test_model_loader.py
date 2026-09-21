"""The capture layer: where the hooks attach, what they reshape, and the write path.

Four things here are load-bearing and none of them needs a checkpoint:

* **Layer resolution.** `decoder_layers` is the one place an architecture's module
  tree is interpreted, including the multimodal wrappers whose text decoder nests a
  level down. An architecture it cannot resolve raises, because the alternative is
  hooks silently attaching to nothing.
* **The hook reshapes.** Keys and values are captured from the projection modules
  pre-RoPE — a post-RoPE key has its position baked in, and the geometric features
  would be reading position — and they reshape by the key/value head count while
  queries reshape by the query head count. Under grouped-query attention those
  differ, so the two reshapes are not interchangeable.
* **Prefill.** Step zero of a generation is the prompt; the banked per-step lists
  begin after it. `get_generation_keys` is where that skip lives.
* **The write path.** `alpha=0` has to reproduce the unperturbed forward exactly,
  positional gating has to read absolute positions so one spec is valid under
  prefill and incremental decoding alike, and an ambiguous incremental step has to
  raise rather than inject at a guessed position.

The attention-implementation requirement is validated in configuration, not here,
which is what the last test states.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from anamnesis.config import ATTENTION_WITHOUT_WEIGHTS, EAGER_ATTENTION, ModelConfig
from anamnesis.extraction.model_loader import (
    HookState,
    ResidualWriteSpec,
    _make_k_proj_hook,
    _make_q_proj_hook,
    _make_v_proj_hook,
    attach_residual_write,
    decoder_layers,
    default_sampled_layers,
)
from synthetic_runtime import HookPlan, TinyCausalLM, loaded_tiny_model


# ── layer resolution ──────────────────────────────────────────────────────────


def test_a_plain_decoder_resolves_at_model_layers() -> None:
    model = TinyCausalLM(num_layers=3)
    assert len(decoder_layers(model)) == 3


def test_a_multimodal_wrapper_resolves_through_its_text_decoder() -> None:
    """A wrapper class nests the text decoder under `language_model`, so the hook paths
    have to resolve through it rather than assuming a fixed attribute path."""
    inner = TinyCausalLM(num_layers=2)
    wrapper = SimpleNamespace(model=SimpleNamespace(language_model=inner))
    assert len(decoder_layers(wrapper)) == 2

    sibling = SimpleNamespace(language_model=inner)
    assert len(decoder_layers(sibling)) == 2


def test_an_unresolvable_architecture_raises_rather_than_hooking_nothing() -> None:
    with pytest.raises(AttributeError, match="extend"):
        decoder_layers(SimpleNamespace(something_else=1))


# ── the layer plan ────────────────────────────────────────────────────────────


def test_the_layer_plan_comes_from_the_config_preset() -> None:
    """A layer index means a different fraction of the network in each model, so the
    plan is read from the row the config was built from."""
    assert default_sampled_layers(ModelConfig.from_preset("8b")) == [0, 8, 16, 20, 24, 28, 31]
    assert default_sampled_layers(ModelConfig.from_preset("3b")) == [0, 7, 14, 18, 21, 24, 27]


def test_a_config_with_no_preset_has_no_layer_plan_to_fall_back_on() -> None:
    hand_built = ModelConfig(
        model_id="tiny", torch_dtype="float32", num_layers=4, hidden_dim=16,
        num_attention_heads=4, num_kv_heads=2, head_dim=4,
    )
    with pytest.raises(ValueError, match="carries no preset name"):
        default_sampled_layers(hand_built)


# ── the hook reshapes ─────────────────────────────────────────────────────────


def test_a_key_capture_reshapes_by_the_key_value_head_count() -> None:
    state = HookState()
    hook = _make_k_proj_hook(layer_idx=3, hook_state=state, num_kv_heads=2, head_dim=4)
    projected = torch.arange(1 * 5 * 8, dtype=torch.float32).reshape(1, 5, 8)
    hook(nn.Identity(), (), projected)
    captured = state.pre_rope_keys[3][0]
    assert captured.shape == (1, 2, 5, 4)
    # Head h at position p is the contiguous slice the projection wrote there.
    assert torch.equal(captured[0, 1, 2], projected[0, 2, 4:8])


def test_a_query_capture_reshapes_by_the_query_head_count() -> None:
    """The two counts differ under grouped-query attention, so using one hook's
    arithmetic for the other surface silently reinterprets the projection's output."""
    state = HookState()
    hook = _make_q_proj_hook(
        layer_idx=0, hook_state=state, num_attention_heads=4, head_dim=4
    )
    hook(nn.Identity(), (), torch.zeros(1, 5, 16))
    assert state.queries[0][0].shape == (1, 4, 5, 4)


def test_a_value_capture_shares_the_key_layout() -> None:
    state = HookState()
    hook = _make_v_proj_hook(layer_idx=0, hook_state=state, num_kv_heads=2, head_dim=4)
    hook(nn.Identity(), (), torch.zeros(1, 5, 8))
    assert state.v_proj_values[0][0].shape == (1, 2, 5, 4)


def test_a_disabled_hook_state_captures_nothing() -> None:
    state = HookState()
    state.enabled = False
    hook = _make_k_proj_hook(layer_idx=0, hook_state=state, num_kv_heads=2, head_dim=4)
    hook(nn.Identity(), (), torch.zeros(1, 5, 8))
    assert state.pre_rope_keys == {}


# ── prefill, and the lifecycle ────────────────────────────────────────────────


def test_the_prefill_capture_is_skipped_by_the_generation_accessors() -> None:
    """Step zero is the prompt, and it is not one of the banked per-step entries."""
    state = HookState()
    for step in range(4):
        state.pre_rope_keys[0].append(torch.full((1, 2, 1, 4), float(step)))
        state.gate_activations[0].append(torch.full((1, 1, 6), float(step)))
    keys = state.get_generation_keys(0)
    gates = state.get_generation_gates(0)
    assert len(keys) == 3 and float(keys[0].flatten()[0]) == 1.0
    assert len(gates) == 3 and float(gates[0].flatten()[0]) == 1.0


def test_a_capture_with_only_a_prefill_step_yields_no_generation_entries() -> None:
    state = HookState()
    state.pre_rope_keys[0].append(torch.zeros(1, 2, 5, 4))
    assert state.get_generation_keys(0) == []
    assert state.get_generation_gates(0) == []


def test_clearing_releases_every_surface() -> None:
    state = HookState()
    state.pre_rope_keys[0].append(torch.zeros(1))
    state.gate_activations[0].append(torch.zeros(1))
    state.v_proj_values[0].append(torch.zeros(1))
    state.queries[0].append(torch.zeros(1))
    state.attn_outputs[0].append(torch.zeros(1))
    state.router_dist[0].append(torch.zeros(1))
    state.flush_to_cpu()
    state.clear()
    for store in (state.pre_rope_keys, state.gate_activations, state.v_proj_values,
                  state.queries, state.attn_outputs, state.router_dist):
        assert store == {}
    assert state._on_cpu is False


def test_flushing_twice_is_harmless() -> None:
    state = HookState()
    state.pre_rope_keys[0].append(torch.zeros(1, 2, 1, 4))
    state.flush_to_cpu()
    state.flush_to_cpu()
    assert len(state.pre_rope_keys[0]) == 1


def test_removing_hooks_stops_capture_and_empties_the_handle_list() -> None:
    loaded, model = loaded_tiny_model(HookPlan(key_layers=[0, 1, 2]))
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))
    assert sorted(loaded.hook_state.pre_rope_keys) == [0, 1, 2]

    loaded.remove_hooks()
    loaded.clear_hook_state()
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))
    assert loaded.hook_state.pre_rope_keys == {}
    assert loaded.hook_handles == []


def test_disabling_hooks_leaves_them_registered() -> None:
    """Calibration runs with capture off, and then the same handles are used again."""
    loaded, model = loaded_tiny_model(HookPlan(key_layers=[0]))
    loaded.disable_hooks()
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))
    assert loaded.hook_state.pre_rope_keys == {}

    loaded.enable_hooks()
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))
    assert sorted(loaded.hook_state.pre_rope_keys) == [0]
    assert len(loaded.hook_handles) == 1


# ── the activation-write path ─────────────────────────────────────────────────


def unperturbed(model: TinyCausalLM, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(ids).logits.clone()


def test_a_write_validates_its_layer_and_its_width() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    with pytest.raises(ValueError, match="out of range"):
        attach_residual_write(model, ResidualWriteSpec(
            layer_idx=9, vector=torch.ones(16), alpha=1.0))
    with pytest.raises(ValueError, match="expected hidden_dim=16"):
        attach_residual_write(model, ResidualWriteSpec(
            layer_idx=1, vector=torch.ones(5), alpha=1.0))


def test_a_zero_dose_reproduces_the_unperturbed_forward() -> None:
    """The floor of the write path: a dose ladder's zero rung has to be the baseline
    exactly, or every effect on the ladder is measured against a moved baseline."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    baseline = unperturbed(model, ids)

    handle = attach_residual_write(model, ResidualWriteSpec(
        layer_idx=1, vector=torch.randn(16), alpha=0.0))
    try:
        assert torch.equal(unperturbed(model, ids), baseline)
    finally:
        handle.remove()


def test_a_disabled_handle_reproduces_the_unperturbed_forward() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    baseline = unperturbed(model, ids)

    handle = attach_residual_write(model, ResidualWriteSpec(
        layer_idx=1, vector=torch.randn(16), alpha=2.0))
    try:
        assert not torch.equal(unperturbed(model, ids), baseline)
        handle.disable()
        assert torch.equal(unperturbed(model, ids), baseline)
        handle.enable()
        assert not torch.equal(unperturbed(model, ids), baseline)
    finally:
        handle.remove()


def test_removing_a_write_restores_the_forward() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    baseline = unperturbed(model, ids)
    handle = attach_residual_write(model, ResidualWriteSpec(
        layer_idx=1, vector=torch.randn(16), alpha=2.0))
    handle.remove()
    assert torch.equal(unperturbed(model, ids), baseline)


def test_positional_bounds_inject_at_exactly_the_named_positions() -> None:
    """Gating reads the absolute `cache_position`, so the same spec means the same
    positions under a full-sequence forward and an incremental step."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    spec = ResidualWriteSpec(
        layer_idx=1, vector=torch.randn(16), alpha=3.0, start_pos=2, end_pos=4)
    handle = attach_residual_write(model, spec)
    try:
        unperturbed(model, ids)
        assert handle.stats["saw_cache_position"] is True
        assert handle.stats["positions"] == 2           # positions 2 and 3, end exclusive
    finally:
        handle.remove()


def test_an_unbounded_write_injects_at_every_position() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    handle = attach_residual_write(model, ResidualWriteSpec(
        layer_idx=0, vector=torch.randn(16), alpha=1.0))
    try:
        unperturbed(model, ids)
        assert handle.stats["positions"] == 6
    finally:
        handle.remove()


def test_the_start_position_may_be_moved_on_a_live_handle() -> None:
    """Prompt lengths vary between generations, and the hook is registered once, so
    `start_pos` is read at every call rather than captured at registration."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    spec = ResidualWriteSpec(
        layer_idx=1, vector=torch.randn(16), alpha=1.0, start_pos=4)
    handle = attach_residual_write(model, spec)
    try:
        unperturbed(model, ids)
        assert handle.stats["positions"] == 2
        handle.reset_stats()
        spec.start_pos = 1
        unperturbed(model, ids)
        assert handle.stats["positions"] == 5
    finally:
        handle.remove()


def test_a_bounded_write_on_an_ambiguous_single_token_step_raises() -> None:
    """Without absolute positions a seq_len=1 forward could be any position, so the
    hook refuses rather than injecting at a guessed one."""
    from anamnesis.extraction.model_loader import _make_residual_write_pre_hook

    spec = ResidualWriteSpec(
        layer_idx=0, vector=torch.ones(16), alpha=1.0, start_pos=3, end_pos=5)
    hook = _make_residual_write_pre_hook(spec, {"enabled": True})
    with pytest.raises(RuntimeError, match="cannot determine the absolute"):
        hook(nn.Identity(), (torch.zeros(1, 1, 16),), {})


def test_an_unbounded_write_without_absolute_positions_still_applies() -> None:
    """The fallback is correct for a full-sequence forward, which is what an unbounded
    spec asks for."""
    from anamnesis.extraction.model_loader import _make_residual_write_pre_hook

    vector = torch.ones(16)
    spec = ResidualWriteSpec(layer_idx=0, vector=vector, alpha=2.0)
    hook = _make_residual_write_pre_hook(spec, {"enabled": True})
    args, _ = hook(nn.Identity(), (torch.zeros(1, 3, 16),), {})
    expected = 2.0 / (16 ** 0.5)                         # alpha × unit(vector)
    assert torch.allclose(args[0], torch.full((1, 3, 16), expected))


def test_a_swapped_vector_is_not_served_from_the_previous_one() -> None:
    """Pilot gates iterate vectors on one live spec, so the cached delta is keyed by
    the tensor as well as by the dose."""
    from anamnesis.extraction.model_loader import _make_residual_write_pre_hook

    spec = ResidualWriteSpec(layer_idx=0, vector=torch.ones(16), alpha=1.0, normalize=False)
    hook = _make_residual_write_pre_hook(spec, {"enabled": True})
    first_args, _ = hook(nn.Identity(), (torch.zeros(1, 2, 16),), {})
    spec.vector = torch.full((16,), 5.0)
    second_args, _ = hook(nn.Identity(), (torch.zeros(1, 2, 16),), {})
    assert not torch.allclose(first_args[0], second_args[0])
    assert torch.allclose(second_args[0], torch.full((1, 2, 16), 5.0))


def test_hidden_states_arriving_as_a_keyword_are_still_injected_into() -> None:
    from anamnesis.extraction.model_loader import _make_residual_write_pre_hook

    spec = ResidualWriteSpec(layer_idx=0, vector=torch.ones(16), alpha=1.0, normalize=False)
    hook = _make_residual_write_pre_hook(spec, {"enabled": True})
    args, kwargs = hook(nn.Identity(), (), {"hidden_states": torch.zeros(1, 2, 16)})
    assert torch.allclose(kwargs["hidden_states"], torch.ones(1, 2, 16))


# ── the attention-implementation requirement ──────────────────────────────────


def test_a_kernel_that_returns_no_attention_weights_is_refused_in_configuration() -> None:
    """Fused attention kernels return no weights, and extraction without them drops the
    attention source silently. The loader passes the config's choice through, so the
    constraint is enforced once, where the choice is made."""
    for kernel in sorted(ATTENTION_WITHOUT_WEIGHTS):
        with pytest.raises(ValidationError, match="no attention weights"):
            ModelConfig.from_preset("8b", attn_implementation=kernel)
    assert ModelConfig.from_preset("8b").attn_implementation == EAGER_ATTENTION


def test_the_loader_reads_the_attention_implementation_off_the_config() -> None:
    """A hard-coded kernel in the loader would put the requirement in two places, and
    the second one is the one that goes stale."""
    import inspect

    from anamnesis.extraction import model_loader

    source = inspect.getsource(model_loader.load_model)
    assert "attn_implementation=config.attn_implementation" in source
    assert '"eager"' not in source and "'eager'" not in source
