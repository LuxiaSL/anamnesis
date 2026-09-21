"""The streaming loop: the same computation as the framework's, collected affordably.

The reason this module exists is cost, not behaviour: asking the framework's
generation loop for hidden states builds one tensor object per step per layer, which
for a 512-token generation on a 32-layer model is tens of thousands of objects and
roughly an order of magnitude of overhead. The loop here stacks each step's states on
the device and transfers once.

So what has to be true of it is that the states it hands back are the ones the
alignment contract names. Its lists begin AFTER the prefill step, because the prefill
produced the first token and no banked state did; its `generated_token_ids` carries
every token including that first one; and the hooks registered on the model's modules
fire during it exactly as they do under the framework's loop, since they are attached
to modules rather than to a generation call.

The calibration variant runs the same loop and accumulates into caller-owned arrays
instead of returning states, which is what lets a calibration pass over hundreds of
generations not hold them all.

No checkpoint: see `synthetic_runtime`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.streaming_generate import (
    StreamingOutput,
    _sample_top_p,
    compute_pca_step_indices,
    streaming_calibrate,
    streaming_generate,
)
from synthetic_runtime import HookPlan, TinyCausalLM, loaded_tiny_model

PROMPT = torch.tensor([[1, 2, 3, 4]])
PROMPT_LENGTH = 4


# ── sampling ──────────────────────────────────────────────────────────────────


def test_a_nucleus_of_one_token_is_the_argmax() -> None:
    """A vanishing top_p leaves only the most probable token in the nucleus, which is
    the sampler's degenerate case and the one a determinism claim leans on."""
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    for _ in range(8):
        assert _sample_top_p(logits, temperature=1.0, top_p=1e-6) == 1


def test_sampling_is_a_function_of_the_seed() -> None:
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    torch.manual_seed(11)
    first = [_sample_top_p(logits, 1.0, 0.9) for _ in range(6)]
    torch.manual_seed(11)
    second = [_sample_top_p(logits, 1.0, 0.9) for _ in range(6)]
    assert first == second


def test_the_sampler_returns_an_index_into_the_vocabulary() -> None:
    logits = torch.randn(32)
    for _ in range(20):
        token = _sample_top_p(logits, temperature=0.7, top_p=0.9)
        assert isinstance(token, int) and 0 <= token < 32


# ── the loop's alignment ──────────────────────────────────────────────────────


def greedy(model: TinyCausalLM, **kwargs: object) -> StreamingOutput:
    """Generate with the nucleus collapsed to one token, so the run is reproducible."""
    return streaming_generate(
        model, PROMPT, temperature=1.0, top_p=1e-6, **kwargs,  # type: ignore[arg-type]
    )


def test_the_state_lists_are_one_shorter_than_the_generated_tokens() -> None:
    model = TinyCausalLM(num_layers=3)
    out = greedy(model, max_new_tokens=5, output_attentions=True)
    assert len(out.generated_token_ids) == 5
    assert len(out.hidden_states) == 4
    assert len(out.logits) == 4
    assert len(out.attentions) == 4


def test_the_sequence_is_the_prompt_followed_by_every_generated_token() -> None:
    model = TinyCausalLM(num_layers=3)
    out = greedy(model, max_new_tokens=5)
    assert out.prompt_length == PROMPT_LENGTH
    assert out.sequences.shape == (1, PROMPT_LENGTH + 5)
    assert out.sequences[0, :PROMPT_LENGTH].tolist() == PROMPT[0].tolist()
    assert out.sequences[0, PROMPT_LENGTH:].tolist() == out.generated_token_ids


def test_each_banked_hidden_state_spans_the_layer_axis_with_the_embedding_first() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    out = greedy(model, max_new_tokens=4)
    for step in out.hidden_states:
        assert step.shape == (4, 16)


def test_each_banked_attention_row_grows_by_one_column_per_step() -> None:
    model = TinyCausalLM(num_layers=3)
    out = greedy(model, max_new_tokens=5, output_attentions=True)
    widths = [row.shape[-1] for row in out.attentions]
    assert widths == [PROMPT_LENGTH + 1 + i for i in range(len(widths))]


def test_attention_is_not_collected_unless_it_is_asked_for() -> None:
    model = TinyCausalLM(num_layers=3)
    out = greedy(model, max_new_tokens=4, output_attentions=False)
    assert out.attentions == []


def test_the_prefill_states_are_collected_only_for_calibration() -> None:
    """The prompt's own states are what positional means are built from, and they are
    not part of a generation's feature vector, so they are collected on request."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    without = greedy(model, max_new_tokens=3)
    assert without.prefill_hidden_states is None

    with_prefill = greedy(model, max_new_tokens=3, collect_prefill_hidden_states=True)
    assert with_prefill.prefill_hidden_states is not None
    assert with_prefill.prefill_hidden_states.shape == (4, PROMPT_LENGTH, 16)


def test_generation_stops_at_an_end_of_sequence_token() -> None:
    """The loop ends on the token, and that token is part of the generation: replay has
    to see what the model actually emitted."""
    model = TinyCausalLM(num_layers=3)
    unstopped = greedy(model, max_new_tokens=6).generated_token_ids
    # The first token that has not already been emitted: greedy decoding repeats, and a
    # repeated token would stop the run at its first occurrence instead.
    index = next(i for i in range(1, len(unstopped)) if unstopped[i] not in unstopped[:i])
    stop_at = unstopped[index]

    stopped = greedy(model, max_new_tokens=6, eos_token_ids=[stop_at])
    assert stopped.generated_token_ids == unstopped[: index + 1]
    assert stopped.generated_token_ids[-1] == stop_at
    assert len(stopped.hidden_states) == index


def test_an_end_of_sequence_id_the_model_never_emits_does_not_stop_it() -> None:
    model = TinyCausalLM(num_layers=3, vocab_size=32)
    out = greedy(model, max_new_tokens=4, eos_token_ids=[31, 30])
    assert len(out.generated_token_ids) == 4 or out.generated_token_ids[-1] in (30, 31)


def test_the_loop_is_deterministic_under_a_seed() -> None:
    model = TinyCausalLM(num_layers=3)
    torch.manual_seed(5)
    first = streaming_generate(model, PROMPT, max_new_tokens=5, temperature=0.8, top_p=0.9)
    torch.manual_seed(5)
    second = streaming_generate(model, PROMPT, max_new_tokens=5, temperature=0.8, top_p=0.9)
    assert first.generated_token_ids == second.generated_token_ids
    for a, b in zip(first.hidden_states, second.hidden_states):
        assert np.array_equal(a, b)


def test_hooks_registered_on_the_modules_fire_through_the_loop() -> None:
    """The hooks are attached to projection modules, not to a generation call, which is
    why this loop can replace the framework's without touching the capture surface."""
    loaded, model = loaded_tiny_model(HookPlan(key_layers=[0, 1, 2], gate_layers=[0]))
    out = streaming_generate(model, PROMPT, max_new_tokens=4, temperature=1.0, top_p=1e-6)
    loaded.flush_hooks_to_cpu()

    # One capture per forward: the prefill plus one per generated token.
    assert len(loaded.hook_state.pre_rope_keys[0]) == len(out.generated_token_ids)
    # The generation accessor drops the prefill, leaving the banked per-step count.
    assert len(loaded.hook_state.get_generation_keys(0)) == len(out.hidden_states)
    assert len(loaded.hook_state.get_generation_gates(0)) == len(out.hidden_states)
    # The prefill capture spans the prompt; each later one is a single token.
    assert loaded.hook_state.pre_rope_keys[0][0].shape[-2] == PROMPT_LENGTH
    assert loaded.hook_state.pre_rope_keys[0][1].shape[-2] == 1


# ── the calibration variant ───────────────────────────────────────────────────


def test_calibration_accumulates_the_prompt_span_and_each_generated_position() -> None:
    """Positional means are per absolute position, so the prefill contributes to every
    prompt position at once and each later step to exactly one."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    n_layers_plus_embed = 4
    max_positions = 32
    pos_sums = np.zeros((n_layers_plus_embed, max_positions, 16), dtype=np.float64)
    pos_counts = np.zeros((n_layers_plus_embed, max_positions), dtype=np.int64)
    pca_samples: list[np.ndarray] = []

    torch.manual_seed(3)
    generated = streaming_calibrate(
        model, PROMPT, max_new_tokens=6, temperature=1.0, top_p=1e-6,
        pos_sums=pos_sums, pos_counts=pos_counts, pca_samples=pca_samples,
        pca_layers=[1, 2],
    )

    assert generated == 6
    # Every prompt position saw the prefill exactly once.
    assert pos_counts[:, :PROMPT_LENGTH].tolist() == [[1] * PROMPT_LENGTH] * n_layers_plus_embed
    # Generation step i lands at absolute position prompt_length + i - 1, so the five
    # post-prefill steps cover positions 4..8.
    assert pos_counts[0, PROMPT_LENGTH:PROMPT_LENGTH + 5].tolist() == [1] * 5
    assert pos_counts[0, PROMPT_LENGTH + 5] == 0


def test_calibration_leaves_positions_past_the_table_alone() -> None:
    """A generation longer than the calibrated span must not write past the table."""
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    pos_sums = np.zeros((4, PROMPT_LENGTH + 2, 16), dtype=np.float64)
    pos_counts = np.zeros((4, PROMPT_LENGTH + 2), dtype=np.int64)
    streaming_calibrate(
        model, PROMPT, max_new_tokens=8, temperature=1.0, top_p=1e-6,
        pos_sums=pos_sums, pos_counts=pos_counts, pca_samples=[], pca_layers=[],
    )
    assert pos_counts[0].tolist() == [1, 1, 1, 1, 1, 1]


def test_calibration_samples_the_residual_stream_for_the_projection() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    pos_sums = np.zeros((4, 64, 16), dtype=np.float64)
    pos_counts = np.zeros((4, 64), dtype=np.int64)
    pca_samples: list[np.ndarray] = []
    streaming_calibrate(
        model, PROMPT, max_new_tokens=8, temperature=1.0, top_p=1e-6,
        pos_sums=pos_sums, pos_counts=pos_counts, pca_samples=pca_samples,
        pca_layers=[1, 2],
    )
    assert pca_samples, "no residual-stream samples were taken for the projection"
    assert all(sample.shape == (16,) for sample in pca_samples)


def test_calibration_takes_no_samples_when_no_layers_are_named() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    pca_samples: list[np.ndarray] = []
    streaming_calibrate(
        model, PROMPT, max_new_tokens=4, temperature=1.0, top_p=1e-6,
        pos_sums=np.zeros((4, 64, 16)), pos_counts=np.zeros((4, 64), dtype=np.int64),
        pca_samples=pca_samples, pca_layers=[],
    )
    assert pca_samples == []


def test_calibration_stops_at_an_end_of_sequence_token() -> None:
    model = TinyCausalLM(num_layers=3, hidden_size=16)
    torch.manual_seed(3)
    tokens = streaming_generate(
        model, PROMPT, max_new_tokens=6, temperature=1.0, top_p=1e-6
    ).generated_token_ids
    generated = streaming_calibrate(
        model, PROMPT, max_new_tokens=6, temperature=1.0, top_p=1e-6,
        pos_sums=np.zeros((4, 64, 16)), pos_counts=np.zeros((4, 64), dtype=np.int64),
        pca_samples=[], pca_layers=[], eos_token_ids=[tokens[1]],
    )
    assert generated == 2


@pytest.mark.parametrize("budget", [4, 100, 512])
def test_the_projection_sample_steps_span_the_generation(budget: int) -> None:
    """Beginning, middle and end, from the token budget rather than the realized length,
    because the length is not known when the loop starts."""
    steps = compute_pca_step_indices(budget)
    assert 1 in steps
    assert budget in steps
    assert all(step >= 1 for step in steps)
    assert len(steps) <= 5
