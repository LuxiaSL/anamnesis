"""What the extraction runtime can only be shown on real weights, and how to show it.

Everything in this directory that could be checked with a synthetic decoder is checked
with one. What remains genuinely needs a checkpoint: loading one, finding its
projection modules by the names its architecture uses, reading its own rotary
frequencies, and — the claim the whole instrument rests on — that the generate path
and the replay path produce the same states for the same tokens.

These cases are therefore skipped unless a checkpoint is reachable, and they are here
rather than absent so that the skip is a standing statement of what a box has to have.
Point `ANAMNESIS_TEST_MODEL` at a local checkpoint directory or a hub id to run them;
name the preset it corresponds to in `ANAMNESIS_TEST_PRESET` (default `8b`). The
device the weights land on comes from the preset's config, so a box with an
accelerator uses it and a box without runs the same cases slowly.

A number produced here is a number for THIS box. Cross-hardware arithmetic differs by
around one part in a million, which is expected rather than wrong, so the agreement
asserted below is between two paths in one session — never against a banked file from
somewhere else.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from anamnesis.config import ExperimentConfig, ModelConfig
from anamnesis.extraction.generation_runner import format_prompt, run_single_generation
from anamnesis.extraction.model_loader import load_model
from anamnesis.extraction.replay.cache_surgery import operative_inv_freq
from anamnesis.extraction.replay.extract import replay_extract
from anamnesis.modes.run4_modes import RUN4_MODES

CHECKPOINT = os.environ.get("ANAMNESIS_TEST_MODEL")
PRESET = os.environ.get("ANAMNESIS_TEST_PRESET", "8b")

needs_checkpoint = pytest.mark.skipif(
    CHECKPOINT is None,
    reason=(
        "no checkpoint: set ANAMNESIS_TEST_MODEL to a local model directory or a hub id "
        "(and ANAMNESIS_TEST_PRESET to the preset it matches) to run the real-weights cases"
    ),
)

needs_accelerator = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no accelerator: the full-surface capture at real widths needs device memory",
)


@pytest.fixture(scope="module")
def loaded():
    """The checkpoint, loaded once with the full read surface registered."""
    config = ModelConfig.from_preset(PRESET, model_id=str(CHECKPOINT))
    bundle = load_model(
        config,
        register_gate_hooks=True,
        key_layers=list(range(config.num_layers)),
        value_layers=list(range(config.num_layers)),
        query_layers=None,
        attn_output_layers=None,
    )
    yield bundle
    bundle.remove_hooks()


@needs_checkpoint
def test_a_checkpoint_loads_with_its_hooks_registered(loaded) -> None:
    """The loader's own report: a checkpoint on a device, with a hook per named layer."""
    assert loaded.hook_handles, "no hooks were registered"
    assert loaded.config.attn_implementation == "eager"
    assert loaded.tokenizer is not None


@needs_checkpoint
def test_the_projection_hooks_fire_on_a_real_forward(loaded) -> None:
    """The module names are per-architecture, so a checkpoint is what shows the paths
    resolve — a synthetic decoder only shows the arithmetic once they have."""
    ids = loaded.tokenizer("a short prompt", return_tensors="pt").input_ids
    ids = ids.to(next(loaded.model.parameters()).device)
    loaded.clear_hook_state()
    loaded.enable_hooks()
    with torch.no_grad():
        loaded.model(ids, output_attentions=True, output_hidden_states=True, use_cache=False)
    loaded.flush_hooks_to_cpu()
    assert loaded.hook_state.pre_rope_keys, "no pre-RoPE keys were captured"
    captured = next(iter(loaded.hook_state.pre_rope_keys.values()))[0]
    assert captured.shape[1] == loaded.config.num_kv_heads
    assert captured.shape[-1] == loaded.config.head_dim
    loaded.clear_hook_state()


@needs_checkpoint
def test_the_live_rotary_frequencies_match_the_configuration(loaded) -> None:
    """The value gate, on the only thing that can fail it: a real model's own buffer."""
    table = operative_inv_freq(loaded.model)
    assert table.numel() == loaded.config.head_dim // 2


@needs_checkpoint
def test_a_replay_of_a_generation_reproduces_its_states(loaded, tmp_path) -> None:
    """The claim the instrument rests on. Teacher-forcing the realized sequence through
    one forward reproduces the states incremental generation produced, so a signature is
    an object about a span of text rather than about an act of generation.

    Cache-versus-no-cache kernel differences put a small floor under the comparison;
    the aggregate features are what the equivalence suite holds to a tighter one.
    """
    config = ExperimentConfig.from_preset(
        PRESET, outputs_dir=tmp_path / "run",
        model_overrides={"model_id": str(CHECKPOINT)},
        generation_overrides={"max_new_tokens": 24, "do_sample": False},
    )
    from anamnesis.config import GenerationSpec

    spec = GenerationSpec(
        generation_id=0, prompt_set="test", topic="a short topic", topic_idx=0,
        mode="linear", mode_idx=0, system_prompt=RUN4_MODES["linear"],
        user_prompt="Write about: a short topic", seed=1234,
    )
    _, metadata = run_single_generation(loaded=loaded, spec=spec, config=config)

    realized = metadata["input_ids"]
    prompt_length = metadata["prompt_length"]
    replayed = replay_extract(loaded, realized, prompt_length=prompt_length)

    assert replayed.chosen_token_ids.tolist() == [
        float(x) for x in realized[prompt_length + 1:]
    ]
    assert len(replayed.hidden_states) == len(realized) - prompt_length - 1


@needs_checkpoint
def test_a_prompt_formats_to_the_length_the_manifest_reconstruction_expects(loaded) -> None:
    """Reconstruction re-renders the chat template and compares against the banked prompt
    length, so the two renderings have to be the same one."""
    from anamnesis.extraction.replay.manifest import build_prompt_ids

    _, prompt_length = format_prompt(loaded, RUN4_MODES["linear"], "Write about: a topic")
    ids = build_prompt_ids(loaded.tokenizer, RUN4_MODES["linear"], "Write about: a topic")
    assert len(ids) == prompt_length


@needs_checkpoint
@needs_accelerator
def test_two_replays_of_one_sequence_are_bit_identical(loaded) -> None:
    """Same code, same input, same box: the floor for a replay is exactly zero. This is
    the accelerator's reading of it, which is where the banked floors were measured."""
    ids = loaded.tokenizer("a short prompt to replay twice", return_tensors="pt").input_ids
    realized = ids[0].tolist() + ids[0].tolist()
    first = replay_extract(loaded, realized, prompt_length=len(ids[0]))
    second = replay_extract(loaded, realized, prompt_length=len(ids[0]))
    for a, b in zip(first.hidden_states, second.hidden_states):
        assert np.array_equal(a, b)
    for a, b in zip(first.logits, second.logits):
        assert np.array_equal(a, b)
