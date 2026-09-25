"""Onboarding a model: the claims the smoke checks, and how it refuses.

The smoke's value is its refusals, so the refusals are what is tested. Each check is driven
against a stand-in whose shape is deliberately wrong in one way, because that is the shape a
bad preset row produces:

  * a checkpoint whose decoder-layer count is not the preset's — every layer-indexed
    feature would then name a different depth;
  * an attention block without the modules the hooks attach to, and a feed-forward without
    the gate projection, checked separately for an ordinary attention block and for a
    latent-attention one whose key hook attaches somewhere else;
  * a routed layer whose expert branches are absent;
  * ``generate`` returning **no attention weights at all**, which is what a fused kernel
    does and which would otherwise produce a full-width vector of zeros;
  * a head count or a hidden-state depth that disagrees with the preset.

The probe layers are arithmetic over the sampled plan, so they are checked by value.

The checks are then composed once, end to end: :func:`onboard_model` is run on a
random-weight Llama saved to disk with a word-level tokenizer beside it, through the real
loader. The weights mean nothing; what the case pins is that the command assembles — the
loader places the checkpoint, the hooks fire, and a finite vector comes out. What a trained
checkpoint adds — that its generate and replay paths agree — is covered by
``test_runtime_on_a_real_checkpoint.py`` and by this command's own run on a box with a model.

CPU only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from anamnesis.config import resolve_preset
from anamnesis.extraction.onboarding import (
    DENSE_ATTENTION_MODULES,
    MLA_ATTENTION_MODULES,
    MOE_MLP_MODULES,
    PROMPT,
    OnboardingError,
    OnboardingReport,
    check_capture_surface,
    check_hook_targets,
    onboard_model,
    probe_layers,
)


class FakeAttention(nn.Module):
    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        for name in names:
            setattr(self, name, nn.Linear(2, 2))


class FakeMLP(nn.Module):
    def __init__(self, names: tuple[str, ...], *, shared_gate: bool = True) -> None:
        super().__init__()
        for name in names:
            if name == "shared_experts":
                shared = nn.Module()
                if shared_gate:
                    shared.gate_proj = nn.Linear(2, 2)
                setattr(self, name, shared)
            else:
                setattr(self, name, nn.Linear(2, 2))


class FakeLayer(nn.Module):
    def __init__(self, attention: tuple[str, ...], mlp: tuple[str, ...],
                 *, shared_gate: bool = True) -> None:
        super().__init__()
        self.self_attn = FakeAttention(attention)
        self.mlp = FakeMLP(mlp, shared_gate=shared_gate)


class FakeConfig:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class FakeModel(nn.Module):
    """A stand-in with the module names the loader resolves layers through."""

    def __init__(self, layers: list[FakeLayer], config: FakeConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.config = config


def dense_model(n_layers: int, *, attention: tuple[str, ...] = DENSE_ATTENTION_MODULES,
                mlp: tuple[str, ...] = ("gate_proj",)) -> FakeModel:
    return FakeModel(
        [FakeLayer(attention, mlp) for _ in range(n_layers)],
        FakeConfig(model_type="llama"),
    )


def test_the_probe_layers_are_the_first_middle_and_last_sampled() -> None:
    preset = resolve_preset("8b")
    sampled = list(preset.sampled_layers)
    assert probe_layers(preset) == sorted(
        {sampled[0], sampled[len(sampled) // 2], sampled[-1]}
    )
    assert len(probe_layers(preset)) <= 3


def test_a_layer_count_that_is_not_the_presets_is_refused() -> None:
    preset = resolve_preset("8b")
    with pytest.raises(OnboardingError, match="decoder layers"):
        check_hook_targets(dense_model(preset.num_layers - 1), preset, [0])


def test_an_attention_block_without_the_hook_targets_is_refused() -> None:
    preset = resolve_preset("8b")
    model = dense_model(preset.num_layers, attention=("q_proj", "o_proj"))
    with pytest.raises(OnboardingError, match="k_proj"):
        check_hook_targets(model, preset, [0])


def test_a_feed_forward_without_the_gate_projection_is_refused() -> None:
    preset = resolve_preset("8b")
    model = dense_model(preset.num_layers, mlp=("up_proj",))
    with pytest.raises(OnboardingError, match="no gate projection"):
        check_hook_targets(model, preset, [0])


def test_a_dense_checkpoint_names_itself_dense() -> None:
    preset = resolve_preset("8b")
    assert check_hook_targets(dense_model(preset.num_layers), preset, [0, 16]) == "dense"


def latent_attention_model(preset: Any, *, first_dense: int, shared_gate: bool = True) -> FakeModel:
    layers = []
    for index in range(preset.num_layers):
        mlp = ("gate_proj",) if index < first_dense else MOE_MLP_MODULES
        layers.append(
            FakeLayer(MLA_ATTENTION_MODULES, mlp, shared_gate=shared_gate)
        )
    return FakeModel(
        layers,
        FakeConfig(model_type="deepseek_v2", first_k_dense_replace=first_dense),
    )


def test_a_latent_attention_checkpoint_is_checked_against_its_own_modules() -> None:
    preset = resolve_preset("8b")
    model = latent_attention_model(preset, first_dense=1)
    assert check_hook_targets(model, preset, [0, 16]) == "deepseek_v2"

    dense_named = dense_model(preset.num_layers)
    dense_named.config = FakeConfig(model_type="deepseek_v2", first_k_dense_replace=1)
    with pytest.raises(OnboardingError, match="kv_a_proj_with_mqa"):
        check_hook_targets(dense_named, preset, [0])


def test_a_routed_layer_missing_its_expert_branches_is_refused() -> None:
    preset = resolve_preset("8b")
    model = latent_attention_model(preset, first_dense=1)
    model.layers[16].mlp = FakeMLP(("gate", "experts"))
    with pytest.raises(OnboardingError, match="shared_experts"):
        check_hook_targets(model, preset, [16])

    no_shared_gate = latent_attention_model(preset, first_dense=1, shared_gate=False)
    with pytest.raises(OnboardingError, match="shared expert branch"):
        check_hook_targets(no_shared_gate, preset, [16])


class Generated:
    """What ``generate`` returns, in the two fields the surface check reads."""

    def __init__(self, attentions: Any, hidden_depth: int) -> None:
        self.attentions = attentions
        self.hidden_states = [tuple(torch.zeros(1) for _ in range(hidden_depth))]


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, _messages: Any, **_kwargs: Any) -> torch.Tensor:
        return torch.zeros((1, 3), dtype=torch.long)


class FakeLoaded:
    """Just enough of a loaded model for the capture-surface check to run."""

    def __init__(self, generated: Generated) -> None:
        self.tokenizer = FakeTokenizer()
        self.generated = generated
        self.flushed = False

        class Inner:
            device = "cpu"

            def generate(inner_self: Any, *_args: Any, **_kwargs: Any) -> Generated:
                return generated

        self.model = Inner()

    def clear_hook_state(self) -> None:
        return None

    def enable_hooks(self) -> None:
        return None

    def flush_hooks_to_cpu(self) -> None:
        self.flushed = True


def attention_steps(heads: int, *, seq: int = 3, layers: int = 2) -> list[Any]:
    """Two steps, each a per-layer tuple — the shape ``generate`` returns."""
    step = tuple(torch.zeros((1, heads, 1, seq)) for _ in range(layers))
    return [step, step]


def test_generate_returning_no_attention_weights_is_the_refusal_that_matters() -> None:
    preset = resolve_preset("8b")
    for attentions in (None, [(torch.zeros((1, 1, 1, 3)),)]):
        loaded = FakeLoaded(Generated(attentions, preset.num_layers + 1))
        with pytest.raises(OnboardingError, match="no attention weights"):
            check_capture_surface(loaded, preset)


def test_a_head_count_that_disagrees_with_the_preset_is_refused() -> None:
    preset = resolve_preset("8b")
    loaded = FakeLoaded(
        Generated(attention_steps(preset.num_attention_heads - 1), preset.num_layers + 1)
    )
    with pytest.raises(OnboardingError, match="heads"):
        check_capture_surface(loaded, preset)


def test_a_hidden_state_depth_that_disagrees_with_the_preset_is_refused() -> None:
    preset = resolve_preset("8b")
    loaded = FakeLoaded(
        Generated(attention_steps(preset.num_attention_heads), preset.num_layers)
    )
    with pytest.raises(OnboardingError, match="hidden states carry"):
        check_capture_surface(loaded, preset)


def test_a_capture_surface_that_agrees_reports_its_shape_and_flushes() -> None:
    preset = resolve_preset("8b")
    loaded = FakeLoaded(
        Generated(attention_steps(preset.num_attention_heads), preset.num_layers + 1)
    )
    shape, depth = check_capture_surface(loaded, preset)
    assert shape[1] == preset.num_attention_heads
    assert depth == preset.num_layers + 1
    assert loaded.flushed, "the hook state is flushed once the surface is established"


def test_the_report_reads_as_a_pass_or_a_fail_and_says_which() -> None:
    report = OnboardingReport(
        model="8b",
        model_id="local/snapshot",
        model_class="LlamaForCausalLM",
        n_decoder_layers=32,
        probed_layers=[0, 16, 31],
        architecture="dense",
        attention_shape=[1, 32, 1, 7],
        n_hidden_state_layers=33,
        n_features=3358,
        n_generated_tokens=12,
        features_finite=True,
        sample_text="a rainbow forms when...",
    )
    assert report.passed
    assert report.lines()[-1] == "SMOKE PASS"
    assert any("hook targets present" in line for line in report.lines())

    failed = report.model_copy(update={"features_finite": False})
    assert not failed.passed
    assert failed.lines()[-1] == "SMOKE FAIL"

    routed = report.model_copy(
        update={
            "architecture": "deepseek_v2",
            "routing": {
                "n_routed_layers": 5,
                "n_layers_fired": 5,
                "distribution_width": 64,
                "n_features": 50,
                "n_unclassified": 0,
            },
        }
    )
    assert any("router fired on 5/5" in line for line in routed.lines())


def save_tiny_tokenizer(directory: Path, vocab_size: int) -> None:
    """A word-level tokenizer over the smoke prompt, with the chat template the smoke formats by.

    Every id it emits is below the tiny model's vocabulary, and id 0 is the end-of-text
    token the tiny preset stops on.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<eos>", "<unk>", *dict.fromkeys(PROMPT.replace(".", " .").split())]
    words += [f"w{i}" for i in range(vocab_size - len(words))]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer.save_pretrained(directory)


def test_the_command_runs_end_to_end_on_a_saved_checkpoint(tmp_path: Path) -> None:
    """The composed command on a checkpoint on disk: real loader, real hooks, a finite vector.

    Each check above is driven in isolation; this is the one case that runs them in the
    order :func:`onboard_model` does, with the configuration it builds for the smoke
    generation, so a check that cannot be composed with the next fails here.
    """
    from test_loaded_model_seam import save_tiny_checkpoint, tiny_preset

    checkpoint = tmp_path / "checkpoint"
    loaded = save_tiny_checkpoint(checkpoint)
    save_tiny_tokenizer(checkpoint, loaded.model.config.vocab_size)

    report = onboard_model(tiny_preset(), str(checkpoint))

    assert report.model_id == str(checkpoint)
    assert report.architecture == "dense"
    assert report.n_decoder_layers == 3
    assert report.features_finite
    assert report.passed
    assert report.lines()[-1] == "SMOKE PASS"
