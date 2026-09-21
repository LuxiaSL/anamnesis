"""A tiny real transformer, so the capture layer can be tested without a checkpoint.

The extraction runtime is defined by where it reads from and how it aligns what it
reads: hooks on the projection modules, hidden states indexed with the embedding
output at zero, attention rows sliced per generated position, the prefill step
skipped. None of that needs a trained model — it needs a module tree shaped like
one, with real `nn.Linear` projections so the hooks fire on real tensors and the
grouped-query reshape is exercised on the widths a checkpoint would have.

So this is a working decoder: a handful of layers, real attention with returned
weights, a key/value cache that can be handed back in, and a config carrying the
fields the RoPE gate and the residual-write validator read. It is small enough to
run a forward on a CPU in milliseconds, and it is the fixture every test in this
directory that needs a model uses.

What it deliberately is NOT: a stand-in for a checkpoint in a numerical claim. Its
weights are random. It proves alignment, shapes, hook wiring and control flow; the
numbers a signature is made of are the numeric anchor's business, and agreement
between a real model's two paths is the equivalence suite's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class TinyRopeConfig:
    """The attributes `cache_surgery.inv_freq_from_config` and the write validator read."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rope_theta: float = 10_000.0
    model_type: str = "tiny"


class TinyAttention(nn.Module):
    """Grouped-query attention with the four projections a hook targets."""

    def __init__(self, config: TinyRopeConfig) -> None:
        super().__init__()
        self.config = config
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False)

    def _heads(self, projected: Tensor, n_heads: int) -> Tensor:
        batch, seq, _ = projected.shape
        return projected.view(batch, seq, n_heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self, hidden_states: Tensor, past: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        cfg = self.config
        queries = self._heads(self.q_proj(hidden_states), cfg.num_attention_heads)
        keys = self._heads(self.k_proj(hidden_states), cfg.num_key_value_heads)
        values = self._heads(self.v_proj(hidden_states), cfg.num_key_value_heads)
        if past is not None:
            keys = torch.cat([past[0], keys], dim=-2)
            values = torch.cat([past[1], values], dim=-2)
        present = (keys, values)

        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        keys_full = keys.repeat_interleave(groups, dim=1)
        values_full = values.repeat_interleave(groups, dim=1)

        scores = queries @ keys_full.transpose(-1, -2) / (cfg.head_dim ** 0.5)
        n_query, n_key = scores.shape[-2], scores.shape[-1]
        offset = n_key - n_query
        causal = torch.ones(n_query, n_key, dtype=torch.bool).tril(diagonal=offset)
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = scores.softmax(dim=-1)

        context = weights @ values_full
        merged = context.transpose(1, 2).reshape(hidden_states.shape[0], n_query, -1)
        return self.o_proj(merged), weights, present


class TinyMLP(nn.Module):
    """A gated MLP, so the SwiGLU gate hook has its own projection to fire on."""

    def __init__(self, config: TinyRopeConfig, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TinyLayer(nn.Module):
    """One decoder layer, taking hidden states positionally and `cache_position` by keyword.

    The residual-write pre-hook reaches `args[0]` and reads `cache_position` out of
    the keyword arguments, which is how one write spec is valid under a full-sequence
    forward and an incremental step alike. Both are therefore part of the signature
    here rather than conveniences.
    """

    def __init__(self, config: TinyRopeConfig, intermediate: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(config)
        self.mlp = TinyMLP(config, intermediate)

    def forward(
        self,
        hidden_states: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        attn_out, weights, present = self.self_attn(hidden_states, past)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states, weights, present


class TinyInner(nn.Module):
    """The `model.layers` level, which is where `decoder_layers()` looks first."""

    def __init__(self, config: TinyRopeConfig, intermediate: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyLayer(config, intermediate) for _ in range(config.num_hidden_layers)
        )


@dataclass
class TinyOutput:
    """What the forward returns, named as the transformers outputs are."""

    logits: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None
    past_key_values: list[tuple[Tensor, Tensor]] | None = None


class TinyCausalLM(nn.Module):
    """A causal language model with the surface the extraction runtime calls.

    `hidden_states` is `num_hidden_layers + 1` entries with the embedding output at
    index zero, which is the indexing every layer-indexed feature depends on.
    """

    def __init__(
        self,
        num_layers: int = 3,
        hidden_size: int = 16,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 4,
        vocab_size: int = 32,
        intermediate: int = 24,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = TinyRopeConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
        )
        self.model = TinyInner(self.config, intermediate)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.eval()

    def forward(
        self,
        input_ids: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
    ) -> TinyOutput:
        assert input_ids is not None
        hidden_states = self.model.embed_tokens(input_ids)
        past = _as_pairs(past_key_values)
        if cache_position is None:
            start = past[0][0].shape[-2] if past else 0
            cache_position = torch.arange(start, start + input_ids.shape[1])

        collected_hidden: list[Tensor] = [hidden_states]
        collected_attn: list[Tensor] = []
        present: list[tuple[Tensor, Tensor]] = []
        for index, layer in enumerate(self.model.layers):
            layer_past = past[index] if past else None
            hidden_states, weights, kv = layer(
                hidden_states, layer_past, cache_position=cache_position
            )
            collected_hidden.append(hidden_states)
            collected_attn.append(weights)
            present.append(kv)

        return TinyOutput(
            logits=self.lm_head(hidden_states),
            hidden_states=tuple(collected_hidden) if output_hidden_states else None,
            attentions=tuple(collected_attn) if output_attentions else None,
            past_key_values=present if use_cache else None,
        )

    def prefill_cache(self, input_ids: Tensor) -> list[tuple[Tensor, Tensor]]:
        """A key/value cache for a context, for the cached-replay path to be handed."""
        with torch.no_grad():
            out = self(input_ids, use_cache=True)
        assert out.past_key_values is not None
        return out.past_key_values


def _as_pairs(past_key_values: Any) -> list[tuple[Tensor, Tensor]]:
    """Accept the cache forms the runtime passes: a list of pairs, or nothing."""
    if past_key_values is None:
        return []
    if isinstance(past_key_values, (list, tuple)):
        return [(pair[0], pair[1]) for pair in past_key_values]
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")


class TinyTokenizer:
    """One token per character, with a two-token chat template and a pad token."""

    pad_token: str | None = "<pad>"
    eos_token: str = "<eos>"

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True,
        return_tensors: Any = None,
    ) -> Tensor | list[int]:
        ids = [1]
        for message in messages:
            ids += [ord(c) % 32 for c in message["content"]]
        if add_generation_prompt:
            ids += [2]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 32 for c in text]

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i) + 64) for i in ids)


@dataclass
class HookPlan:
    """Which layers each read hook is registered at, mirroring `load_model`'s arguments."""

    key_layers: list[int] = field(default_factory=list)
    value_layers: list[int] = field(default_factory=list)
    query_layers: list[int] = field(default_factory=list)
    attn_output_layers: list[int] = field(default_factory=list)
    gate_layers: list[int] = field(default_factory=list)


def loaded_tiny_model(
    plan: HookPlan | None = None, **model_kwargs: Any
) -> tuple[Any, TinyCausalLM]:
    """A `LoadedModel` around a tiny model, with read hooks registered as `load_model` does.

    Returns the bundle and the underlying model. The hook registration mirrors
    `load_model`'s Llama-class branch — the same private hook factories on the same
    projection modules — so a test of the capture surface tests the wiring that runs
    in production, not a copy of it.
    """
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import (
        HookState,
        LoadedModel,
        _make_gate_proj_hook,
        _make_k_proj_hook,
        _make_o_proj_hook,
        _make_q_proj_hook,
        _make_v_proj_hook,
        decoder_layers,
    )

    model = TinyCausalLM(**model_kwargs)
    cfg = model.config
    plan = plan or HookPlan(key_layers=list(range(cfg.num_hidden_layers)))
    config = ModelConfig(
        model_id="tiny",
        torch_dtype="float32",
        num_layers=cfg.num_hidden_layers,
        hidden_dim=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )

    hook_state = HookState()
    handles: list[Any] = []
    layers = decoder_layers(model)
    for index in plan.key_layers:
        handles.append(layers[index].self_attn.k_proj.register_forward_hook(
            _make_k_proj_hook(layer_idx=index, hook_state=hook_state,
                              num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim)))
    for index in plan.value_layers:
        handles.append(layers[index].self_attn.v_proj.register_forward_hook(
            _make_v_proj_hook(layer_idx=index, hook_state=hook_state,
                              num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim)))
    for index in plan.query_layers:
        handles.append(layers[index].self_attn.q_proj.register_forward_hook(
            _make_q_proj_hook(layer_idx=index, hook_state=hook_state,
                              num_attention_heads=cfg.num_attention_heads,
                              head_dim=cfg.head_dim)))
    for index in plan.attn_output_layers:
        handles.append(layers[index].self_attn.o_proj.register_forward_hook(
            _make_o_proj_hook(layer_idx=index, hook_state=hook_state)))
    for index in plan.gate_layers:
        handles.append(layers[index].mlp.gate_proj.register_forward_hook(
            _make_gate_proj_hook(layer_idx=index, hook_state=hook_state)))

    return (
        LoadedModel(
            model=model, tokenizer=TinyTokenizer(), hook_state=hook_state,
            hook_handles=handles, config=config,
        ),
        model,
    )
