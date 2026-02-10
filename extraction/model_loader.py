"""Model loading and pre-RoPE hook management for Llama 3.2 3B Instruct.

Responsibilities:
  - Load model + tokenizer with eager attention (required for attn weights)
  - Register forward hooks on k_proj linear layers to capture pre-RoPE keys
  - Provide hook lifecycle management (register, collect, clear, remove)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """Mutable storage for pre-RoPE key captures across generation steps."""

    # layer_idx → list of tensors, one per generation step
    # Each tensor shape: [1, num_kv_heads, seq_len_at_step, head_dim]
    #   - During prefill: seq_len_at_step = prompt_length
    #   - During generation: seq_len_at_step = 1
    pre_rope_keys: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    enabled: bool = True

    def clear(self) -> None:
        """Release all captured tensors."""
        for tensors in self.pre_rope_keys.values():
            del tensors[:]
        self.pre_rope_keys.clear()

    def get_generation_keys(self, layer_idx: int) -> list[Tensor]:
        """Return captured keys for a layer (excluding prefill step 0)."""
        all_keys = self.pre_rope_keys.get(layer_idx, [])
        if len(all_keys) <= 1:
            return []
        # Skip index 0 (prefill), return generation steps only
        return all_keys[1:]


def _make_k_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a k_proj linear layer.

    k_proj output shape: [batch, seq_len, num_kv_heads * head_dim]
    We reshape to [batch, seq_len, num_kv_heads, head_dim] and detach to CPU.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        # output: [batch, seq_len, num_kv_heads * head_dim]
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_kv_heads, head_dim)
        # Transpose to [batch, num_kv_heads, seq_len, head_dim] for consistency
        reshaped = reshaped.transpose(1, 2).cpu()
        hook_state.pre_rope_keys[layer_idx].append(reshaped)

    return hook_fn


@dataclass
class LoadedModel:
    """Bundle of model, tokenizer, hooks, and their state."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hook_state: HookState
    hook_handles: list[torch.utils.hooks.RemovableHook]
    config: ModelConfig

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("All hooks removed")

    def clear_hook_state(self) -> None:
        """Clear captured hook data without removing hooks."""
        self.hook_state.clear()

    def disable_hooks(self) -> None:
        """Temporarily disable hook capture (e.g. during calibration)."""
        self.hook_state.enabled = False

    def enable_hooks(self) -> None:
        """Re-enable hook capture."""
        self.hook_state.enabled = True


def load_model(
    config: ModelConfig | None = None,
    sampled_layers: list[int] | None = None,
) -> LoadedModel:
    """Load model, tokenizer, and register pre-RoPE hooks.

    Args:
        config: Model configuration. Defaults to ModelConfig().
        sampled_layers: Which layers to hook for pre-RoPE key extraction.
            Defaults to {0, 7, 14, 18, 21, 24, 27}.

    Returns:
        LoadedModel with everything wired up.
    """
    if config is None:
        config = ModelConfig()
    if sampled_layers is None:
        sampled_layers = [0, 7, 14, 18, 21, 24, 27]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    logger.info(f"Loading model: {config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register pre-RoPE hooks on k_proj for sampled layers
    hook_state = HookState()
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    for layer_idx in sampled_layers:
        if layer_idx < 0 or layer_idx >= config.num_layers:
            logger.warning(f"Skipping invalid layer index {layer_idx} (model has {config.num_layers} layers)")
            continue

        layer = model.model.layers[layer_idx]
        k_proj = layer.self_attn.k_proj

        hook_fn = _make_k_proj_hook(
            layer_idx=layer_idx,
            hook_state=hook_state,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )
        handle = k_proj.register_forward_hook(hook_fn)
        hook_handles.append(handle)
        logger.debug(f"Registered pre-RoPE hook on layer {layer_idx}")

    logger.info(f"Model loaded. {len(hook_handles)} pre-RoPE hooks registered on layers {sampled_layers}")

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hook_state=hook_state,
        hook_handles=hook_handles,
        config=config,
    )
