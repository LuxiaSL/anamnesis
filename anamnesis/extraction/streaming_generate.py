"""Streaming generation with efficient internal state collection.

:func:`streaming_generate` stands in for the framework's ``model.generate()``
because of cost, not behaviour: asking that loop for hidden states builds one
tensor object per step per layer — roughly 17,000 of them for a 512-token
generation on a 32-layer model, and about ten times the overhead of a plain
generation. This loop runs the same computation and hands back pre-stacked numpy
arrays.

Where it differs from the framework's loop:
  - One model() call per step (which is what generate() does internally)
  - States are stacked on the device per step → one .cpu() transfer per step
  - No tuple-of-tuples accumulation — numpy list appends
  - k_proj hooks fire normally (they are registered on modules, not on a call)

Generation is all this module does. The calibration pass — positional means and a
residual basis — is :mod:`anamnesis.extraction.calibration_fit`, which generates at
the decode policy its preset row states. A second loop here with decode defaults of
its own would calibrate a checkpoint at a nucleus mass or a temperature it is never
run at, and nothing downstream could see that it happened, because a mean is a mean
whatever produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


@dataclass
class StreamingOutput:
    """Efficient generation output with pre-stacked numpy arrays.

    All lists are indexed by generation step, starting AFTER the prefill step.
    Step i corresponds to the model processing gen_token_i and producing
    gen_token_{i+1}. This matches the alignment contract of _convert_outputs_to_raw.

    For a generation producing N tokens:
      - generated_token_ids has N entries: [token_0, token_1, ..., token_{N-1}]
      - hidden_states has N-1 entries (prefill produces token_0, excluded)
      - logits has N-1 entries
      - attentions has N-1 entries (if collected)
    """

    sequences: torch.Tensor                    # [1, prompt_len + N]
    hidden_states: list[F32]                   # (N-1) × [n_layers+1, hidden_dim]
    attentions: list[F32]                      # (N-1) × [n_layers, n_heads, seq_len_at_step]
    logits: list[F32]                          # (N-1) × [vocab_size]
    generated_token_ids: list[int]             # N tokens (all generated, including first)
    prompt_length: int
    prefill_hidden_states: F32 | None = None   # [n_layers+1, prompt_len, hidden_dim], on request


def _sample_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """Sample a single token from logits with temperature and nucleus sampling.

    Args:
        logits: [vocab_size] raw logits (not softmaxed)
        temperature: Sampling temperature (>0)
        top_p: Nucleus sampling threshold

    Returns:
        Sampled token ID
    """
    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Zero out tokens beyond the nucleus
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order and sample
    logits_filtered = torch.full_like(scaled, float("-inf"))
    logits_filtered.scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def streaming_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    eos_token_ids: list[int] | None = None,
    output_hidden_states: bool = True,
    output_attentions: bool = False,
    collect_prefill_hidden_states: bool = False,
) -> StreamingOutput:
    """Generate tokens with efficient streaming state collection.

    Performs the same autoregressive generation as HF's model.generate(), but
    collects internal states (hidden_states, attentions, logits) efficiently
    by doing per-step GPU stacking and immediate CPU transfer.

    The k_proj hooks registered on the model fire normally during each forward
    pass — no special handling needed.

    Args:
        model: HuggingFace causal LM (already on GPU, eval mode)
        input_ids: [1, prompt_len] input token IDs on the model's device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_ids: Token IDs that signal end of generation
        output_hidden_states: Collect per-step hidden states
        output_attentions: Collect per-step attention weights
        collect_prefill_hidden_states: Also collect the prompt's own states,
            which are the one capture whose cost scales with the prompt rather
            than with a step, and which no feature of a generation reads

    Returns:
        StreamingOutput with pre-stacked numpy arrays
    """
    device = input_ids.device
    eos_set = set(eos_token_ids or [])
    prompt_length = input_ids.shape[1]

    # GPU accumulators — kept on device during generation to avoid
    # per-step synchronous CPU transfers. Batch-converted after the loop.
    generated_ids: list[int] = []
    hidden_gpu: list[torch.Tensor] = []
    attn_gpu: list[torch.Tensor] = []
    logit_gpu: list[torch.Tensor] = []
    prefill_hs: F32 | None = None

    past_key_values = None
    current_input = input_ids

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

            step_logits = outputs.logits[0, -1]  # [vocab_size]

            next_token_id = _sample_top_p(step_logits, temperature, top_p)
            generated_ids.append(next_token_id)

            # ── Collect states on GPU ──────────────────────────────

            if step_idx == 0:
                if collect_prefill_hidden_states and output_hidden_states:
                    prefill_stacked = torch.stack([
                        outputs.hidden_states[l][0]
                        for l in range(len(outputs.hidden_states))
                    ])
                    prefill_hs = prefill_stacked.cpu().float().numpy()
                    del prefill_stacked

            else:
                if output_hidden_states and outputs.hidden_states is not None:
                    hs_stacked = torch.stack([
                        outputs.hidden_states[l][0, -1]
                        for l in range(len(outputs.hidden_states))
                    ])  # [n_layers+1, hidden_dim]
                    hidden_gpu.append(hs_stacked)

                if output_attentions and outputs.attentions is not None:
                    attn_stacked = torch.stack([
                        outputs.attentions[l][0, :, -1, :]
                        for l in range(len(outputs.attentions))
                    ])  # [n_layers, n_heads, current_seq_len]
                    attn_gpu.append(attn_stacked)

                logit_gpu.append(step_logits.clone())

            # ── Advance state ───────────────────────────────────────

            past_key_values = outputs.past_key_values
            current_input = torch.tensor(
                [[next_token_id]], device=device, dtype=input_ids.dtype
            )
            del outputs

            if next_token_id in eos_set:
                break

    # ── Batch GPU → CPU transfer ───────────────────────────────────
    hidden_list: list[F32] = []
    if hidden_gpu:
        hs_batch = torch.stack(hidden_gpu).cpu().float().numpy()
        hidden_list = [hs_batch[i] for i in range(hs_batch.shape[0])]
        del hidden_gpu, hs_batch

    logit_list: list[F32] = []
    if logit_gpu:
        lg_batch = torch.stack(logit_gpu).cpu().float().numpy()
        logit_list = [lg_batch[i] for i in range(lg_batch.shape[0])]
        del logit_gpu, lg_batch

    # Attention has variable seq_len per step — can't stack, transfer individually
    attn_list: list[F32] = [a.cpu().float().numpy() for a in attn_gpu]
    del attn_gpu

    if generated_ids:
        gen_tensor = torch.tensor(
            generated_ids, device=device, dtype=input_ids.dtype
        ).unsqueeze(0)
        sequences = torch.cat([input_ids, gen_tensor], dim=1)
    else:
        sequences = input_ids

    return StreamingOutput(
        sequences=sequences,
        hidden_states=hidden_list,
        attentions=attn_list,
        logits=logit_list,
        generated_token_ids=generated_ids,
        prompt_length=prompt_length,
        prefill_hidden_states=prefill_hs,
    )
