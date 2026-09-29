"""Cached-bridge replay: teacher-force a continuation against an INJECTED
(possibly surgically-modified) KV cache and extract per-step states.

The capture surface is the full one — pre-RoPE keys and gates plus values, queries
and attention outputs — so a cached replay and a generate-path run feed the same
feature families.

Alignment contract: prompt_length = cache length, so the extractor's
prompt/generated split lands exactly on the cache/continuation boundary — the
surgered-cache readout the key/attention features need. T = N-1 per-step entries
for an N-token continuation (same convention as :mod:`..replay.extract`).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from anamnesis.extraction.model_loader import LoadedModel
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def cache_length(past_key_values) -> int:
    from anamnesis.extraction.replay.cache_surgery import _extract_kv

    keys, _ = _extract_kv(past_key_values)
    if not keys:
        raise ValueError("cache has no layers")
    return int(keys[0].shape[-2])


def remap_positional_means(
    positional_means: F32 | None, cache_len: int, position_offset: int, n_steps: int
) -> F32 | None:
    """Make extractor lookups (index cache_len + t) return the means of the TRUE
    absolute positions (position_offset + t). Identity for FULL/ROT/REC; only
    NAIVE (survivors keep original positions => offset > cache_len) with a short
    cache needs the copy."""
    if positional_means is None or position_offset == cache_len:
        return positional_means
    if position_offset < cache_len:
        raise ValueError(f"position_offset {position_offset} < cache_len {cache_len}")
    max_pos = positional_means.shape[1]
    if cache_len >= max_pos - 1:
        return positional_means
    remapped = positional_means.copy()
    for t in range(n_steps):
        dst = cache_len + t
        if dst >= max_pos:
            break
        src = min(position_offset + t, max_pos - 1)
        remapped[:, dst, :] = positional_means[:, src, :]
    return remapped


def _slice_head_hook(d: dict[int, list[Tensor]], t_steps: int) -> dict[int, list[F32]] | None:
    """[1, heads, N, head_dim] captures (single cached forward) → {layer: T × [heads, head_dim]}."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in d.items():
        if not tensors:
            continue
        full = tensors[0]
        rows = full[0, :, :t_steps, :].float().cpu().numpy()  # [heads, T, head_dim]
        result[int(layer_idx)] = [rows[:, i, :] for i in range(t_steps)]
    return result or None


def _slice_seq_hook(d: dict[int, list[Tensor]], t_steps: int) -> dict[int, list[F32]] | None:
    """[1, N, width] captures → {layer: T × [width]} (gate_proj / o_proj shapes)."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in d.items():
        if not tensors:
            continue
        full = tensors[0]
        rows = full[0, :t_steps, :].float().cpu().numpy()  # [T, width]
        result[int(layer_idx)] = [rows[i] for i in range(t_steps)]
    return result or None


def replay_extract_cached(
    loaded: LoadedModel,
    past_key_values,
    cont_ids: Tensor | list[int] | NDArray,
    position_offset: int,
    positional_means: F32 | None = None,
) -> RawGenerationData:
    """Teacher-force cont_ids against an injected cache; extract per-step states.

    past_key_values is CONSUMED (the forward appends continuation KV) — pass a
    fresh cache per call (rebuild from the KVSnapshot; snapshot tensors are not
    mutated by to_hf_dynamic_cache). position_offset = absolute RoPE position of
    the first continuation token: cache length for FULL/ROT/REC, survivor-max+1
    for NAIVE.
    """
    device = next(loaded.model.parameters()).device
    if isinstance(cont_ids, torch.Tensor):
        ids = cont_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(np.asarray(cont_ids), dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError(f"replay_extract_cached expects batch=1, got {ids.shape[0]}")

    n = int(ids.shape[1])
    if n < 2:
        raise ValueError(f"need >= 2 continuation tokens, got {n}")
    t_steps = n - 1

    cache_len = cache_length(past_key_values)
    if cache_len <= 0:
        raise ValueError("injected cache is empty — prefill the context first")
    if position_offset < cache_len:
        raise ValueError(f"position_offset {position_offset} < cache_len {cache_len}")
    pm_eff = remap_positional_means(positional_means, cache_len, position_offset, t_steps)

    loaded.clear_hook_state()
    loaded.enable_hooks()
    try:
        with torch.no_grad():
            out = loaded.model(
                ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                position_ids=torch.arange(
                    position_offset, position_offset + n, device=device
                ).unsqueeze(0),
                cache_position=torch.arange(cache_len, cache_len + n, device=device),
            )
        loaded.flush_hooks_to_cpu()

        hs = out.hidden_states  # tuple(num_layers+1) of [1, N, hidden]
        n_hs_layers = len(hs)
        hs_rows = [hs[l][0, :t_steps].float().cpu().numpy() for l in range(n_hs_layers)]
        hidden_states: list[F32] = [
            np.stack([hs_rows[l][i] for l in range(n_hs_layers)]) for i in range(t_steps)
        ]

        att = out.attentions  # tuple(num_layers) of [1, H, N, cache_len + N]
        if att is None or len(att) == 0:
            raise RuntimeError("no attentions returned — model must be loaded eager")
        exp_cols = cache_len + n
        if int(att[0].shape[-1]) != exp_cols:
            raise RuntimeError(
                f"attention columns {int(att[0].shape[-1])} != cache_len + N ({exp_cols}) "
                "— cache injection did not take effect"
            )
        att_rows = [a[0, :, :t_steps, :].float().cpu().numpy() for a in att]  # [H, T, C+N]
        attentions: list[F32] = [
            np.stack([att_rows[l][:, i, : cache_len + i + 1] for l in range(len(att))])
            for i in range(t_steps)
        ]

        logits_rows = out.logits[0, :t_steps].float().cpu().numpy()
        logits: list[F32] = [logits_rows[i] for i in range(t_steps)]
        del out

        pre_rope_keys = _slice_head_hook(loaded.hook_state.pre_rope_keys, t_steps) or {}
        v_proj_values = _slice_head_hook(loaded.hook_state.v_proj_values, t_steps)
        queries = _slice_head_hook(loaded.hook_state.queries, t_steps)
        attn_outputs = _slice_seq_hook(loaded.hook_state.attn_outputs, t_steps)
        gate_activations = _slice_seq_hook(loaded.hook_state.gate_activations, t_steps)
    finally:
        loaded.clear_hook_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chosen_token_ids = ids[0, 1:n].float().cpu().numpy()

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_token_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=cache_len,
        positional_means=pm_eff,
        gate_activations=gate_activations,
        v_proj_values=v_proj_values,
        queries=queries,
        attn_outputs=attn_outputs,
    )


def replay_extract_incremental(
    loaded: LoadedModel,
    full_token_ids: list[int] | NDArray | Tensor,
    prompt_length: int,
    positional_means: F32 | None = None,
) -> RawGenerationData:
    """:func:`anamnesis.extraction.replay.extract.replay_extract`'s states and alignment,
    computed the generate path's way: prompt prefill, then one cached forward per
    generated position. The distance between the two paths' vectors is a row's path
    floor. Dense models only: a router surface is refused, not dropped.

    Raises
    ------
    ValueError
        On a batch other than one sequence, or a span with no prompt or fewer than
        two generated tokens; on a CUDA model outside the lane's arithmetic, where a
        path floor is not defined (:func:`anamnesis.extraction.fast.runtime.refuse_unpinned_floor`).
    NotImplementedError
        When the loaded model captures a router surface.
    """
    from anamnesis.extraction.fast.runtime import refuse_unpinned_floor

    device = next(loaded.model.parameters()).device
    refuse_unpinned_floor(device)
    if isinstance(full_token_ids, Tensor):
        ids = full_token_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(np.asarray(full_token_ids), dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError(f"replay_extract_incremental expects one sequence, got {ids.shape[0]}")
    length, start = int(ids.shape[1]), int(prompt_length)
    if not 0 < start < length - 1:
        raise ValueError(f"prompt_length {start} leaves fewer than two generated tokens "
                         f"of {length}")
    hidden: list[F32] = []
    attentions: list[F32] = []
    logits: list[F32] = []
    stores: dict[str, dict[int, list[F32]]] = {
        name: {} for name in ("pre_rope_keys", "v_proj_values", "queries",
                              "gate_activations", "attn_outputs")}
    loaded.clear_hook_state()
    loaded.disable_hooks()
    try:
        with torch.no_grad():
            prefill = loaded.model(ids[:, :start], use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        del prefill
        for position in range(start, length - 1):
            loaded.clear_hook_state()
            loaded.enable_hooks()
            with torch.no_grad():
                out = loaded.model(
                    ids[:, position:position + 1], past_key_values=cache, use_cache=True,
                    output_hidden_states=True, output_attentions=True, return_dict=True,
                    position_ids=torch.tensor([[position]], device=device),
                    cache_position=torch.tensor([position], device=device),
                )
            if loaded.hook_state.router_dist or loaded.hook_state.router_logit_norm:
                raise NotImplementedError("the incremental path captures dense-model "
                                          "surfaces only; this model routes to experts")
            cache = out.past_key_values
            hidden.append(np.stack([h[0, 0].float().cpu().numpy() for h in out.hidden_states]))
            attentions.append(np.stack([a[0, :, 0, :].float().cpu().numpy()
                                        for a in out.attentions]))
            logits.append(out.logits[0, 0].float().cpu().numpy())
            for name, dest in stores.items():
                slicer = _slice_seq_hook if name in ("gate_activations", "attn_outputs") \
                    else _slice_head_hook
                for layer, values in (slicer(getattr(loaded.hook_state, name), 1) or {}).items():
                    dest.setdefault(layer, []).extend(values)
            del out
    finally:
        loaded.clear_hook_state()
    return RawGenerationData(
        hidden_states=hidden,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=ids[0, start + 1:length].float().cpu().numpy(),
        pre_rope_keys=stores["pre_rope_keys"],
        prompt_length=start,
        positional_means=positional_means,
        gate_activations=stores["gate_activations"] or None,
        v_proj_values=stores["v_proj_values"] or None,
        queries=stores["queries"] or None,
        attn_outputs=stores["attn_outputs"] or None,
    )
