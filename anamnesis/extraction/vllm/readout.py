"""Reduce a captured row to the fast lane's full feature vector, outside the engine.

A capture holds what the engine computed; this turns it into the same named
vector the fast lane produces, with the fast lane's own reducers. The residual,
key, value, query, gate and output families run through
:class:`anamnesis.extraction.fast.families.FamilyReducer` and the lane's
operators exactly as :class:`anamnesis.extraction.fast.features.GpuFeatureLane`
runs them; the attention families come from the instrumented backend's
statistics and products through :class:`AttentionFeatureAdapter`.

The per-layer loop restates ``GpuFeatureLane._reduce_capture`` rather than
calling it, because the fast lane's source bytes are part of the fast lane's own
identity: a shared helper would change every fast-lane id already banked. A
test holds the two to identical output on the same capture.

The reduction refuses to run in a process that has imported the engine or
enabled its batch-invariant overrides, which replace torch's matrix products,
and it runs with deterministic algorithms, TF32 off and the cuBLAS workspace
:data:`anamnesis.extraction.vllm.envelope.READOUT_WORKSPACE` fixed before CUDA
starts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.families import FamilyReducer
from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    rowcos,
    std,
    trajectory_indices,
)
from anamnesis.extraction.vllm.adapter import AttentionFeatureAdapter
from anamnesis.extraction.vllm.envelope import READOUT_WORKSPACE
from anamnesis.extraction.vllm.receipts import assert_substrate_fields
from anamnesis.extraction.vllm.rows import SPECTRAL_STRIDE, request_row_schema


@dataclass(frozen=True)
class LaneReadout:
    """One row's reduced vector, its feature names, and what it was read from."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict


def assert_clean_readout_process(device: torch.device | str) -> None:
    """Refuse a process whose arithmetic is not the readout's.

    Raises
    ------
    RuntimeError
        When the engine has been imported here, its batch-invariant overrides
        are enabled or installed, or (on CUDA) the workspace, deterministic
        algorithms or TF32 settings differ from the readout's.
    """
    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise RuntimeError("the readout must run outside the process that imports vLLM")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        raise RuntimeError(
            "vLLM batch-invariant overrides are forbidden in the readout process"
        )
    for fn in (torch.mm, torch.matmul, torch.nn.functional.linear):
        if "vllm" in getattr(fn, "__module__", ""):
            raise RuntimeError("a vLLM override of a torch matrix product is installed")
    if torch.device(device).type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != READOUT_WORKSPACE:
            raise RuntimeError(
                f"set CUBLAS_WORKSPACE_CONFIG={READOUT_WORKSPACE} before CUDA starts"
            )
        if (
            not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
        ):
            raise RuntimeError("the readout requires deterministic algorithms")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("the readout forbids TF32")


def _validate(lane, capture, start, end, model):
    if not isinstance(model, str) or not model:
        raise ValueError("model identity required")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end - start < 2
    ):
        raise ValueError("invalid start/end prediction span")
    assert_substrate_fields(capture, context="feature readout")
    h = capture["hidden"]
    if not isinstance(h, torch.Tensor) or h.ndim != 3 or not h.is_floating_point():
        raise ValueError("hidden must be native floating [N,T,D] tensor")
    n, t, d = h.shape
    if t != end - start - 1 or n < 1 or d < 1:
        raise ValueError("hidden span shape mismatch")
    if (
        lane.pm.dtype != np.float32
        or lane.pm.ndim != 3
        or lane.pm.shape[0] != n + 1
        or lane.pm.shape[2] != d
        or lane.pm.shape[1] < end - 1
    ):
        raise ValueError("positional means must cover [N+1,positions,D] in float32")
    if h.device != lane.device:
        raise ValueError("capture device differs from reference lane")
    layers = set(lane.config.sampled_layers)
    if (
        not layers.issubset(range(n))
        or not set(lane.config.pca_layers).issubset(range(n))
        or not set(lane.families.trajectory_layers).issubset(range(n))
    ):
        raise ValueError("configured layer outside capture")
    tensors = [h]
    shapes = {}
    for kind in ("keys", "values", "queries", "gates"):
        values = capture[kind]
        if not isinstance(values, Mapping) or set(values) != layers:
            raise ValueError(f"missing/extra {kind} layers")
        for layer, x in values.items():
            ndim = 2 if kind == "gates" else 3
            if (
                not isinstance(x, torch.Tensor)
                or x.ndim != ndim
                or x.shape[0] != t
                or min(x.shape) < 1
            ):
                raise ValueError(f"invalid {kind} shape at layer {layer}")
            if x.dtype != h.dtype:
                raise ValueError("mixed native substrate dtypes")
            shape = tuple(x.shape[1:])
            if kind in shapes and shapes[kind] != shape:
                raise ValueError(f"mixed {kind} shapes across layers")
            shapes[kind] = shape
            tensors.append(x)
    if layers and (
        shapes["keys"] != shapes["values"]
        or shapes["queries"][-1] != shapes["keys"][-1]
        or np.prod(shapes["queries"]) != d
        or shapes["queries"][0] % shapes["keys"][0]
    ):
        raise ValueError("incompatible query/key/value head shapes")
    logits, chosen = capture["logits"], capture["chosen"]
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] != t
        or logits.shape[1] < 5
    ):
        raise ValueError("logits must be [T,full_vocab] with vocabulary >=5")
    if (
        not isinstance(chosen, torch.Tensor)
        or chosen.dtype != torch.int64
        or chosen.shape != (t,)
    ):
        raise ValueError("chosen must contain one int64 next-token ID per logit row")
    if chosen.device != h.device or bool(
        ((chosen < 0) | (chosen >= logits.shape[1])).any()
    ):
        raise ValueError("chosen token outside vocabulary or on wrong device")
    tensors.append(logits)
    for x in tensors:
        if (
            x.device != h.device
            or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())
        ):
            raise ValueError("mixed devices or nonfinite/nonfloating capture")
    every = set(range(n))
    checks = [(key, every) for key in
              ("attn_stats", "attn_h_mean", "attn_h_heads", "attn_entropy_rows")]
    checks += [(key, layers) for key in
               ("attn_coverage", "attn_spectral_rows", "attn_decay_rows",
                "attn_span_rows", "attn_head_ent", "attn_head_sink",
                "attn_head_prompt", "attn_head_recency")]
    for key, expected in checks:
        values = capture[key]
        if not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError(f"missing/extra {key} layers")
        if not all(isinstance(x, torch.Tensor) for x in values.values()):
            raise ValueError(f"{key} must map layers to tensors")
    return n, t


@torch.no_grad()
def reduce_capture(lane: GpuFeatureLane, capture: dict, *, start: int, end: int,
                   model: str) -> LaneReadout:
    """Reduce one captured row to the lane's full feature vector.

    ``start`` is the prompt length and ``end`` the row's full length: query
    positions ``start:end-1`` are read against labels ``start+1:end``, and
    ``hidden[i]`` is the output of block ``i`` (the last one after the final
    norm). ``lane`` supplies the configuration, calibration and schema; its
    names fix the vector's order. The caller has already checked the
    capture's receipt against its bytes.

    Raises
    ------
    RuntimeError
        When the process is not a clean readout process (see
        :func:`assert_clean_readout_process`).
    ValueError
        When the capture's fields, shapes, devices or values are not a complete
        finite capture of this span, or the lane's spectral stride differs from
        the one the capture selected rows at.
    """
    assert_clean_readout_process(lane.device)
    if (
        lane.device.type == "cuda"
        and lane.identity["cublas_workspace_config"] != READOUT_WORKSPACE
    ):
        raise RuntimeError("the lane's workspace differs from the readout workspace")
    n, steps = _validate(lane, capture, start, end, model)
    if lane.config.spectral_subsample_step != SPECTRAL_STRIDE:
        raise ValueError(
            "the capture selected spectral rows at a stride the lane configuration "
            "does not use"
        )
    names = tuple(lane.names)
    if not names or len(set(names)) != len(names):
        raise ValueError("empty or duplicate feature schema")
    out = FeatureCollector(lane.device)
    reducer = FamilyReducer(out, lane.config, lane.families)
    adapter = AttentionFeatureAdapter(
        out,
        schema=request_row_schema(prompt_length=start, end=end),
        num_layers=n,
        sampled_layers=lane.config.sampled_layers,
        n_windows=lane.families.temporal_n_windows,
        include_stft=lane.families.enable_stft,
    )
    sampled = set(lane.config.sampled_layers)
    previous = None
    for layer in range(n):
        h = capture["hidden"][layer].float()
        pm = np.ascontiguousarray(lane.pm[layer + 1, start : start + steps])
        corrected = h - torch.as_tensor(pm, device=lane.device)
        norms = corrected.norm(dim=-1).float()
        out.put(f"activation_norm_mean_L{layer}", norms.mean())
        out.put(f"activation_norm_std_L{layer}", std(norms))
        for i, t in enumerate(trajectory_indices(steps, lane.config.trajectory_points)):
            out.put(f"activation_norm_traj{i}_L{layer}", norms[t])
        if previous is not None:
            delta = corrected - previous
            dn = delta.norm(dim=-1).float()
            dc = rowcos(delta.double(), previous.double()).float()
            out.put(f"delta_norm_mean_L{layer - 1}", dn.mean())
            out.put(f"delta_norm_std_L{layer - 1}", std(dn))
            out.put(f"delta_cosine_mean_L{layer - 1}", dc.mean())
        previous = corrected
        extras = {}
        if layer in sampled:
            extras = {
                key: capture[f"attn_{key}"][layer].to(lane.device)
                for key in ("coverage", "spectral_rows", "decay_rows", "span_rows",
                            "head_ent", "head_sink", "head_prompt", "head_recency")
            }
        adapter.consume(
            layer,
            stats=capture["attn_stats"][layer].to(lane.device),
            h_mean=capture["attn_h_mean"][layer].to(lane.device),
            h_heads=capture["attn_h_heads"][layer].to(lane.device),
            entropy_rows=capture["attn_entropy_rows"][layer].to(lane.device),
            **extras,
        )
        if layer in sampled:
            adapter.smoothness(layer, corrected)
        if layer in lane.families.trajectory_layers:
            reducer.residual_trajectory(layer, corrected, h[0].norm())
        if layer in lane.config.pca_layers:
            components = lane.pca[layer] if isinstance(lane.pca, dict) else lane.pca
            mean = (
                lane.pca_mean[layer]
                if isinstance(lane.pca_mean, dict)
                else lane.pca_mean
            )
            temporal = trajectory_indices(steps, lane.config.pca_temporal_samples)
            projections = (corrected[temporal].double() - mean) @ components[
                : lane.config.pca_components
            ].T
            for ti, projection in enumerate(projections):
                for ci, value in enumerate(projection):
                    out.put(f"pca_L{layer}_t{ti}_c{ci}", value)
    lane._outputs(out, capture["logits"], capture["chosen"])
    for layer in lane.config.sampled_layers:
        keys = capture["keys"][layer].float()
        reducer.key(layer, keys)
        if steps < 4:
            head_mean = head_std = 0.0
        else:
            k = keys.double()
            center = k.mean(dim=0, keepdim=True)
            sims = (k * center).sum(dim=-1) / (
                k.norm(dim=-1) * center.norm(dim=-1)
            ).clamp_min(1e-12)
            spread = (1 - sims).mean(dim=0)
            head_mean, head_std = spread.mean(), std(spread)
        out.put(f"ph_L{layer}_kv_key_spread_head_mean", head_mean)
        out.put(f"ph_L{layer}_kv_key_spread_head_std", head_std)
        reducer.value(layer, capture["values"][layer].float())
        reducer.query(layer, capture["queries"][layer].float())
        reducer.gate(layer, capture["gates"][layer].float())
    reducer.finish()
    adapter.assert_complete()
    vector = out.finish(list(names)).cpu().numpy()
    if not np.isfinite(vector).all():
        raise ValueError("the readout produced a nonfinite feature")
    return LaneReadout(
        vector,
        names,
        {
            "model": model,
            "start": start,
            "end": end,
            "steps": steps,
            "workspace": READOUT_WORKSPACE,
            "substrate_dtype": str(capture["hidden"].dtype),
            "logits_dtype": str(capture["logits"].dtype),
        },
    )
