"""The fast lane's reduction of a capture an outside model produced, on the device.

:class:`~anamnesis.extraction.fast.features.GpuFeatureLane` drives a Hugging Face
decoder itself: its hooks read each layer's attention weights as the forward runs,
and its ``_reduce_capture`` reads the hidden states, projections and gates the
forward left behind. A model that is not a Hugging Face decoder (one whose layers
route over committed blocks rather than one running residual, say) runs its own
instrumented forward instead and hands this module what that forward computed:

* :meth:`ExternalSpan.attention`, once per layer while the forward runs, with that
  layer's attention weights over the span's rows, so the all-layer weights never
  coexist on the device;
* :meth:`ExternalSpan.finish`, once, with the span's block outputs, the sampled
  layers' keys, values, queries and gate activations, the logits, and, for a
  block-routing architecture, its routing weights and committed blocks.

The reduction is the fast lane's: the same collector, attention reducer and family
reducer, the same positional correction, PCA projection and output statistics, in
the same order. The per-layer loop restates ``GpuFeatureLane._reduce_capture``
rather than calling it, because the fast lane's source bytes are part of its own
identity and a shared helper would move every fast-lane id already banked; a test
holds the two to identical bytes on one capture. The block-routing family runs as
:mod:`anamnesis.extraction.fast.attn_res`, a device port of the canonical family,
and its coordinates follow the battery's, where the canonical pipeline puts them.

What the capture means is the caller's to state: :meth:`ExternalSpan.finish` takes
a ``capture_identity`` mapping (the producing code and its arithmetic) and folds it
into the row's ``lane_id`` with the fast lane's own id and this module's source
digest. Two captures of one architecture by different code are different lanes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.attn_res import attn_res_features
from anamnesis.extraction.fast.families import FamilyReducer
from anamnesis.extraction.fast.features import GpuFeatureLane, GpuFeatureResult
from anamnesis.extraction.fast.ops import FeatureCollector, rowcos, std, trajectory_indices

READOUT_SOURCE_SHA256 = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__), Path(__file__).with_name("attn_res.py"))
}
"""This module's and the routing family's source digests, part of every row's lane id."""

RoutingPoint = tuple[str, int, Tensor]


class ExternalCaptureError(ValueError):
    """A capture that is not a complete, finite capture of the span the lane reads."""


def external_lane_id(lane: GpuFeatureLane, capture_identity: Mapping[str, Any]) -> str:
    """The lane id of rows reduced here: the fast lane's id, this readout's source and
    the producing capture's identity, hashed together."""
    payload = dict(fast_lane_id=lane.lane_id, readout_source_sha256=READOUT_SOURCE_SHA256,
                   capture=dict(capture_identity))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class ExternalSpan:
    """One span through the fast lane's reducers, fed by an outside forward.

    ``start`` is the prompt length and ``end`` the row's full length: the span's
    ``end - start - 1`` steps are positions ``start .. end - 2``, each read against the
    token after it, as in the fast lane's full replay.
    """

    def __init__(self, lane: GpuFeatureLane, *, start: int, end: int, num_layers: int):
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end - start < 2:
            raise ExternalCaptureError(f"invalid span start={start} end={end}")
        if num_layers < 1:
            raise ExternalCaptureError("a capture has at least one layer")
        self.lane, self.start, self.end, self.num_layers = lane, start, end, num_layers
        self.steps = end - start - 1
        self.collector = FeatureCollector(lane.device)
        self.attention_reducer = AttentionReducer(
            self.collector,
            steps=self.steps,
            prefix_length=start,
            sampled_layers=lane.config.sampled_layers,
            spectral_stride=lane.config.spectral_subsample_step,
            n_windows=lane.families.temporal_n_windows,
            include_stft=lane.families.enable_stft,
        )
        self.reducer = FamilyReducer(self.collector, lane.config, lane.families)
        self.finished = False

    def attention(self, layer: int, weights: Tensor) -> None:
        """Reduce one layer's attention weights now, so they need not be kept.

        ``weights`` is ``[heads, rows, columns]`` or ``[1, heads, rows, columns]`` over the
        forward's whole sequence; rows ``start .. end - 1`` and columns ``0 .. end - 1``
        are read, the rows the fast lane's full replay hands its reducer.
        """
        if self.finished:
            raise ExternalCaptureError("attention after the span was finished")
        if not 0 <= layer < self.num_layers:
            raise ExternalCaptureError(f"attention layer {layer} outside 0..{self.num_layers - 1}")
        w = weights if weights.ndim == 4 else weights.unsqueeze(0)
        if w.ndim != 4 or w.shape[0] != 1 or w.shape[2] < self.end or w.shape[3] < self.end:
            raise ExternalCaptureError(f"attention weights {tuple(weights.shape)} do not cover the span")
        self.attention_reducer.consume(layer, w[:, :, self.start:self.end, :self.end].to(self.lane.device))

    @torch.no_grad()
    def finish(
        self,
        *,
        hidden: Tensor,
        keys: Mapping[int, Tensor],
        values: Mapping[int, Tensor],
        queries: Mapping[int, Tensor],
        gates: Mapping[int, Tensor],
        logits: Tensor,
        token_ids: Sequence[int],
        capture_identity: Mapping[str, Any],
        routing: Sequence[RoutingPoint] | None = None,
        committed: Sequence[Tensor] | None = None,
    ) -> GpuFeatureResult:
        """The span's feature vector, in the fast lane's schema then the routing family's.

        ``hidden`` is ``[layers, steps, width]``: each block's output at the span's
        positions (the embedding is not included). ``keys``, ``values`` and
        ``queries`` map each sampled layer to ``[steps, heads, head_dim]`` (keys before
        rotary embedding); ``gates`` to ``[steps, intermediate]``; ``logits`` is
        ``[steps, vocabulary]``. ``routing`` is ``(tag, layer, weights[steps, sources])``
        per routing point, tag ``"final"`` for the pre-output read, and ``committed``
        each committed block's ``[steps, width]``; both or neither.

        Raises
        ------
        ExternalCaptureError
            When the capture is incomplete, mis-shaped or nonfinite, an attention layer
            was not reduced, or the arithmetic is not the lane's.
        """
        if self.finished:
            raise ExternalCaptureError("a span is finished once")
        self.finished = True
        lane, steps, start, end = self.lane, self.steps, self.start, self.end
        self._validate(hidden, keys, values, queries, gates, logits, token_ids, routing, committed)
        if self.attention_reducer.seen != set(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - self.attention_reducer.seen)
            raise ExternalCaptureError(f"attention layers {missing} were not reduced")
        out, attention, reducer = self.collector, self.attention_reducer, self.reducer
        pm = np.ascontiguousarray(lane.pm[1:self.num_layers + 1, start:start + steps])
        positional = torch.as_tensor(pm, device=lane.device)
        previous = None
        for layer in range(self.num_layers):
            h = hidden[layer].to(lane.device).float()
            corrected = h - positional[layer]
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
            if layer in lane.config.sampled_layers:
                attention.smoothness(layer, corrected)
            if layer in lane.families.trajectory_layers:
                reducer.residual_trajectory(layer, corrected, h[0].norm())
            if layer in lane.config.pca_layers:
                components = lane.pca[layer] if isinstance(lane.pca, dict) else lane.pca
                mean = lane.pca_mean[layer] if isinstance(lane.pca_mean, dict) else lane.pca_mean
                indices = trajectory_indices(steps, lane.config.pca_temporal_samples)
                projections = (corrected[indices].double() - mean) @ components[
                    : lane.config.pca_components
                ].T
                for ti, projection in enumerate(projections):
                    for ci, value in enumerate(projection):
                        out.put(f"pca_L{layer}_t{ti}_c{ci}", value)
        chosen = torch.as_tensor(list(token_ids[start + 1:end]), device=lane.device, dtype=torch.long)
        logprob = lane._outputs(out, logits.to(lane.device), chosen)
        for layer in lane.config.sampled_layers:
            k = keys[layer].to(lane.device).float()
            reducer.key(layer, k)
            attention.head_features(layer, k)
            reducer.value(layer, values[layer].to(lane.device).float())
            reducer.query(layer, queries[layer].to(lane.device).float())
            reducer.gate(layer, gates[layer].to(lane.device).float())
        reducer.finish()
        vector = out.finish(list(lane.names))
        names = list(lane.names)
        if routing is not None:
            routed_names, routed = attn_res_features(
                [(tag, layer, w.to(lane.device)) for tag, layer, w in routing],
                [c.to(lane.device) for c in committed or []],
                lane.config.sampled_layers,
            )
            vector = torch.cat([vector, routed.to(vector.dtype)])
            names += routed_names
        features = vector.cpu().numpy()
        if not np.isfinite(features).all():
            raise ExternalCaptureError("the reduction produced a nonfinite feature")
        knnlm = (hidden[-1][steps - 1].float().cpu().numpy().copy()
                 if lane.config.enable_knnlm_baseline else None)
        receipt = dict(
            lane_id=external_lane_id(lane, capture_identity),
            fast_lane_id=lane.lane_id,
            lane_source_sha256=dict(lane.source_sha256),
            readout_source_sha256=dict(READOUT_SOURCE_SHA256),
            capture=dict(capture_identity),
            replay_id=str(uuid.uuid4()),
            replay_path="external",
            span_start=start,
            span_end=end,
            input_tokens_sha256=hashlib.sha256(
                np.asarray(list(token_ids[:end]), dtype="<i8").tobytes()).hexdigest(),
            feature_schema_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            calibration_sha256=lane.calibration_sha256,
            channel="probe-free",
            certified=False,
            batch_size=1,
            attention_layers_reduced=len(attention.seen),
            retained_attention_summary_bytes=attention.retained_bytes,
            routing_family=routing is not None,
        )
        return GpuFeatureResult(features, tuple(names), float(logprob.cpu()), knnlm, receipt)

    def _validate(self, hidden, keys, values, queries, gates, logits, token_ids, routing,
                  committed) -> None:
        lane, steps = self.lane, self.steps
        if not torch.are_deterministic_algorithms_enabled():
            raise ExternalCaptureError("the reduction runs with deterministic algorithms on")
        if lane.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ExternalCaptureError("the reduction runs with TF32 off")
        if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[:2] != (self.num_layers, steps):
            raise ExternalCaptureError(
                f"hidden must be [{self.num_layers}, {steps}, width], got "
                f"{tuple(hidden.shape) if isinstance(hidden, Tensor) else type(hidden).__name__}")
        if (lane.pm.ndim != 3 or lane.pm.shape[0] < self.num_layers + 1
                or lane.pm.shape[1] < self.start + steps or lane.pm.shape[2] != hidden.shape[2]):
            raise ExternalCaptureError("the lane's positional means do not cover this capture")
        sampled = set(lane.config.sampled_layers)
        if not sampled <= set(range(self.num_layers)):
            raise ExternalCaptureError("a sampled layer lies outside the capture")
        for name, mapping, ndim in (("keys", keys, 3), ("values", values, 3), ("queries", queries, 3),
                                    ("gates", gates, 2)):
            if set(mapping) != sampled:
                raise ExternalCaptureError(f"{name} must hold exactly the sampled layers")
            for layer, x in mapping.items():
                if x.ndim != ndim or x.shape[0] != steps:
                    raise ExternalCaptureError(f"{name} layer {layer} has shape {tuple(x.shape)}")
        if logits.ndim != 2 or logits.shape[0] != steps or logits.shape[1] < 5:
            raise ExternalCaptureError("logits must be [steps, vocabulary]")
        if len(token_ids) < self.end:
            raise ExternalCaptureError("token_ids do not cover the span")
        if (routing is None) != (committed is None):
            raise ExternalCaptureError("routing and committed blocks come together")
        for x in (hidden, logits, *keys.values(), *values.values(), *queries.values(), *gates.values()):
            if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
                raise ExternalCaptureError("the capture holds a nonfloating or nonfinite tensor")
