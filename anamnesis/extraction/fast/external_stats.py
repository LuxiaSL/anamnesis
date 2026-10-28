"""The fast lane's reduction of a capture that hands over attention statistics, not weights.

:class:`~anamnesis.extraction.fast.external.ExternalSpan` reduces each layer's attention
weights as an outside forward produces them. A forward whose attention kernel never
materialises the weights (a fused decode kernel, say) can instead emit, per span
position, the handful of statistics every attention-family feature is a function of.
:class:`ExternalStatsSpan` takes those and reduces them into the same named features;
everything else (block outputs, keys, values, queries, gates, logits, routing and
committed blocks) is :meth:`ExternalSpan.finish`'s, taken and reduced as it takes them.

**The contract.** One span of ``n = end - start - 1`` steps; step ``j`` is position
``start + j``, whose causal row has ``len = start + j + 1`` columns (``0 .. start + j``,
itself included), and ``p`` is a row's attention, summing to one per head. Every tensor
is float32:

* ``head_entropy [n, L, H]``: per query head, ``-sum(p ln p)`` over the row;
* ``mean_entropy [n, L]``: the same entropy of the head-mean row ``m = mean_h p_h``;
* at the lane's sampled layers only, ``S`` of them in ``sampled_layers`` order:

  * ``head_prompt_mass [n, S, H]``: per head, ``sum(p)`` over columns ``< start``;
  * ``head_recency_mass [n, S, H]``: per head, ``sum(p)`` over columns
    ``>= max(1, floor(0.8 * len))``;
  * ``head_sink [n, S, H]``: ``p`` at column 0;
  * ``mean_rows [n, S, W]``: the head-mean row at absolute columns, ``W >= end - 1``,
    zero past column ``start + j``.

:func:`stats_from_weights` builds the contract from eager weights, which makes it the
reference a producer of these statistics is checked against.

**What is derived from what.** The entropy and head-agreement features read the two
entropies; recency, sink, coverage, lookback, the decay fit, the attention-flow regions
and their operators, the recency trajectory and the spectral graph read the mean rows;
attention-flow head diversity reads the per-head masses; the per-head family reads the
per-head entropy, normalised by ``ln(max(len, 2))``, and the sink. The arithmetic over
the mean rows is :meth:`AttentionReducer.consume`'s, restated here in
:meth:`StatsAttentionReducer.consume_stats` beside the derivations that replace its
reads of the weights, because the fast lane's source bytes are part of its identity;
a test holds the two to agreement. Smoothness and the per-head key features are the
attention reducer's own, inherited.

The statistics are not the weights, so the vector agrees with
:class:`ExternalSpan`'s on the same forward to rounding, not to the byte. A row reduced
here is therefore its own lane: its ``lane_id`` is :func:`stats_lane_id`, the capture's
external lane id with this module's digest and the capture kind ``engine-stats``, and
it carries a prefix no :class:`ExternalSpan` id has.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.external import (
    ExternalCaptureError,
    ExternalSpan,
    RoutingPoint,
    external_lane_id,
)
from anamnesis.extraction.fast.features import GpuFeatureLane, GpuFeatureResult
from anamnesis.extraction.fast.ops import decay, entropy, slope, std

STATS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""This module's source digest, part of every row's lane id."""

CAPTURE_KIND = "engine-stats"
"""The capture kind a row reduced here declares in its lane identity and receipt."""

LANE_PREFIX = "engine-stats-"
"""Every lane id of a row reduced here starts with this; no ExternalSpan id does."""

CONTRACT = ("head_entropy", "mean_entropy", "head_prompt_mass", "head_recency_mass",
            "head_sink", "mean_rows")
"""The contract's tensors, by name, in the order :meth:`ExternalStatsSpan.attention_stats` takes them."""


def stats_lane_id(lane: GpuFeatureLane, capture_identity: Mapping[str, Any]) -> str:
    """The lane id of a row reduced from attention statistics."""
    payload = dict(external_lane_id=external_lane_id(lane, capture_identity),
                   stats_source_sha256=STATS_SOURCE_SHA256, capture_kind=CAPTURE_KIND)
    return LANE_PREFIX + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def stats_from_weights(weights: Sequence[Tensor], start: int, end: int,
                       sampled_layers: Sequence[int]) -> dict[str, Tensor]:
    """The contract's tensors for one span, from each layer's eager weights.

    ``weights[layer]`` is ``[heads, rows, columns]`` or ``[1, heads, rows, columns]``
    over the forward's sequence, as :meth:`ExternalSpan.attention` takes it. Each row is
    read the way the attention reducer reads it: float32, masked to its causal length,
    then renormalised in float64 where the reducer renormalises, so the statistics are
    the reducer's own intermediates.
    """
    n = end - start - 1
    if n < 1 or start < 1:
        raise ExternalCaptureError(f"invalid span start={start} end={end}")
    sampled = list(sampled_layers)
    entropies: dict[str, list[Tensor]] = {"head_entropy": [], "mean_entropy": []}
    at_sampled: dict[int, dict[str, Tensor]] = {}
    for layer, w in enumerate(weights):
        w = w if w.ndim == 4 else w.unsqueeze(0)
        if w.ndim != 4 or w.shape[0] != 1 or w.shape[2] < end or w.shape[3] < end:
            raise ExternalCaptureError(f"layer {layer} weights {tuple(w.shape)} do not cover the span")
        a = w[0, :, start:start + n, :end].permute(1, 0, 2).float()
        pos = torch.arange(end, device=a.device)
        lengths = torch.arange(start + 1, start + n + 1, device=a.device)
        valid = pos[None, :] < lengths[:, None]
        a = a * valid[:, None, :]
        a64 = a.double()
        norm = a64 / a64.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropies["head_entropy"].append(entropy(a))
        entropies["mean_entropy"].append(entropy(norm.mean(dim=1)))
        if layer in sampled:
            cutoff = (lengths.double() * 0.8).long().clamp_min(1)
            at_sampled[layer] = dict(
                head_prompt_mass=norm[:, :, :start].sum(dim=-1).float(),
                head_recency_mass=(norm * (pos[None, None, :] >= cutoff[:, None, None]))
                .sum(dim=-1).float(),
                head_sink=norm[:, :, 0].float(),
                mean_rows=a.mean(dim=1)[:, :end - 1],
            )
    missing = [layer for layer in sampled if layer not in at_sampled]
    if missing:
        raise ExternalCaptureError(f"the weights do not cover sampled layers {missing}")
    stats = {name: torch.stack(tensors, dim=1) for name, tensors in entropies.items()}
    for name in CONTRACT[2:]:
        stats[name] = torch.stack([at_sampled[layer][name] for layer in sampled], dim=1)
    return stats


class StatsAttentionReducer(AttentionReducer):
    """The attention reducer fed a span's statistics instead of its weights."""

    def consume(self, layer: int, weights: Tensor) -> None:
        raise ExternalCaptureError("a statistics span takes no attention weights")

    def consume_stats(self, layer: int, head_entropy: Tensor, mean_entropy: Tensor,
                      sampled: dict[str, Tensor] | None) -> None:
        """One layer's attention features from its statistics: ``head_entropy [t, H]``,
        ``mean_entropy [t]`` and, at a sampled layer, the per-head masses ``[t, H]`` and
        the mean rows ``[t, W]``."""
        if layer in self.seen:
            raise ValueError(f"attention layer {layer} captured twice in one replay")
        self.seen.add(layer)
        t, c = self.t, self.prefix
        heads = head_entropy.shape[1]
        ent = head_entropy.float().contiguous()
        # Twin of AttentionReducer.consume: entropy summary and head agreement.
        sample = ent[:: max(1, t // 60)] if t > 60 else ent
        self.out.put(f"attn_entropy_mean_L{layer}", sample.mean())
        self.out.put(f"attn_entropy_std_L{layer}", std(sample))
        stride = max(1, t // 30) if t > 30 else 1
        h_mean = mean_entropy.float().contiguous()[::stride].double()
        h_heads = ent[::stride].mean(dim=1).double()
        agreement = (
            1 - (h_mean - h_heads).clamp_min(0) / max(math.log(heads), 1e-12)
        ).float()
        self.out.put(f"head_agreement_mean_L{layer}", agreement.mean())
        self.out.put(f"head_agreement_std_L{layer}", std(agreement))
        if layer not in self.sampled:
            return
        if sampled is None:
            raise ExternalCaptureError(f"sampled layer {layer} has no sampled statistics")
        device = ent.device
        pos = torch.arange(c + t + 1, device=device)
        lengths = torch.arange(c + 1, c + t + 1, device=device)
        valid = pos[None, :] < lengths[:, None]
        rows = sampled["mean_rows"].float()[:, : c + t]
        mean32 = torch.cat([rows, rows.new_zeros(t, 1)], dim=1) * valid
        # Twin of AttentionReducer.consume from here to the per-head summaries: the same
        # arithmetic over the head-mean rows, which it computes and this is handed.
        mean = mean32.double()
        total = mean.sum(dim=-1).clamp_min(1e-12)
        cutoff = (lengths.double() * 0.8).long().clamp_min(1)
        rec = (mean * (pos[None, :] >= cutoff[:, None])).sum(dim=-1) / total
        sink = mean[:, 0]
        coverage = ((mean > 1 / lengths.double()[:, None]) & valid).sum(
            dim=-1
        ) / lengths.double()
        prompt = mean[:, :c].sum(dim=-1)
        generated = mean[:, c:].sum(dim=-1)
        for name, series in (
            ("recency_bias", rec),
            ("sink_mass", sink),
            ("cache_coverage", coverage),
        ):
            self.out.put(f"cache_{name}_L{layer}", series.mean())
        self.out.put(
            f"cache_lookback_ratio_L{layer}",
            prompt.sum() / generated.sum().clamp_min(1e-12),
        )
        if t < 10:
            fit = mean.new_zeros(())
        else:
            sample_steps = list(range(t // 4, t, max(1, t // 10)))[:10]
            picked = mean[sample_steps, 1:]
            distances = (
                (lengths[sample_steps, None] - 1 - pos[None, 1:]).clamp_min(1).double()
            )
            mask = (picked > 1e-10) & valid[sample_steps, 1:]
            fit = -slope(
                picked.clamp_min(1e-300).log().flatten(),
                distances.flatten(),
                mask.flatten(),
            )
        self.out.put(f"cache_attn_decay_rate_L{layer}", fit)
        width = t // 4
        for i in range(4):
            series = rec[i * width : (i + 1) * width if i < 3 else t] if t >= 4 else rec
            self.out.put(f"cache_recency_traj{i}_L{layer}", series.mean())

        steps = list(range(0, t, self.stride))
        if len(steps) < 3:
            steps = list(range(t))
        matrix = mean[steps, : c + steps[-1] + 1]
        matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        graph = (matrix @ matrix.T).clamp_min(0)
        graph = (graph + graph.T) * 0.5
        graph.diagonal().zero_()
        degrees = graph.sum(dim=-1)
        lap = (
            torch.diag(degrees)
            - graph
            + torch.eye(len(steps), device=device, dtype=torch.float64) * 1e-10
        )
        ev = torch.linalg.eigvalsh(lap).clamp_min(0)
        energy = ev.sum()
        radius = ev[-1]
        fiedler = (
            torch.where(
                radius > 1e-12, ev[1] / radius.clamp_min(1e-30), ev.new_zeros(())
            )
            if len(ev) > 1
            else ev.new_zeros(())
        )
        mid = len(ev) // 2
        median = ev[mid] if len(ev) % 2 else (ev[mid - 1] + ev[mid]) * 0.5
        p = ev / energy.clamp_min(1e-30) + 1e-12
        p = p / p.sum()
        spec = (
            -(p * p.log()).sum() / math.log(len(ev))
            if len(ev) > 1
            else ev.new_zeros(())
        )
        self.out.put(f"spectral_fiedler_L{layer}", fiedler)
        self.out.put(f"spectral_hfer_L{layer}", self.spectral_hfer(ev, median, energy))
        self.out.put(
            f"spectral_spectral_entropy_L{layer}",
            torch.where(energy > 1e-12, spec, spec.new_zeros(())),
        )
        inv = degrees.clamp_min(1e-12).rsqrt()
        normalized = (
            torch.eye(len(steps), device=device, dtype=torch.float64)
            - inv[:, None] * graph * inv[None, :]
        )
        self.spectral[layer] = (steps, normalized)
        self.retained_bytes += normalized.numel() * normalized.element_size()

        prefix = f"attn_flow_L{layer}"
        if t < 2:
            from anamnesis.extraction.feature_families.attention_flow import (
                _attention_flow_names,
            )

            for name in _attention_flow_names(layer, self.n_windows, self.include_stft):
                self.out.put(name, 0.0)
        else:
            sys = prompt / total
            self.out.moments(prefix + "_prompt_mass", sys)
            self.out.put(prefix + "_prompt_decay_rate", decay(sys))
            third = ((lengths - c) // 3).clamp_min(1)
            regions = (
                sys,
                (
                    mean * ((pos[None, :] >= c) & (pos[None, :] < c + third[:, None]))
                ).sum(dim=1)
                / total,
                (
                    mean
                    * (
                        (pos[None, :] >= c + third[:, None])
                        & (pos[None, :] < c + 2 * third[:, None])
                    )
                ).sum(dim=1)
                / total,
                (mean * (pos[None, :] >= c + 2 * third[:, None])).sum(dim=1) / total,
            )
            for label, series in zip(
                ("prompt", "early_gen", "mid_gen", "recent"), regions, strict=True
            ):
                self.out.moments(prefix + "_region_" + label, series)
            # The reducer divides each head's masses by its row total; the contract's
            # rows already sum to one.
            head_prompt = sampled["head_prompt_mass"].double()
            head_recency = sampled["head_recency_mass"].double()
            self.out.put(
                prefix + "_head_diversity_prompt", std(head_prompt, dim=1).mean()
            )
            self.out.put(
                prefix + "_head_diversity_recency", std(head_recency, dim=1).mean()
            )
            self.out.operators(
                prefix + "_prompt_mass", sys, self.n_windows, self.include_stft
            )
            self.out.operators(
                prefix + "_recency_bias", rec, self.n_windows, self.include_stft
            )

        per_ent = ent.double() / lengths.clamp_min(2).double().log()[:, None]
        per_sink = sampled["head_sink"].double()
        self.per_head[layer] = (per_ent, per_sink)
        self.retained_bytes += sum(
            v.numel() * v.element_size() for v in (per_ent, per_sink)
        )


class ExternalStatsSpan(ExternalSpan):
    """One span through the fast lane's reducers, its attention given as statistics.

    ``start`` and ``end`` are :class:`ExternalSpan`'s. :meth:`attention_stats` takes the
    whole span's statistics once; :meth:`finish` is :meth:`ExternalSpan.finish`, with
    the row's own lane id.
    """

    def __init__(self, lane: GpuFeatureLane, *, start: int, end: int, num_layers: int):
        super().__init__(lane, start=start, end=end, num_layers=num_layers)
        self.attention_reducer = StatsAttentionReducer(
            self.collector,
            steps=self.steps,
            prefix_length=start,
            sampled_layers=lane.config.sampled_layers,
            spectral_stride=lane.config.spectral_subsample_step,
            n_windows=lane.families.temporal_n_windows,
            include_stft=lane.families.enable_stft,
        )

    def attention(self, layer: int, weights: Tensor) -> None:
        raise ExternalCaptureError("a statistics span takes attention_stats, not weights")

    def attention_stats(self, *, head_entropy: Tensor, mean_entropy: Tensor,
                        head_prompt_mass: Tensor, head_recency_mass: Tensor,
                        head_sink: Tensor, mean_rows: Tensor) -> None:
        """Reduce the span's attention statistics (the module's contract) for every layer.

        Raises
        ------
        ExternalCaptureError
            When a tensor is missing, not float32, mis-shaped or nonfinite, or the
            statistics were already given.
        """
        if self.finished:
            raise ExternalCaptureError("attention statistics after the span was finished")
        if self.attention_reducer.seen:
            raise ExternalCaptureError("a span's attention statistics are given once")
        given = dict(head_entropy=head_entropy, mean_entropy=mean_entropy,
                     head_prompt_mass=head_prompt_mass, head_recency_mass=head_recency_mass,
                     head_sink=head_sink, mean_rows=mean_rows)
        n, layers = self.steps, self.num_layers
        sampled = list(self.lane.config.sampled_layers)
        if not set(sampled) <= set(range(layers)):
            raise ExternalCaptureError("a sampled layer lies outside the capture")
        if not isinstance(head_entropy, Tensor) or head_entropy.ndim != 3:
            raise ExternalCaptureError("head_entropy must be [steps, layers, heads]")
        heads = head_entropy.shape[2]
        expected = dict(head_entropy=(n, layers, heads), mean_entropy=(n, layers),
                        head_prompt_mass=(n, len(sampled), heads),
                        head_recency_mass=(n, len(sampled), heads),
                        head_sink=(n, len(sampled), heads))
        for name, value in given.items():
            if not isinstance(value, Tensor):
                raise ExternalCaptureError(f"{name} is not a tensor")
            if value.dtype != torch.float32:
                raise ExternalCaptureError(f"{name} must be float32, got {value.dtype}")
            if name == "mean_rows":
                if (value.ndim != 3 or value.shape[:2] != (n, len(sampled))
                        or value.shape[2] < self.end - 1):
                    raise ExternalCaptureError(
                        f"mean_rows must be [{n}, {len(sampled)}, >={self.end - 1}], "
                        f"got {tuple(value.shape)}")
            elif tuple(value.shape) != expected[name]:
                raise ExternalCaptureError(
                    f"{name} must be {list(expected[name])}, got {list(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise ExternalCaptureError(f"{name} holds a nonfinite value")
        device = self.lane.device
        given = {name: value.to(device) for name, value in given.items()}
        for layer in range(layers):
            at = None
            if layer in sampled:
                s = sampled.index(layer)
                at = {name: given[name][:, s] for name in
                      ("head_prompt_mass", "head_recency_mass", "head_sink", "mean_rows")}
            self.attention_reducer.consume_stats(
                layer, given["head_entropy"][:, layer], given["mean_entropy"][:, layer], at)

    def finish(self, *, hidden: Tensor, keys: Mapping[int, Tensor],
               values: Mapping[int, Tensor], queries: Mapping[int, Tensor],
               gates: Mapping[int, Tensor], logits: Tensor, token_ids: Sequence[int],
               capture_identity: Mapping[str, Any],
               routing: Sequence[RoutingPoint] | None = None,
               committed: Sequence[Tensor] | None = None) -> GpuFeatureResult:
        """:meth:`ExternalSpan.finish`, receipted as a statistics row: its lane id is
        :func:`stats_lane_id`, and the receipt names the capture kind and keeps the
        external lane id the weights would have had beside it."""
        result = super().finish(
            hidden=hidden, keys=keys, values=values, queries=queries, gates=gates,
            logits=logits, token_ids=token_ids, capture_identity=capture_identity,
            routing=routing, committed=committed)
        receipt = result.metadata
        receipt.update(
            external_lane_id=receipt["lane_id"],
            lane_id=stats_lane_id(self.lane, capture_identity),
            capture_kind=CAPTURE_KIND,
            stats_source_sha256=STATS_SOURCE_SHA256,
            replay_path="external-stats",
        )
        return result
