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

**Output and routing reductions.** :meth:`ExternalStatsSpan.finish` also takes, in place
of the full logits, ``logit_stats [n, 6]``: per step the output entropy, top-1
probability, top-5 mass, the chosen token's rank and probability, and its log
probability, as the fast lane's output reduction computes them
(:func:`logit_stats_from_logits` is their reference); and in place of every routing
point's weights, :class:`RoutingPointStats`: the weights in full at the lane's sampled
layers and the final read, and at every other point only the three per-position
quantities the depth profile reads (:func:`routing_stats_from_routing`). Committed
blocks are taken as they are.

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

The output arithmetic after the per-step quantities restates
``GpuFeatureLane._outputs``, and the routing family's assembly restates
:func:`~anamnesis.extraction.fast.attn_res.attn_res_features` around that module's own
per-point and summary functions; tests pin the digests of both twins' sources.

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
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.attn_res import _point_quantities, _slope, _summ
from anamnesis.extraction.fast.external import (
    READOUT_SOURCE_SHA256,
    ExternalCaptureError,
    ExternalSpan,
    RoutingPoint,
    external_lane_id,
)
from anamnesis.extraction.fast.features import GpuFeatureLane, GpuFeatureResult
from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    decay,
    entropy,
    rowcos,
    slope,
    std,
    trajectory_indices,
)
from anamnesis.extraction.feature_families.attn_res import EPS, QNAMES, SUMM, committed_pairs

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


LOGIT_STATS = ("entropy", "top1_prob", "top5_mass", "chosen_rank", "chosen_prob", "logprob")
"""The columns of ``logit_stats``, in order."""


def logit_stats_from_logits(logits: Tensor, chosen: Tensor) -> Tensor:
    """``logit_stats [steps, 6]`` (float64) from ``logits [steps, vocabulary]`` and the
    chosen next tokens, as the fast lane's output reduction computes each step: in
    float64 after subtracting the row maximum, sixteen rows at a time, the first five
    columns rounded to float32 where the reduction rounds them."""
    rows = []
    for first in range(0, len(logits), 16):
        x = logits[first : first + 16].float().double()
        x = x - x.max(dim=1, keepdim=True).values
        exp = x.exp()
        z = exp.sum(dim=1)
        logz = z.log()
        cid = chosen[first : first + 16, None]
        xc = x.gather(1, cid).squeeze(1)
        rows.append(torch.stack([
            (logz - (x * exp).sum(dim=1) / z).float().double(),
            (exp.max(dim=1).values / z).float().double(),
            (exp.topk(5, dim=1).values.sum(dim=1) / z).float().double(),
            (x > xc[:, None]).sum(dim=1).float().double(),
            (exp.gather(1, cid).squeeze(1) / z).float().double(),
            xc - logz,
        ], dim=1))
    return torch.cat(rows)


def _outputs_from_stats(out: FeatureCollector, lane: GpuFeatureLane, stats: Tensor) -> Tensor:
    """The output family from ``logit_stats``. Twin of ``GpuFeatureLane._outputs`` after
    its per-step quantities; returns the mean log probability."""
    ent, top, topfive, rank, chosen_prob = (
        stats[:, i].float().contiguous() for i in range(5))
    logprob = stats[:, 5].double().contiguous()
    n = len(stats)
    surprise = -chosen_prob.clamp_min(1e-10).log()
    for name, values in (("logit_entropy", ent), ("top1_prob", top)):
        out.moments(name, values)
        for i, t in enumerate(trajectory_indices(n, lane.config.trajectory_points)):
            out.put(f"{name}_traj{i}", values[t])
    out.put("top5_mass_mean", topfive.mean())
    for name, values in (("chosen_rank", rank), ("surprise", surprise)):
        out.put("mean_" + name, values.mean())
        out.put("std_" + name, std(values))
    for i, t in enumerate(trajectory_indices(n, lane.config.trajectory_points)):
        out.put(f"surprise_traj{i}", surprise[t])
    window = lane.config.surprise_window
    if n >= window:
        values = surprise.unfold(0, window, 1)
        running_mean = values.double().mean(dim=1)
        running_std = std(values, dim=1).float()
        threshold = (
            running_mean
            + lane.config.surprise_threshold_sigma * running_std.double().clamp_min(1e-6)
        )
        out.put("surprise_boundary_count", (surprise[window - 1 :].double() > threshold).sum())
    else:
        out.put("surprise_boundary_count", 0.0)
    return logprob.mean()


@dataclass(frozen=True)
class RoutingPointStats:
    """One routing point, in the routing order: its weights ``[steps, sources]`` in full,
    or only ``quantities [steps, 3]`` (entropy over ``ln max(sources, 2)``, anchor weight,
    partial weight, after renormalising), which is all the depth profile reads."""

    tag: str
    layer: int
    weights: Tensor | None = None
    quantities: Tensor | None = None


def routing_stats_from_routing(routing: Sequence[RoutingPoint],
                               sampled_layers: Sequence[int]) -> list[RoutingPointStats]:
    """The routing reductions: full weights at the sampled layers and the final read,
    the three depth quantities everywhere else, each as the routing family computes it."""
    sampled = {int(layer) for layer in sampled_layers}
    points = []
    for tag, layer, w in routing:
        if tag == "final" or int(layer) in sampled:
            points.append(RoutingPointStats(tag, int(layer), weights=w))
        else:
            q = _point_quantities(w)
            points.append(RoutingPointStats(tag, int(layer), quantities=torch.stack(
                [q["entropy"], q["anchor_w"], q["partial_w"]], dim=1)))
    return points


def _routing_from_stats(points: Sequence[RoutingPointStats], committed: Sequence[Tensor],
                        sampled_layers: Sequence[int], device) -> tuple[list[str], Tensor]:
    """The routing family from routing reductions. Twin of ``attn_res_features``, with
    its per-point quantities and summaries computed by that module's own functions."""
    values: list[Tensor] = []
    names: list[str] = []
    by_layer: dict[int, list[Tensor]] = {}
    stats: list[tuple[int, Tensor, Tensor, Tensor]] = []
    final_vec: Tensor | None = None
    for point in points:
        if point.weights is not None:
            q = _point_quantities(point.weights.to(device))
            vec = torch.cat([_summ(q[qn]) for qn in QNAMES])
            if point.tag == "final":
                final_vec = vec
                continue
            by_layer.setdefault(int(point.layer), []).append(vec)
            stats.append((int(point.layer), q["entropy"].mean(), q["anchor_w"].mean(),
                          q["partial_w"].mean()))
        else:
            q = point.quantities.to(device).double()
            stats.append((int(point.layer), *(q[:, i].contiguous().mean() for i in range(3))))

    zeros18 = torch.zeros(len(QNAMES) * 3, dtype=torch.float64, device=device)
    for layer in sampled_layers:
        vecs = by_layer.get(int(layer))
        v = torch.stack(vecs).mean(0) if vecs else zeros18
        values.append(v)
        names += [f"attnres_L{layer}_{qn}_{sm}" for qn in QNAMES for sm in SUMM]

    zero = torch.zeros((), dtype=torch.float64, device=device)
    if len(stats) >= 2:
        lay = torch.tensor([s[0] for s in stats], dtype=torch.float64, device=device)
        cols = [torch.stack([s[i] for s in stats]) for i in (1, 2, 3)]
        flat = bool(lay.std(correction=0) <= EPS)
        values.append(torch.stack([zero if flat else _slope(lay, c) for c in cols] + [cols[0].std(correction=0)]))
    else:
        values.append(torch.stack([zero, zero, zero, zero]))
    names += [f"attnres_depthslope_{cn}" for cn in ("entropy", "anchor_w", "partial_w")]
    names.append("attnres_xpoint_entropy_std")

    values.append(final_vec[0::3] if final_vec is not None
                  else torch.zeros(len(QNAMES), dtype=torch.float64, device=device))
    names += [f"attnres_final_{qn}_mean" for qn in QNAMES]

    n_pairs = committed_pairs(len(committed))
    cos_means = [zero] * n_pairs
    overall = [zero, zero]
    if len(committed) >= 2:
        cos_all = []
        for b in range(len(committed) - 1):
            a, c = committed[b].double(), committed[b + 1].double()
            if a.ndim == 2 and a.shape == c.shape and a.shape[0] > 0:
                cos = (a * c).sum(1) / (a.norm(dim=1) * c.norm(dim=1) + EPS)
                cos_means[b] = cos.mean()
                cos_all.append(cos)
        if cos_all:
            allc = torch.cat(cos_all)
            overall = [allc.mean(), allc.std(correction=0)]
    values.append(torch.stack(cos_means + overall))
    names += [f"attnres_committed_cos_b{b}b{b + 1}_mean" for b in range(n_pairs)]
    names += ["attnres_committed_cos_overall_mean", "attnres_committed_cos_overall_std"]
    return names, torch.nan_to_num(torch.cat(values).float())


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
               gates: Mapping[int, Tensor], logits: Tensor | None = None,
               token_ids: Sequence[int], capture_identity: Mapping[str, Any],
               routing: Sequence[RoutingPoint] | None = None,
               committed: Sequence[Tensor] | None = None,
               logit_stats: Tensor | None = None,
               routing_stats: Sequence[RoutingPointStats] | None = None) -> GpuFeatureResult:
        """:meth:`ExternalSpan.finish`, with the logits or their ``logit_stats``, the
        routing points or their ``routing_stats``, receipted as a statistics row.

        The block-output, key, value, query and gate reduction restates
        :meth:`ExternalSpan.finish`'s. The receipt's lane id is :func:`stats_lane_id`;
        it names the capture kind, which reductions arrived, and keeps beside it the
        external lane id the weights would have had.

        Raises
        ------
        ExternalCaptureError
            As :meth:`ExternalSpan.finish` does, and when both or neither of the logits
            and ``logit_stats`` are given, both routing forms are given, or a reduction
            is mis-shaped or nonfinite.
        """
        if self.finished:
            raise ExternalCaptureError("a span is finished once")
        self.finished = True
        lane, steps, start, end = self.lane, self.steps, self.start, self.end
        if (logits is None) == (logit_stats is None):
            raise ExternalCaptureError("give the logits or logit_stats, one of the two")
        if routing is not None and routing_stats is not None:
            raise ExternalCaptureError("give the routing points or routing_stats, not both")
        if logit_stats is not None:
            if (not isinstance(logit_stats, Tensor) or tuple(logit_stats.shape) != (steps, 6)
                    or logit_stats.dtype not in (torch.float32, torch.float64)
                    or not bool(torch.isfinite(logit_stats).all())):
                raise ExternalCaptureError(
                    f"logit_stats must be finite float32 or float64 [{steps}, 6]")
        routed = routing if routing is not None else routing_stats
        if routing_stats is not None:
            for point in routing_stats:
                if not isinstance(point, RoutingPointStats) or (
                        (point.weights is None) == (point.quantities is None)):
                    raise ExternalCaptureError("routing_stats holds a point with neither or both forms")
                x = point.weights if point.weights is not None else point.quantities
                if x.ndim != 2 or x.shape[0] != steps or (
                        point.quantities is not None and x.shape[1] != 3) or not bool(
                        torch.isfinite(x).all()):
                    raise ExternalCaptureError(
                        f"routing_stats point {point.tag} L{point.layer} has shape {tuple(x.shape)}")
                if point.quantities is not None and (
                        point.tag == "final" or point.layer in lane.config.sampled_layers):
                    raise ExternalCaptureError(
                        f"routing point {point.tag} L{point.layer} needs its full weights")
        # ExternalSpan's validation of everything else; a reduction stands in for the
        # tensor it replaces, so only what this span actually holds is checked there.
        self._validate(hidden, keys, values, queries, gates,
                       logits if logits is not None else torch.zeros(steps, 5), token_ids,
                       routed, committed)
        if self.attention_reducer.seen != set(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - self.attention_reducer.seen)
            raise ExternalCaptureError(f"attention layers {missing} were not reduced")
        out, attention, reducer = self.collector, self.attention_reducer, self.reducer
        # Twin of ExternalSpan.finish from here to the routing family.
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
        if logit_stats is not None:
            logprob = _outputs_from_stats(out, lane, logit_stats.to(lane.device))
        else:
            chosen = torch.as_tensor(list(token_ids[start + 1:end]), device=lane.device,
                                     dtype=torch.long)
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
        if routed is not None:
            points = (routing_stats if routing_stats is not None else
                      [RoutingPointStats(tag, int(layer), weights=w) for tag, layer, w in routing])
            routed_names, routed_values = _routing_from_stats(
                points, [c.to(lane.device) for c in committed or []],
                lane.config.sampled_layers, lane.device)
            vector = torch.cat([vector, routed_values.to(vector.dtype)])
            names += routed_names
        features = vector.cpu().numpy()
        if not np.isfinite(features).all():
            raise ExternalCaptureError("the reduction produced a nonfinite feature")
        knnlm = (hidden[-1][steps - 1].float().cpu().numpy().copy()
                 if lane.config.enable_knnlm_baseline else None)
        receipt = dict(
            lane_id=stats_lane_id(lane, capture_identity),
            external_lane_id=external_lane_id(lane, capture_identity),
            fast_lane_id=lane.lane_id,
            lane_source_sha256=dict(lane.source_sha256),
            readout_source_sha256=dict(READOUT_SOURCE_SHA256),
            stats_source_sha256=STATS_SOURCE_SHA256,
            capture=dict(capture_identity),
            capture_kind=CAPTURE_KIND,
            reductions=dict(attention="stats",
                            logits="stats" if logit_stats is not None else "full",
                            routing=("stats" if routing_stats is not None else
                                     "full" if routing is not None else "none")),
            replay_id=str(uuid.uuid4()),
            replay_path="external-stats",
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
            routing_family=routed is not None,
        )
        return GpuFeatureResult(features, tuple(names), float(logprob.cpu()), knnlm, receipt)
