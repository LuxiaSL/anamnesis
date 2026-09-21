"""Per-layer eager-attention reduction; never retains all-layer attention raws.

Only the sampled-step spectral Gram and small per-head statistics survive the
hook. Smoothness waits for the corresponding position-corrected residuals.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    corr,
    decay,
    entropy,
    slope,
    std,
)


class AttentionReducer:
    def __init__(
        self,
        collector: FeatureCollector,
        *,
        steps: int,
        prefix_length: int,
        sampled_layers: list[int],
        spectral_stride: int = 10,
        n_windows: int = 4,
        include_stft: bool = True,
    ):
        self.out = collector
        self.t = steps
        self.prefix = prefix_length
        self.sampled = set(sampled_layers)
        self.stride = spectral_stride
        self.n_windows = n_windows
        self.include_stft = include_stft
        self.spectral: dict[int, tuple[list[int], Tensor]] = {}
        self.per_head: dict[int, tuple[Tensor, Tensor]] = {}
        self.seen: set[int] = set()
        self.retained_bytes = 0

    def consume(self, layer: int, weights: Tensor) -> None:
        if layer in self.seen:
            raise ValueError(f"attention layer {layer} captured twice in one replay")
        self.seen.add(layer)
        t, c = self.t, self.prefix
        if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2] != t + 1:
            raise ValueError(f"unexpected eager attention shape {tuple(weights.shape)}")
        if weights.shape[-1] != c + t + 1:
            raise ValueError("attention cache width mismatch")
        a = weights[0, :, :t, :].permute(1, 0, 2).float()
        pos = torch.arange(a.shape[-1], device=a.device)
        lengths = torch.arange(c + 1, c + t + 1, device=a.device)
        valid = pos[None, :] < lengths[:, None]
        a = a * valid[:, None, :]
        ent = entropy(a)
        sample = ent[:: max(1, t // 60)] if t > 60 else ent
        self.out.put(f"attn_entropy_mean_L{layer}", sample.mean())
        self.out.put(f"attn_entropy_std_L{layer}", std(sample))
        selected = a[:: max(1, t // 30)] if t > 30 else a
        norm = selected.double()
        norm = norm / norm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        h_mean = entropy(norm.mean(dim=1)).double()
        h_heads = entropy(norm).mean(dim=1).double()
        agreement = (
            1 - (h_mean - h_heads).clamp_min(0) / max(math.log(a.shape[1]), 1e-12)
        ).float()
        self.out.put(f"head_agreement_mean_L{layer}", agreement.mean())
        self.out.put(f"head_agreement_std_L{layer}", std(agreement))
        if layer not in self.sampled:
            return
        mean = a.mean(dim=1).double()  # IMPORTANT: mean is float32 before promotion
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
            sampled = mean[sample_steps, 1:]
            distances = (
                (lengths[sample_steps, None] - 1 - pos[None, 1:]).clamp_min(1).double()
            )
            mask = (sampled > 1e-10) & valid[sample_steps, 1:]
            fit = -slope(
                sampled.clamp_min(1e-300).log().flatten(),
                distances.flatten(),
                mask.flatten(),
            )
        self.out.put(f"cache_attn_decay_rate_L{layer}", fit)
        width = t // 4
        for i in range(4):
            series = rec[i * width : (i + 1) * width if i < 3 else t] if t >= 4 else rec
            self.out.put(f"cache_recency_traj{i}_L{layer}", series.mean())

        # Spectral exception: sampled head means -> small Gram, then discard rows.
        steps = list(range(0, t, self.stride))
        if len(steps) < 3:
            steps = list(range(t))
        matrix = mean[steps, : c + steps[-1] + 1]
        matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        graph = (matrix @ matrix.T).clamp_min(0)
        graph = (graph + graph.T) * 0.5
        graph.fill_diagonal_(0)
        degrees = graph.sum(dim=-1)
        lap = (
            torch.diag(degrees)
            - graph
            + torch.eye(len(steps), device=a.device, dtype=torch.float64) * 1e-10
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
        # torch.median selects the lower middle element; NumPy averages the two.
        mid = len(ev) // 2
        median = ev[mid] if len(ev) % 2 else (ev[mid - 1] + ev[mid]) * 0.5
        hfer = ev[ev > median].sum() / energy.clamp_min(1e-30)
        p = ev / energy.clamp_min(1e-30) + 1e-12
        p = p / p.sum()
        spec = (
            -(p * p.log()).sum() / math.log(len(ev))
            if len(ev) > 1
            else ev.new_zeros(())
        )
        self.out.put(f"spectral_fiedler_L{layer}", fiedler)
        self.out.put(
            f"spectral_hfer_L{layer}",
            torch.where(energy > 1e-12, hfer, hfer.new_zeros(())),
        )
        self.out.put(
            f"spectral_spectral_entropy_L{layer}",
            torch.where(energy > 1e-12, spec, spec.new_zeros(())),
        )
        inv = degrees.clamp_min(1e-12).rsqrt()
        normalized = (
            torch.eye(len(steps), device=a.device, dtype=torch.float64)
            - inv[:, None] * graph * inv[None, :]
        )
        self.spectral[layer] = (steps, normalized)
        self.retained_bytes += normalized.numel() * normalized.element_size()

        # Attention flow, including the reference's short-series name contract.
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
            a64 = a.double()
            head_total = a64.sum(dim=-1).clamp_min(1e-12)
            head_prompt = a64[:, :, :c].sum(dim=-1) / head_total
            head_recency = (a64 * (pos[None, None, :] >= cutoff[:, None, None])).sum(
                dim=-1
            ) / head_total
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

        a64 = a.double()
        norm = a64 / a64.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        safe = norm.clamp_min(1e-30)
        per_ent = (
            -(safe * safe.log() * valid[:, None, :]).sum(dim=-1)
            / lengths.clamp_min(2).double().log()[:, None]
        )
        per_sink = norm[:, :, 0]
        # Only these per-head summaries must wait for sampled keys.
        self.per_head[layer] = (per_ent, per_sink)
        self.retained_bytes += sum(
            v.numel() * v.element_size() for v in (per_ent, per_sink)
        )

    def smoothness(self, layer: int, corrected_hidden: Tensor) -> None:
        steps, lap = self.spectral.pop(layer)
        signal = corrected_hidden[steps].double().norm(dim=-1)
        self.out.put(
            f"spectral_smoothness_L{layer}",
            (signal @ lap @ signal) / (signal @ signal).clamp_min(1e-12),
        )

    def head_features(self, layer: int, keys: Tensor) -> None:
        prefix = f"ph_L{layer}"
        if self.t < 4:
            from anamnesis.extraction.feature_families.per_head import _per_head_names

            for name in _per_head_names(layer):
                self.out.put(name, 0.0)
            self.per_head.pop(layer)
            return
        ent, sink = self.per_head.pop(layer)
        means = ent.mean(dim=0)
        for name, value in (
            ("mean", means.mean()),
            ("std", std(means)),
            ("min", means.min()),
            ("max", means.max()),
        ):
            self.out.put(prefix + "_head_entropy_" + name, value)
        half = len(ent) // 2
        self.out.put(
            prefix + "_head_role_stability",
            corr(ent[:half].mean(dim=0), ent[half:].mean(dim=0), min_n=2),
        )
        self.out.put(prefix + "_sink_head_std", std(sink.mean(dim=0)))
        k = keys.double()
        center = k.mean(dim=0, keepdim=True)
        sims = (k * center).sum(dim=-1) / (
            k.norm(dim=-1) * center.norm(dim=-1)
        ).clamp_min(1e-12)
        spread = (1 - sims).mean(dim=0)
        self.out.put(prefix + "_kv_key_spread_head_mean", spread.mean())
        self.out.put(prefix + "_kv_key_spread_head_std", std(spread))
