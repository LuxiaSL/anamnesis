"""Residual, key/value/query, gate and output operators for the GPU lane."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import Tensor

from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    cos,
    corr,
    rowcos,
    std,
    trajectory_indices,
)


def spread(x: Tensor) -> Tensor:
    return (1 - rowcos(x, x.mean(dim=0, keepdim=True))).mean()


def eff_dim(x: Tensor) -> Tensor:
    s2 = torch.linalg.svdvals(x).square()
    total = s2.sum()
    return torch.where(
        total > 1e-12,
        total.square() / s2.square().sum().clamp_min(1e-30) / min(x.shape),
        total.new_zeros(()),
    )


def novelty(x: Tensor) -> Tensor:
    running = (
        x.cumsum(dim=0)
        / torch.arange(1, len(x) + 1, device=x.device, dtype=torch.float64)[:, None]
    )
    return 1 - rowcos(x[1:], running[:-1])


def drift(x: Tensor) -> Tensor:
    return 1 - rowcos(x[:-1], x[1:])


def half_drift(x: Tensor) -> Tensor:
    mid = len(x) // 2
    return 1 - cos(x[:mid].mean(dim=0).float(), x[mid:].mean(dim=0).float())


class FamilyReducer:
    def __init__(self, out: FeatureCollector, extraction, families):
        self.out = out
        self.config = extraction
        self.families = families
        self.keys: dict[int, Tensor] = {}
        self.values: dict[int, Tensor] = {}
        self.gate_means: dict[int, Tensor] = {}
        self.gate_sparsity: dict[int, Tensor] = {}
        self.value_spreads: list[Tensor] = []
        self.query_spreads: list[Tensor] = []
        self.epochs: list[tuple[Tensor, Tensor, Tensor]] = []
        self.gate_tie_exception_d2h_bytes = 0
        self.gate_family_present = False

    def operators(self, prefix: str, x: Tensor) -> None:
        self.out.operators(
            prefix, x, self.families.temporal_n_windows, self.families.enable_stft
        )

    def key(self, layer: int, keys: Tensor) -> None:
        matrix32 = keys.float().mean(dim=1)
        x = matrix32.double()
        self.keys[layer] = x
        prefix = "kv_"
        if len(x) < 2:
            for name in (
                "key_spread",
                "key_eff_dim",
                "key_drift",
                "key_novelty_mean",
                "key_novelty_std",
                *[f"key_novelty_traj{i}" for i in range(5)],
            ):
                self.out.put(f"{prefix}{name}_L{layer}", 0.0)
            return
        # Baseline key centroid is computed in float32 before promotion;
        # value/query geometry use a float64 centroid instead.
        center = matrix32.mean(dim=0, keepdim=True).double()
        self.out.put(f"kv_key_spread_L{layer}", (1 - rowcos(x, center)).mean())
        self.out.put(f"kv_key_eff_dim_L{layer}", eff_dim(x))
        self.out.put(f"kv_key_drift_L{layer}", half_drift(x))
        nov = novelty(x).float()
        self.out.put(f"kv_key_novelty_mean_L{layer}", nov.mean())
        self.out.put(f"kv_key_novelty_std_L{layer}", std(nov))
        for i, index in enumerate(trajectory_indices(len(nov))):
            self.out.put(f"kv_key_novelty_traj{i}_L{layer}", nov[index])
        window, stride = self.config.epoch_window_size, self.config.epoch_stride
        if len(x) >= window + stride:
            centers = torch.stack(
                [
                    matrix32[start : start + window].mean(dim=0)
                    for start in range(0, len(x) - window + 1, stride)
                ]
            )
            boundaries = (1 - cos(centers[:-1], centers[1:])).float()
            threshold = boundaries.mean().double() + 1.5 * std(
                boundaries
            ).double().clamp_min(1e-6)
            # NumPy compares f32 array with Python scalar under its scalar rules;
            # cast threshold to f32, as the reference's current NumPy does.
            count = (boundaries > threshold.float()).double().mean()
            self.epochs.append(
                (count, boundaries.max().double(), std(boundaries).double())
            )

    def value(self, layer: int, values: Tensor) -> None:
        from anamnesis.extraction.feature_families.value_geometry import (
            _value_layer_names,
        )

        x = values.float().mean(dim=1).double()
        self.values[layer] = x
        if len(x) < 2:
            for name in _value_layer_names(
                layer, self.families.temporal_n_windows, self.families.enable_stft
            ):
                self.out.put(name, 0.0)
            return
        nov = novelty(x)
        dr = drift(x)
        sp = spread(x)
        self.value_spreads.append(sp)
        norms = values.double().norm(dim=-1)
        mean = norms.mean(dim=1)
        cv = torch.where(
            mean > 1e-12, std(norms, dim=1) / mean.clamp_min(1e-30), mean.new_zeros(())
        )
        vals = {
            "spread": sp,
            "eff_dim": eff_dim(x),
            "drift_halfcentroid": half_drift(x),
            "novelty_mean": nov.mean(),
            "novelty_std": std(nov),
            "crosshead_normcv_mean": cv.mean(),
            "crosshead_normcv_std": std(cv),
            "key_drift_corr": corr(dr, drift(self.keys[layer])),
        }
        for name, value in vals.items():
            self.out.put(f"value_L{layer}_{name}", value)
        self.operators(f"value_L{layer}_novelty", nov)
        self.operators(f"value_L{layer}_drift", dr)

    def query(self, layer: int, queries: Tensor) -> None:
        from anamnesis.extraction.feature_families.qk_geometry import _qk_layer_names

        q = queries.float().mean(dim=1).double()
        if len(q) < 2:
            for name in _qk_layer_names(
                layer, self.families.temporal_n_windows, self.families.enable_stft
            ):
                self.out.put(name, 0.0)
            return
        k = self.keys[layer]
        running = (
            k.cumsum(dim=0)
            / torch.arange(1, len(k) + 1, device=k.device, dtype=torch.float64)[:, None]
        )
        self_align = rowcos(q, k)
        cache_align = rowcos(q, running)
        nov = novelty(q)
        sp = spread(q)
        self.query_spreads.append(sp)
        vals = {
            "q_spread": sp,
            "q_eff_dim": eff_dim(q),
            "q_drift_halfcentroid": half_drift(q),
            "q_novelty_mean": nov.mean(),
            "q_novelty_std": std(nov),
            "qk_self_align_mean": self_align.mean(),
            "qk_self_align_std": std(self_align),
            "qk_cache_align_mean": cache_align.mean(),
            "qk_cache_align_std": std(cache_align),
        }
        for name, value in vals.items():
            self.out.put(f"qk_L{layer}_{name}", value)
        self.operators(f"qk_L{layer}_q_novelty", nov)
        self.operators(f"qk_L{layer}_qk_self_align", self_align)

    def gate(self, layer: int, pre_silu: Tensor) -> None:
        if len(pre_silu) < 2:
            return
        self.gate_family_present = True
        x = pre_silu.double()
        g = x / (1 + torch.exp(-x.clamp(-88, 88)))
        sp = (g.abs() > self.families.gate_sparsity_threshold).double().mean(dim=1)
        s2 = g.square().sum(dim=1)
        dim = torch.where(
            s2 > 1e-12,
            g.abs().sum(dim=1).square() / s2.clamp_min(1e-30),
            s2.new_zeros(()),
        )
        dr = 1 - cos(g[:-1].float(), g[1:].float())
        prefix = f"gate_L{layer}"
        self.out.moments(prefix + "_sparsity", sp)
        self.out.moments(prefix + "_eff_dim", dim)
        self.out.moments(prefix + "_drift", dr)
        self.gate_means[layer] = g.mean(dim=0).float()
        self.gate_sparsity[layer] = sp.mean()
        # Named CPU exception: NumPy's unstable argsort tie ordering is part of
        # the historical top-k Jaccard. bf16 gate ties are common. CUDA topk or
        # a newly chosen stable sort would change the feature definition.
        # Transfer only this sampled gate surface, reproduce its exact SiLU and
        # argsort, and price the bytes/time; all other gate reductions stay here.
        host = pre_silu.float().cpu().numpy()
        self.gate_tie_exception_d2h_bytes += host.nbytes
        activated = host.astype(np.float64)
        activated = activated * (1 / (1 + np.exp(-np.clip(activated, -88, 88))))
        k = min(100, activated.shape[1] // 10)
        top = np.argsort(np.abs(activated), axis=1)[:, -k:]
        overlaps = []
        for a, b in zip(top[:-1], top[1:], strict=True):
            sa, sb = set(a), set(b)
            overlaps.append(len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
        self.out.put(prefix + "_topk_overlap_mean", float(np.mean(overlaps)))
        self.operators(prefix + "_sparsity", sp)
        self.operators(prefix + "_drift", dr)

    def residual_trajectory(
        self, layer: int, corrected: Tensor, raw_first_norm: Tensor
    ) -> None:
        from anamnesis.extraction.feature_families.residual_stream import (
            _trajectory_feature_names,
        )

        before = set(self.out.values)
        prefix = f"res_traj_L{layer}"
        if len(corrected) < 3:
            for name in _trajectory_feature_names(
                layer, self.families.temporal_n_windows, self.families.enable_stft
            ):
                self.out.put(name, 0.0)
            return
        x = corrected.double()
        vel = x[1:] - x[:-1]
        vn = vel.norm(dim=-1)
        angles = cos(vel[:-1].float(), vel[1:].float()).clamp(-1, 1).acos()
        self.out.moments(prefix + "_velocity_norm", vn)
        self.out.moments(prefix + "_direction_change", angles)
        self.out.put(
            prefix + "_directness", (x[-1] - x[0]).norm() / vn.sum().clamp_min(1e-12)
        )
        self.out.moments(
            prefix + "_acceleration_norm", (vel[1:] - vel[:-1]).norm(dim=-1)
        )
        self.operators(prefix + "_velocity_norm", vn)
        self.operators(prefix + "_direction_change", angles)
        for name in set(self.out.values) - before:
            value = self.out.values[name]
            self.out.values[name] = torch.where(
                raw_first_norm < 1e-6, value.new_zeros(()), value
            )

    def finish(self) -> None:
        for index, name in enumerate(
            ("epoch_n_transitions", "epoch_max_transition", "epoch_regularity")
        ):
            series = (
                torch.stack([e[index] for e in self.epochs])
                if self.epochs
                else torch.zeros(1, device=self.out.device, dtype=torch.float64)
            )
            self.out.put(name + "_mean", series.mean())
            self.out.put(name + "_max", series.max())
            self.out.put(name + "_std", std(series))
        if len(self.gate_means) >= 2:
            self.out.put(
                "gate_cross_layer_sparsity_diversity",
                std(torch.stack(list(self.gate_sparsity.values()))),
            )
            sims = [
                cos(self.gate_means[a], self.gate_means[b])
                for a, b in combinations(sorted(self.gate_means), 2)
            ]
            self.out.put("gate_cross_layer_agreement", torch.stack(sims).mean())
        elif self.gate_family_present:
            self.out.put("gate_cross_layer_sparsity_diversity", 0.0)
            self.out.put("gate_cross_layer_agreement", 0.0)
        for name, values in (
            ("value_crosslayer_spread_diversity", self.value_spreads),
            ("q_crosslayer_spread_diversity", self.query_spreads),
        ):
            self.out.put(name, std(torch.stack(values)) if len(values) >= 2 else 0.0)
        for label, matrices in (("key", self.keys), ("value", self.values)):
            available = [
                layer
                for layer in self.config.sampled_layers
                if len(matrices[layer]) >= 2
            ]
            if not available:
                for suffix in ("early_late", "adjacent_mean", "overall_mean"):
                    self.out.put(f"kv_{label}_cka_{suffix}", 0.0)
                continue
            centered = {
                layer: matrices[layer] - matrices[layer].mean(dim=0, keepdim=True)
                for layer in available
            }
            self_norm = {layer: (x.T @ x).norm() for layer, x in centered.items()}
            pairs = {}
            for a, b in combinations(available, 2):
                num = (centered[b].T @ centered[a]).norm().square()
                value = num / (self_norm[a] * self_norm[b]).clamp_min(1e-12)
                pairs[(a, b)] = value
                self.out.put(f"kv_{label}_cka_L{a}_L{b}", value)
            third = max(1, len(available) // 3)
            early, late = set(available[:third]), set(available[-third:])
            groups = {
                "early_late": [
                    v for (a, b), v in pairs.items() if a in early and b in late
                ],
                "adjacent_mean": [
                    pairs[(a, b)]
                    for a, b in zip(available[:-1], available[1:], strict=True)
                ],
                "overall_mean": list(pairs.values()),
            }
            for name, values in groups.items():
                self.out.put(
                    f"kv_{label}_cka_{name}",
                    torch.stack(values).mean() if values else 0.0,
                )
