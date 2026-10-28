"""Residual, key/value/query, gate and output operators for the GPU lane."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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


def norms_eff_dims(matrices: Tensor) -> Tensor:
    """:func:`eff_dim` of each ``[m, n]`` matrix in ``matrices[..., m, n]`` from two
    Frobenius norms: the sum of squared singular values is ``|A|_F**2`` and the sum of
    their fourth powers is the squared Frobenius norm of the smaller Gram matrix. The
    guards are :func:`eff_dim`'s. It agrees with :func:`eff_dim` to rounding."""
    m, n = matrices.shape[-2:]
    total = matrices.square().sum(dim=(-2, -1))
    gram = matrices.mT @ matrices if n <= m else matrices @ matrices.mT
    fourth = gram.square().sum(dim=(-2, -1))
    return torch.where(
        total > 1e-12,
        total.square() / fourth.clamp_min(1e-30) / min(m, n),
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


_TOPK_POOL: ThreadPoolExecutor | None = None


def _topk_pool() -> ThreadPoolExecutor:
    """One small pool per process for the gate's host-side top-k work. NumPy releases
    the GIL in its ufuncs and sorts, so the sampled layers' work runs concurrently."""
    global _TOPK_POOL
    if _TOPK_POOL is None:
        _TOPK_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gate-topk")
    return _TOPK_POOL


def gate_topk_overlap(host: np.ndarray) -> float:
    """The historical top-k Jaccard of one gate surface ``[T, I]`` (float32, on host).

    The SiLU is ``a * (1 / (1 + exp(-clip(a, -88, 88))))`` in float64, applied through
    the same ufuncs in the same operand order, in place rather than through
    temporaries; the top-k comes from NumPy's ``argsort``, whose tie order is part of
    the feature. Each row's top-k holds distinct columns, so the union of two
    rows' sets is twice the set size less their intersection, and every adjacent pair's
    intersection is counted at once through a membership mask.
    """
    activated = host.astype(np.float64)
    scale = np.clip(activated, -88, 88)
    np.negative(scale, out=scale)
    np.exp(scale, out=scale)
    np.add(1, scale, out=scale)
    np.divide(1, scale, out=scale)
    np.multiply(activated, scale, out=activated)
    del scale
    k = min(100, activated.shape[1] // 10)
    top = np.argsort(np.abs(activated, out=activated), axis=1)[:, -k:]
    del activated
    member = np.zeros(host.shape, dtype=bool)
    np.put_along_axis(member, top, True, axis=1)
    shared = np.count_nonzero(member[:-1] & member[1:], axis=1)
    union = 2 * top.shape[1] - shared
    overlaps = np.zeros(len(shared), dtype=np.float64)
    np.divide(shared, union, out=overlaps, where=union > 0)
    return float(np.mean(overlaps))


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
        self.gate_topk: dict[str, Future] = {}
        self.eff_dim_method = getattr(families, "_fast_lane_eff_dim", "svd")
        self.held_eff_dims: list[tuple[str, Tensor]] = []

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
        self.effective_dimension(f"kv_key_eff_dim_L{layer}", x)
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
            "drift_halfcentroid": half_drift(x),
            "novelty_mean": nov.mean(),
            "novelty_std": std(nov),
            "crosshead_normcv_mean": cv.mean(),
            "crosshead_normcv_std": std(cv),
            "key_drift_corr": corr(dr, drift(self.keys[layer])),
        }
        for name, value in vals.items():
            self.out.put(f"value_L{layer}_{name}", value)
        self.effective_dimension(f"value_L{layer}_eff_dim", x)
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
        self.effective_dimension(f"qk_L{layer}_q_eff_dim", q)
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
        self.gate_topk_on_host(prefix + "_topk_overlap_mean", pre_silu)
        self.operators(prefix + "_sparsity", sp)
        self.operators(prefix + "_drift", dr)

    def gate_topk_on_host(self, name: str, pre_silu: Tensor) -> None:
        """Start the gate's top-k Jaccard on the host; :meth:`finish` collects it.

        Named CPU exception: NumPy's unstable argsort tie ordering is part of the
        historical top-k Jaccard. bf16 gate ties are common. CUDA topk or a newly
        chosen stable sort would change the feature definition. Only this sampled
        gate surface is transferred, and its exact SiLU and argsort run on the host
        concurrently with the other layers and the device reductions. All other gate
        reductions stay on the device.
        """
        host = pre_silu.float().cpu().numpy()
        self.gate_tie_exception_d2h_bytes += host.nbytes
        self.gate_topk[name] = _topk_pool().submit(gate_topk_overlap, host)

    def residual_trajectory(
        self, layer: int, corrected: Tensor, raw_first_norm: Tensor
    ) -> None:
        from anamnesis.extraction.feature_families.residual_stream import (
            _trajectory_feature_names,
        )

        before = self.out.mark()
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
        self.out.zero_since(before, raw_first_norm < 1e-6)

    def effective_dimension(self, name: str, x: Tensor) -> None:
        """The feature ``name``, ``x``'s effective dimension, by the lane's method: from
        the singular values now, or held for :meth:`finish`, which computes every held
        matrix of one shape with one batched product."""
        if self.eff_dim_method == "svd":
            self.out.put(name, eff_dim(x))
        else:
            self.held_eff_dims.append((name, x))

    def finish(self) -> None:
        groups: dict[tuple, list[tuple[str, Tensor]]] = {}
        for name, x in self.held_eff_dims:
            groups.setdefault((tuple(x.shape), x.dtype, x.device), []).append((name, x))
        self.held_eff_dims.clear()
        for held in groups.values():
            values = norms_eff_dims(torch.stack([x for _, x in held]))
            for (name, _), value in zip(held, values, strict=True):
                self.out.put(name, value)
        for name, overlap in self.gate_topk.items():
            self.out.put(name, overlap.result())
        self.gate_topk.clear()
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
