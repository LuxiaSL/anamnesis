"""The attention-residual routing family on the device.

A port of :func:`anamnesis.extraction.feature_families.attn_res.extract_attn_res`, the
canonical definition: the same names in the same order, the same quantities per
routing point, the same summaries, depth profile, final read and committed-block
geometry, computed in float64 like the canonical family and returned as float32.
Inputs are the routing weights and committed blocks a block-routing forward leaves
on the device, at the span's positions, so nothing moves to the host first.

The depth slope is the least-squares line's slope in closed form, which is what
``numpy.polyfit(x, y, 1)[0]`` solves; the two agree to rounding, which is the
whole difference between this family and the canonical one on the same inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from anamnesis.extraction.feature_families.attn_res import EPS, QNAMES, SUMM, committed_pairs


def _point_quantities(w: Tensor) -> dict[str, Tensor]:
    """``w[positions, sources]`` routing weights to six bounded per-position series."""
    w = w.double().clamp_min(0.0)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        z = torch.zeros(max(int(w.shape[0]) if w.ndim else 1, 1), dtype=torch.float64, device=w.device)
        return {q: z for q in QNAMES}
    n_src = w.shape[1]
    w = w / (w.sum(1, keepdim=True) + EPS)
    ent = -(w * torch.log(w + EPS)).sum(1) / torch.log(torch.tensor(float(max(n_src, 2)), dtype=torch.float64))
    eff = (1.0 / (w.square().sum(1) + EPS)) / n_src
    idx = torch.arange(n_src, dtype=torch.float64, device=w.device) / max(n_src - 1, 1)
    return {"entropy": ent, "top1": w.max(1).values, "eff_src": eff,
            "anchor_w": w[:, 0], "partial_w": w[:, -1], "recency_cent": (w * idx[None, :]).sum(1)}


def _summ(ts: Tensor) -> Tensor:
    """Mean, population standard deviation and half-split drift of a series."""
    ts = ts.double()
    zero = torch.zeros((), dtype=torch.float64, device=ts.device)
    if ts.numel() == 0:
        return torch.stack([zero, zero, zero])
    if ts.numel() < 2:
        return torch.stack([ts.mean(), zero, zero])
    h = ts.numel() // 2
    return torch.stack([ts.mean(), ts.std(correction=0), (ts[:h].mean() - ts[h:].mean()).abs()])


def _slope(x: Tensor, y: Tensor) -> Tensor:
    xc = x - x.mean()
    return (xc * (y - y.mean())).sum() / xc.square().sum()


def attn_res_features(
    routing: Sequence[tuple[str, int, Tensor]],
    committed: Sequence[Tensor],
    sampled_layers: Sequence[int],
) -> tuple[list[str], Tensor]:
    """The routing family's names and float32 values for one span, on the inputs' device."""
    device = routing[0][2].device if routing else (committed[0].device if committed else "cpu")
    values: list[Tensor] = []
    names: list[str] = []
    by_layer: dict[int, list[Tensor]] = {}
    stats: list[tuple[int, Tensor, Tensor, Tensor]] = []
    final_vec: Tensor | None = None
    for tag, layer, w in routing:
        q = _point_quantities(w)
        vec = torch.cat([_summ(q[qn]) for qn in QNAMES])
        if tag == "final":
            final_vec = vec
            continue
        by_layer.setdefault(int(layer), []).append(vec)
        stats.append((int(layer), q["entropy"].mean(), q["anchor_w"].mean(), q["partial_w"].mean()))

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
