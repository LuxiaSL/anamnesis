"""Attention-residual routing: how a block-routing architecture allocates across its own history.

Some architectures do not carry one running residual stream. They commit intermediate states as
*blocks* and, at each routing point, attend over those committed blocks plus the running partial —
so the model chooses, per position, how much of its own history to read from and how far back. That
choice is a substrate no dense-residual model has, and this family reads it: per-producing-position
routing weights over source blocks (source 0 = earliest committed = ANCHOR, source -1 = the running
`partial` = RECENCY), plus the block-boundary committed snapshots themselves.

The read is MID-GRAIN distributional, on the per-head template: never average a structured axis
away, never emit per-token. Six bounded quantities per routing point, each summarised coarsely over
position (mean / std / half-split drift), fanned across the sampled layers, with an explicit
ANCHOR-versus-RECENCY contrast — the block-routing echo of the anamnesis how-axis, which is
anchor/sink against recency at mid layers. Every value is bounded in [0,1] or [-1,1], so the family
carries no length confound, and the output dimension is fixed (a missing slot is zero-filled) so
banks stay column-comparable. Fanning across blocks is deliberate; pruning depth is the decompose's
job, not the extractor's.

**This family is the reference example of an optional per-architecture hook.** The pattern, which any
new architecture-specific capability should follow:

1. The substrate arrives as its own optional fields on
   :class:`~anamnesis.extraction.state_extractor.RawGenerationData` — here ``attn_res_routing`` and
   ``attn_res_committed`` — defaulting to ``None``.
2. A capture supplies those fields when the architecture has the substrate to give. An architecture
   without it leaves them ``None``; nothing is faked and nothing is zero-filled at that level.
3. The orchestrator gates on the substrate's PRESENCE, not on a model name
   (``enable_attn_res and raw_data.attn_res_routing is not None``), so the family is simply absent
   from a vector it has nothing to say about. Adding an architecture is a capture change plus a
   preset row; this module never learns a model's name.
4. The family's own arity is fixed by ``sampled_layers``, so two architectures that both supply the
   substrate produce comparable columns.

Absent is the honest reading of a substrate the architecture does not have — not zero, which would
claim a measured value, and not an error, which would make one architecture's feature set a
precondition for reading another's.

Inputs:
  routing   : sequence of (tag, layer, weights[n_pos, n_src]) — per-producing-position routing
              weights. Source 0 = anchor (block-0/embed), source -1 = partial (recency); n_src grows
              with depth. The tag ``"final"`` marks the pre-output consolidation read, which is
              summarised separately rather than folded into a layer.
  committed : block-boundary snapshots, each [n_pos, hidden_dim], or None.
  sampled_layers : the layers to emit per-layer features for (fixed in, fixed dim out).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult

F32 = NDArray[np.float32]
RoutingPoint = tuple[str, int, NDArray[np.floating]]
EPS = 1e-8
QNAMES = ("entropy", "top1", "eff_src", "anchor_w", "partial_w", "recency_cent")
SUMM = ("mean", "std", "drift")
N_COMMITTED_PAIRS = 6                      # consecutive committed-block pairs read, which fixes the dim


def _summ(ts: np.ndarray) -> list[float]:
    """Coarse-temporal summary of a per-position series: mean, std, half-split drift (NOT per-token)."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0:
        return [0.0, 0.0, 0.0]
    if ts.size < 2:
        return [float(ts.mean()), 0.0, 0.0]
    h = ts.size // 2
    return [float(ts.mean()), float(ts.std()), abs(float(ts[:h].mean()) - float(ts[h:].mean()))]


def _point_quantities(w: np.ndarray) -> dict[str, np.ndarray]:
    """w (n_pos, n_src) routing weights → 6 bounded per-position series. src0=anchor, src-1=recency."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        z = np.zeros(max(w.shape[0], 1))
        return {q: z for q in QNAMES}
    n_src = w.shape[1]
    w = w / (w.sum(1, keepdims=True) + EPS)                          # renormalize (safety)
    ent = -(w * np.log(w + EPS)).sum(1) / np.log(max(n_src, 2))       # ∈[0,1]: 0=focused, 1=diffuse
    eff = (1.0 / (np.square(w).sum(1) + EPS)) / n_src                 # participation ratio / n_src ∈(0,1]
    idx = np.arange(n_src) / max(n_src - 1, 1)                        # 0=anchor … 1=most-recent
    return {"entropy": ent, "top1": w.max(1), "eff_src": eff,
            "anchor_w": w[:, 0], "partial_w": w[:, -1], "recency_cent": (w * idx[None, :]).sum(1)}


def extract_attn_res(
    routing: Sequence[RoutingPoint] | None,
    committed: Sequence[NDArray[np.floating]] | None = None,
    sampled_layers: Sequence[int] | None = None,
) -> FeatureFamilyResult:
    """Routing-allocation features for one generation, or an empty result when the substrate is absent."""
    feats: list[float] = []
    names: list[str] = []
    by_layer: dict[int, list[np.ndarray]] = {}
    point_stats: list[tuple[int, float, float, float]] = []          # (layer, entropy_mean, anchor_mean, partial_mean)
    final_vec: np.ndarray | None = None

    for tag, layer, w in (routing or []):
        q = _point_quantities(w)
        vec = np.array([v for qn in QNAMES for v in _summ(q[qn])], dtype=np.float64)   # 6*3 = 18
        if tag == "final":
            final_vec = vec
            continue
        by_layer.setdefault(int(layer), []).append(vec)
        point_stats.append((int(layer), float(q["entropy"].mean()),
                            float(q["anchor_w"].mean()), float(q["partial_w"].mean())))

    sl = list(sampled_layers) if sampled_layers is not None else sorted(by_layer.keys())

    # (A–E) per-sampled-layer averaged distribution-summaries (the mid-grain core)
    for L in sl:
        vecs = by_layer.get(int(L))
        v = np.mean(vecs, axis=0) if vecs else np.zeros(len(QNAMES) * 3)
        for qi, qn in enumerate(QNAMES):
            for si, sm in enumerate(SUMM):
                feats.append(float(v[qi * 3 + si]))
                names.append(f"attnres_L{L}_{qn}_{sm}")

    # (D) depth profile: slope of entropy/anchor/partial vs layer + cross-point entropy spread
    if len(point_stats) >= 2:
        pm = np.array(point_stats, dtype=np.float64)                  # (n, 4)
        lay = pm[:, 0]
        for ci, cn in ((1, "entropy"), (2, "anchor_w"), (3, "partial_w")):
            slope = float(np.polyfit(lay, pm[:, ci], 1)[0]) if lay.std() > EPS else 0.0
            feats.append(slope)
            names.append(f"attnres_depthslope_{cn}")
        feats.append(float(pm[:, 1].std()))
        names.append("attnres_xpoint_entropy_std")
    else:
        for cn in ("entropy", "anchor_w", "partial_w"):
            feats.append(0.0)
            names.append(f"attnres_depthslope_{cn}")
        feats.append(0.0)
        names.append("attnres_xpoint_entropy_std")

    # (E) the final consolidation read (pre-output routing): its 6 mean quantities
    for qi, qn in enumerate(QNAMES):
        feats.append(float(final_vec[qi * 3]) if final_vec is not None else 0.0)
        names.append(f"attnres_final_{qn}_mean")

    # (F) committed-state geometry: consecutive-block direction preservation (cosine ∈[-1,1], bounded)
    cos_means = [0.0] * N_COMMITTED_PAIRS
    overall = [0.0, 0.0]
    if committed is not None and len(committed) >= 2:
        cos_all = []
        for b in range(min(len(committed) - 1, N_COMMITTED_PAIRS)):
            a = np.asarray(committed[b], dtype=np.float64)
            c = np.asarray(committed[b + 1], dtype=np.float64)
            if a.ndim == 2 and a.shape == c.shape and a.shape[0] > 0:
                den = np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + EPS
                cos = (a * c).sum(1) / den
                cos_means[b] = float(cos.mean())
                cos_all.append(cos)
        if cos_all:
            allc = np.concatenate(cos_all)
            overall = [float(allc.mean()), float(allc.std())]
    for b in range(N_COMMITTED_PAIRS):
        feats.append(cos_means[b])
        names.append(f"attnres_committed_cos_b{b}b{b + 1}_mean")
    feats.append(overall[0]); names.append("attnres_committed_cos_overall_mean")
    feats.append(overall[1]); names.append("attnres_committed_cos_overall_std")

    return FeatureFamilyResult(
        features=np.nan_to_num(np.array(feats, dtype=np.float32)),
        feature_names=names,
        family_name="attn_res",
    )
