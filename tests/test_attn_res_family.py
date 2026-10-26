"""The attention-residual routing family: its committed-pair width and its earlier output.

The committed-state read covers ``max(6, n_committed - 1)`` consecutive block pairs. An
architecture of up to seven committed blocks reads the same six pairs the family always read,
byte for byte; one with more blocks reads every pair it has. A frozen copy of the earlier
function holds the first half of that.
"""
from __future__ import annotations

import numpy as np
import pytest

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.attn_res import (
    EPS,
    N_COMMITTED_PAIRS,
    QNAMES,
    SUMM,
    _point_quantities,
    _summ,
    committed_pairs,
    extract_attn_res,
)

HIDDEN = 16
N_POS = 9


def _earlier_extract_attn_res(routing, committed=None, sampled_layers=None):
    """The family before the committed-pair read was sized by block count (frozen copy)."""
    feats, names = [], []
    by_layer: dict[int, list[np.ndarray]] = {}
    point_stats = []
    final_vec = None
    for tag, layer, w in (routing or []):
        q = _point_quantities(w)
        vec = np.array([v for qn in QNAMES for v in _summ(q[qn])], dtype=np.float64)
        if tag == "final":
            final_vec = vec
            continue
        by_layer.setdefault(int(layer), []).append(vec)
        point_stats.append((int(layer), float(q["entropy"].mean()),
                            float(q["anchor_w"].mean()), float(q["partial_w"].mean())))
    sl = list(sampled_layers) if sampled_layers is not None else sorted(by_layer.keys())
    for L in sl:
        vecs = by_layer.get(int(L))
        v = np.mean(vecs, axis=0) if vecs else np.zeros(len(QNAMES) * 3)
        for qi, qn in enumerate(QNAMES):
            for si, sm in enumerate(SUMM):
                feats.append(float(v[qi * 3 + si]))
                names.append(f"attnres_L{L}_{qn}_{sm}")
    if len(point_stats) >= 2:
        pm = np.array(point_stats, dtype=np.float64)
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
    for qi, qn in enumerate(QNAMES):
        feats.append(float(final_vec[qi * 3]) if final_vec is not None else 0.0)
        names.append(f"attnres_final_{qn}_mean")
    cos_means = [0.0] * 6
    overall = [0.0, 0.0]
    if committed is not None and len(committed) >= 2:
        cos_all = []
        for b in range(min(len(committed) - 1, 6)):
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
    for b in range(6):
        feats.append(cos_means[b])
        names.append(f"attnres_committed_cos_b{b}b{b + 1}_mean")
    feats.append(overall[0]); names.append("attnres_committed_cos_overall_mean")
    feats.append(overall[1]); names.append("attnres_committed_cos_overall_std")
    return FeatureFamilyResult(features=np.nan_to_num(np.array(feats, dtype=np.float32)),
                               feature_names=names, family_name="attn_res")


def _substrate(n_committed: int, seed: int = 0):
    """Routing over a growing source count, a final read, and ``n_committed`` block snapshots."""
    rng = np.random.default_rng(seed)
    routing = []
    for layer in range(6):
        for _point in range(2):
            n_src = min(layer + 2, max(n_committed, 2))
            w = rng.random((N_POS, n_src))
            routing.append(("layer", layer, w / w.sum(1, keepdims=True)))
    w = rng.random((N_POS, max(n_committed, 2)))
    routing.append(("final", -1, w / w.sum(1, keepdims=True)))
    committed = [rng.standard_normal((N_POS, HIDDEN)).astype(np.float32) for _ in range(n_committed)]
    return routing, committed


@pytest.mark.parametrize("n_committed", [0, 1, 2, 4, 7])
def test_up_to_seven_blocks_is_byte_identical_to_the_earlier_family(n_committed):
    routing, committed = _substrate(n_committed, seed=n_committed)
    sampled = [0, 2, 5]
    new = extract_attn_res(routing, committed=committed or None, sampled_layers=sampled)
    old = _earlier_extract_attn_res(routing, committed=committed or None, sampled_layers=sampled)
    assert new.feature_names == old.feature_names
    assert new.features.tobytes() == old.features.tobytes()


def test_eight_blocks_read_the_seventh_pair():
    routing, committed = _substrate(8, seed=8)
    result = extract_attn_res(routing, committed=committed, sampled_layers=[0, 3])
    names = list(result.feature_names)
    pair_names = [n for n in names if n.startswith("attnres_committed_cos_b")]
    assert pair_names == [f"attnres_committed_cos_b{b}b{b + 1}_mean" for b in range(7)]
    a, c = (np.asarray(committed[i], np.float64) for i in (6, 7))
    expected = float(((a * c).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + EPS)).mean())
    got = float(result.features[names.index("attnres_committed_cos_b6b7_mean")])
    assert got == pytest.approx(expected, abs=1e-6)


def test_eight_blocks_keep_every_shared_column_of_the_first_seven():
    routing, committed = _substrate(8, seed=3)
    new = extract_attn_res(routing, committed=committed, sampled_layers=[1, 4])
    old = _earlier_extract_attn_res(routing, committed=committed, sampled_layers=[1, 4])
    new_by_name = dict(zip(new.feature_names, new.features))
    shared = [n for n in old.feature_names if not n.startswith("attnres_committed_cos_overall")]
    assert all(new_by_name[n] == v for n, v in zip(old.feature_names, old.features) if n in shared)
    assert len(new) == len(old) + 1


def test_the_overall_committed_statistics_cover_every_pair_read():
    routing, committed = _substrate(8, seed=5)
    result = extract_attn_res(routing, committed=committed, sampled_layers=[0])
    names = list(result.feature_names)
    cos = []
    for b in range(7):
        a, c = (np.asarray(committed[i], np.float64) for i in (b, b + 1))
        cos.append((a * c).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + EPS))
    allc = np.concatenate(cos)
    assert float(result.features[names.index("attnres_committed_cos_overall_mean")]) == pytest.approx(
        float(allc.mean()), abs=1e-6)
    assert float(result.features[names.index("attnres_committed_cos_overall_std")]) == pytest.approx(
        float(allc.std()), abs=1e-6)


def test_the_width_is_the_architectures_not_the_rows():
    """Two rows of one architecture give one schema, whatever their routing values."""
    a = extract_attn_res(*_substrate(8, seed=11), sampled_layers=[0, 2])
    b = extract_attn_res(*_substrate(8, seed=12), sampled_layers=[0, 2])
    assert a.feature_names == b.feature_names


@pytest.mark.parametrize("n_committed,pairs", [(0, 6), (1, 6), (7, 6), (8, 7), (12, 11)])
def test_committed_pairs(n_committed, pairs):
    assert committed_pairs(n_committed) == pairs
    assert committed_pairs(n_committed) >= N_COMMITTED_PAIRS


def test_committed_pairs_refuses_a_negative_count():
    with pytest.raises(ValueError):
        committed_pairs(-1)
