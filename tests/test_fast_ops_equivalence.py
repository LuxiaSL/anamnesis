"""The fast lane's operators and collector, held to their earlier forms byte for byte.

The slope reuses the terms that depend only on a series' length, the STFT reuses its
window and band indices, and the collector casts every feature of one dtype in one
operation rather than one at a time. None of that may move a feature: the functions
below are the earlier forms, frozen verbatim, and the shipped ones must equal them
exactly on the same inputs, across the lengths where the branches change (under 4
points no windows, under 8 no spectrum, under 64 a padded frame).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from anamnesis.extraction.fast.families import gate_topk_overlap
from anamnesis.extraction.fast.ops import FeatureCollector, slope, std, temporal


def _historical_slope(y: Tensor, x: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
    y = y.double()
    if len(y) < 2:
        return y.new_zeros(())
    if x is None:
        x = torch.arange(len(y), dtype=torch.float64, device=y.device)
    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)
    w = mask.double()
    count = w.sum()
    xm = (x * w).sum() / count.clamp_min(1)
    ym = (y * w).sum() / count.clamp_min(1)
    xc = x - xm
    value = (xc * (y - ym) * w).sum() / (xc.square() * w).sum().clamp_min(1e-30)
    return torch.where(count >= 2, value, value.new_zeros(()))


def _historical_temporal(x: Tensor, n_windows: int = 4, include_stft: bool = True) -> Tensor:
    x = x.float().double()
    n = len(x)
    pieces = []
    if n < 4:
        pieces.extend([x.new_zeros(()) for _ in range(3 * n_windows)])
    else:
        width = n // n_windows
        for i in range(n_windows):
            w = x[i * width : (i + 1) * width if i < n_windows - 1 else n]
            pieces.extend((w.mean(), std(w), _historical_slope(w)))
    if include_stft:
        if n < 8:
            pieces.extend([x.new_zeros(()) for _ in range(5)])
        else:
            if n < 64:
                x = F.pad(x, (0, 64 - n))
            extended = F.pad(x, (32, 32 + (-len(x)) % 32))
            window = torch.hann_window(64, periodic=True, dtype=torch.float64, device=x.device)
            fft = torch.fft.rfft(extended.unfold(0, 64, 32) * window, dim=-1) / window.sum()
            power = fft.abs().square().mean(dim=0)
            total = power.sum()
            p = power / total.clamp_min(1e-30)
            freq = torch.fft.rfftfreq(64, dtype=torch.float64, device=x.device)
            centroid = (p * freq).sum()
            values = (
                centroid,
                (p * (freq - centroid).square()).sum().sqrt(),
                p[freq < 0.1].sum(),
                p[(freq >= 0.1) & (freq < 0.3)].sum(),
                p[freq >= 0.3].sum(),
            )
            pieces.extend(torch.where(total < 1e-12, v.new_zeros(()), v) for v in values)
    return torch.stack(pieces).float()


def _historical_put(value, device="cpu") -> Tensor:
    return torch.as_tensor(value, device=device).reshape(()).float()


def _same(a: Tensor, b: Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.numpy().tobytes() == b.numpy().tobytes()


LENGTHS = [1, 2, 3, 4, 5, 7, 8, 9, 31, 63, 64, 65, 97, 128, 129, 200, 511]


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_the_unmasked_slope_is_the_historical_one(n, dtype):
    g = torch.Generator().manual_seed(n)
    y = (torch.randn(n, generator=g, dtype=torch.float64) * 3 + 1).to(dtype)
    assert _same(slope(y), _historical_slope(y))
    # a window of a longer series, as the temporal operator hands it over
    longer = torch.randn(n + 9, generator=g, dtype=torch.float64)
    assert _same(slope(longer[5:5 + n]), _historical_slope(longer[5:5 + n]))


def test_the_masked_slope_is_unchanged():
    g = torch.Generator().manual_seed(3)
    y = torch.randn(40, generator=g, dtype=torch.float64)
    x = torch.rand(40, generator=g, dtype=torch.float64) * 9
    mask = torch.rand(40, generator=g) > 0.3
    assert _same(slope(y, x, mask), _historical_slope(y, x, mask))
    assert _same(slope(y, mask=torch.zeros(40, dtype=torch.bool)),
                 _historical_slope(y, mask=torch.zeros(40, dtype=torch.bool)))


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("include_stft", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_temporal_is_the_historical_operator(n, include_stft, dtype):
    g = torch.Generator().manual_seed(100 + n)
    x = (torch.randn(n, generator=g) * 2).to(dtype)
    assert _same(temporal(x, 4, include_stft), _historical_temporal(x, 4, include_stft))


def test_temporal_of_a_silent_series_is_zero_as_before():
    x = torch.zeros(70)
    assert _same(temporal(x), _historical_temporal(x))


def test_the_collector_emits_each_feature_as_its_own_cast():
    g = torch.Generator().manual_seed(9)
    f64 = torch.randn(6, generator=g, dtype=torch.float64) * 1e3
    f32 = torch.randn(6, generator=g)
    entries = {
        "a": f64[0], "b": f32[2], "c": 0.0, "d": 0.1234567890123, "e": 3,
        "f": (f64 > 0).sum(), "g": np.float32(0.3), "h": f64[1:2], "i": True,
    }
    out = FeatureCollector("cpu")
    for name, value in entries.items():
        out.put(name, value)
    order = ["i", "c", "h", "a", "g", "e", "b", "f", "d"]
    vector = out.finish(order)
    expected = torch.stack([_historical_put(entries[name]) for name in order])
    assert _same(vector, expected)
    assert _same(out.values["d"], _historical_put(entries["d"]))


def test_a_block_is_its_elements_and_zeroing_matches_the_per_value_where():
    g = torch.Generator().manual_seed(4)
    out = FeatureCollector("cpu")
    out.put("before", torch.tensor(5.0, dtype=torch.float64))
    mark = out.mark()
    block = torch.randn(4, generator=g)
    out.put_block(["b0", "b1", "b2", "b3"], block)
    out.put("after", torch.tensor(-2.5, dtype=torch.float64))
    out.zero_since(mark, torch.tensor(True))
    assert out.finish(["after", "b2", "before", "b0", "b3", "b1"]).tolist() == [
        0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    kept = FeatureCollector("cpu")
    mark = kept.mark()
    kept.put_block(["b0", "b1", "b2", "b3"], block)
    kept.zero_since(mark, torch.tensor(False))
    assert _same(kept.finish(["b3", "b0", "b1", "b2"]), block[[3, 0, 1, 2]])


def test_the_collector_refuses_duplicates_and_a_schema_it_was_not_given():
    out = FeatureCollector("cpu")
    out.put("a", 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        out.put("a", 2.0)
    with pytest.raises(ValueError, match="duplicate"):
        out.put_block(["b", "a"], torch.zeros(2))
    with pytest.raises(ValueError, match="names for a block"):
        out.put_block(["c"], torch.zeros(2))
    with pytest.raises(ValueError, match="schema mismatch"):
        out.finish(["a", "z"])


def _historical_topk_loop(host: np.ndarray) -> float:
    """The top-k Jaccard's earlier pairwise set loop, verbatim, after the same SiLU."""
    activated = host.astype(np.float64)
    activated = activated * (1 / (1 + np.exp(-np.clip(activated, -88, 88))))
    k = min(100, activated.shape[1] // 10)
    top = np.argsort(np.abs(activated), axis=1)[:, -k:]
    overlaps = []
    for a, b in zip(top[:-1], top[1:], strict=True):
        sa, sb = set(a), set(b)
        overlaps.append(len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
    return float(np.mean(overlaps))


@pytest.mark.parametrize("width", [0, 5, 9, 10, 11, 160, 2000])
def test_the_topk_jaccard_at_every_set_size(width):
    """Below ten columns the set size is zero and the slice keeps every column; both
    forms must agree there as they do above it."""
    rng = np.random.default_rng(width)
    host = (rng.integers(-6, 7, size=(12, width)) / 4).astype(np.float32)
    assert np.float64(gate_topk_overlap(host.copy())).tobytes() == np.float64(
        _historical_topk_loop(host.copy())).tobytes()
