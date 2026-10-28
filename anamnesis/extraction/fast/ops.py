"""Torch counterparts of the reference's small scalar/time-series operators.

The NumPy extractor remains the reference. These operators deliberately retain
its float32 -> float64 -> float32 boundaries and population standard deviation.

Constants that depend only on a series' length or the device (the slope's
abscissa, the STFT window and its frequency bands) are built once and reused;
they are the tensors each call would otherwise rebuild, so reusing them moves no
byte.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F


def std(x: Tensor, dim=None) -> Tensor:
    return x.std(dim=dim, correction=0)


def cos(a: Tensor, b: Tensor) -> Tensor:
    a, b = a.double(), b.double()
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    value = (a * b).sum(dim=-1) / (na * nb).clamp_min(1e-30)
    return torch.where((na < 1e-12) | (nb < 1e-12), torch.zeros_like(value), value)


def rowcos(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def corr(a: Tensor, b: Tensor, min_n: int = 3) -> Tensor:
    n = min(len(a), len(b))
    if n < min_n:
        return a.new_zeros(())
    a, b = a[:n].double(), b[:n].double()
    ac, bc = a - a.mean(), b - b.mean()
    value = (ac * bc).sum() / (ac.norm() * bc.norm()).clamp_min(1e-30)
    return torch.where(
        (std(a) < 1e-12) | (std(b) < 1e-12), value.new_zeros(()), value.clamp(-1, 1)
    )


@lru_cache(maxsize=1024)
def _unit_abscissa(n: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The unmasked slope's terms that depend only on ``n``: the all-ones weight, the
    clamped count, the centred abscissa ``0..n-1`` and its clamped squared sum."""
    x = torch.arange(n, dtype=torch.float64, device=device)
    w = torch.ones(n, dtype=torch.bool, device=device).double()
    count = w.sum().clamp_min(1)
    xc = x - (x * w).sum() / count
    return w, count, xc, (xc.square() * w).sum().clamp_min(1e-30)


def slope(y: Tensor, x: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
    y = y.double()
    if len(y) < 2:
        return y.new_zeros(())
    if x is None and mask is None:
        # Every point kept: the count is n >= 2, so the guard below always selects
        # the value, and the abscissa terms are the cached ones.
        w, count, xc, denominator = _unit_abscissa(len(y), y.device)
        ym = (y * w).sum() / count
        return (xc * (y - ym) * w).sum() / denominator
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


def decay(y: Tensor) -> Tensor:
    mask = y > 1e-10
    return -slope(y.clamp_min(1e-30).log(), mask=mask)


def entropy(x: Tensor) -> Tensor:
    x = x.double().clamp_min(0)
    p = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(p * p.clamp_min(1e-300).log()).sum(dim=-1).float()


@lru_cache(maxsize=None)
def _stft_constants(device: torch.device) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
    """The periodic Hann window, its sum, the bins' frequencies and the indices of the
    three bands (below 0.1, 0.1 to 0.3, from 0.3), in ascending bin order."""
    window = torch.hann_window(64, periodic=True, dtype=torch.float64, device=device)
    freq = torch.fft.rfftfreq(64, dtype=torch.float64, device=device)
    bands = tuple(
        torch.nonzero(band).flatten()
        for band in (freq < 0.1, (freq >= 0.1) & (freq < 0.3), freq >= 0.3)
    )
    return window, window.sum(), freq, bands


def temporal(x: Tensor, n_windows: int = 4, include_stft: bool = True) -> Tensor:
    x = x.float().double()
    n = len(x)
    if n < 4:
        windowed = x.new_zeros(3 * n_windows)
    else:
        width = n // n_windows
        pieces = []
        for i in range(n_windows):
            w = x[i * width : (i + 1) * width if i < n_windows - 1 else n]
            if not len(w):
                raise ValueError(
                    "empty temporal window (unsupported reference configuration)"
                )
            pieces.extend((w.mean(), std(w), slope(w)))
        windowed = torch.stack(pieces)
    if not include_stft:
        return windowed.float()
    if n < 8:
        spectral = x.new_zeros(5)
    else:
        if n < 64:
            x = F.pad(x, (0, 64 - n))
        # scipy.signal.stft defaults: periodic Hann, nperseg64, overlap32,
        # boundary=zeros, padded=True, spectrum scaling by sum(window).
        extended = F.pad(x, (32, 32 + (-len(x)) % 32))
        window, window_sum, freq, bands = _stft_constants(x.device)
        fft = torch.fft.rfft(extended.unfold(0, 64, 32) * window, dim=-1) / window_sum
        power = fft.abs().square().mean(dim=0)
        total = power.sum()
        p = power / total.clamp_min(1e-30)
        centroid = (p * freq).sum()
        spectral = torch.stack(
            (
                centroid,
                (p * (freq - centroid).square()).sum().sqrt(),
                *(p[band].sum() for band in bands),
            )
        )
        spectral = torch.where(total < 1e-12, spectral.new_zeros(()), spectral)
    return torch.cat((windowed, spectral)).float()


def trajectory_indices(n: int, count: int = 5) -> list[int]:
    if n <= 0:
        return []
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


_BlockRef = tuple[Tensor, int]


class FeatureCollector:
    """A span's named features, assembled into the declared schema once.

    Each feature is kept as it was computed: a 0-d tensor in its own dtype, an
    element of a block of features computed together, or a host number. The
    float32 vector is assembled in :meth:`finish`, where every tensor feature of
    one dtype is stacked and cast once, the host numbers are uploaded once, and one
    gather puts the coordinates in schema order. Each coordinate takes exactly the
    cast it would take alone, so the vector's bytes do not depend on how the
    features were grouped; a feature costs no device launch of its own.
    """

    def __init__(self, device: torch.device | str):
        self.device = device
        self._device = torch.device(device)
        self._raw: dict[str, Tensor | float | int | _BlockRef] = {}
        self._zeroed: list[tuple[list[str], Tensor]] = []

    def put(self, name: str, value: Tensor | float | int) -> None:
        if name in self._raw:
            raise ValueError(f"duplicate GPU feature {name}")
        if isinstance(value, Tensor):
            if value.dim() != 0:
                value = value.reshape(())
            if value.device != self._device:
                value = value.to(self._device)
        elif not isinstance(value, (int, float)):
            value = torch.as_tensor(value, device=self._device).reshape(())
        self._raw[name] = value

    def put_block(self, names: list[str], values: Tensor) -> None:
        """One feature per element of the 1-d tensor ``values``, named in order."""
        if values.dim() != 1 or values.shape[0] != len(names):
            raise ValueError(f"{len(names)} names for a block of shape {tuple(values.shape)}")
        if values.device != self._device:
            values = values.to(self._device)
        for i, name in enumerate(names):
            if name in self._raw:
                raise ValueError(f"duplicate GPU feature {name}")
            self._raw[name] = (values, i)

    @property
    def names(self) -> list[str]:
        """The names put so far, in the order they were put."""
        return list(self._raw)

    def mark(self) -> int:
        """A position in the order features were put, for :meth:`zero_since`."""
        return len(self._raw)

    def zero_since(self, mark: int, condition: Tensor) -> None:
        """Every feature put after ``mark`` becomes zero where ``condition`` holds."""
        self._zeroed.append((list(self._raw)[mark:], condition))

    def moments(self, prefix: str, x: Tensor) -> None:
        self.put(prefix + "_mean", x.mean())
        self.put(prefix + "_std", std(x))

    def operators(
        self, prefix: str, x: Tensor, n_windows: int = 4, include_stft: bool = True
    ) -> None:
        from anamnesis.extraction.feature_families.operators import (
            _windowed_names,
            _stft_names,
        )

        names = list(_windowed_names(prefix, n_windows))
        if include_stft:
            names.extend(_stft_names(prefix))
        self.put_block(names, temporal(x, n_windows, include_stft))

    @property
    def values(self) -> dict[str, Tensor]:
        """Every feature put so far, as the float32 0-d tensor it is emitted as."""
        names = list(self._raw)
        vector = self._assemble(names)
        return {name: vector[i] for i, name in enumerate(names)}

    def _assemble(self, names: list[str]) -> Tensor:
        by_dtype: dict[torch.dtype, list[Tensor]] = {}
        blocks: dict[int, tuple[int, Tensor]] = {}
        host: list[float | int] = []
        slots: list[tuple[object, int]] = []
        for name in names:
            value = self._raw[name]
            if isinstance(value, tuple):
                block, i = value
                key = blocks.setdefault(id(block), (len(blocks), block))[0]
                slots.append((("block", key), i))
            elif isinstance(value, Tensor):
                group = by_dtype.setdefault(value.dtype, [])
                slots.append((("dtype", value.dtype), len(group)))
                group.append(value)
            else:
                slots.append((("host", None), len(host)))
                host.append(value)
        segments: list[Tensor] = []
        offsets: dict[object, int] = {}
        width = 0
        for dtype, group in by_dtype.items():
            offsets[("dtype", dtype)] = width
            segments.append(torch.stack(group).float())
            width += len(group)
        for key, block in blocks.values():
            offsets[("block", key)] = width
            segments.append(block.float())
            width += block.shape[0]
        if host:
            offsets[("host", None)] = width
            segments.append(torch.tensor(host, dtype=torch.float32, device=self._device))
            width += len(host)
        if not segments:
            return torch.zeros(0, dtype=torch.float32, device=self._device)
        pool = torch.cat(segments) if len(segments) > 1 else segments[0]
        order = torch.tensor(
            [offsets[key] + i for key, i in slots], dtype=torch.long, device=self._device
        )
        vector = pool.index_select(0, order)
        if self._zeroed:
            position = {name: i for i, name in enumerate(names)}
            for zeroed, condition in self._zeroed:
                selected = [False] * len(names)
                for name in zeroed:
                    if name in position:
                        selected[position[name]] = True
                mask = torch.tensor(selected, dtype=torch.bool, device=self._device)
                vector = torch.where(mask & condition, vector.new_zeros(()), vector)
        return vector

    def finish(self, names: list[str]) -> Tensor:
        if set(names) != set(self._raw) or len(names) != len(self._raw):
            missing = set(names) - set(self._raw)
            extra = set(self._raw) - set(names)
            raise ValueError(
                f"GPU feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return self._assemble(names)
