"""Torch counterparts of the reference's small scalar/time-series operators.

The NumPy extractor remains the reference. These operators deliberately retain
its float32 -> float64 -> float32 boundaries and population standard deviation.
"""

from __future__ import annotations

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


def slope(y: Tensor, x: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
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


def decay(y: Tensor) -> Tensor:
    mask = y > 1e-10
    return -slope(y.clamp_min(1e-30).log(), mask=mask)


def entropy(x: Tensor) -> Tensor:
    x = x.double().clamp_min(0)
    p = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(p * p.clamp_min(1e-300).log()).sum(dim=-1).float()


def temporal(x: Tensor, n_windows: int = 4, include_stft: bool = True) -> Tensor:
    x = x.float().double()
    n = len(x)
    pieces = []
    if n < 4:
        pieces.extend([x.new_zeros(()) for _ in range(3 * n_windows)])
    else:
        width = n // n_windows
        for i in range(n_windows):
            w = x[i * width : (i + 1) * width if i < n_windows - 1 else n]
            if not len(w):
                raise ValueError(
                    "empty temporal window (unsupported reference configuration)"
                )
            pieces.extend((w.mean(), std(w), slope(w)))
    if include_stft:
        if n < 8:
            pieces.extend([x.new_zeros(()) for _ in range(5)])
        else:
            if n < 64:
                x = F.pad(x, (0, 64 - n))
            # scipy.signal.stft defaults: periodic Hann, nperseg64, overlap32,
            # boundary=zeros, padded=True, spectrum scaling by sum(window).
            extended = F.pad(x, (32, 32 + (-len(x)) % 32))
            window = torch.hann_window(
                64, periodic=True, dtype=torch.float64, device=x.device
            )
            fft = (
                torch.fft.rfft(extended.unfold(0, 64, 32) * window, dim=-1)
                / window.sum()
            )
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
            pieces.extend(
                torch.where(total < 1e-12, v.new_zeros(()), v) for v in values
            )
    return torch.stack(pieces).float()


def trajectory_indices(n: int, count: int = 5) -> list[int]:
    if n <= 0:
        return []
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


class FeatureCollector:
    def __init__(self, device: torch.device | str):
        self.device = device
        self.values: dict[str, Tensor] = {}

    def put(self, name: str, value: Tensor | float | int) -> None:
        if name in self.values:
            raise ValueError(f"duplicate GPU feature {name}")
        self.values[name] = (
            torch.as_tensor(value, device=self.device).reshape(()).float()
        )

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
        result = temporal(x, n_windows, include_stft)
        for name, value in zip(names, result, strict=True):
            self.put(name, value)

    def finish(self, names: list[str]) -> Tensor:
        if set(names) != set(self.values) or len(names) != len(self.values):
            missing = set(names) - set(self.values)
            extra = set(self.values) - set(names)
            raise ValueError(
                f"GPU feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return torch.stack([self.values[n] for n in names])
