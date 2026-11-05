"""A batch's finiteness check: per row, exact, without a copy of the surfaces.

``finite_rows`` reads a row as finite exactly when its smallest and largest values are. It must
agree with ``torch.isfinite(...).all()`` per row for NaN, +inf and -inf at any position, for
huge finite values (which a sum would overflow), for empty rows and for every float dtype the
surfaces arrive in, and on a GPU it must not allocate a tensor the size of its input.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast.engine_reducer import finite_rows

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_finite_rows_agrees_with_isfinite(device, dtype):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(12, 7, 33, generator=g).to(dtype)
    big = torch.finfo(dtype).max
    x[1, 3, 4] = float("nan")
    x[2, 0, 0] = float("inf")
    x[3, 6, 32] = float("-inf")
    x[4] = big  # finite, and a sum of it would overflow
    x[5, 2, :] = -big
    x[6, 1, 1] = float("nan")
    x[6, 1, 2] = float("inf")
    x = x.to(device)
    want = torch.isfinite(x).reshape(12, -1).all(dim=1)
    got = finite_rows(x)
    assert got.dtype == torch.bool and torch.equal(got, want)
    assert got.tolist() == [True, False, False, False, True, True, False] + [True] * 5


def test_empty_rows_are_finite():
    assert finite_rows(torch.empty(3, 0)).tolist() == [True, True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_finite_rows_does_not_copy_its_input():
    x = torch.randn(64, 256, 1024, device="cuda")
    finite_rows(x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    finite_rows(x)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < x.numel() * x.element_size() // 16
