"""The batched families' head mean: a fixed-order elementwise fold over the head axis."""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast.batched_families import _head_mean

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads", [1, 6, 12])
def test_agrees_with_torch_mean_to_rounding(device, heads):
    gen = torch.Generator().manual_seed(heads)
    x = torch.randn(3, 17, heads, 64, generator=gen).to(device)
    got = _head_mean(x)
    assert got.shape == (3, 17, 64) and got.dtype == torch.float32
    torch.testing.assert_close(got, x.double().mean(-2).float(), rtol=0, atol=4 * torch.finfo(torch.float32).eps * 8)


@pytest.mark.parametrize("device", DEVICES)
def test_batchmate_invariant(device):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(16, 33, 6, 128, generator=gen).to(device)
    whole = _head_mean(x)
    for b in range(16):
        assert torch.equal(_head_mean(x[b : b + 1])[0], whole[b])
    assert torch.equal(_head_mean(x[:, 5:9]), whole[:, 5:9])  # positions read only themselves


@pytest.mark.parametrize("device", DEVICES)
def test_input_dtype_is_read_as_float32(device):
    x = torch.randn(2, 4, 6, 8, dtype=torch.bfloat16).to(device)
    assert torch.equal(_head_mean(x), _head_mean(x.float()))
