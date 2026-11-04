"""row_gram: the Gram matrix in row_sum's fixed order, without materializing the products.

The CUDA kernel agrees with the materialized path (an elementwise product, then :func:`row_sum`)
to rounding, with and without lengths, in both dtypes, at widths below, at and across the tile.
Its bytes are the property that must hold exactly: the same alone and inside any batch, and the
same across processes.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast import invariant_ops
from anamnesis.extraction.fast.invariant_ops import BLOCK, row_gram, row_sum

needs_gpu = pytest.mark.skipif(
    not (invariant_ops.HAVE_TRITON and torch.cuda.is_available()),
    reason="no GPU with triton: the Triton Gram kernel needs a CUDA device",
)
WIDTHS = [1, 7, 128, BLOCK, BLOCK + 3, 2 * BLOCK + 129]
DTYPES = [torch.float32, torch.float64]


def materialized(x: torch.Tensor, lengths: torch.Tensor | None, backend: str) -> torch.Tensor:
    product = x[:, :, None, :] * x[:, None, :, :]
    lens = None if lengths is None else lengths.view(-1, 1, 1).expand(product.shape[:-1])
    return row_sum(product, lengths=lens, backend=backend)


def data(batch: int, m: int, n: int, dtype: torch.dtype, seed: int, device: str = "cpu"):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(batch, m, n, generator=gen, dtype=torch.float64) * 3).to(dtype)
    lengths = torch.randint(0, n + 1, (batch,), generator=gen)
    return x.to(device), lengths.to(device)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize("masked", [False, True], ids=["full", "lengths"])
def test_torch_path_is_the_materialized_row_sum(dtype, masked):
    x, lengths = data(3, 5, 37, dtype, seed=1)
    lengths = lengths if masked else None
    got = row_gram(x, lengths=lengths, backend="torch", chunk_bytes=64)  # several chunks
    assert torch.equal(got, materialized(x, lengths, "torch"))


def test_refusals():
    x = torch.zeros(2, 3, 4)
    with pytest.raises(TypeError):
        row_gram(x.half())
    with pytest.raises(ValueError):
        row_gram(x[0])
    with pytest.raises(ValueError):
        row_gram(x, lengths=torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError):
        row_gram(x, lengths=torch.zeros(2))


@needs_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "fp64"])
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("masked", [False, True], ids=["full", "lengths"])
def test_gpu_kernel_agrees_with_the_materialized_row_sum(dtype, width, masked):
    x, lengths = data(4, 9, width, dtype, seed=width, device="cuda")
    lengths = lengths if masked else None
    got = row_gram(x, lengths=lengths, backend="triton")
    want = materialized(x, lengths, "triton")
    assert got.dtype == want.dtype and got.shape == want.shape
    scale = (x.abs().amax() ** 2) * max(width, 1)
    eps = torch.finfo(dtype).eps
    assert ((got - want).abs() <= 4 * eps * scale).all()


@needs_gpu
def test_gpu_kernel_is_batchmate_invariant():
    x, lengths = data(6, 5, BLOCK + 3, torch.float64, seed=3, device="cuda")
    whole = row_gram(x, lengths=lengths, backend="triton")
    for b in range(6):
        alone = row_gram(x[b : b + 1], lengths=lengths[b : b + 1], backend="triton")
        assert torch.equal(alone[0], whole[b])


@needs_gpu
def test_gpu_kernel_is_batchmate_invariant_where_a_plain_matmul_is_not():
    """The falsifier for the invariance test: a plain batched matmul's entries can depend on
    the batch it ran in, so the comparison above is one that can fail."""
    x, _ = data(64, 32, 4096, torch.float32, seed=5, device="cuda")
    whole = x @ x.transpose(1, 2)
    alone = torch.stack([x[b : b + 1] @ x[b : b + 1].transpose(1, 2) for b in range(64)])[:, 0]
    gram_whole = row_gram(x, backend="triton")
    gram_alone = torch.stack([row_gram(x[b : b + 1], backend="triton")[0] for b in range(64)])
    assert torch.equal(gram_whole, gram_alone)
    if torch.equal(whole, alone):
        pytest.skip("this GPU's matmul happened to be batch-invariant at this shape; falsifier not shown")
