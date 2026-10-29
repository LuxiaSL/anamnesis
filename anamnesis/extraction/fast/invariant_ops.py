"""Row reductions whose bytes depend only on the row: batchmate-invariant primitives.

A plain torch reduction over the last dimension is not batchmate-invariant. Its
kernel chooses how to split the reduced dimension across threads from the number
of outputs, so the same row is associated differently when it shares a launch with
seven other rows than when it is reduced alone, and its result can move by an ulp.
A batched reducer built on such reductions gives a row different bytes depending on
which rows it was batched with.

The primitives here reduce each row of ``x [..., W]`` over its last dimension in a
fixed order that depends only on ``W`` and :data:`BLOCK`:

1. the row is cut into tiles of ``BLOCK`` consecutive elements, the last tile
   zero-filled past ``W``;
2. a ``BLOCK``-wide accumulator adds the tiles elementwise, first tile to last;
3. a fixed tree folds the accumulator to one value.

Masked positions are loaded as exact zeros at their own positions, so a mask changes
which terms are zero, never the order of the additions. Nothing in that order reads
the batch size, the row's index in the batch, or another row's values. A row's
result is therefore the same bytes alone and inside any batch.

Two implementations carry the order:

* **CUDA**: one Triton program per row, with ``BLOCK``, ``num_warps`` and
  ``num_stages`` fixed (:data:`BLOCK`, :data:`NUM_WARPS`, :data:`NUM_STAGES`),
  never autotuned, so every process compiles the same kernel for a given
  ``(W, dtype)``. The fold in step 3 is ``tl.sum`` over the accumulator, whose tree
  is fixed by that compiled layout. The launcher hands the kernel contiguous,
  16-byte-aligned buffers, so the compiler's alignment specialisation is the same
  for every call with the same ``W``.
* **torch**: the same tiles and the same accumulator in plain elementwise torch,
  with step 3 a pairwise halving fold (``acc[:half] + acc[half:]`` until one value
  is left). Every operation is elementwise across rows, so it is invariant on any
  device. It is the path for CPU tensors and the fallback where Triton is missing.

The two implementations share the order of steps 1 and 2, not the tree of step 3,
and the CUDA compiler may fuse a multiply and an add. Each is batchmate-invariant
and restart-invariant on its own. They agree with each other, and with torch's
reductions, to rounding rather than to the byte.

**The primitives.** Every function takes ``x`` (and ``row_dot`` a second tensor of
the same shape) in float32 or float64, reduces the last dimension, and returns a
tensor of shape ``x.shape[:-1]`` in the input dtype. Each takes one optional mask:

* ``lengths``: an integer tensor of shape ``x.shape[:-1]``; positions ``>= length``
  are masked. A length is clamped to ``[0, W]``.
* ``mask``: a boolean tensor of the shape of ``x``; ``False`` positions are masked.

The count ``n`` is the number of unmasked positions. ``row_mean`` divides by
``max(n, 1)`` and ``row_std`` is the population deviation about that mean, so a row
with nothing unmasked has mean and deviation zero. Accumulation is in the input
dtype; a caller that wants float64 arithmetic passes float64.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

try:  # Triton is needed to launch on a GPU, not to import this module.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only without triton
    HAVE_TRITON = False

BLOCK = 1024
"""Tile width and accumulator width. Fixed: the reduction order is a function of it."""

NUM_WARPS = 4
"""Warps per Triton program. Fixed: the accumulator's fold tree follows the layout."""

NUM_STAGES = 1
"""Software pipelining stages per Triton program. Fixed for the same reason."""

SUPPORTED_DTYPES = (torch.float32, torch.float64)

Backend = Literal["auto", "triton", "torch"]

# Operation codes shared by the kernel and the torch implementation.
_SUM, _SUMSQ, _DOT, _MEAN, _NORM, _STD = range(6)
_MASK_NONE, _MASK_LENGTHS, _MASK_BOOL = range(3)
_ALIGN_BYTES = 16


if HAVE_TRITON:

    @triton.jit
    def _row_reduce_kernel(
        x_ptr,
        y_ptr,
        len_ptr,
        mask_ptr,
        out_ptr,
        W,
        OP: tl.constexpr,
        MASK_KIND: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * W
        offs = tl.arange(0, BLOCK_W)
        if MASK_KIND == 1:
            limit = tl.minimum(tl.maximum(tl.load(len_ptr + row), 0), W)
        else:
            limit = W

        acc = tl.zeros([BLOCK_W], dtype=out_ptr.dtype.element_ty)
        count = tl.zeros([BLOCK_W], dtype=tl.int32)
        for start in range(0, W, BLOCK_W):
            cols = start + offs
            keep = cols < limit
            if MASK_KIND == 2:
                keep = keep & (tl.load(mask_ptr + base + cols, mask=cols < W, other=0) != 0)
            x = tl.load(x_ptr + base + cols, mask=keep, other=0.0)
            if OP == 2:
                y = tl.load(y_ptr + base + cols, mask=keep, other=0.0)
                acc += x * y
            elif OP == 1 or OP == 4:
                acc += x * x
            else:
                acc += x
            count += keep.to(tl.int32)

        total = tl.sum(acc, axis=0)
        n = tl.maximum(tl.sum(count, axis=0), 1).to(total.dtype)
        if OP == 3:
            total = tl.div_rn(total, n)
        elif OP == 4:
            total = tl.sqrt_rn(total)
        elif OP == 5:
            mean = tl.div_rn(total, n)
            acc = tl.zeros([BLOCK_W], dtype=out_ptr.dtype.element_ty)
            for start in range(0, W, BLOCK_W):
                cols = start + offs
                keep = cols < limit
                if MASK_KIND == 2:
                    keep = keep & (
                        tl.load(mask_ptr + base + cols, mask=cols < W, other=0) != 0
                    )
                x = tl.load(x_ptr + base + cols, mask=keep, other=0.0)
                d = tl.where(keep, x - mean, 0.0)
                acc += d * d
            total = tl.sqrt_rn(tl.div_rn(tl.sum(acc, axis=0), n))
        tl.store(out_ptr + row, total)


def _aligned_contiguous(t: Tensor) -> Tensor:
    """``t`` as a contiguous buffer whose address is 16-byte aligned.

    The Triton compiler specialises on pointer alignment; a buffer that is aligned
    on every call keeps the compiled kernel, and with it the fold tree, the same.
    """
    t = t.contiguous()
    if t.numel() and t.data_ptr() % _ALIGN_BYTES:
        t = t.clone()
    return t


def _prepare(
    x: Tensor, y: Tensor | None, lengths: Tensor | None, mask: Tensor | None
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, int]:
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"row reductions take float32 or float64, not {x.dtype}")
    if x.dim() < 1:
        raise ValueError("row reductions need at least one dimension to reduce")
    if y is not None:
        if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
            raise ValueError(
                f"row_dot needs two tensors of one shape, dtype and device; got "
                f"{tuple(x.shape)} {x.dtype} {x.device} and {tuple(y.shape)} {y.dtype} {y.device}"
            )
    if lengths is not None and mask is not None:
        raise ValueError("pass at most one of lengths and mask")
    width = x.shape[-1]
    rows = x.reshape(-1, width)
    other = y.reshape(-1, width) if y is not None else None
    if lengths is not None:
        if lengths.shape != x.shape[:-1]:
            raise ValueError(
                f"lengths must have shape {tuple(x.shape[:-1])}, not {tuple(lengths.shape)}"
            )
        if lengths.dtype.is_floating_point or lengths.dtype == torch.bool:
            raise TypeError(f"lengths must be an integer tensor, not {lengths.dtype}")
        lengths = lengths.reshape(-1).to(device=x.device, dtype=torch.int64)
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError(f"mask must have shape {tuple(x.shape)}, not {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be boolean, not {mask.dtype}")
        mask = mask.reshape(-1, width).to(device=x.device)
    return rows, other, lengths, mask, width


def _keep(rows: Tensor, lengths: Tensor | None, mask: Tensor | None) -> Tensor | None:
    """The unmasked positions as a boolean ``[R, W]``, or ``None`` with no mask."""
    if lengths is not None:
        cols = torch.arange(rows.shape[-1], device=rows.device)
        return cols.unsqueeze(0) < lengths.clamp(0, rows.shape[-1]).unsqueeze(1)
    return mask


def _tiled_sum(terms: Tensor) -> Tensor:
    """Each row of ``terms [R, W]`` summed in the fixed order, using elementwise ops only."""
    count, width = terms.shape
    tiles = max(1, -(-width // BLOCK))
    padded = terms.new_zeros(count, tiles * BLOCK)
    padded[:, :width] = terms
    padded = padded.view(count, tiles, BLOCK)
    acc = padded[:, 0].clone()
    for t in range(1, tiles):
        acc = acc + padded[:, t]
    half = BLOCK
    while half > 1:
        half //= 2
        acc = acc[:, :half] + acc[:, half:]
    return acc[:, 0]


def _torch_reduce(
    op: int,
    rows: Tensor,
    other: Tensor | None,
    lengths: Tensor | None,
    mask: Tensor | None,
) -> Tensor:
    keep = _keep(rows, lengths, mask)
    zero = rows.new_zeros(())
    x = rows if keep is None else torch.where(keep, rows, zero)
    if op == _DOT:
        assert other is not None
        y = other if keep is None else torch.where(keep, other, zero)
        terms = x * y
    elif op in (_SUMSQ, _NORM):
        terms = x * x
    else:
        terms = x
    total = _tiled_sum(terms)
    if op in (_SUM, _SUMSQ, _DOT):
        return total
    if op == _NORM:
        return total.sqrt()
    # Integer counting is exact in any order, so a plain reduction is safe here.
    if keep is None:
        n = torch.full_like(total, max(rows.shape[-1], 1))
    else:
        n = keep.sum(dim=-1).clamp_min(1).to(rows.dtype)
    mean = total / n
    if op == _MEAN:
        return mean
    d = x - mean.unsqueeze(-1)
    if keep is not None:
        d = torch.where(keep, d, zero)
    return (_tiled_sum(d * d) / n).sqrt()


def _triton_reduce(
    op: int,
    rows: Tensor,
    other: Tensor | None,
    lengths: Tensor | None,
    mask: Tensor | None,
    width: int,
) -> Tensor:
    count = rows.shape[0]
    out = torch.empty(count, dtype=rows.dtype, device=rows.device)
    if count == 0:
        return out
    if width == 0:
        return out.zero_()
    x = _aligned_contiguous(rows)
    y = _aligned_contiguous(other) if other is not None else x
    if lengths is not None:
        kind, len_arg, mask_arg = _MASK_LENGTHS, _aligned_contiguous(lengths.to(torch.int32)), x
    elif mask is not None:
        kind, len_arg, mask_arg = _MASK_BOOL, x, _aligned_contiguous(mask.to(torch.int8))
    else:
        kind, len_arg, mask_arg = _MASK_NONE, x, x
    with torch.cuda.device(rows.device):
        _row_reduce_kernel[(count,)](
            x,
            y,
            len_arg,
            mask_arg,
            out,
            width,
            OP=op,
            MASK_KIND=kind,
            BLOCK_W=BLOCK,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return out


def _reduce(
    op: int,
    x: Tensor,
    y: Tensor | None,
    lengths: Tensor | None,
    mask: Tensor | None,
    backend: Backend,
) -> Tensor:
    rows, other, lengths, mask, width = _prepare(x, y, lengths, mask)
    use_triton = backend == "triton" or (
        backend == "auto" and x.is_cuda and HAVE_TRITON
    )
    if use_triton:
        if not HAVE_TRITON:
            raise RuntimeError("backend='triton' needs triton importable")
        if not x.is_cuda:
            raise RuntimeError("backend='triton' needs a CUDA tensor")
        out = _triton_reduce(op, rows, other, lengths, mask, width)
    elif backend in ("auto", "torch"):
        out = _torch_reduce(op, rows, other, lengths, mask)
    else:
        raise ValueError(f"backend must be 'auto', 'triton' or 'torch', not {backend!r}")
    return out.reshape(x.shape[:-1])


def row_sum(
    x: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The sum of each row's unmasked values."""
    return _reduce(_SUM, x, None, lengths, mask, backend)


def row_sumsq(
    x: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The sum of each row's squared unmasked values."""
    return _reduce(_SUMSQ, x, None, lengths, mask, backend)


def row_dot(
    a: Tensor, b: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The dot product of each row of ``a`` with the same row of ``b`` over unmasked positions."""
    return _reduce(_DOT, a, b, lengths, mask, backend)


def row_mean(
    x: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The mean of each row's unmasked values; zero for a row with none."""
    return _reduce(_MEAN, x, None, lengths, mask, backend)


def row_norm(
    x: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The Euclidean norm of each row's unmasked values."""
    return _reduce(_NORM, x, None, lengths, mask, backend)


def row_std(
    x: Tensor, *, lengths: Tensor | None = None, mask: Tensor | None = None,
    backend: Backend = "auto",
) -> Tensor:
    """The population standard deviation (``correction=0``) of each row's unmasked values.

    Two passes over the row: the mean, then the mean squared deviation from it.
    Zero for a row with no unmasked values.
    """
    return _reduce(_STD, x, None, lengths, mask, backend)


__all__ = (
    "BLOCK",
    "HAVE_TRITON",
    "NUM_STAGES",
    "NUM_WARPS",
    "SUPPORTED_DTYPES",
    "row_dot",
    "row_mean",
    "row_norm",
    "row_std",
    "row_sum",
    "row_sumsq",
)
