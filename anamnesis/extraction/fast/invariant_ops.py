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
  is fixed by that compiled layout. The kernel reads each operand (``x``, ``y``,
  ``lengths``, ``mask``) through its own strides over up to three leading
  dimensions, so an expanded, sliced or transposed view is reduced where it lies,
  with no copy to lay it out; a row reads the same values at the same positions as
  it would from a contiguous copy, so its bytes are the same. An operand whose
  address is not 16-byte aligned, or whose leading dimensions do not merge into
  three, is copied to an aligned contiguous buffer first, so the compiler's
  alignment specialisation never follows a caller's view.
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

GRAM_BLOCK_M = 32
"""Output tile edge of :func:`row_gram`'s kernel. Fixed: an entry's order is a function of it."""

GRAM_BLOCK_K = 32
"""Width chunk of :func:`row_gram`'s kernel. Fixed for the same reason."""

GRAM_USE_DOT = True
"""Whether :func:`row_gram`'s kernel accumulates each chunk with ``tl.dot`` (IEEE precision)
or with an explicit multiply and sum. Fixed for the same reason."""

Backend = Literal["auto", "triton", "torch"]

# Operation codes shared by the kernel and the torch implementation.
_SUM, _SUMSQ, _DOT, _MEAN, _NORM, _STD = range(6)
_MASK_NONE, _MASK_LENGTHS, _MASK_BOOL = range(3)
_ALIGN_BYTES = 16


if HAVE_TRITON:

    @triton.jit
    def _div(a, b, IS_FP64: tl.constexpr):
        # float64 division and square root round correctly as written; float32 needs
        # the _rn forms, or the compiler emits the approximate instructions.
        if IS_FP64:
            return a / b
        else:
            return tl.div_rn(a, b)

    @triton.jit
    def _sqrt(a, IS_FP64: tl.constexpr):
        if IS_FP64:
            return tl.sqrt(a)
        else:
            return tl.sqrt_rn(a)

    @triton.jit(do_not_specialize=["D1", "D2", "SX0", "SX1", "SX2", "SY0", "SY1", "SY2",
                                   "SL0", "SL1", "SL2", "SM0", "SM1", "SM2"])
    def _row_reduce_kernel(
        x_ptr,
        y_ptr,
        len_ptr,
        mask_ptr,
        out_ptr,
        W,
        D1,
        D2,
        SX0, SX1, SX2,
        SY0, SY1, SY2,
        SL0, SL1, SL2,
        SM0, SM1, SM2,
        OP: tl.constexpr,
        MASK_KIND: tl.constexpr,
        BLOCK_W: tl.constexpr,
        IS_FP64: tl.constexpr,
        ROW_ALIGN: tl.constexpr,
    ):
        # Row ``row`` of the [D0, D1, D2] leading index, each operand read through its own
        # strides over the leading dimensions (a stride of zero repeats it) and contiguously
        # along the row; the values a row reads, and the order it adds them in, are those of
        # the same row laid out contiguously. The strides are not specialised on: the
        # compiler's knowledge of a row's alignment is ROW_ALIGN, a function of W alone, so
        # the kernel (and with it the fold tree) is the same for every layout of one width.
        row = tl.program_id(0).to(tl.int64)
        i2 = row % D2
        i1 = (row // D2) % D1
        i0 = row // (D1 * D2)
        x_row = tl.multiple_of(i0 * SX0 + i1 * SX1 + i2 * SX2, ROW_ALIGN)
        y_row = tl.multiple_of(i0 * SY0 + i1 * SY1 + i2 * SY2, ROW_ALIGN)
        m_row = tl.multiple_of(i0 * SM0 + i1 * SM1 + i2 * SM2, ROW_ALIGN)
        offs = tl.arange(0, BLOCK_W)
        if MASK_KIND == 1:
            n = tl.load(len_ptr + i0 * SL0 + i1 * SL1 + i2 * SL2).to(tl.int64)
            limit = tl.minimum(tl.maximum(n, 0), W)
        else:
            limit = W

        acc = tl.zeros([BLOCK_W], dtype=out_ptr.dtype.element_ty)
        count = tl.zeros([BLOCK_W], dtype=tl.int32)
        for start in range(0, W, BLOCK_W):
            cols = start + offs
            keep = cols < limit
            if MASK_KIND == 2:
                keep = keep & (tl.load(mask_ptr + m_row + cols, mask=cols < W, other=0) != 0)
            x = tl.load(x_ptr + x_row + cols, mask=keep, other=0.0)
            if OP == 2:
                y = tl.load(y_ptr + y_row + cols, mask=keep, other=0.0)
                acc += x * y
            elif OP == 1 or OP == 4:
                acc += x * x
            else:
                acc += x
            count += keep.to(tl.int32)

        total = tl.sum(acc, axis=0)
        n = tl.maximum(tl.sum(count, axis=0), 1).to(total.dtype)
        if OP == 3:
            total = _div(total, n, IS_FP64)
        elif OP == 4:
            total = _sqrt(total, IS_FP64)
        elif OP == 5:
            mean = _div(total, n, IS_FP64)
            acc = tl.zeros([BLOCK_W], dtype=out_ptr.dtype.element_ty)
            for start in range(0, W, BLOCK_W):
                cols = start + offs
                keep = cols < limit
                if MASK_KIND == 2:
                    keep = keep & (
                        tl.load(mask_ptr + m_row + cols, mask=cols < W, other=0) != 0
                    )
                x = tl.load(x_ptr + x_row + cols, mask=keep, other=0.0)
                d = tl.where(keep, x - mean, 0.0)
                acc += d * d
            total = _sqrt(_div(tl.sum(acc, axis=0), n, IS_FP64), IS_FP64)
        tl.store(out_ptr + row, total)


if HAVE_TRITON:

    @triton.jit
    def _gram_kernel(
        x_ptr,
        len_ptr,
        out_ptr,
        M,
        W,
        HAS_LENGTHS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        # One program per (batch entry b, tile pair ti <= tj): the BLOCK_M x BLOCK_M block of
        # x[b] x[b]^T at rows ti and columns tj, accumulated over the width in BLOCK_K chunks
        # in a fixed order, then written to both (ti, tj) and its mirror. The order depends
        # only on the tile, so an entry is the same bytes alone and inside any batch.
        b = tl.program_id(0).to(tl.int64)
        ti = tl.program_id(1)
        tj = tl.program_id(2)
        if ti > tj:
            return
        rows_i = ti * BLOCK_M + tl.arange(0, BLOCK_M)
        rows_j = tj * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = tl.arange(0, BLOCK_K)
        if HAS_LENGTHS:
            limit = tl.minimum(tl.maximum(tl.load(len_ptr + b), 0), W)
        else:
            limit = W
        base = b * M * W
        acc = tl.zeros([BLOCK_M, BLOCK_M], dtype=out_ptr.dtype.element_ty)
        for start in range(0, W, BLOCK_K):
            cols = start + ks
            keep = cols < limit
            a = tl.load(x_ptr + base + rows_i[:, None] * W + cols[None, :],
                        mask=(rows_i[:, None] < M) & keep[None, :], other=0.0)
            c = tl.load(x_ptr + base + rows_j[:, None] * W + cols[None, :],
                        mask=(rows_j[:, None] < M) & keep[None, :], other=0.0)
            if USE_DOT:
                acc += tl.dot(a, tl.trans(c), input_precision="ieee")
            else:
                acc += tl.sum(a[:, None, :] * c[None, :, :], axis=2)
        out = out_ptr + b * M * M
        inside = (rows_i[:, None] < M) & (rows_j[None, :] < M)
        tl.store(out + rows_i[:, None] * M + rows_j[None, :], acc, mask=inside)
        tl.store(out + rows_j[None, :] * M + rows_i[:, None], acc, mask=inside)

def _aligned_contiguous(t: Tensor) -> Tensor:
    """``t`` as a contiguous buffer whose address is 16-byte aligned.

    The Triton compiler specialises on pointer alignment; a buffer that is aligned
    on every call keeps the compiled kernel, and with it the fold tree, the same.
    """
    t = t.contiguous()
    if t.numel() and t.data_ptr() % _ALIGN_BYTES:
        t = t.clone()
    return t


def _collapsed_layout(shape: tuple[int, ...], operands: list[tuple[int, ...]]
                      ) -> tuple[tuple[int, ...], list[tuple[int, ...]]] | None:
    """``shape`` (the leading dimensions) and each operand's strides over it, with adjacent
    dimensions merged where every operand allows, padded at the front to three; ``None`` when
    more than three remain."""
    dims = [(size, [st[i] for st in operands]) for i, size in enumerate(shape) if size != 1]
    merged: list[tuple[int, list[int]]] = []
    for size, strides in dims:
        if merged:
            prev_size, prev = merged[-1]
            if all(p == st * size for p, st in zip(prev, strides, strict=True)):
                merged[-1] = (prev_size * size, strides)
                continue
        merged.append((size, strides))
    if len(merged) > 3:
        return None
    merged = [(1, [0] * len(operands))] * (3 - len(merged)) + merged
    return (tuple(m[0] for m in merged),
            [tuple(m[1][k] for m in merged) for k in range(len(operands))])


def _aligned(t: Tensor) -> bool:
    return t.numel() == 0 or t.data_ptr() % _ALIGN_BYTES == 0


_ROW_ALIGN_ELEMENTS = 16


def _row_align(width: int) -> int:
    """What the kernel is told every row's first element is a multiple of (in elements): 16
    when the width is, else 1. A function of the width alone, so one width compiles one
    kernel whatever the operands' layout, as a contiguous ``[rows, W]`` buffer does."""
    return _ROW_ALIGN_ELEMENTS if width % _ROW_ALIGN_ELEMENTS == 0 else 1


def _check(x: Tensor, y: Tensor | None, lengths: Tensor | None, mask: Tensor | None) -> None:
    """Refuse operands the primitives do not take; nothing is copied."""
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
    if lengths is not None:
        if lengths.shape != x.shape[:-1]:
            raise ValueError(
                f"lengths must have shape {tuple(x.shape[:-1])}, not {tuple(lengths.shape)}"
            )
        if lengths.dtype.is_floating_point or lengths.dtype == torch.bool:
            raise TypeError(f"lengths must be an integer tensor, not {lengths.dtype}")
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError(f"mask must have shape {tuple(x.shape)}, not {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be boolean, not {mask.dtype}")


def _prepare(
    x: Tensor, y: Tensor | None, lengths: Tensor | None, mask: Tensor | None
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, int]:
    """The operands as ``[rows, W]`` (copied where a view cannot be), after :func:`_check`."""
    _check(x, y, lengths, mask)
    width = x.shape[-1]
    rows = x.reshape(-1, width)
    other = y.reshape(-1, width) if y is not None else None
    if lengths is not None:
        lengths = lengths.reshape(-1).to(device=x.device, dtype=torch.int64)
    if mask is not None:
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


def _triton_strided(
    op: int, x: Tensor, y: Tensor | None, lengths: Tensor | None, mask: Tensor | None
) -> Tensor | None:
    """The reduction launched on the operands as they are laid out (expanded, sliced or
    transposed views included), or ``None`` when their layout needs the contiguous path:
    more than three leading dimensions after merging, or an operand not 16-byte aligned."""
    width = x.shape[-1]
    lead = tuple(x.shape[:-1])
    if lengths is not None:
        lengths = lengths.to(device=x.device)
        if lengths.dtype not in (torch.int32, torch.int64):
            lengths = lengths.to(torch.int64)
    if mask is not None:
        mask = mask.to(device=x.device).view(torch.int8)
    operands = [x] + ([y] if y is not None else []) + ([mask] if mask is not None else [])
    if not all(_aligned(t) for t in operands + ([lengths] if lengths is not None else [])):
        return None
    if width > 1 and any(t.stride(-1) != 1 for t in operands):
        return None
    align = _row_align(width)
    if any(st % align for t in operands for st in t.stride()[:-1]):
        return None
    strides = [tuple(t.stride()[:-1]) for t in operands]
    if lengths is not None:
        strides.append(tuple(lengths.stride()))
    layout = _collapsed_layout(lead, strides)
    if layout is None:
        return None
    (_, d1, d2), lead_strides = layout
    count = 1
    for size in lead:
        count *= size
    out = torch.empty(count, dtype=x.dtype, device=x.device)
    if count == 0:
        return out.reshape(lead)
    if width == 0:
        return out.zero_().reshape(lead)
    sx = lead_strides[0]
    k = 1
    if y is not None:
        sy, k = lead_strides[k], k + 1
    else:
        sy = sx
    if mask is not None:
        sm, k = lead_strides[k], k + 1
        kind = _MASK_BOOL
    else:
        sm = (0, 0, 0)
        kind = _MASK_LENGTHS if lengths is not None else _MASK_NONE
    sl = lead_strides[k] if lengths is not None else (0, 0, 0)
    with torch.cuda.device(x.device):
        _row_reduce_kernel[(count,)](
            x,
            y if y is not None else x,
            lengths if lengths is not None else x,
            mask if mask is not None else x,
            out,
            width,
            d1,
            d2,
            *sx,
            *sy,
            *sl,
            *sm,
            OP=op,
            MASK_KIND=kind,
            BLOCK_W=BLOCK,
            IS_FP64=x.dtype == torch.float64,
            ROW_ALIGN=align,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return out.reshape(lead)


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
            1,
            1,
            width, 0, 0,
            width, 0, 0,
            1, 0, 0,
            width, 0, 0,
            OP=op,
            MASK_KIND=kind,
            BLOCK_W=BLOCK,
            IS_FP64=rows.dtype == torch.float64,
            ROW_ALIGN=_row_align(width),
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
    use_triton = backend == "triton" or (
        backend == "auto" and x.is_cuda and HAVE_TRITON
    )
    if use_triton:
        if not HAVE_TRITON:
            raise RuntimeError("backend='triton' needs triton importable")
        if not x.is_cuda:
            raise RuntimeError("backend='triton' needs a CUDA tensor")
        _check(x, y, lengths, mask)
        strided = _triton_strided(op, x, y, lengths, mask)
        if strided is not None:
            return strided
    rows, other, lengths, mask, width = _prepare(x, y, lengths, mask)
    if use_triton:
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


def row_gram(
    x: Tensor, *, lengths: Tensor | None = None, backend: Backend = "auto",
    chunk_bytes: int = 1 << 30,
) -> Tensor:
    """``x x^T`` of each batch entry of ``x [B, m, n]``: ``[B, m, m]``.

    Entry ``(b, i, j)`` is the dot product of ``x[b, i]`` and ``x[b, j]`` over their
    unmasked positions, accumulated in a fixed order that depends only on the entry's
    tile, so it is batchmate- and restart-invariant like the other primitives, and the
    matrix is exactly symmetric. ``lengths`` (integer,
    shape ``[B]``) masks positions ``>= lengths[b]`` in every row of batch entry ``b``.

    On CUDA one program computes a ``GRAM_BLOCK_M``-square tile of entries from rows it
    loads once per width chunk, and only tiles on or above the diagonal are computed; the
    products are never stored, so the working set is the Gram matrix, not the
    ``[B, m, m, n]`` product. The torch path materializes the product in
    chunks of the first index (at most ``chunk_bytes`` each) and reduces it with
    :func:`row_sum`. The two agree to rounding, not to the byte.
    """
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"row_gram takes float32 or float64, not {x.dtype}")
    if x.dim() != 3:
        raise ValueError(f"row_gram takes [B, m, n], not {tuple(x.shape)}")
    batch, m, n = x.shape
    if lengths is not None:
        if lengths.shape != (batch,):
            raise ValueError(f"lengths must have shape ({batch},), not {tuple(lengths.shape)}")
        if lengths.dtype.is_floating_point or lengths.dtype == torch.bool:
            raise TypeError(f"lengths must be an integer tensor, not {lengths.dtype}")
    use_triton = backend == "triton" or (backend == "auto" and x.is_cuda and HAVE_TRITON)
    if use_triton:
        if not HAVE_TRITON:
            raise RuntimeError("backend='triton' needs triton importable")
        if not x.is_cuda:
            raise RuntimeError("backend='triton' needs a CUDA tensor")
        out = torch.empty(batch, m, m, dtype=x.dtype, device=x.device)
        if out.numel() == 0:
            return out
        if n == 0:
            return out.zero_()
        xs = _aligned_contiguous(x)
        if lengths is not None:
            has_len, len_arg = True, _aligned_contiguous(lengths.to(device=x.device, dtype=torch.int32))
        else:
            has_len, len_arg = False, xs
        tiles = -(-m // GRAM_BLOCK_M)
        with torch.cuda.device(x.device):
            _gram_kernel[(batch, tiles, tiles)](
                xs, len_arg, out, m, n,
                HAS_LENGTHS=has_len, BLOCK_M=GRAM_BLOCK_M, BLOCK_K=GRAM_BLOCK_K,
                USE_DOT=GRAM_USE_DOT, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
            )
        return out
    if backend not in ("auto", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton' or 'torch', not {backend!r}")
    chunk = max(1, min(m, chunk_bytes // max(1, batch * m * n * x.element_size())))
    parts = []
    for first in range(0, m, chunk):
        last = min(m, first + chunk)
        product = x[:, first:last, None, :] * x[:, None, :, :]
        lens = None if lengths is None else lengths.view(-1, 1, 1).expand(product.shape[:-1])
        parts.append(row_sum(product, lengths=lens, backend="torch"))
    return torch.cat(parts, dim=1)


__all__ = (
    "BLOCK",
    "HAVE_TRITON",
    "NUM_STAGES",
    "NUM_WARPS",
    "GRAM_BLOCK_K",
    "GRAM_BLOCK_M",
    "GRAM_USE_DOT",
    "SUPPORTED_DTYPES",
    "row_dot",
    "row_gram",
    "row_mean",
    "row_norm",
    "row_std",
    "row_sum",
    "row_sumsq",
)
