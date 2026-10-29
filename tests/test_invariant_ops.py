"""The invariant row reductions: a row's bytes alone equal its bytes in any batch.

The test of record places random rows alone and inside batches of 2, 8, 32 and 128
random mates (other values, other scales, other mask lengths) at random positions,
and requires every primitive to return the row's bytes unchanged. It runs on the
torch implementation everywhere and on the Triton kernel where a GPU is present.
The same inputs in two fresh processes, each compiling its own kernels, give the
same bytes. The values agree with torch's own reductions to a tolerance scaled by
the size of the terms summed.

The falsifier shows the premise on a GPU: torch's ``x.sum(-1)`` gives some row
different bytes alone than inside a batch. Without a GPU, the cases that launch the
kernel skip and say so.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from anamnesis.extraction.fast import invariant_ops
from anamnesis.extraction.fast.invariant_ops import (
    BLOCK,
    row_dot,
    row_mean,
    row_norm,
    row_std,
    row_sum,
    row_sumsq,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_gpu = pytest.mark.skipif(
    not (invariant_ops.HAVE_TRITON and torch.cuda.is_available()),
    reason="no GPU with triton: the Triton row kernel needs a CUDA device",
)

PRIMITIVES: dict[str, Callable[..., torch.Tensor]] = {
    "row_sum": row_sum,
    "row_sumsq": row_sumsq,
    "row_dot": row_dot,
    "row_mean": row_mean,
    "row_norm": row_norm,
    "row_std": row_std,
}
MASK_KINDS = ("none", "lengths", "mask")
BATCH_SIZES = (2, 8, 32, 128)
# Widths below one tile, at one tile, across a ragged tile edge, and several tiles.
WIDTHS = (37, BLOCK, 2048, 3 * BLOCK + 129)
TOLERANCE = {torch.float32: 1e-6, torch.float64: 1e-12}
BACKENDS_CPU = ("torch",)
BACKENDS_GPU = ("triton", "torch")


def _call(name: str, x, y, lengths, mask, backend: str) -> torch.Tensor:
    fn = PRIMITIVES[name]
    kwargs = {"lengths": lengths, "mask": mask, "backend": backend}
    return fn(x, y, **kwargs) if name == "row_dot" else fn(x, **kwargs)


def _random_rows(count: int, width: int, dtype, device, gen: torch.Generator):
    """``count`` rows of mixed scale and offset, a partner for ``row_dot``, and both mask kinds."""
    scale = torch.pow(10.0, torch.randint(-3, 4, (count, 1), generator=gen).double())
    offset = torch.randn(count, 1, generator=gen, dtype=torch.float64)
    x = (torch.randn(count, width, generator=gen, dtype=torch.float64) + offset) * scale
    y = torch.randn(count, width, generator=gen, dtype=torch.float64)
    lengths = torch.randint(0, width + 1, (count,), generator=gen)
    # The edges are the cases a fixed tile order is most likely to get wrong.
    lengths[: min(count, 2)] = torch.tensor([0, width])[: min(count, 2)]
    density = torch.rand(count, 1, generator=gen, dtype=torch.float64)
    mask = torch.rand(count, width, generator=gen, dtype=torch.float64) < density
    return (
        x.to(dtype).to(device),
        y.to(dtype).to(device),
        lengths.to(device),
        mask.to(device),
    )


def _masks(kind: str, lengths, mask):
    return (lengths if kind == "lengths" else None, mask if kind == "mask" else None)


def _check_batchmate_invariance(device: str, backend: str, dtype, width: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    targets = 4
    tx, ty, tlen, tmask = _random_rows(targets, width, dtype, device, gen)
    for kind in MASK_KINDS:
        for name in PRIMITIVES:
            for t in range(targets):
                alone_len, alone_mask = _masks(kind, tlen[t : t + 1], tmask[t : t + 1])
                alone = _call(name, tx[t : t + 1], ty[t : t + 1], alone_len, alone_mask, backend)
                alone_bytes = alone.cpu().numpy().tobytes()
                for batch in BATCH_SIZES:
                    x, y, lengths, mask = _random_rows(batch, width, dtype, device, gen)
                    pos = int(torch.randint(0, batch, (1,), generator=gen))
                    x[pos], y[pos], lengths[pos], mask[pos] = tx[t], ty[t], tlen[t], tmask[t]
                    blen, bmask = _masks(kind, lengths, mask)
                    got = _call(name, x, y, blen, bmask, backend)[pos : pos + 1]
                    assert got.cpu().numpy().tobytes() == alone_bytes, (
                        f"{name} ({backend}, {dtype}, W={width}, mask={kind}) moved for target {t} "
                        f"at position {pos} of {batch}: alone {alone.item()!r}, batched {got.item()!r}"
                    )


def _reference(name: str, x, y, keep) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch's own reduction of each row's kept values, and the scale its error is measured against."""
    xk = torch.where(keep, x, torch.zeros_like(x))
    n = keep.sum(-1).clamp_min(1).to(x.dtype)
    if name == "row_sum":
        return xk.sum(-1), xk.abs().sum(-1)
    if name == "row_sumsq":
        value = xk.square().sum(-1)
        return value, value
    if name == "row_dot":
        prod = xk * torch.where(keep, y, torch.zeros_like(y))
        return prod.sum(-1), prod.abs().sum(-1)
    if name == "row_mean":
        return xk.sum(-1) / n, xk.abs().sum(-1) / n
    if name == "row_norm":
        value = torch.linalg.vector_norm(xk, dim=-1)
        return value, value
    stds = []
    for row, kept in zip(x, keep, strict=True):
        values = row[kept]
        stds.append(values.std(correction=0) if values.numel() else values.new_zeros(()))
    # Subtracting the mean costs rounding at the size of the values, not of the deviation.
    return torch.stack(stds), (xk.square().sum(-1) / n).sqrt()


def _check_against_torch(device: str, backend: str, dtype, width: int) -> None:
    gen = torch.Generator().manual_seed(width)
    x, y, lengths, mask = _random_rows(64, width, dtype, device, gen)
    cols = torch.arange(width, device=device)
    keeps = {
        "none": torch.ones_like(mask),
        "lengths": cols.unsqueeze(0) < lengths.unsqueeze(1),
        "mask": mask,
    }
    rtol = TOLERANCE[dtype]
    for kind in MASK_KINDS:
        blen, bmask = _masks(kind, lengths, mask)
        for name in PRIMITIVES:
            got = _call(name, x, y, blen, bmask, backend)
            want, scale = _reference(name, x, y, keeps[kind])
            excess = (got - want).abs() - rtol * scale
            assert bool((excess <= 0).all()), (
                f"{name} ({backend}, {dtype}, W={width}, mask={kind}) is off torch's reduction "
                f"by up to {float(((got - want).abs() / scale.clamp_min(1e-300)).max()):.3g} "
                f"of the term scale, over the {rtol} allowed"
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("width", WIDTHS)
def test_torch_rows_are_batchmate_invariant(dtype, width):
    _check_batchmate_invariance("cpu", "torch", dtype, width, seed=width)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("width", WIDTHS)
def test_torch_rows_agree_with_torch_reductions(dtype, width):
    _check_against_torch("cpu", "torch", dtype, width)


def test_torch_rows_are_invariant_where_plain_cpu_sum_is_not():
    """A row long enough that torch's CPU ``sum(-1)`` splits it across threads.

    At this width a row's plain sum alone and in a batch of two can differ by an ulp
    (the split follows the number of outputs); the fixed order does not move.
    """
    width = 65536
    gen = torch.Generator().manual_seed(5)
    tx, ty, tlen, _ = _random_rows(3, width, torch.float64, "cpu", gen)
    for name in PRIMITIVES:
        for t in range(3):
            for kind_len in (None, tlen[t : t + 1]):
                alone = _call(name, tx[t : t + 1], ty[t : t + 1], kind_len, None, "torch")
                for batch in (2, 8):
                    x, y, lengths, _ = _random_rows(batch, width, torch.float64, "cpu", gen)
                    pos = int(torch.randint(0, batch, (1,), generator=gen))
                    x[pos], y[pos], lengths[pos] = tx[t], ty[t], tlen[t]
                    blen = None if kind_len is None else lengths
                    got = _call(name, x, y, blen, None, "torch")[pos : pos + 1]
                    assert got.numpy().tobytes() == alone.numpy().tobytes(), (name, t, batch)


def test_leading_dimensions_are_rows():
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, 40, generator=gen, dtype=torch.float64)
    lengths = torch.randint(0, 41, (3, 5), generator=gen)
    got = row_mean(x, lengths=lengths, backend="torch")
    flat = row_mean(x.reshape(15, 40), lengths=lengths.reshape(15), backend="torch")
    assert got.shape == (3, 5)
    assert got.reshape(15).numpy().tobytes() == flat.numpy().tobytes()


def test_a_row_with_nothing_unmasked_has_zero_mean_and_deviation():
    x = torch.randn(2, 50, dtype=torch.float64)
    lengths = torch.tensor([0, 50])
    for fn in (row_sum, row_mean, row_std, row_norm, row_sumsq):
        assert fn(x, lengths=lengths, backend="torch")[0].item() == 0.0
    assert row_mean(x, mask=torch.zeros_like(x, dtype=torch.bool), backend="torch").eq(0).all()


def test_refusals():
    x = torch.randn(4, 10)
    with pytest.raises(TypeError):
        row_sum(x.half())
    with pytest.raises(ValueError):
        row_sum(x, lengths=torch.full((4,), 3), mask=torch.ones(4, 10, dtype=torch.bool))
    with pytest.raises(ValueError):
        row_sum(x, lengths=torch.full((3,), 3))
    with pytest.raises(TypeError):
        row_sum(x, lengths=torch.full((4,), 3.0))
    with pytest.raises(TypeError):
        row_sum(x, mask=torch.ones(4, 10))
    with pytest.raises(ValueError):
        row_dot(x, torch.randn(4, 11))
    with pytest.raises(ValueError):
        row_sum(x, backend="numpy")
    with pytest.raises(RuntimeError):
        row_sum(x, backend="triton")


_RESTART_SCRIPT = """
import hashlib, sys, torch
from anamnesis.extraction.fast import invariant_ops as ops
device, backend = sys.argv[1], sys.argv[2]
gen = torch.Generator().manual_seed(1234)
digest = hashlib.sha256()
for dtype in (torch.float32, torch.float64):
    for width in (37, 1024, 3201):
        x = torch.randn(33, width, generator=gen, dtype=torch.float64).to(dtype).to(device)
        y = torch.randn(33, width, generator=gen, dtype=torch.float64).to(dtype).to(device)
        lengths = torch.randint(0, width + 1, (33,), generator=gen).to(device)
        mask = (torch.rand(33, width, generator=gen) < 0.6).to(device)
        for masks in ({}, {"lengths": lengths}, {"mask": mask}):
            for fn in (ops.row_sum, ops.row_sumsq, ops.row_mean, ops.row_norm, ops.row_std):
                digest.update(fn(x, backend=backend, **masks).cpu().numpy().tobytes())
            digest.update(ops.row_dot(x, y, backend=backend, **masks).cpu().numpy().tobytes())
print(digest.hexdigest())
"""


def _restart_digests(device: str, backend: str, tmp_path: Path) -> list[str]:
    digests = []
    for run in range(2):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
        )
        # A fresh kernel cache per process: each compiles its own kernel.
        cache = tmp_path / f"triton-cache-{run}"
        cache.mkdir()
        env["TRITON_CACHE_DIR"] = str(cache)
        result = subprocess.run(
            [sys.executable, "-c", _RESTART_SCRIPT, device, backend],
            env=env, capture_output=True, text=True, timeout=600, check=False,
        )
        assert result.returncode == 0, result.stderr
        digests.append(result.stdout.strip())
    return digests


def test_torch_rows_are_restart_invariant(tmp_path):
    first, second = _restart_digests("cpu", "torch", tmp_path)
    assert first == second and len(first) == 64


@needs_gpu
@pytest.mark.parametrize("backend", BACKENDS_GPU)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("width", WIDTHS)
def test_gpu_rows_are_batchmate_invariant(backend, dtype, width):
    _check_batchmate_invariance("cuda", backend, dtype, width, seed=width + 1)


@needs_gpu
@pytest.mark.parametrize("backend", BACKENDS_GPU)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("width", WIDTHS)
def test_gpu_rows_agree_with_torch_reductions(backend, dtype, width):
    _check_against_torch("cuda", backend, dtype, width)


@needs_gpu
@pytest.mark.parametrize("backend", BACKENDS_GPU)
def test_gpu_rows_are_restart_invariant(backend, tmp_path):
    first, second = _restart_digests("cuda", backend, tmp_path)
    assert first == second and len(first) == 64


@needs_gpu
def test_gpu_unaligned_views_give_the_aligned_bytes():
    gen = torch.Generator().manual_seed(3)
    width = 1031  # odd, so a one-row slice of float32 starts off a 16-byte boundary
    x = torch.randn(9, width, generator=gen).cuda()
    whole = row_sum(x)
    for r in range(9):
        view = x[r : r + 1]
        assert row_sum(view).cpu().numpy().tobytes() == whole[r : r + 1].cpu().numpy().tobytes()


def _plain_sum_moves(dtype, width: int, batch: int, gen: torch.Generator) -> int:
    """How many of 8 target rows torch's own ``sum(-1)`` gives different bytes alone and in a batch."""
    moved = 0
    for _ in range(8):
        x = torch.randn(batch, width, generator=gen, dtype=torch.float64).to(dtype).cuda()
        pos = int(torch.randint(0, batch, (1,), generator=gen))
        alone = x[pos : pos + 1].clone().sum(-1)
        batched = x.sum(-1)[pos : pos + 1]
        moved += alone.cpu().numpy().tobytes() != batched.cpu().numpy().tobytes()
    return moved


@needs_gpu
def test_falsifier_plain_torch_sum_is_not_batchmate_invariant():
    """The premise: torch's reduction gives a row different bytes depending on its batchmates."""
    gen = torch.Generator().manual_seed(11)
    report = []
    for dtype in (torch.float32, torch.float64):
        for width in (256, 1024, 2048, 4096, 16384, 65536):
            for batch in BATCH_SIZES:
                moved = _plain_sum_moves(dtype, width, batch, gen)
                report.append((str(dtype).removeprefix("torch."), width, batch, moved))
    print("\nplain x.sum(-1), rows of 8 that moved alone vs batched (dtype, W, B, moved):")
    for row in report:
        print("  ", *row)
    assert any(moved for *_, moved in report), (
        "torch's x.sum(-1) gave every row the same bytes alone and batched on this GPU; "
        "the premise that plain reductions are batch-variant needs re-checking here"
    )
    # The same cases through the invariant kernel move nothing.
    gen = torch.Generator().manual_seed(11)
    for dtype in (torch.float32, torch.float64):
        for width in (256, 2048, 65536):
            for batch in BATCH_SIZES:
                x = torch.randn(batch, width, generator=gen, dtype=torch.float64).to(dtype).cuda()
                alone = torch.cat([row_sum(x[r : r + 1].clone()) for r in range(batch)])
                assert alone.cpu().numpy().tobytes() == row_sum(x).cpu().numpy().tobytes()


@needs_gpu
def test_gpu_one_launch_per_batch_and_speed():
    """A 128-row reduction is one kernel launch, and faster than 128 single-row launches."""
    x = torch.randn(128, 2048, dtype=torch.float64, device="cuda")
    row_sum(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        row_sum(x)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names = sorted({e.name for e in kernels})
    assert len(kernels) == 1 and "_row_reduce_kernel" in names[0], names

    def timed(fn, repeats: int = 200) -> float:
        for _ in range(10):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats * 1000.0  # microseconds

    rows = [x[r : r + 1] for r in range(128)]
    batched = timed(lambda: row_sum(x))
    separate = timed(lambda: [row_sum(r) for r in rows], repeats=20)
    plain = timed(lambda: x.sum(-1))
    std_batched = timed(lambda: row_std(x))
    print(
        f"\nB=128 W=2048 fp64: row_sum one launch {batched:.1f} us; "
        f"128 single-row launches {separate:.1f} us ({separate / batched:.0f}x); "
        f"torch x.sum(-1) {plain:.1f} us; row_std one launch {std_batched:.1f} us"
    )
    assert batched < separate / 10

