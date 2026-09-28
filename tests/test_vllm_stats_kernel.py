"""The instrumented statistics kernel: its host contract, and its references on a device.

The launcher refuses everything outside the lane's configuration rather than
computing it differently from the reference; the completion arithmetic in
`synthetic_attention.finalize_stats` turns the accumulators into the quantities the
references state; the synthetic geometry (paged packing, controlled scores, the rows
that are comparable at all) is proven on the host, so a device run only ever tests
the kernel itself; and the kernel is pinned to the exact upstream source it is an
instrumented copy of.

Without a GPU the kernel itself cannot run: the cases that launch it — the kernel
against the float64 materialized reference, padding lanes never emitting statistics,
and bit-identical statistics across chunk boundaries and batch composition — skip
and say so. The launch refusals need Triton importable but no device. The upstream
hash check needs vLLM installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from anamnesis.extraction.vllm import stats_kernel
from anamnesis.extraction.vllm.stats_kernel import (
    NUM_STATS,
    REGION_NAMES,
    STAT_FIELDS,
    UPSTREAM_KERNEL_SHA256,
    unified_attention_stats,
)
from synthetic_attention import (
    REGION_NAMES as REFERENCE_REGION_NAMES,
)
from synthetic_attention import (
    PagedBatch,
    SequenceSpec,
    bitwise_case,
    evaluate_case,
    finalize_stats,
    materialized_stats,
    paged_batch,
    reference_rows,
    run_batch,
    scored_sequence,
)

UPSTREAM_MODULE = "vllm.v1.attention.ops.triton_unified_attention"

needs_triton = pytest.mark.skipif(
    not stats_kernel.HAVE_TRITON,
    reason="no triton: the launcher's refusals run after it checks that triton is importable",
)

needs_gpu = pytest.mark.skipif(
    not (stats_kernel.HAVE_TRITON and torch.cuda.is_available()),
    reason="no GPU with triton: launching the instrumented kernel needs a CUDA device",
)


def small_batch(dtype: torch.dtype = torch.float16, device: str = "cpu") -> PagedBatch:
    return paged_batch([scored_sequence(
        context=20, steps=4, num_q=4, num_kv=2, head_size=64, dtype=dtype,
        prompt=8, recent_window=4, seed=1)], device=device)


def launch_args(batch: PagedBatch) -> dict:
    out = torch.empty_like(batch.q)
    stats = torch.empty(
        (batch.q.shape[0], batch.q.shape[1], NUM_STATS),
        dtype=torch.float32, device=batch.q.device)
    return dict(out=out, stats=stats, cu_seqlens_q=batch.cu_seqlens_q,
                seqused_k=batch.seqused_k, block_table=batch.block_table,
                softmax_scale=0.125, prefix_lengths=batch.prefix_lengths,
                recency_cutoffs=batch.recency_cutoffs)


def test_stat_fields_carry_the_reference_region_order() -> None:
    assert STAT_FIELDS[:3] == ("running_max", "mass", "entropy_moment")
    assert STAT_FIELDS[3:] == REGION_NAMES == REFERENCE_REGION_NAMES
    assert NUM_STATS == 9


def test_finalize_matches_the_completion_arithmetic() -> None:
    stats = torch.zeros((1, 1, NUM_STATS))
    stats[0, 0, 0] = 0.5  # running max
    stats[0, 0, 1] = 4.0  # mass
    stats[0, 0, 2] = 0.0  # entropy moment
    stats[0, 0, 3] = 1.0  # sink mass accumulator
    final = finalize_stats(stats)
    assert final["entropy"][0, 0] == pytest.approx(math.log(4.0))
    assert final["log_mass"][0, 0] == pytest.approx(math.log(4.0) + 0.5)
    assert final["sink"][0, 0] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="fields"):
        finalize_stats(torch.zeros((2, 3)))


@needs_triton
def test_unsupported_configurations_are_refused() -> None:
    batch = small_batch()
    args = launch_args(batch)
    with pytest.raises(ValueError, match="16-bit model dtypes"):
        unified_attention_stats(
            batch.q.float(), batch.k_cache.float(), batch.v_cache.float(),
            **dict(args, out=args["out"].float()))
    with pytest.raises(ValueError, match="quantized"):
        unified_attention_stats(
            batch.q, batch.k_cache.to(torch.bfloat16), batch.v_cache, **args)
    with pytest.raises(ValueError, match="contiguous last dimension"):
        bad = torch.empty(
            (batch.q.shape[0], batch.q.shape[1], NUM_STATS * 2), dtype=torch.float32)[..., ::2]
        unified_attention_stats(batch.q, batch.k_cache, batch.v_cache, **dict(args, stats=bad))
    with pytest.raises(ValueError, match="one entry per query token"):
        unified_attention_stats(
            batch.q, batch.k_cache, batch.v_cache,
            **dict(args, prefix_lengths=batch.prefix_lengths.long()))
    with pytest.raises(ValueError, match="multiple of KV heads"):
        unified_attention_stats(
            batch.q[:, :3], batch.k_cache, batch.v_cache,
            **dict(args, out=args["out"][:, :3], stats=args["stats"][:, :3]))


def test_paged_packing_reproduces_the_sequence() -> None:
    batch = small_batch()
    seq = batch.seqs[0]
    table = batch.block_table[0]
    gathered = [batch.k_cache[int(table[pos // 16]), pos % 16]
                for pos in range(seq.keys.shape[0])]
    assert torch.equal(torch.stack(gathered), seq.keys)
    assert int(batch.seqused_k[0]) == seq.keys.shape[0]
    assert batch.prefix_lengths.tolist() == [8, 8, 8, 8]
    # The recency cutoff is row_length - window per span row.
    assert batch.recency_cutoffs.tolist() == [17, 18, 19, 20]


def test_controlled_scores_give_exact_ties() -> None:
    scores = torch.zeros(24)
    scores[3] = 2.0
    scores[19] = 2.0
    seq = scored_sequence(
        context=20, steps=4, num_q=2, num_kv=2, head_size=64, dtype=torch.float16,
        prompt=8, recent_window=4, seed=2, key_scores=scores)
    rows = list(reference_rows(seq, scale=1.0))
    assert len(rows) == 4 * 2  # four span rows, two heads, all past the prompt
    for _, _, row_scores, regions in rows:
        assert row_scores[3] == row_scores[19]  # the tie survives rounding
        reference = materialized_stats(
            row_scores, torch.ones(len(row_scores), dtype=torch.bool), regions)
        assert math.isfinite(reference["entropy"])


def test_rows_inside_the_prompt_are_not_compared() -> None:
    seq = SequenceSpec(
        keys=torch.randn(12, 2, 64).half(), values=torch.randn(12, 2, 64).half(),
        queries=torch.randn(12, 2, 64).half(), prompt_length=5, recent_window=4)
    compared = {t for t, _, _, _ in reference_rows(seq, scale=1.0)}
    assert compared == {5, 6, 7, 8, 9, 10, 11}


def test_bitwise_case_reports_mismatches() -> None:
    a = torch.zeros((2, 1, NUM_STATS))
    b = a.clone()
    assert bitwise_case("same", a, b)["passed"]
    b[1, 0, 4] = 1.0
    verdict = bitwise_case("diff", a, b)
    assert not verdict["passed"]
    assert verdict["mismatched_entries"] == 1
    c = torch.zeros((2, 1, NUM_STATS))
    c[0, 0, 0] = float("nan")
    assert bitwise_case("nan-pair", c, c.clone())["passed"]
    with pytest.raises(ValueError, match="identical shapes"):
        bitwise_case("shape", a, a[:1])


def test_the_kernel_is_a_copy_of_the_installed_upstream_source() -> None:
    """The instrumented kernel keeps the upstream structure line for line, so the
    engine's own kernel file must be the exact bytes it was copied from."""
    if importlib.util.find_spec("vllm") is None:
        pytest.skip(
            "no vllm: install the engine version anamnesis.extraction.vllm.envelope pins "
            "to check its kernel source")
    spec = importlib.util.find_spec(UPSTREAM_MODULE)
    assert spec is not None and spec.origin is not None, UPSTREAM_MODULE
    digest = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    assert digest == UPSTREAM_KERNEL_SHA256


@needs_gpu
def test_kernel_matches_the_materialized_reference_on_device() -> None:
    result = evaluate_case(
        "device_smoke",
        [scored_sequence(context=100, steps=12, num_q=6, num_kv=2, head_size=64,
                         dtype=torch.float16, prompt=40, recent_window=8, seed=5)],
        scale=0.125, tolerance_class="standard", device="cuda:0")
    assert result["passed"], result


@needs_gpu
def test_padding_lanes_never_emit_statistics_on_device() -> None:
    batch = small_batch(device="cuda:0")
    args = launch_args(batch)
    unified_attention_stats(batch.q, batch.k_cache, batch.v_cache, **args)
    stats = args["stats"].cpu()
    assert not bool(torch.isnan(stats).any())  # every real row is written
    final = finalize_stats(stats)
    sums = sum(final[name] for name in ("prompt", "gen_third_1", "gen_third_2", "gen_third_3"))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@needs_gpu
def test_statistics_are_invariant_to_chunking_and_batch_composition_on_device() -> None:
    device = "cuda:0"
    generator = torch.Generator().manual_seed(31)
    total, split = 40, 32
    keys, values, queries = (
        torch.randn((total, 4, 64), generator=generator).half() for _ in range(3))
    whole = SequenceSpec(keys=keys, values=values, queries=queries,
                         prompt_length=10, recent_window=8)
    second = SequenceSpec(keys=keys, values=values, queries=queries[split:],
                          prompt_length=10, recent_window=8)
    _, stats_whole = run_batch(paged_batch([whole], device=device), scale=0.125)
    _, stats_second = run_batch(paged_batch([second], device=device), scale=0.125)
    chunked = bitwise_case("chunk_boundary", stats_whole[split:].cpu(), stats_second.cpu())
    assert chunked["passed"], chunked

    subject = scored_sequence(context=64, steps=6, num_q=4, num_kv=4, head_size=64,
                              dtype=torch.float16, prompt=20, recent_window=8, seed=32)
    neighbor = scored_sequence(context=48, steps=9, num_q=4, num_kv=4, head_size=64,
                               dtype=torch.float16, prompt=16, recent_window=8, seed=33)
    _, alone = run_batch(paged_batch([subject], device=device), scale=0.125)
    _, batched = run_batch(paged_batch([neighbor, subject], device=device), scale=0.125)
    composed = bitwise_case("batch_composition", alone.cpu(), batched[9:].cpu())
    assert composed["passed"], composed
