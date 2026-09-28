"""The bounded second pass: its refusals, and the coverage definition it applies.

Coverage finalization goes through `second_pass.coverage_reference`, which restates
the fast lane's coverage expression: a uniform head-mean row is strictly not covered,
one fp32 ulp above the threshold is covered, and the head mean is taken over heads
in fp32 before the double promotion. Those, and the launcher's slot validation, run
on the host.

Without a GPU the kernel itself cannot run: the cases that launch it — a step's row
plan driven through the second pass and checked against float64-materialized rows,
the series scratch reproducing the span scratch bitwise, the products of that
scratch, and the row-sum invariant tripping when the first-pass mass is corrupted —
skip and say so. The slot refusals need Triton importable but no device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.vllm import stats_kernel
from anamnesis.extraction.vllm.rows import request_row_schema, step_row_plan
from anamnesis.extraction.vllm.second_pass import (
    ROW_SUM_TOLERANCE,
    coverage_of_selected_rows,
    second_pass_rows,
)
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS
from anamnesis.extraction.vllm.step_products import agreement_pair, span_products
from synthetic_attention import PagedBatch, paged_batch, run_batch, scored_sequence

needs_triton = pytest.mark.skipif(
    not stats_kernel.HAVE_TRITON,
    reason="no triton: the launcher's refusals run after it checks that triton is importable",
)

needs_gpu = pytest.mark.skipif(
    not (stats_kernel.HAVE_TRITON and torch.cuda.is_available()),
    reason="no GPU with triton: launching the second-pass kernel needs a CUDA device",
)


def launch_args(batch: PagedBatch, *, row_slots: torch.Tensor, num_slots: int) -> dict:
    tokens, heads = batch.q.shape[0], batch.q.shape[1]
    return dict(
        stats=torch.zeros((tokens, heads, NUM_STATS), dtype=torch.float32),
        row_slots=row_slots, num_slots=num_slots, cu_seqlens_q=batch.cu_seqlens_q,
        seqused_k=batch.seqused_k, block_table=batch.block_table, softmax_scale=0.125,
        round_to_model_dtype=False)


@needs_triton
def test_slot_misuse_is_refused() -> None:
    batch = paged_batch([scored_sequence(
        context=20, steps=4, num_q=4, num_kv=2, head_size=64, dtype=torch.float16,
        prompt=8, recent_window=4, seed=71)], device="cpu")
    slots = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    with pytest.raises(ValueError, match="at most one row"):
        second_pass_rows(batch.q, batch.k_cache, **launch_args(
            batch, row_slots=torch.tensor([0, 0, -1, -1], dtype=torch.int32), num_slots=2))
    with pytest.raises(ValueError, match="exceeds the scratch"):
        second_pass_rows(batch.q, batch.k_cache, **launch_args(
            batch, row_slots=slots, num_slots=1))
    with pytest.raises(ValueError, match="no rows are selected"):
        second_pass_rows(batch.q, batch.k_cache, **launch_args(
            batch, row_slots=torch.full((4,), -1, dtype=torch.int32), num_slots=2))
    with pytest.raises(ValueError, match="one entry per query token"):
        second_pass_rows(batch.q, batch.k_cache, **launch_args(
            batch, row_slots=slots[:2], num_slots=2))


def test_coverage_finalization_is_the_banked_definition() -> None:
    # Uniform row: strictly not covered. One fp32 ulp above the threshold on a
    # single key: exactly that key is covered. The mean is over heads.
    length = 8
    uniform = torch.full((1, 4, length), 1 / length, dtype=torch.float32)
    lengths = torch.tensor([length])
    assert coverage_of_selected_rows(uniform, lengths).tolist() == [0.0]
    above = uniform.clone()
    above[0, :, 3] = torch.nextafter(torch.tensor(1 / length), torch.tensor(1.0))
    assert coverage_of_selected_rows(above, lengths).tolist() == [1 / length]
    with pytest.raises(ValueError, match="fit the scratch"):
        coverage_of_selected_rows(uniform, torch.tensor([length + 1]))
    with pytest.raises(ValueError, match="one row length"):
        coverage_of_selected_rows(uniform, torch.tensor([length, length]))


def test_the_mean_is_over_heads_before_the_threshold() -> None:
    # Two heads on opposite sides of the threshold: the mean decides, not either
    # head alone.
    length = 4
    scratch = torch.zeros((1, 2, length), dtype=torch.float32)
    scratch[0, 0] = torch.tensor([0.40, 0.30, 0.20, 0.10])
    scratch[0, 1] = torch.tensor([0.10, 0.30, 0.20, 0.40])
    coverage = coverage_of_selected_rows(scratch, torch.tensor([length]))
    # head mean = [0.25, 0.30, 0.20, 0.25]; the 1/4 threshold is strict
    assert coverage.tolist() == [0.25]


@needs_gpu
def test_plan_driven_second_pass_reduces_the_right_rows() -> None:
    """The row plan through the kernel, against materialized rows.

    Two sequences with a GQA ratio, one arriving mid-span: every span slot's scratch
    row must be the float64-materialized softmax row rounded to the model dtype (to
    rounding-scale tolerance), with exact zeros beyond the row's valid keys; the
    series launch must reproduce the span scratch's rows bitwise at the rows they
    share; and the recorded products must be the products of that scratch.
    """
    device = "cuda:0"
    scale = 0.125
    seqs = [
        scored_sequence(context=37, steps=21, num_q=8, num_kv=2, head_size=64,
                        dtype=torch.float16, prompt=30, recent_window=8, seed=311),
        scored_sequence(context=12, steps=30, num_q=8, num_kv=2, head_size=64,
                        dtype=torch.float16, prompt=9, recent_window=8, seed=312),
    ]
    batch = paged_batch(seqs, device=device)
    schemas, items, offset = {}, [], 0
    for index, seq in enumerate(seqs):
        rid = f"r{index}"
        schemas[rid] = request_row_schema(prompt_length=seq.prompt_length,
                                          end=seq.keys.shape[0])
        items.append(dict(request_id=rid, start=seq.context_length,
                          count=seq.queries.shape[0], offset=offset))
        offset += seq.queries.shape[0]
    plan = step_row_plan(items, schemas)
    _, stats = run_batch(batch, scale=scale)

    def launch(slots: np.ndarray, num_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
        return second_pass_rows(
            batch.q, batch.k_cache, stats=stats,
            row_slots=torch.from_numpy(slots).to(device), num_slots=num_slots,
            cu_seqlens_q=batch.cu_seqlens_q, seqused_k=batch.seqused_k,
            block_table=batch.block_table, softmax_scale=scale, round_to_model_dtype=True)

    scratch, rowsum = launch(plan.span_slots, len(plan.span_members))
    # Every span slot's row is the materialized rounded reference row.
    for slot in plan.span_slots:
        if slot < 0:
            continue
        rid, row = plan.span_members[slot]
        seq = seqs[int(rid[1:])]
        t = row + seq.prompt_length - seq.context_length
        length = seq.row_length(t)
        assert length == int(plan.span_lengths[slot])
        k64 = seq.keys.double().to(device)
        group = seq.queries.shape[1] // seq.keys.shape[1]
        for head in range(seq.queries.shape[1]):
            scores = scale * (k64[:length, head // group]
                              @ seq.queries[t, head].double().to(device))
            rounded = torch.softmax(scores, dim=-1).to(torch.float16).float()
            got = scratch[slot, head]
            assert float((got[:length] - rounded).abs().max()) < 1e-3
            assert float(got[length:].abs().max()) == 0.0

    # The series launch reproduces the span rows bitwise at the rows they share.
    assert len(plan.series_members)
    series_scratch, _ = launch(plan.series_slots, len(plan.series_members))
    shared = plan.series_slots >= 0
    span_of_series = torch.from_numpy(
        plan.span_slots[shared][np.argsort(plan.series_slots[shared])].astype(np.int64)
    ).to(device)
    assert torch.equal(series_scratch, scratch[span_of_series])

    # The recorded products are the products of exactly that scratch.
    def index(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).to(device)

    products = span_products(
        scratch, lengths=index(plan.span_lengths),
        agreement_index=index(plan.span_agreement),
        spectral_index=index(plan.span_spectral), decay_index=index(plan.span_decay),
        entropy_index=index(plan.span_entropy), prefixes=index(plan.span_prefixes),
        widths=index(plan.span_widths), rowsum=rowsum, slots=index(plan.span_slots))
    assert products["coverage"].shape == (len(plan.span_members),)
    assert bool(torch.isfinite(products["coverage"]).all())
    assert products["spectral_rows"].shape[0] == plan.span_spectral.size
    pair = agreement_pair(scratch[index(plan.span_agreement)])
    assert torch.allclose(products["h_mean"], pair[0], atol=1e-12)
    assert float(products["row_sum_worst"]) <= ROW_SUM_TOLERANCE


@needs_gpu
def test_second_pass_normalizes_with_the_first_pass_state() -> None:
    batch = paged_batch([scored_sequence(
        context=40, steps=6, num_q=6, num_kv=2, head_size=64, dtype=torch.float16,
        prompt=15, recent_window=8, seed=72)], device="cuda:0")
    _, stats = run_batch(batch, scale=0.125)
    slots = torch.arange(6, dtype=torch.int32, device="cuda:0")

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return second_pass_rows(
            batch.q, batch.k_cache, stats=stats, row_slots=slots, num_slots=6,
            cu_seqlens_q=batch.cu_seqlens_q, seqused_k=batch.seqused_k,
            block_table=batch.block_table, softmax_scale=0.125, round_to_model_dtype=False)

    _, rowsum = launch()
    assert float((rowsum[slots.long()] - 1.0).abs().max()) <= ROW_SUM_TOLERANCE
    # Corrupting the first-pass mass must trip the online invariant.
    stats[:, :, 1] *= 2.0
    with pytest.raises(RuntimeError, match="diverged"):
        launch()
