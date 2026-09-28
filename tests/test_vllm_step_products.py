"""The step products against the fast lane's reducer and the banked coverage.

The second pass's scratch is reduced inside the capture step to what the attention
consumers keep, and every product must be the reducer's own number. The h-pair and
the entropy series are compared bit for bit with
`anamnesis.extraction.fast.attention.AttentionReducer` on the same rows; the per-head
summaries likewise, including the prompt-mass slice and the recency truncation.
Every product is also blind to the scratch width, because the capture scratch is as
wide as the step's longest scheduled row while the reducer materializes each
request's own row. Coverage is `second_pass.coverage_reference` applied verbatim.

Everything here is host arithmetic on fp32 tensors and runs on a CPU; the scratch
the capture step feeds these reductions comes from the second-pass kernel, which
`test_vllm_second_pass` exercises on a device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.ops import FeatureCollector, std
from anamnesis.extraction.vllm.rows import request_row_schema
from anamnesis.extraction.vllm.second_pass import coverage_reference
from anamnesis.extraction.vllm.step_products import (
    SERIES_PRODUCT_KEYS,
    SPAN_PRODUCT_KEYS,
    agreement_pair,
    entropy_of_rows,
    per_head_summaries,
    row_sum_worst,
    series_products,
    span_products,
)

PREFIX = 7


def masked_rows(weights: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The reducer's span rows as a [steps, heads, keys] fp32 scratch, masked to
    each row's valid keys, and those row lengths."""
    a = weights[0, :, :steps, :].permute(1, 0, 2).float()
    pos = torch.arange(a.shape[-1])
    lengths = torch.arange(PREFIX + 1, PREFIX + steps + 1)
    return (a * (pos[None, :] < lengths[:, None])[:, None, :]).contiguous(), lengths


@pytest.mark.parametrize("steps", [3, 31, 127])
def test_agreement_pair_reproduces_the_reducer_bitwise(steps: int) -> None:
    torch.manual_seed(steps)
    heads = 3
    weights = torch.rand(1, heads, steps + 1, PREFIX + steps + 1)
    out = FeatureCollector("cpu")
    AttentionReducer(out, steps=steps, prefix_length=PREFIX, sampled_layers=[0]).consume(
        0, weights)
    a, _ = masked_rows(weights, steps)
    schema = request_row_schema(prompt_length=PREFIX, end=PREFIX + steps + 1)
    rows = a[::max(1, steps // 30)] if steps > 30 else a
    assert rows.shape[0] == len(schema.agreement)
    h_mean, h_heads = agreement_pair(rows.contiguous())
    agreement = (1 - (h_mean - h_heads).clamp_min(0) / max(np.log(heads), 1e-12)).float()
    assert torch.equal(out.values["head_agreement_mean_L0"], agreement.mean())
    assert torch.equal(out.values["head_agreement_std_L0"], std(agreement))
    # Width independence: rows padded with extra zero keys give the exact same
    # pair, because zeros contribute exactly zero everywhere.
    padded = torch.cat([rows, torch.zeros(rows.shape[0], heads, 9)], dim=-1)
    h_mean_pad, h_heads_pad = agreement_pair(padded.contiguous())
    assert torch.equal(h_mean, h_mean_pad)
    assert torch.equal(h_heads, h_heads_pad)


def scratch_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    scratch = torch.rand(3, 2, 6).float()
    lengths = torch.tensor([4, 5, 6], dtype=torch.int32)
    positions = torch.arange(6)
    scratch = scratch * (positions[None, :] < lengths[:, None])[:, None, :]
    rowsum = torch.full((5, 2), float("nan"))
    slots = torch.tensor([-1, 0, 1, 2, -1], dtype=torch.int32)
    rowsum[1:4] = 1.0
    rowsum[2, 1] = 1.0 + 3e-4
    return scratch, lengths, rowsum, slots


def span_arguments(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every span_products argument but the scratch, for `scratch_fixture`."""
    _, lengths, rowsum, slots = scratch_fixture()
    arguments = dict(
        lengths=lengths,
        agreement_index=torch.tensor([0, 2], dtype=torch.int64),
        spectral_index=torch.tensor([1], dtype=torch.int64),
        decay_index=torch.zeros(0, dtype=torch.int64),
        entropy_index=torch.tensor([0, 2], dtype=torch.int64),
        prefixes=torch.tensor([2, 2, 3], dtype=torch.int32),
        widths=torch.tensor([6, 6, 6], dtype=torch.int32),
        rowsum=rowsum, slots=slots)
    arguments.update(overrides)
    return arguments


def test_span_products_shapes_and_definitions() -> None:
    scratch, lengths, _, _ = scratch_fixture()
    arguments = span_arguments()
    products = span_products(scratch, **arguments)
    assert set(products) == set(SPAN_PRODUCT_KEYS)
    mean32 = scratch.mean(dim=1)
    valid = torch.arange(6)[None, :] < lengths[:, None]
    assert torch.equal(products["coverage"], coverage_reference(mean32, lengths, valid))
    expected_pair = agreement_pair(scratch[[0, 2]])
    assert torch.equal(products["h_mean"], expected_pair[0])
    assert torch.equal(products["h_heads"], expected_pair[1])
    assert torch.equal(products["spectral_rows"], mean32[[1]])
    assert products["decay_rows"].shape == (0, 6)
    assert torch.isclose(products["row_sum_worst"], torch.tensor(3e-4), atol=1e-7)
    # Every span row's fp32 head mean is kept for the cache and flow families.
    assert torch.equal(products["span_rows"], mean32)
    # The family products are the family reductions of the same scratch.
    entropy_index, widths = arguments["entropy_index"], arguments["widths"]
    assert torch.equal(
        products["entropy_rows"],
        entropy_of_rows(scratch[entropy_index], widths=widths[entropy_index]))
    summaries = per_head_summaries(
        scratch, lengths=lengths, prefixes=arguments["prefixes"], widths=widths)
    for key, value in summaries.items():
        assert torch.equal(products[key], value), key


def test_span_products_with_no_agreement_rows() -> None:
    scratch, _, _, _ = scratch_fixture()
    products = span_products(scratch, **span_arguments(
        agreement_index=torch.zeros(0, dtype=torch.int64),
        spectral_index=torch.tensor([0, 1, 2], dtype=torch.int64),
        decay_index=torch.tensor([2], dtype=torch.int64)))
    assert products["h_mean"].shape == (0,)
    assert products["h_mean"].dtype == torch.float64
    assert products["spectral_rows"].shape == (3, 6)


def test_products_refusals() -> None:
    scratch, lengths, rowsum, _ = scratch_fixture()
    with pytest.raises(ValueError, match="exceeds the scratch"):
        span_products(scratch, **span_arguments(
            agreement_index=torch.tensor([3], dtype=torch.int64)))
    with pytest.raises(ValueError, match="1D int64"):
        span_products(scratch, **span_arguments(
            agreement_index=torch.tensor([0], dtype=torch.int32)))
    with pytest.raises(ValueError, match="entropy index exceeds the scratch"):
        span_products(scratch, **span_arguments(
            entropy_index=torch.tensor([5], dtype=torch.int64)))
    with pytest.raises(ValueError, match="fp32"):
        agreement_pair(scratch.double())
    with pytest.raises(ValueError, match="no selected rows"):
        row_sum_worst(rowsum, torch.full((5,), -1, dtype=torch.int32))
    with pytest.raises(ValueError, match="one entry per span row"):
        per_head_summaries(
            scratch, lengths=lengths, prefixes=torch.tensor([2, 2], dtype=torch.int32),
            widths=torch.tensor([6, 6, 6], dtype=torch.int32))


@pytest.mark.parametrize("steps", [4, 31, 127])
def test_per_head_summaries_reproduce_the_reducer_bitwise(steps: int) -> None:
    """The reducer's per-head summaries from the same rows, bit for bit.

    The reducer keeps per_ent and per_sink for its per-head family and the two
    head-diversity ratios inside the flow family; identical rows must give identical
    float64 values, including the prompt-mass slice and the recency cutoff
    truncation.
    """
    torch.manual_seed(steps + 1)
    heads = 3
    weights = torch.softmax(torch.randn(1, heads, steps + 1, PREFIX + steps + 1), dim=-1)
    out = FeatureCollector("cpu")
    reducer = AttentionReducer(out, steps=steps, prefix_length=PREFIX, sampled_layers=[0])
    reducer.consume(0, weights)
    per_ent, per_sink = reducer.per_head[0]
    a, lengths = masked_rows(weights, steps)
    summaries = per_head_summaries(
        a, lengths=lengths.to(torch.int32),
        prefixes=torch.full((steps,), PREFIX, dtype=torch.int32),
        widths=torch.full((steps,), PREFIX + steps + 1, dtype=torch.int32))
    assert torch.equal(summaries["head_ent"], per_ent)
    assert torch.equal(summaries["head_sink"], per_sink)
    # The collector stores fp32 scalars; compare after its own cast.
    assert torch.equal(
        std(summaries["head_prompt"], dim=1).mean().float(),
        out.values["attn_flow_L0_head_diversity_prompt"])
    assert torch.equal(
        std(summaries["head_recency"], dim=1).mean().float(),
        out.values["attn_flow_L0_head_diversity_recency"])


def test_entropy_of_rows_matches_the_reducer_series() -> None:
    torch.manual_seed(41)
    steps, heads = 127, 2
    weights = torch.softmax(torch.randn(1, heads, steps + 1, PREFIX + steps + 1), dim=-1)
    out = FeatureCollector("cpu")
    AttentionReducer(out, steps=steps, prefix_length=PREFIX, sampled_layers=[0]).consume(
        0, weights)
    a, _ = masked_rows(weights, steps)
    schema = request_row_schema(prompt_length=PREFIX, end=PREFIX + steps + 1)
    widths = torch.full((len(schema.entropy),), PREFIX + steps + 1, dtype=torch.int32)
    values = entropy_of_rows(a[list(schema.entropy)], widths=widths)
    full = torch.zeros(steps, heads)
    full[list(schema.entropy)] = values
    sample = full[::max(1, steps // 60)]
    assert torch.equal(out.values["attn_entropy_mean_L0"], sample.mean())
    assert torch.equal(out.values["attn_entropy_std_L0"], std(sample))
    with pytest.raises(ValueError, match="fp32"):
        entropy_of_rows(a[list(schema.entropy)].double(), widths=widths)


def test_series_products_reduce_both_subsets_from_one_scratch() -> None:
    scratch, _, rowsum, slots = scratch_fixture()
    agreement_index = torch.tensor([0, 2], dtype=torch.int64)
    entropy_index = torch.tensor([0, 1, 2], dtype=torch.int64)
    widths = torch.tensor([6, 6, 6], dtype=torch.int32)
    products = series_products(
        scratch, agreement_index=agreement_index, entropy_index=entropy_index,
        widths=widths, rowsum=rowsum, slots=slots)
    assert set(products) == set(SERIES_PRODUCT_KEYS)
    expected_pair = agreement_pair(scratch[agreement_index])
    assert torch.equal(products["h_mean"], expected_pair[0])
    assert torch.equal(products["h_heads"], expected_pair[1])
    assert torch.equal(
        products["entropy_rows"],
        entropy_of_rows(scratch[entropy_index], widths=widths[entropy_index]))
    assert torch.equal(products["row_sum_worst"], row_sum_worst(rowsum, slots))
    empty = series_products(
        scratch, agreement_index=torch.zeros(0, dtype=torch.int64),
        entropy_index=entropy_index, widths=widths, rowsum=rowsum, slots=slots)
    assert empty["h_mean"].shape == (0,)
    with pytest.raises(ValueError, match="exceeds the scratch"):
        series_products(
            scratch, agreement_index=torch.tensor([9], dtype=torch.int64),
            entropy_index=entropy_index, widths=widths, rowsum=rowsum, slots=slots)


def test_per_head_summaries_mixed_prefix_runs_slice_exactly() -> None:
    """Rows from different requests carry different prompt boundaries; the prompt
    mass must slice each run at its own boundary."""
    scratch, lengths, _, _ = scratch_fixture()
    summaries = per_head_summaries(
        scratch, lengths=lengths, prefixes=torch.tensor([2, 2, 4], dtype=torch.int32),
        widths=torch.tensor([6, 6, 7], dtype=torch.int32))
    a64 = scratch.double()
    head_total = a64.sum(dim=-1).clamp_min(1e-12)
    assert torch.equal(summaries["head_prompt"][:2],
                       a64[:2, :, :2].sum(dim=-1) / head_total[:2])
    assert torch.equal(summaries["head_prompt"][2:],
                       a64[2:, :, :4].sum(dim=-1) / head_total[2:])


def test_family_products_are_blind_to_the_scratch_width() -> None:
    """A batched or chunked schedule widens or narrows the step scratch against a
    request's own width, and a per-head entropy reduction over a different trailing
    extent can move by an ulp. Every family product must be byte-identical between a
    scratch at the request's own width and the same rows at a wider or narrower
    step width."""
    torch.manual_seed(71)
    steps, heads, width = 33, 3, PREFIX + 33 + 1
    weights = torch.softmax(torch.randn(1, heads, steps + 1, width), dim=-1)
    a, lengths = masked_rows(weights, steps)
    widths = torch.full((steps,), width, dtype=torch.int32)
    prefixes = torch.full((steps,), PREFIX, dtype=torch.int32)
    narrow = dict(lengths=lengths.to(torch.int32), prefixes=prefixes, widths=widths)
    reference = per_head_summaries(a, **narrow)
    ent_reference = entropy_of_rows(a, widths=widths)
    for extra in (2, 17):
        wide = torch.cat([a, torch.zeros(steps, heads, extra)], dim=-1).contiguous()
        padded = per_head_summaries(wide, **narrow)
        for key, value in reference.items():
            assert torch.equal(padded[key], value), (key, extra)
        assert torch.equal(entropy_of_rows(wide, widths=widths), ent_reference), extra
    # A narrower step scratch (a chunked schedule) reduces the same.
    short = a[:7, :, :int(lengths[6])].contiguous()
    chunked = per_head_summaries(
        short, lengths=lengths[:7].to(torch.int32), prefixes=prefixes[:7], widths=widths[:7])
    for key, value in reference.items():
        assert torch.equal(chunked[key], value[:7]), key
    assert torch.equal(entropy_of_rows(short, widths=widths[:7]), ent_reference[:7])
