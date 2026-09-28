"""The row schema against the fast lane reducer's own observable row choices.

`anamnesis.extraction.vllm.rows` restates which query rows each attention consumer
reads. A restatement can drift, so the tests here run the real reducer
(`anamnesis.extraction.fast.attention.AttentionReducer`) and check its observable
behavior against the schema: the spectral steps it stores, and the strided series
values recomputed from exactly the schema's rows. Span lengths sit on both sides of
every selection boundary — the spectral all-rows fallback below three steps, the
decay minimum at ten, and the agreement and entropy stride onsets. The step plan is
then checked for what the capture layer relies on: packed-order metadata, slot maps
that name their members, family selections that agree with the schema, and chunks
that concatenate to the whole.

Everything here is host bookkeeping and runs on a CPU; the kernels that consume a
plan are exercised on a device by `test_vllm_second_pass`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.ops import FeatureCollector, entropy, slope, std
from anamnesis.extraction.vllm.rows import (
    StepRowPlan,
    agreement_rows,
    entropy_rows,
    recency_cutoffs,
    request_row_schema,
    series_rows,
    step_row_plan,
)

PREFIX = 11


def run_reducer(steps: int, heads: int = 3):
    torch.manual_seed(93 + steps)
    weights = torch.rand(1, heads, steps + 1, PREFIX + steps + 1)
    out = FeatureCollector("cpu")
    reducer = AttentionReducer(out, steps=steps, prefix_length=PREFIX, sampled_layers=[0])
    reducer.consume(0, weights)
    a = weights[0, :, :steps, :].permute(1, 0, 2).float()
    pos = torch.arange(a.shape[-1])
    lengths = torch.arange(PREFIX + 1, PREFIX + steps + 1)
    a = a * (pos[None, :] < lengths[:, None])[:, None, :]
    return out.values, reducer, a, lengths


@pytest.mark.parametrize("steps", [1, 2, 7, 9, 10, 19, 20, 21, 31, 61, 127])
def test_schema_rows_match_reducer_observables(steps: int) -> None:
    values, reducer, a, lengths = run_reducer(steps)
    schema = request_row_schema(prompt_length=PREFIX, end=PREFIX + steps + 1)
    assert schema.steps == steps

    # Spectral: the reducer stores the step list it actually used.
    stored_steps, _ = reducer.spectral[0]
    assert stored_steps == list(schema.spectral)
    assert schema.spectral_width == PREFIX + stored_steps[-1] + 1

    # Entropy series: the reducer's own slice must select the schema rows, and
    # its exact values must follow (same strided view, same reduction).
    ent = entropy(a)
    stride = max(1, steps // 60)
    sample = ent[::stride] if steps > 60 else ent
    rows = torch.arange(steps)[::stride] if steps > 60 else torch.arange(steps)
    assert list(schema.entropy) == rows.tolist()
    assert torch.equal(values["attn_entropy_mean_L0"], sample.mean())
    assert torch.equal(values["attn_entropy_std_L0"], std(sample))

    # Head agreement: the h-pair recomputed at exactly the schema rows.
    stride = max(1, steps // 30)
    selected = a[::stride] if steps > 30 else a
    rows = torch.arange(steps)[::stride] if steps > 30 else torch.arange(steps)
    assert list(schema.agreement) == rows.tolist()
    norm = selected.double()
    norm = norm / norm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    h_mean = entropy(norm.mean(dim=1)).double()
    h_heads = entropy(norm).mean(dim=1).double()
    agreement = (
        1 - (h_mean - h_heads).clamp_min(0) / max(np.log(a.shape[1]), 1e-12)).float()
    assert torch.equal(values["head_agreement_mean_L0"], agreement.mean())
    assert torch.equal(values["head_agreement_std_L0"], std(agreement))

    # Decay fit: recomputed from the schema's rows, or the short-span zero.
    mean = a.mean(dim=1).double()
    if not schema.decay:
        assert steps < 10
        assert torch.equal(values["cache_attn_decay_rate_L0"], torch.zeros(()))
    else:
        rows = list(schema.decay)
        sampled = mean[rows, 1:]
        pos = torch.arange(mean.shape[-1])
        distances = (lengths[rows, None] - 1 - pos[None, 1:]).clamp_min(1).double()
        valid = (pos[None, :] < lengths[:, None])[rows, 1:]
        mask = (sampled > 1e-10) & valid
        fit = -slope(sampled.clamp_min(1e-300).log().flatten(),
                     distances.flatten(), mask.flatten())
        assert torch.equal(values["cache_attn_decay_rate_L0"], fit.float())


def test_recency_cutoffs_match_reducer_arithmetic() -> None:
    lengths = np.arange(1, 4097, dtype=np.int64)
    expected = (torch.arange(1, 4097).double() * 0.8).long().clamp_min(1)
    assert np.array_equal(recency_cutoffs(lengths), expected.numpy().astype(np.int32))


def test_request_schema_validation() -> None:
    with pytest.raises(ValueError):
        request_row_schema(prompt_length=5, end=6)  # empty span
    with pytest.raises(ValueError):
        request_row_schema(prompt_length=-1, end=10)
    with pytest.raises(ValueError):
        request_row_schema(prompt_length=np.int64(3), end=10)
    schema = request_row_schema(prompt_length=3, end=5)
    assert schema.steps == 1
    assert schema.agreement == (0,)
    assert schema.entropy == (0,)
    assert schema.series == (0,)
    assert schema.spectral == (0,)
    assert schema.decay == ()
    assert schema.spectral_width == 4


def plan_for(items: list[dict], schemas: dict, **kwargs) -> StepRowPlan:
    plan = step_row_plan(items, schemas, **kwargs)
    assert isinstance(plan, StepRowPlan)
    return plan


def test_step_plan_chunked_two_requests() -> None:
    schemas = {
        "a": request_row_schema(prompt_length=4, end=40),
        "b": request_row_schema(prompt_length=2, end=8),
    }
    # Request a arrives mid-span (positions 10..24), b whole (0..7).
    items = [
        dict(request_id="a", start=10, count=15, offset=0),
        dict(request_id="b", start=0, count=8, offset=15),
    ]
    plan = plan_for(items, schemas)
    assert plan.total_tokens == 23
    assert list(plan.prefix_lengths) == [4] * 15 + [2] * 8
    expected_lengths = np.array(list(range(11, 26)) + list(range(1, 9)))
    assert np.array_equal(
        plan.recency_cutoffs,
        np.maximum((expected_lengths.astype(np.float64) * 0.8).astype(int), 1))
    # Span rows: a positions 10..24 (rows 6..20), b positions 2..6 (rows 0..4).
    assert plan.span_members[:15] == tuple(("a", r) for r in range(6, 21))
    assert plan.span_members[15:] == tuple(("b", r) for r in range(0, 5))
    assert list(plan.span_lengths) == list(range(11, 26)) + list(range(3, 8))
    # Every span token points at its slot; non-span tokens carry -1.
    for token, slot in enumerate(plan.span_slots):
        if slot >= 0:
            request_id, row = plan.span_members[slot]
            item = items[0 if request_id == "a" else 1]
            position = schemas[request_id].prompt_length + row
            assert token == item["offset"] + position - item["start"]
    # b's positions 0..1 are prompt rows and its position 7 is the final token:
    # none of them may own a span slot.
    assert plan.span_slots[15] == -1 and plan.span_slots[16] == -1
    assert plan.span_slots[22] == -1
    # Agreement/spectral/decay membership agrees with the schemas.
    for index, slot in enumerate(plan.span_agreement):
        request_id, row = plan.span_members[slot]
        assert row in schemas[request_id].agreement
        assert plan.agreement_members[index] == (request_id, row)
    scheduled = set(plan.span_members)
    agreement_expected = {
        (r, row) for r, schema in schemas.items() for row in schema.agreement}
    assert set(plan.agreement_members) == agreement_expected & scheduled
    for slot in plan.span_spectral:
        request_id, row = plan.span_members[slot]
        assert row in schemas[request_id].spectral
    spectral_expected = {
        (r, row) for r, schema in schemas.items() for row in schema.spectral}
    assert {tuple(plan.span_members[s]) for s in plan.span_spectral} \
        == spectral_expected & scheduled
    decay_expected = {(r, row) for r, schema in schemas.items() for row in schema.decay}
    assert {tuple(plan.span_members[s]) for s in plan.span_decay} \
        == decay_expected & scheduled
    # Each slot's width is its request's materialized width.
    assert list(plan.span_widths) == [40] * 15 + [8] * 5
    assert list(plan.span_prefixes) == [4] * 15 + [2] * 5


def test_step_plan_prompt_only_chunk_selects_nothing() -> None:
    schemas = {"a": request_row_schema(prompt_length=20, end=30)}
    plan = plan_for([dict(request_id="a", start=0, count=10, offset=0)], schemas)
    assert (plan.span_slots == -1).all()
    assert (plan.series_slots == -1).all()
    assert plan.span_members == ()
    assert plan.series_members == ()
    assert plan.span_agreement.size == 0


def test_step_plan_refusals() -> None:
    schemas = {"a": request_row_schema(prompt_length=2, end=8)}
    with pytest.raises(ValueError):
        step_row_plan([dict(request_id="a", start=0, count=4, offset=1)], schemas)
    with pytest.raises(ValueError):
        step_row_plan([dict(request_id="a", start=0, count=0, offset=0)], schemas)
    with pytest.raises(ValueError):
        step_row_plan([dict(request_id="a", start=0, count=9, offset=0)], schemas)
    with pytest.raises(ValueError):
        step_row_plan([], schemas)


def test_step_plan_selected_requests_keep_metadata_but_own_no_slots() -> None:
    """An unretained occupant of a step is scheduled and described, never kept."""
    schemas = {
        "a": request_row_schema(prompt_length=2, end=12),
        "b": request_row_schema(prompt_length=2, end=12),
    }
    items = [dict(request_id="a", start=0, count=12, offset=0),
             dict(request_id="b", start=0, count=12, offset=12)]
    whole = plan_for(items, schemas)
    kept = plan_for(items, schemas, selected=frozenset({"b"}))
    assert np.array_equal(kept.prefix_lengths, whole.prefix_lengths)
    assert np.array_equal(kept.recency_cutoffs, whole.recency_cutoffs)
    assert {request for request, _ in kept.span_members} == {"b"}
    assert {request for request, _ in kept.series_members} == {"b"}
    assert (kept.span_slots[:12] == -1).all() and (kept.series_slots[:12] == -1).all()
    assert kept.span_members == tuple(m for m in whole.span_members if m[0] == "b")


def test_series_union_is_not_a_superset_relation() -> None:
    """The stride families are independent truncations: at 511 steps the entropy
    stride is 8 and the agreement stride 17, so neither selection contains the
    other and the union is strictly larger than each."""
    agreement = set(agreement_rows(511))
    entropy_set = set(entropy_rows(511))
    union = set(series_rows(511))
    assert not agreement <= entropy_set
    assert not entropy_set <= agreement
    assert union == agreement | entropy_set
    assert len(union) > max(len(agreement), len(entropy_set))
    assert series_rows(511) == tuple(sorted(union))
    # Short spans collapse every family onto every row.
    assert series_rows(12) == tuple(range(12))


def test_step_plan_family_selections() -> None:
    schemas = {"r": request_row_schema(prompt_length=4, end=200)}
    plan = plan_for([dict(request_id="r", start=0, count=200, offset=0)], schemas)
    schema = schemas["r"]
    assert tuple(row for _, row in plan.series_members) == schema.series
    assert tuple(plan.series_lengths) == tuple(4 + row + 1 for row in schema.series)
    assert tuple(plan.series_widths) == (200,) * len(schema.series)
    assert tuple(plan.series_members[s][1] for s in plan.series_agreement) == schema.agreement
    assert tuple(plan.series_members[s][1] for s in plan.series_entropy) == schema.entropy
    assert tuple(plan.span_members[s][1] for s in plan.span_entropy) == schema.entropy
    assert tuple(plan.span_prefixes) == (4,) * len(plan.span_members)
    # Slot maps agree with the members they name.
    for token, slot in enumerate(plan.series_slots):
        if slot >= 0:
            request, row = plan.series_members[slot]
            assert request == "r" and row == token - 4


def test_step_plan_family_selections_split_across_chunks() -> None:
    schemas = {"r": request_row_schema(prompt_length=4, end=200)}
    whole = plan_for([dict(request_id="r", start=0, count=200, offset=0)], schemas)
    first = plan_for([dict(request_id="r", start=0, count=77, offset=0)], schemas)
    second = plan_for([dict(request_id="r", start=77, count=123, offset=0)], schemas)
    assert first.series_members + second.series_members == whole.series_members
    assert (tuple(first.span_prefixes) + tuple(second.span_prefixes)
            == tuple(whole.span_prefixes))
    joined_entropy = tuple(
        first.span_members[s][1] for s in first.span_entropy
    ) + tuple(second.span_members[s][1] for s in second.span_entropy)
    assert joined_entropy == schemas["r"].entropy
