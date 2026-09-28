"""The statistics collector enforces the step protocol, never tolerance.

One arming per step, one buffer per layer, token counts that agree with the armed
metadata, a second-pass plan whose every field is well formed, each layer routed to
the span or the series launch it owes, and a close that refuses to succeed with
declared layers or their products missing: the collector is the seam between the
capture layer and the kernels, and every misuse is an error at the seam rather than
a silent gap in the statistics downstream. The collector is engine-free, so all of
that runs on a CPU.

Without vLLM installed the instrumented backend cannot exist: its registration
refuses, and that refusal is what runs here. With vLLM installed, the registry
override and its restoration run instead. The instrumented forward itself launches
the kernels inside the engine and needs a GPU; it is not exercised here.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from anamnesis.extraction.vllm.backend import (
    HAVE_VLLM,
    StatsCollector,
    StepSecondPass,
    current_collector,
    install_collector,
    register_instrumented_backend,
    uninstall_collector,
)
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS


def metadata(tokens: int = 4) -> dict[str, torch.Tensor]:
    return dict(prefix_lengths=torch.full((tokens,), 3, dtype=torch.int32),
                recency_cutoffs=torch.full((tokens,), 5, dtype=torch.int32))


def int32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def int64(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def plan(**overrides) -> StepSecondPass:
    """A four-token step: three span slots, two series slots, layer a sampled."""
    fields = dict(
        round_to_model_dtype=True,
        sampled_layers=frozenset({"a.attn"}),
        span_slots=int32([0, 1, 2, -1]),
        span_lengths=int32([4, 5, 6]),
        span_agreement=int64([0]),
        span_spectral=int64([0]),
        span_decay=int64([]),
        span_entropy=int64([0, 2]),
        span_prefixes=int32([3, 3, 3]),
        span_widths=int32([8, 8, 8]),
        series_slots=int32([0, -1, 1, -1]),
        series_lengths=int32([4, 6]),
        series_agreement=int64([0]),
        series_entropy=int64([0, 1]),
        series_widths=int32([8, 8]),
    )
    fields.update(overrides)
    return StepSecondPass(**fields)


EMPTY_SPAN = dict(span_slots=int32([-1, -1, -1, -1]), span_lengths=int32([]),
                  span_agreement=int64([]), span_spectral=int64([]),
                  span_entropy=int64([]), span_prefixes=int32([]), span_widths=int32([]))
EMPTY_SERIES = dict(series_slots=int32([-1, -1, -1, -1]), series_lengths=int32([]),
                    series_agreement=int64([]), series_entropy=int64([]),
                    series_widths=int32([]))


def draw_all(collector: StatsCollector, layers=("a.attn", "b.attn")) -> None:
    for layer in layers:
        collector.stats_buffer(layer_name=layer, num_tokens=4, num_query_heads=2,
                               device="cpu")


def test_the_step_protocol_round_trips() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    assert not collector.armed
    collector.arm_step(**metadata())
    prefix, cutoffs = collector.step_metadata(4)
    assert prefix.tolist() == [3, 3, 3, 3]
    assert cutoffs.tolist() == [5, 5, 5, 5]
    for layer in ("a.attn", "b.attn"):
        buffer = collector.stats_buffer(layer_name=layer, num_tokens=4, num_query_heads=2,
                                        device="cpu")
        assert buffer.shape == (4, 2, NUM_STATS)
        assert buffer.dtype == torch.float32
    step = collector.close_step()
    assert set(step.stats) == {"a.attn", "b.attn"}
    assert step.second == {}
    assert collector.steps == [step]
    assert not collector.armed


def test_misuse_is_refused_at_the_seam() -> None:
    collector = StatsCollector(expected_layers=("a.attn",))
    with pytest.raises(RuntimeError, match="no statistics step"):
        collector.step_metadata(4)
    collector.arm_step(**metadata())
    with pytest.raises(RuntimeError, match="already armed"):
        collector.arm_step(**metadata())
    with pytest.raises(RuntimeError, match="schedules 6"):
        collector.step_metadata(6)
    with pytest.raises(RuntimeError, match="not in the declared"):
        collector.stats_buffer(layer_name="c.attn", num_tokens=4, num_query_heads=2,
                               device="cpu")
    collector.stats_buffer(layer_name="a.attn", num_tokens=4, num_query_heads=2, device="cpu")
    with pytest.raises(RuntimeError, match="already produced"):
        collector.stats_buffer(layer_name="a.attn", num_tokens=4, num_query_heads=2,
                               device="cpu")
    step = collector.close_step()
    assert set(step.stats) == {"a.attn"}


def test_closing_with_layers_missing_is_an_error() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    collector.arm_step(**metadata())
    collector.stats_buffer(layer_name="a.attn", num_tokens=4, num_query_heads=2, device="cpu")
    with pytest.raises(RuntimeError, match="b.attn"):
        collector.close_step()
    collector.abort_step()  # a failed step is dropped, not recorded
    assert not collector.armed
    assert collector.steps == []


def test_arming_validates_the_metadata() -> None:
    collector = StatsCollector()
    with pytest.raises(ValueError, match="int32"):
        collector.arm_step(prefix_lengths=torch.zeros(4, dtype=torch.int64),
                           recency_cutoffs=torch.zeros(4, dtype=torch.int32))
    with pytest.raises(ValueError, match="lengths must agree"):
        collector.arm_step(prefix_lengths=torch.zeros(4, dtype=torch.int32),
                           recency_cutoffs=torch.zeros(5, dtype=torch.int32))
    with pytest.raises(ValueError, match="no tokens"):
        collector.arm_step(prefix_lengths=torch.zeros(0, dtype=torch.int32),
                           recency_cutoffs=torch.zeros(0, dtype=torch.int32))
    with pytest.raises(ValueError, match="empty expected-layer"):
        StatsCollector(expected_layers=())


def test_second_pass_plan_routes_by_layer_class() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    collector.arm_step(**metadata(), second_pass=plan())
    span = collector.second_pass_for("a.attn")
    assert span.kind == "span"
    assert span.slots.tolist() == [0, 1, 2, -1]
    assert span.lengths.numel() == 3
    assert span.round_to_model_dtype is True
    assert span.agreement_index.tolist() == [0]
    assert span.spectral_index.tolist() == [0]
    assert span.entropy_index.tolist() == [0, 2]
    assert span.prefixes.tolist() == [3, 3, 3]
    assert span.widths.tolist() == [8, 8, 8]
    series = collector.second_pass_for("b.attn")
    assert series.kind == "series"
    assert series.slots.tolist() == [0, -1, 1, -1]
    assert series.lengths.numel() == 2
    assert series.agreement_index.tolist() == [0]
    assert series.entropy_index.tolist() == [0, 1]
    assert series.spectral_index is None and series.decay_index is None
    assert series.prefixes is None
    draw_all(collector)
    for layer in ("a.attn", "b.attn"):
        collector.record_second(layer, dict(marker=torch.zeros(())))
    step = collector.close_step()
    assert set(step.second) == {"a.attn", "b.attn"}


def test_empty_selections_owe_no_second_pass() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    collector.arm_step(**metadata(), second_pass=plan(**EMPTY_SPAN, **EMPTY_SERIES))
    assert collector.second_pass_for("a.attn") is None
    assert collector.second_pass_for("b.attn") is None
    with pytest.raises(RuntimeError, match="owes no second pass"):
        collector.record_second("a.attn", dict(marker=torch.zeros(())))
    draw_all(collector)
    assert collector.close_step().second == {}


def test_empty_series_owes_no_second_pass() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    collector.arm_step(**metadata(), second_pass=plan(**EMPTY_SERIES))
    assert collector.second_pass_for("a.attn").kind == "span"
    assert collector.second_pass_for("b.attn") is None
    draw_all(collector)
    collector.record_second("a.attn", dict(marker=torch.zeros(())))
    assert set(collector.close_step().second) == {"a.attn"}


def test_closing_with_products_missing_is_an_error() -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    collector.arm_step(**metadata(), second_pass=plan())
    draw_all(collector)
    collector.record_second("a.attn", dict(marker=torch.zeros(())))
    with pytest.raises(RuntimeError, match="second-pass products missing"):
        collector.close_step()
    collector.abort_step()
    collector.arm_step(**metadata(), second_pass=plan())
    collector.record_second("a.attn", dict(marker=torch.zeros(())))
    with pytest.raises(RuntimeError, match="already recorded"):
        collector.record_second("a.attn", dict(marker=torch.zeros(())))


def test_every_plan_field_is_required() -> None:
    fields = {f.name: getattr(plan(), f.name) for f in dataclasses.fields(StepSecondPass)}
    del fields["series_slots"]
    with pytest.raises(TypeError, match="series_slots"):
        StepSecondPass(**fields)


@pytest.mark.parametrize(("overrides", "message"), [
    (dict(sampled_layers=frozenset({"zz.attn"})), "sampled layers must be expected"),
    (dict(sampled_layers={"a.attn"}), "frozenset"),
    (dict(round_to_model_dtype=1), "rounding switch"),
    (dict(span_slots=int32([0, 1, 2, -1]).to(torch.int64)), "span_slots"),
    (dict(span_slots=int32([0, 1, 2])), "span_slots"),
    (dict(series_slots=int32([0, -1, 1])), "series_slots"),
    (dict(span_lengths=torch.tensor([4.0, 5.0, 6.0])), "span_lengths"),
    (dict(span_agreement=int32([0])), "span_agreement"),
    (dict(series_entropy=int32([0])), "series_entropy"),
    (dict(span_prefixes=int32([3, 3])), "span_prefixes"),
    (dict(span_widths=int64([8, 8, 8])), "span_widths"),
    (dict(series_widths=int32([8])), "series_widths"),
])
def test_plan_validation_refuses_malformed_arming(overrides: dict, message: str) -> None:
    collector = StatsCollector(expected_layers=("a.attn", "b.attn"))
    with pytest.raises(ValueError, match=message):
        collector.arm_step(**metadata(), second_pass=plan(**overrides))
    assert not collector.armed


def test_steps_can_be_handed_over_without_retention() -> None:
    collector = StatsCollector(expected_layers=("a.attn",), retain_steps=False)
    collector.arm_step(**metadata())
    collector.stats_buffer(layer_name="a.attn", num_tokens=4, num_query_heads=2, device="cpu")
    step = collector.close_step()
    assert set(step.stats) == {"a.attn"}
    assert collector.steps == []


def test_installation_is_exclusive() -> None:
    assert current_collector() is None
    collector = StatsCollector()
    install_collector(collector)
    try:
        assert current_collector() is collector
        with pytest.raises(RuntimeError, match="already installed"):
            install_collector(StatsCollector())
    finally:
        assert uninstall_collector() is collector
    assert current_collector() is None
    with pytest.raises(RuntimeError, match="no statistics collector"):
        uninstall_collector()


@pytest.mark.skipif(HAVE_VLLM, reason="vllm is installed: the refusal exists only without it")
def test_registration_requires_the_engine() -> None:
    with pytest.raises(RuntimeError, match="engine is required"):
        register_instrumented_backend()


@pytest.mark.skipif(
    not HAVE_VLLM,
    reason="no vllm: install the engine version anamnesis.extraction.vllm.envelope pins "
           "to register the instrumented backend")
def test_registration_overrides_and_restores_the_registry() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from anamnesis.extraction.vllm.backend import InstrumentedTritonAttentionBackend

    try:
        path = register_instrumented_backend()
        assert path.endswith("InstrumentedTritonAttentionBackend")
        resolved = AttentionBackendEnum.TRITON_ATTN.get_class()
        assert resolved is InstrumentedTritonAttentionBackend
        assert resolved.get_name() == "TRITON_ATTN"
    finally:
        AttentionBackendEnum.TRITON_ATTN.clear_override()
