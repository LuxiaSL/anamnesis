"""Instrumented TRITON_ATTN backend: attention statistics from inside the engine.

The engine's registry override is the supported mechanism: the instrumented
backend registers under the TRITON_ATTN entry inside the capturing process
only, so backend-name dispatch (including the batch-invariant
configuration, which accepts TRITON_ATTN and always takes the 2D kernel)
sees the name it expects while ``get_class`` resolves to this module. The
instrumented impl delegates to the stock forward whenever no collector is
armed, so capture off is the upstream arithmetic path by construction; with
capture on, the instrumented kernel's attention output is bit-identical to the
stock 2D kernel's in the lane configuration, so capturing never changes what
the model computes.

Per scheduler step, the capture layer arms a :class:`StatsCollector` with
the per-token region metadata (absolute prompt boundary and recency cutoff,
in the step's flat packing order) after input preparation and closes it
after the step, which verifies every expected layer produced exactly one
statistics tensor. The collector is engine-free and fully testable on the
host; only the impl/backend classes require the engine.

A step also arms a second-pass plan: per-token scratch slots for the span
rows (the sampled layers' scratch, which coverage consumes whole) and for the
series rows (every other layer's scratch), with the per-series subsets
indexing into them. Each layer with a nonempty selection launches the bounded
second pass right after its statistics pass (the query tensor and paged cache
are only alive inside the forward) and records the reduced products; the
scratch never leaves the step. Closing verifies the products beside the
statistics.

Everything outside the lane's configuration is refused at capture time: non-decoder attention, sinks, alibi, sliding windows, softcap,
quantized KV cache or fused output quantization, and cascade metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from anamnesis.extraction.vllm.second_pass import second_pass_rows
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS, unified_attention_stats
from anamnesis.extraction.vllm.step_products import series_products, span_products

try:
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
    )
    HAVE_VLLM = True
except ImportError:  # pragma: no cover - exercised only without the engine
    HAVE_VLLM = False


@dataclass
class StepSecondPass:
    """One step's second-pass plan, in the step's packed token order.

    The fields mirror :class:`anamnesis.extraction.vllm.rows.StepRowPlan` as
    device tensors: ``span_slots``/``series_slots`` map each packed token to its
    scratch slot or -1, the lengths, prefixes and widths carry one entry per
    slot, and the index tensors select slots per series. ``sampled_layers``
    names the layers that take the span scratch; every other expected layer
    takes the series scratch.
    """

    round_to_model_dtype: bool
    sampled_layers: frozenset[str]
    span_slots: Tensor
    span_lengths: Tensor
    span_agreement: Tensor
    span_spectral: Tensor
    span_decay: Tensor
    span_entropy: Tensor
    span_prefixes: Tensor
    span_widths: Tensor
    series_slots: Tensor
    series_lengths: Tensor
    series_agreement: Tensor
    series_entropy: Tensor
    series_widths: Tensor


@dataclass(frozen=True)
class LayerSecondPass:
    """The launch a single layer owes this step, resolved from the plan."""

    kind: str  # 'span' or 'series'
    slots: Tensor
    lengths: Tensor
    round_to_model_dtype: bool
    agreement_index: Tensor
    entropy_index: Tensor
    widths: Tensor
    spectral_index: Tensor | None = None
    decay_index: Tensor | None = None
    prefixes: Tensor | None = None


@dataclass
class StepStatistics:
    """What one closed step hands back: statistics and products per layer."""

    stats: dict[str, Tensor]
    second: dict[str, dict[str, Tensor]]


@dataclass
class _StepArming:
    prefix_lengths: Tensor
    recency_cutoffs: Tensor
    second_plan: StepSecondPass | None = None
    collected: dict[str, Tensor] = field(default_factory=dict)
    second: dict[str, dict[str, Tensor]] = field(default_factory=dict)


class StatsCollector:
    """Per-step statistics routing between the capture layer and the kernel.

    The capture layer arms a step with per-token metadata, every attention
    layer draws exactly one statistics buffer during that step, and closing
    the step hands the per-layer tensors back while verifying completeness
    against the declared layer set. Arming while a step is open, drawing a
    buffer twice for one layer, drawing with the wrong token count and
    closing with layers missing are all errors, not best-effort tolerance.
    """

    def __init__(self, *, expected_layers: tuple[str, ...] | None = None,
                 retain_steps: bool = True):
        if expected_layers is not None and not expected_layers:
            raise ValueError('an empty expected-layer set collects nothing')
        self.expected_layers = expected_layers
        self.retain_steps = retain_steps
        self._step: _StepArming | None = None
        self.steps: list[StepStatistics] = []

    @property
    def armed(self) -> bool:
        return self._step is not None

    def arm_step(self, *, prefix_lengths: Tensor, recency_cutoffs: Tensor,
                 second_pass: StepSecondPass | None = None) -> None:
        if self._step is not None:
            raise RuntimeError('a statistics step is already armed; the '
                               'previous step was never closed')
        for name, tensor in (('prefix_lengths', prefix_lengths),
                             ('recency_cutoffs', recency_cutoffs)):
            if tensor.ndim != 1 or tensor.dtype != torch.int32:
                raise ValueError(f'{name} must be a 1D int32 tensor')
        if prefix_lengths.shape != recency_cutoffs.shape:
            raise ValueError('per-token metadata lengths must agree')
        if prefix_lengths.numel() == 0:
            raise ValueError('a step with no tokens cannot be armed')
        if second_pass is not None:
            self._validate_plan(second_pass, prefix_lengths.numel())
        self._step = _StepArming(prefix_lengths=prefix_lengths,
                                 recency_cutoffs=recency_cutoffs,
                                 second_plan=second_pass)

    def _validate_plan(self, plan: StepSecondPass, num_tokens: int) -> None:
        if type(plan.round_to_model_dtype) is not bool:
            raise ValueError('the rounding switch must be a plain bool')
        if not isinstance(plan.sampled_layers, frozenset):
            raise ValueError('sampled layers must arrive as a frozenset')
        if self.expected_layers is not None and \
                not plan.sampled_layers <= set(self.expected_layers):
            raise ValueError('sampled layers must be expected layers')
        for name, tensor in (('span_slots', plan.span_slots),
                             ('series_slots', plan.series_slots)):
            if tensor.shape != (num_tokens,) or tensor.dtype != torch.int32:
                raise ValueError(f'{name} must be int32 with one entry per '
                                 'scheduled token')
        for name, tensor in (('span_lengths', plan.span_lengths),
                             ('series_lengths', plan.series_lengths)):
            if tensor.ndim != 1 or tensor.dtype != torch.int32:
                raise ValueError(f'{name} must be a 1D int32 tensor')
        for name, tensor in (('span_agreement', plan.span_agreement),
                             ('span_spectral', plan.span_spectral),
                             ('span_decay', plan.span_decay),
                             ('span_entropy', plan.span_entropy),
                             ('series_agreement', plan.series_agreement),
                             ('series_entropy', plan.series_entropy)):
            if tensor.ndim != 1 or tensor.dtype != torch.int64:
                raise ValueError(f'{name} must be a 1D int64 index tensor')
        for name, tensor, lengths in (
                ('span_prefixes', plan.span_prefixes, plan.span_lengths),
                ('span_widths', plan.span_widths, plan.span_lengths),
                ('series_widths', plan.series_widths, plan.series_lengths)):
            if tensor.shape != lengths.shape or tensor.dtype != torch.int32:
                raise ValueError(f'{name} must be int32 with one entry per slot')

    def step_metadata(self, num_tokens: int) -> tuple[Tensor, Tensor]:
        step = self._require_armed()
        if step.prefix_lengths.numel() != num_tokens:
            raise RuntimeError(
                f'armed metadata covers {step.prefix_lengths.numel()} tokens '
                f'but the step schedules {num_tokens}')
        return step.prefix_lengths, step.recency_cutoffs

    def stats_buffer(self, *, layer_name: str, num_tokens: int,
                     num_query_heads: int, device) -> Tensor:
        step = self._require_armed()
        if not layer_name:
            raise ValueError('a statistics buffer requires a layer name')
        if self.expected_layers is not None \
                and layer_name not in self.expected_layers:
            raise RuntimeError(f'layer {layer_name} is not in the declared '
                               'collection set')
        if layer_name in step.collected:
            raise RuntimeError(f'layer {layer_name} already produced '
                               'statistics this step')
        if step.prefix_lengths.numel() != num_tokens:
            raise RuntimeError(
                f'armed metadata covers {step.prefix_lengths.numel()} tokens '
                f'but layer {layer_name} schedules {num_tokens}')
        buffer = torch.empty((num_tokens, num_query_heads, NUM_STATS),
                             dtype=torch.float32, device=device)
        step.collected[layer_name] = buffer
        return buffer

    def second_pass_for(self, layer_name: str) -> LayerSecondPass | None:
        """The launch this layer owes the armed step, or None."""
        step = self._require_armed()
        plan = step.second_plan
        if plan is None:
            return None
        if layer_name in plan.sampled_layers:
            if plan.span_lengths.numel() == 0:
                return None
            return LayerSecondPass(
                kind='span', slots=plan.span_slots, lengths=plan.span_lengths,
                round_to_model_dtype=plan.round_to_model_dtype,
                agreement_index=plan.span_agreement,
                entropy_index=plan.span_entropy, widths=plan.span_widths,
                spectral_index=plan.span_spectral, decay_index=plan.span_decay,
                prefixes=plan.span_prefixes)
        if plan.series_lengths.numel() == 0:
            return None
        return LayerSecondPass(
            kind='series', slots=plan.series_slots, lengths=plan.series_lengths,
            round_to_model_dtype=plan.round_to_model_dtype,
            agreement_index=plan.series_agreement,
            entropy_index=plan.series_entropy, widths=plan.series_widths)

    def record_second(self, layer_name: str,
                      products: dict[str, Tensor]) -> None:
        step = self._require_armed()
        if self.second_pass_for(layer_name) is None:
            raise RuntimeError(f'layer {layer_name} owes no second pass '
                               'this step')
        if layer_name in step.second:
            raise RuntimeError(f'layer {layer_name} already recorded '
                               'second-pass products this step')
        step.second[layer_name] = products

    def close_step(self) -> StepStatistics:
        step = self._require_armed()
        if self.expected_layers is not None:
            missing = sorted(set(self.expected_layers) - set(step.collected))
            if missing:
                raise RuntimeError(
                    'statistics step closed with layers missing: '
                    + ', '.join(missing))
            owing = {layer for layer in self.expected_layers
                     if self.second_pass_for(layer) is not None}
            unpaid = sorted(owing - set(step.second))
            if unpaid:
                raise RuntimeError(
                    'statistics step closed with second-pass products '
                    'missing: ' + ', '.join(unpaid))
        record = StepStatistics(stats=step.collected, second=step.second)
        self._step = None
        if self.retain_steps:
            self.steps.append(record)
        return record

    def abort_step(self) -> None:
        """Drop an armed step after a failure, without recording it."""
        self._step = None

    def _require_armed(self) -> _StepArming:
        if self._step is None:
            raise RuntimeError('no statistics step is armed')
        return self._step


_COLLECTOR: StatsCollector | None = None


def install_collector(collector: StatsCollector) -> None:
    global _COLLECTOR
    if _COLLECTOR is not None:
        raise RuntimeError('a statistics collector is already installed')
    _COLLECTOR = collector


def uninstall_collector() -> StatsCollector:
    global _COLLECTOR
    if _COLLECTOR is None:
        raise RuntimeError('no statistics collector is installed')
    collector, _COLLECTOR = _COLLECTOR, None
    return collector


def current_collector() -> StatsCollector | None:
    return _COLLECTOR


if HAVE_VLLM:

    class InstrumentedTritonAttentionImpl(TritonAttentionImpl):
        """The stock impl, with statistics when a step is armed."""

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            collector = current_collector()
            if (collector is None or not collector.armed
                    or attn_metadata is None):
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output=output,
                                       output_scale=output_scale,
                                       output_block_scale=output_block_scale)
            if self.attn_type != AttentionType.DECODER:
                raise RuntimeError('statistics capture supports decoder '
                                   'attention only')
            if self.sinks is not None or self.alibi_slopes is not None:
                raise RuntimeError('sinks and alibi are outside the '
                                   'instrumented configuration')
            if self.sliding_window != (-1, -1):
                raise RuntimeError('sliding windows are outside the '
                                   'instrumented configuration')
            if self.logits_soft_cap:
                raise RuntimeError('softcap is outside the instrumented '
                                   'configuration')
            if self.kv_cache_dtype.startswith('fp8'):
                raise RuntimeError('a quantized KV cache is outside the '
                                   'instrumented configuration')
            if output_scale is not None or output_block_scale is not None:
                raise RuntimeError('fused output quantization is outside '
                                   'the instrumented configuration')
            if attn_metadata.use_cascade:
                raise RuntimeError('cascade attention is outside the '
                                   'instrumented configuration')
            if output is None:
                raise RuntimeError('the instrumented forward requires the '
                                   'output buffer')
            num_actual = attn_metadata.num_actual_tokens
            key_cache, value_cache = kv_cache.unbind(1)
            prefix_lengths, recency_cutoffs = collector.step_metadata(
                num_actual)
            stats = collector.stats_buffer(
                layer_name=layer.layer_name, num_tokens=num_actual,
                num_query_heads=self.num_heads, device=query.device)
            unified_attention_stats(
                query[:num_actual], key_cache, value_cache,
                out=output[:num_actual], stats=stats,
                cu_seqlens_q=attn_metadata.query_start_loc,
                seqused_k=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                softmax_scale=self.scale,
                prefix_lengths=prefix_lengths,
                recency_cutoffs=recency_cutoffs)
            selection = collector.second_pass_for(layer.layer_name)
            if selection is not None:
                scratch, rowsum = second_pass_rows(
                    query[:num_actual], key_cache, stats=stats,
                    row_slots=selection.slots,
                    num_slots=selection.lengths.numel(),
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    seqused_k=attn_metadata.seq_lens,
                    block_table=attn_metadata.block_table,
                    softmax_scale=self.scale,
                    round_to_model_dtype=selection.round_to_model_dtype)
                if selection.kind == 'span':
                    products = span_products(
                        scratch, lengths=selection.lengths,
                        agreement_index=selection.agreement_index,
                        spectral_index=selection.spectral_index,
                        decay_index=selection.decay_index,
                        entropy_index=selection.entropy_index,
                        prefixes=selection.prefixes,
                        widths=selection.widths,
                        rowsum=rowsum, slots=selection.slots)
                else:
                    products = series_products(
                        scratch,
                        agreement_index=selection.agreement_index,
                        entropy_index=selection.entropy_index,
                        widths=selection.widths,
                        rowsum=rowsum, slots=selection.slots)
                del scratch
                collector.record_second(layer.layer_name, products)
            return output

    class InstrumentedTritonAttentionBackend(TritonAttentionBackend):
        """TRITON_ATTN with the instrumented impl; the backend name is unchanged."""

        @staticmethod
        def get_impl_cls() -> type[InstrumentedTritonAttentionImpl]:
            return InstrumentedTritonAttentionImpl

    def register_instrumented_backend() -> str:
        """Override TRITON_ATTN's resolved class for this process.

        Runs before engine construction, in the capturing process; the engine
        still selects and names TRITON_ATTN, so every backend-name dispatch
        branch behaves as stock.
        """
        path = (f'{InstrumentedTritonAttentionBackend.__module__}.'
                'InstrumentedTritonAttentionBackend')
        register_backend(AttentionBackendEnum.TRITON_ATTN, path)
        resolved = AttentionBackendEnum.TRITON_ATTN.get_class()
        if resolved is not InstrumentedTritonAttentionBackend:
            raise RuntimeError('the registry did not resolve the '
                               'instrumented backend')
        return path

else:  # pragma: no cover - exercised only without the engine

    def register_instrumented_backend() -> str:
        raise RuntimeError('the engine is required to register the '
                           'instrumented backend')
