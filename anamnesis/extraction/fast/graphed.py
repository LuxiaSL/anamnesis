"""CUDA-graph replay of :class:`~anamnesis.extraction.fast.external.ExternalSpan`'s reduction.

A short span's reduction launches tens of thousands of small kernels, and on a
fast device the host's launch cost, not the device's arithmetic, sets its wall
time. :class:`GraphedExternalSpan` is an ``ExternalSpan`` whose :meth:`finish`
records the reduction once per span shape as CUDA graphs and replays them for
every later span of that shape. The kernels are the reducers' own, in their own
order, on inputs laid out exactly as the caller's: a replayed row is the eager
row, byte for byte, and carries the eager row's lane id.

How a reduction becomes replayable
----------------------------------
The reducers are driven unedited. :func:`_reduce` restates the body of
``ExternalSpan.finish`` (its per-layer loop, the output statistics, the
key/value/query/gate families and the routing family) and runs three times for a
new shape: once eagerly to record what the host reads, once under stream
capture, and once replayed to compare. Three kinds of operation cannot sit in a
graph, and each is handled where the dispatcher sees it:

* **Host reads of a device value** (``bool(t)``, ``t.item()``). A read of a
  value computed only from constants (shapes, layer numbers, configuration) is
  the same for every span of the shape; the value recorded in the eager run is
  returned in its place. A read of anything derived from the span's data refuses
  the capture. Which values derive from data is tracked per storage through every
  operation.
* **Solvers that check their status on the host** (``torch.linalg.eigvalsh``,
  ``torch.linalg.svdvals``). The graph is cut around each call: the solver runs
  eagerly between two graphs on the graph's own input, and its outputs are copied
  into the buffers the next graph reads.
* **Host data uploaded to the device** (``torch.tensor(list, device=...)``, a
  Python list used as an index). The data is a constant of the shape; it is staged
  in pinned host memory, and the graph copies it on every replay.

The gate's top-k overlap is the one feature computed on the host from the span's
data; it runs on the host after the device reduction, as in the eager path, from
the same bytes. Each layer's attention is reduced eagerly as the forward produces
it (:meth:`ExternalSpan.attention`), and its retained summaries enter the graph as
inputs. The collector's assembly of the vector runs eagerly after the replay.

Every new shape is verified before it is kept: the replay's vector is compared
byte for byte with ``ExternalSpan.finish`` on the same inputs, and a shape whose
replay differs, or whose capture is refused, is reduced eagerly from then on.
Without CUDA, or for inputs off the lane's device, :meth:`finish` is
``ExternalSpan.finish``.

Graphs read the cuFFT plans the eager run created; the plan cache
(``torch.backends.cuda.cufft_plan_cache``) must not be cleared while a cache of
graphs is in use. A cache that sees the plan cache shrink drops every graph it
holds. The classes here are not thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import weakref
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.attn_res import attn_res_features
from anamnesis.extraction.fast.external import (
    READOUT_SOURCE_SHA256,
    ExternalCaptureError,
    ExternalSpan,
    RoutingPoint,
    external_lane_id,
)
from anamnesis.extraction.fast.families import FamilyReducer, _topk_pool, gate_topk_overlap
from anamnesis.extraction.fast.features import GpuFeatureLane, GpuFeatureResult
from anamnesis.extraction.fast.ops import FeatureCollector, rowcos, std, trajectory_indices

_aten = torch.ops.aten
_HOST_READ = _aten._local_scalar_dense.default
_SOLVERS = frozenset({_aten._linalg_eigh.default, _aten._linalg_svd.default})
_ALIGNMENT = 512
"""Bytes of address alignment a replica of an input keeps: the caching allocator's
block alignment, past any vector width a kernel selects on."""


class GraphCaptureRefused(RuntimeError):
    """A reduction that cannot be recorded as graphs without moving a byte."""


# --------------------------------------------------------------------------- reduction


class _DeferredTopk(FamilyReducer):
    """The family reducer with the gate's host top-k left to the caller.

    ``deferred_topk`` lists ``(feature name, layer)`` for every gate surface whose
    top-k overlap the eager reducer would compute; the caller computes each with
    :func:`gate_topk_overlap` from the same bytes and adds it to the collector.
    """

    def __init__(self, out: FeatureCollector, extraction, families):
        super().__init__(out, extraction, families)
        self.deferred_topk: list[tuple[str, int]] = []
        self._layer: int | None = None

    def gate(self, layer: int, pre_silu: Tensor) -> None:
        self._layer = layer
        super().gate(layer, pre_silu)

    def gate_topk_on_host(self, name: str, pre_silu: Tensor) -> None:
        if self._layer is None:
            raise GraphCaptureRefused("a gate top-k outside a gate reduction")
        self.deferred_topk.append((name, self._layer))


class _DeferredHfer(AttentionReducer):
    """The attention reducer with the spectral high-frequency energy ratio left to an
    eager step.

    How many eigenvalues pass the median depends on their values, so the ratio
    selects a data-dependent number of elements; each call returns a buffer and
    appends ``(buffer, eigenvalues, median, energy)`` to ``deferred``, and the step
    that fills the buffer computes the ratio with ``AttentionReducer.spectral_hfer``.
    """

    def __init__(self, collector: FeatureCollector, deferred: list, **kwargs):
        super().__init__(collector, **kwargs)
        self.deferred = deferred

    def spectral_hfer(self, ev: Tensor, median: Tensor, energy: Tensor) -> Tensor:
        buffer = torch.empty((), dtype=torch.promote_types(ev.dtype, energy.dtype), device=ev.device)
        self.deferred.append((buffer, ev, median, energy))
        return buffer


def _attention_reducer(span: ExternalSpan, collector: FeatureCollector, deferred: list) -> _DeferredHfer:
    lane = span.lane
    return _DeferredHfer(
        collector,
        deferred,
        steps=span.steps,
        prefix_length=span.start,
        sampled_layers=lane.config.sampled_layers,
        spectral_stride=lane.config.spectral_subsample_step,
        n_windows=lane.families.temporal_n_windows,
        include_stft=lane.families.enable_stft,
    )


class _HferStep:
    """Fills a deferred high-frequency energy ratio from its graph's outputs."""

    def __init__(self, buffer: Tensor, ev: Tensor, median: Tensor, energy: Tensor):
        self.buffer, self.ev, self.median, self.energy = buffer, ev, median, energy

    def __call__(self) -> None:
        self.buffer.copy_(AttentionReducer.spectral_hfer(self.ev, self.median, self.energy))


@dataclass
class _Reduction:
    collector: FeatureCollector
    logprob: Tensor
    routed_names: list[str] | None
    routed: Tensor | None
    deferred_topk: list[tuple[str, int]]


@dataclass
class _Inputs:
    hidden: Tensor
    keys: dict[int, Tensor]
    values: dict[int, Tensor]
    queries: dict[int, Tensor]
    gates: dict[int, Tensor]
    logits: Tensor
    routing: list[RoutingPoint] | None
    committed: list[Tensor] | None
    spectral: dict[int, tuple[list[int], Tensor]]
    per_head: dict[int, tuple[Tensor, Tensor]]

    def tensors(self) -> list[Tensor]:
        """Every tensor, in a fixed order :meth:`rebuilt` reads back."""
        out = [self.hidden, self.logits]
        for mapping in (self.keys, self.values, self.queries, self.gates):
            out += [mapping[layer] for layer in sorted(mapping)]
        out += [w for _tag, _layer, w in self.routing or []]
        out += list(self.committed or [])
        for layer in sorted(self.spectral):
            out.append(self.spectral[layer][1])
        for layer in sorted(self.per_head):
            out += list(self.per_head[layer])
        return out

    def rebuilt(self, tensors: Sequence[Tensor]) -> _Inputs:
        """These inputs with each tensor replaced, in :meth:`tensors` order."""
        it = iter(tensors)
        hidden, logits = next(it), next(it)
        maps = [{layer: next(it) for layer in sorted(m)}
                for m in (self.keys, self.values, self.queries, self.gates)]
        routing = (None if self.routing is None
                   else [(tag, layer, next(it)) for tag, layer, _w in self.routing])
        committed = None if self.committed is None else [next(it) for _ in self.committed]
        spectral = {layer: (list(self.spectral[layer][0]), next(it)) for layer in sorted(self.spectral)}
        per_head = {layer: (next(it), next(it)) for layer in sorted(self.per_head)}
        return _Inputs(hidden, *maps, logits, routing, committed, spectral, per_head)


def _reduce(lane: GpuFeatureLane, span: ExternalSpan, inputs: _Inputs, positional: Tensor,
            chosen: Tensor) -> _Reduction:
    """``ExternalSpan.finish``'s reduction, operation for operation, into a new collector.

    The attention reducer starts from the retained summaries in ``inputs`` (what
    the span's eager attention reduction left), and the gate's host top-k is
    deferred. ``chosen`` holds the span's next tokens on the device.
    """
    out = FeatureCollector(lane.device)
    attention = AttentionReducer(
        out,
        steps=span.steps,
        prefix_length=span.start,
        sampled_layers=lane.config.sampled_layers,
        spectral_stride=lane.config.spectral_subsample_step,
        n_windows=lane.families.temporal_n_windows,
        include_stft=lane.families.enable_stft,
    )
    attention.spectral = dict(inputs.spectral)
    attention.per_head = dict(inputs.per_head)
    reducer = _DeferredTopk(out, lane.config, lane.families)
    steps, hidden = span.steps, inputs.hidden
    previous = None
    for layer in range(span.num_layers):
        h = hidden[layer].to(lane.device).float()
        corrected = h - positional[layer]
        norms = corrected.norm(dim=-1).float()
        out.put(f"activation_norm_mean_L{layer}", norms.mean())
        out.put(f"activation_norm_std_L{layer}", std(norms))
        for i, t in enumerate(trajectory_indices(steps, lane.config.trajectory_points)):
            out.put(f"activation_norm_traj{i}_L{layer}", norms[t])
        if previous is not None:
            delta = corrected - previous
            dn = delta.norm(dim=-1).float()
            dc = rowcos(delta.double(), previous.double()).float()
            out.put(f"delta_norm_mean_L{layer - 1}", dn.mean())
            out.put(f"delta_norm_std_L{layer - 1}", std(dn))
            out.put(f"delta_cosine_mean_L{layer - 1}", dc.mean())
        previous = corrected
        if layer in lane.config.sampled_layers:
            attention.smoothness(layer, corrected)
        if layer in lane.families.trajectory_layers:
            reducer.residual_trajectory(layer, corrected, h[0].norm())
        if layer in lane.config.pca_layers:
            components = lane.pca[layer] if isinstance(lane.pca, dict) else lane.pca
            mean = lane.pca_mean[layer] if isinstance(lane.pca_mean, dict) else lane.pca_mean
            indices = trajectory_indices(steps, lane.config.pca_temporal_samples)
            projections = (corrected[indices].double() - mean) @ components[
                : lane.config.pca_components
            ].T
            for ti, projection in enumerate(projections):
                for ci, value in enumerate(projection):
                    out.put(f"pca_L{layer}_t{ti}_c{ci}", value)
    logprob = lane._outputs(out, inputs.logits.to(lane.device), chosen)
    for layer in lane.config.sampled_layers:
        k = inputs.keys[layer].to(lane.device).float()
        reducer.key(layer, k)
        attention.head_features(layer, k)
        reducer.value(layer, inputs.values[layer].to(lane.device).float())
        reducer.query(layer, inputs.queries[layer].to(lane.device).float())
        reducer.gate(layer, inputs.gates[layer].to(lane.device).float())
    reducer.finish()
    routed_names = routed = None
    if inputs.routing is not None:
        routed_names, routed = attn_res_features(
            [(tag, layer, w.to(lane.device)) for tag, layer, w in inputs.routing],
            [c.to(lane.device) for c in inputs.committed or []],
            lane.config.sampled_layers,
        )
    return _Reduction(out, logprob, routed_names, routed, reducer.deferred_topk)


# --------------------------------------------------------------------------- capture


def _storage(t: Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _written_arguments(func, args, kwargs) -> list[Tensor]:
    """The tensors an operation writes in place, from its schema."""
    written = []
    schema = func._schema
    for i, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        value = args[i] if i < len(args) else kwargs.get(argument.name)
        written += [t for t in tree_leaves(value) if isinstance(t, Tensor)]
    return written


@dataclass
class _Record:
    """What an eager run of the reduction showed the host: each device value it read,
    and each solver's output layout, in order."""

    host_reads: list[tuple[tuple[int, ...], torch.dtype, Any]] = field(default_factory=list)
    solver_outputs: list[tuple[Any, list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype]]]] = field(
        default_factory=list)


class _Recording(TorchDispatchMode):
    """An eager run that records the host's reads of device values and the solvers'
    output layouts."""

    def __init__(self, record: _Record):
        super().__init__()
        self.record = record

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func is _HOST_READ and args[0].is_cuda:
            self.record.host_reads.append((tuple(args[0].shape), args[0].dtype, out))
        elif func in _SOLVERS:
            leaves, spec = tree_flatten(out)
            self.record.solver_outputs.append(
                (spec, [(tuple(t.shape), tuple(t.stride()), t.dtype) for t in leaves]))
        return out


class _SolverStep:
    """A solver run eagerly between two graphs, its outputs copied into the buffers
    the next graph reads."""

    def __init__(self, func, args, kwargs, outputs: list[Tensor]):
        self.func, self.args, self.kwargs, self.outputs = func, args, kwargs, outputs

    def __call__(self) -> None:
        results = [t for t in tree_leaves(self.func(*self.args, **self.kwargs)) if isinstance(t, Tensor)]
        for buffer, result in zip(self.outputs, results, strict=True):
            buffer.copy_(result)


class _Tape:
    """The recorded reduction: graphs, with the eager solver steps between them."""

    def __init__(self, pool, stream: torch.cuda.Stream):
        self.pool = pool
        self.stream = stream
        self.steps: list[Callable[[], None]] = []
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.keep: list[Any] = []
        self.solver_steps = 0

    def replay(self) -> None:
        for step in self.steps:
            step()


class _Capturing(TorchDispatchMode):
    """Stream capture of one reduction into a :class:`_Tape`.

    Storages written from the span's data are tracked through every operation, so
    a host read is answered from the record only when it reads constants. A solver
    ends the current graph, becomes an eager step, and starts the next graph.
    """

    def __init__(self, tape: _Tape, record: _Record, data: set[int], side: torch.cuda.Stream,
                 replay_stream: torch.cuda.Stream):
        super().__init__()
        self.tape, self.record, self.data = tape, record, data
        self.side, self.replay_stream = side, replay_stream
        self.created: set[int] = set()
        self.reads = 0
        self.solvers = 0
        self.graph: torch.cuda.CUDAGraph | None = None

    def begin(self) -> None:
        self.graph = torch.cuda.CUDAGraph()
        self.graph.capture_begin(pool=self.tape.pool, capture_error_mode="thread_local")

    def end(self) -> None:
        graph, self.graph = self.graph, None
        if graph is not None:
            graph.capture_end()
            self.tape.graphs.append(graph)
            self.tape.steps.append(graph.replay)

    def abandon(self) -> None:
        """End a capture that failed, discarding its graph."""
        graph, self.graph = self.graph, None
        if graph is not None:
            try:
                graph.capture_end()
            except Exception:  # the capture is already invalid; nothing to keep
                pass

    def _is_data(self, t: Tensor) -> bool:
        return t.is_cuda and _storage(t) in self.data

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        leaves = [t for t in tree_leaves((args, kwargs)) if isinstance(t, Tensor)]
        from_data = any(self._is_data(t) for t in leaves)
        if func is _HOST_READ and args[0].is_cuda:
            if self.reads >= len(self.record.host_reads):
                raise GraphCaptureRefused("the capture read the device more often than the eager run")
            shape, dtype, value = self.record.host_reads[self.reads]
            self.reads += 1
            if (shape, dtype) != (tuple(args[0].shape), args[0].dtype):
                raise GraphCaptureRefused("the capture's host reads differ from the eager run's")
            if from_data:
                raise GraphCaptureRefused(
                    f"the reduction reads a value derived from the span's data on the host "
                    f"({tuple(args[0].shape)} {args[0].dtype})")
            return value
        if func in _SOLVERS:
            if self.solvers >= len(self.record.solver_outputs):
                raise GraphCaptureRefused("the capture ran more solvers than the eager run")
            spec, layouts = self.record.solver_outputs[self.solvers]
            self.solvers += 1
            self.end()
            with torch.cuda.stream(self.replay_stream):
                outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=self.side.device)
                           for shape, stride, dtype in layouts]
            self.tape.steps.append(_SolverStep(func, args, kwargs, outputs))
            self.tape.keep.append(outputs)
            for t in outputs:
                (self.data.add if from_data else self.data.discard)(_storage(t))
            self.begin()
            return tree_unflatten(outputs, spec)
        for t in leaves:
            if t.is_cuda and _storage(t) not in self.created:
                self.tape.keep.append(t)
        out = func(*args, **kwargs)
        for t in [x for x in tree_leaves(out) if isinstance(x, Tensor)] + _written_arguments(
                func, args, kwargs):
            if not t.is_cuda:
                continue
            ptr = _storage(t)
            self.created.add(ptr)
            (self.data.add if from_data else self.data.discard)(ptr)
        return out


def _int_list(x: Any) -> bool:
    return (isinstance(x, list) and len(x) > 0
            and all(isinstance(i, int) and not isinstance(i, bool) for i in x))


class _HostConstants(TorchFunctionMode):
    """Host data a captured reduction uploads, staged in pinned memory so the graph
    copies it on replay: ``torch.tensor``/``torch.as_tensor`` of host data onto the
    device, and a Python list of positions used as an index."""

    def __init__(self, device: torch.device, keep: list[Any]):
        super().__init__()
        self.device, self.keep = device, keep

    def _upload(self, host: Tensor) -> Tensor:
        pinned = host.pin_memory()
        self.keep.append(pinned)
        out = torch.empty(pinned.shape, dtype=pinned.dtype, device=self.device)
        out.copy_(pinned, non_blocking=True)
        return out

    def _index(self, index: Any) -> Any:
        if _int_list(index):
            return self._upload(torch.tensor(index, dtype=torch.long))
        if isinstance(index, tuple):
            return tuple(self._index(i) for i in index)
        return index

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.tensor, torch.as_tensor) and args and not isinstance(args[0], Tensor):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                return self._upload(func(args[0], **{**kwargs, "device": "cpu"}))
        if func is Tensor.__getitem__ and len(args) == 2:
            args = (args[0], self._index(args[1]))
        return func(*args, **kwargs)


def _record_tape(device: torch.device, pool, data: set[int], warm: Callable[[], Any],
                 run: Callable[[], Any]) -> tuple[_Tape, Any]:
    """``warm()`` run eagerly and recorded, then ``run()`` (the same reduction on fresh
    reducers) captured into a tape on the current stream's behalf; returns the tape
    and what ``run`` returned. The tape has not been replayed."""
    replay_stream = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    side.wait_stream(replay_stream)
    record = _Record()
    with torch.cuda.stream(side):
        with _Recording(record):
            warm()
    torch.cuda.synchronize(device)
    tape = _Tape(pool, replay_stream)
    capturing = _Capturing(tape, record, set(data), side, replay_stream)
    with torch.cuda.stream(side):
        try:
            with _HostConstants(device, tape.keep), capturing:
                capturing.begin()
                result = run()
                capturing.end()
        except BaseException:
            capturing.abandon()
            raise
    if capturing.reads != len(record.host_reads) or capturing.solvers != len(record.solver_outputs):
        raise GraphCaptureRefused("the capture and the eager run read the host differently")
    replay_stream.wait_stream(side)
    torch.cuda.synchronize(device)
    tape.solver_steps = len(record.solver_outputs)
    return tape, result


# --------------------------------------------------------------------------- inputs


def _layout(t: Tensor) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() % _ALIGNMENT)


def _extent(t: Tensor) -> int:
    """Elements from a tensor's first to its last, inclusive."""
    if t.numel() == 0:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True))


class _Replica:
    """A device buffer holding a tensor's elements at the same strides and address
    alignment, and the flat ranges a per-span copy moves."""

    def __init__(self, source: Tensor):
        if any(s < 0 for s in source.stride()):
            raise GraphCaptureRefused("an input with a negative stride")
        size = source.element_size()
        if source.data_ptr() % size:
            raise GraphCaptureRefused("an input not aligned to its element size")
        self.extent = _extent(source)
        self.base = torch.empty(self.extent + _ALIGNMENT // size, dtype=source.dtype, device=source.device)
        start = ((source.data_ptr() - self.base.data_ptr()) % _ALIGNMENT) // size
        self.view = self.base.as_strided(source.shape, source.stride(), start)
        self.flat = self.base.as_strided((self.extent,), (1,), start)
        if self.view.data_ptr() % _ALIGNMENT != source.data_ptr() % _ALIGNMENT:
            raise GraphCaptureRefused("an input's alignment could not be reproduced")

    @staticmethod
    def flat_source(source: Tensor) -> Tensor:
        return source.as_strided((_extent(source),), (1,), source.storage_offset())


def _copy_into(replicas: Sequence[_Replica], sources: Sequence[Tensor]) -> None:
    """Every source's elements into its replica, one multi-tensor copy per dtype."""
    groups: dict[torch.dtype, tuple[list[Tensor], list[Tensor]]] = {}
    for replica, source in zip(replicas, sources, strict=True):
        if replica.extent == 0:
            continue
        dst, src = groups.setdefault(source.dtype, ([], []))
        dst.append(replica.flat)
        src.append(_Replica.flat_source(source))
    for dst, src in groups.values():
        torch._foreach_copy_(dst, src)


def _pool_bytes(pool) -> int:
    """Bytes the caching allocator has reserved for one graph pool."""
    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except Exception:  # an allocator without snapshots reports nothing
        return 0
    return sum(seg["total_size"] for seg in segments
               if tuple(seg.get("segment_pool_id", ())) == tuple(pool))


# --------------------------------------------------------------------------- the cache


@dataclass
class FinishGraphs:
    """One span shape's recorded ``finish`` reduction and what it costs."""

    lane: GpuFeatureLane
    tape: _Tape
    replicas: list[_Replica]
    positional: Tensor
    chosen: Tensor
    reduction: _Reduction
    capture_seconds: float
    input_bytes: int
    pool_bytes: int
    replays: int = 0


@dataclass
class _LayerTape:
    tape: _Tape
    names: list[str]
    retained_bytes: int


@dataclass
class ConsumeGraphs:
    """One span shape's recorded per-layer attention reductions, one tape per layer in
    one memory pool, replayed in layer order.

    The outputs live in the pool and are rewritten by the next span's replay, so one
    span at a time holds them (``owner``)."""

    lane: GpuFeatureLane
    pool: Any
    replica: _Replica
    collector: FeatureCollector
    attention: _DeferredHfer
    layers: list[_LayerTape] = field(default_factory=list)
    capture_seconds: float = 0.0
    pool_bytes: int = 0
    replays: int = 0
    owner: Any = None

    @property
    def input_bytes(self) -> int:
        return self.replica.base.numel() * self.replica.base.element_size()

    def held_by_another(self, span: object) -> bool:
        owner = self.owner() if self.owner is not None else None
        return owner is not None and owner is not span and not owner.finished


@dataclass
class GraphCacheStats:
    finish_shapes: int
    consume_shapes: int
    refused: dict[str, str]
    captures: int
    finish_replays: int
    consume_replays: int
    evictions: int
    capture_seconds: float
    input_bytes: int
    pool_bytes: int


class SpanGraphCache:
    """Recorded reductions keyed by span shape and input layout.

    ``finish`` reductions and per-layer attention reductions are kept apart, each
    least recently used first out with at most ``max_shapes`` held. A shape whose
    capture was refused, or whose replay did not match the eager reduction, is
    remembered and reduced eagerly."""

    def __init__(self, max_shapes: int = 8):
        if max_shapes < 1:
            raise ValueError("a graph cache holds at least one shape")
        self.max_shapes = max_shapes
        self.finish: OrderedDict[tuple, FinishGraphs] = OrderedDict()
        self.consume: OrderedDict[tuple, ConsumeGraphs] = OrderedDict()
        self.refused: dict[tuple, str] = {}
        self.captures = 0
        self.evictions = 0
        self._fft_plans: dict[int, tuple[int, int]] = {}

    @staticmethod
    def _get(table: OrderedDict, key: tuple):
        entry = table.get(key)
        if entry is not None:
            table.move_to_end(key)
        return entry

    def _put(self, table: OrderedDict, key: tuple, entry) -> None:
        table[key] = entry
        table.move_to_end(key)
        self.captures += 1
        while len(table) > self.max_shapes:
            torch.cuda.synchronize()
            table.popitem(last=False)
            self.evictions += 1

    def get_finish(self, key: tuple) -> FinishGraphs | None:
        return self._get(self.finish, key)

    def put_finish(self, key: tuple, entry: FinishGraphs) -> None:
        self._put(self.finish, key, entry)

    def get_consume(self, key: tuple) -> ConsumeGraphs | None:
        return self._get(self.consume, key)

    def put_consume(self, key: tuple, entry: ConsumeGraphs) -> None:
        self._put(self.consume, key, entry)

    def refuse(self, key: tuple, reason: str) -> None:
        self.refused[key] = reason
        self.finish.pop(key, None)
        self.consume.pop(key, None)

    def check_fft_plans(self, device: torch.device) -> None:
        """Drop every graph if the device's cuFFT plan cache shrank, was resized or
        filled since it was last seen: a graph may hold a plan the cache destroyed."""
        index = device.index if device.index is not None else torch.cuda.current_device()
        plans = torch.backends.cuda.cufft_plan_cache[index]
        now = (plans.size, plans.max_size)
        seen = self._fft_plans.get(index)
        if seen is not None and (now[1] != seen[1] or now[0] < seen[0] or now[0] >= now[1]):
            torch.cuda.synchronize()
            self.evictions += len(self.finish) + len(self.consume)
            self.finish.clear()
            self.consume.clear()
        self._fft_plans[index] = now

    def clear(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.finish.clear()
        self.consume.clear()
        self.refused.clear()

    def stats(self) -> GraphCacheStats:
        finish, consume = list(self.finish.values()), list(self.consume.values())
        return GraphCacheStats(
            finish_shapes=len(finish),
            consume_shapes=len(consume),
            refused={repr(key[:5]): reason for key, reason in self.refused.items()},
            captures=self.captures,
            finish_replays=sum(e.replays for e in finish),
            consume_replays=sum(e.replays for e in consume),
            evictions=self.evictions,
            capture_seconds=sum(e.capture_seconds for e in finish + consume),
            input_bytes=sum(e.input_bytes for e in finish + consume),
            pool_bytes=sum(e.pool_bytes for e in finish + consume),
        )


DEFAULT_CACHE = SpanGraphCache()
"""The cache a :class:`GraphedExternalSpan` uses unless it is given one."""


@dataclass
class _ConsumeState:
    """What the per-layer attention reduction leaves for ``finish``: the features it
    put, and the summaries it retained."""

    raw: dict
    zeroed: list
    spectral: dict
    per_head: dict


def _bitwise_equal(a: GpuFeatureResult, b: GpuFeatureResult) -> bool:
    return (a.feature_names == b.feature_names
            and a.features.tobytes() == b.features.tobytes()
            and np.float64(a.mean_logprob).tobytes() == np.float64(b.mean_logprob).tobytes())


def _differing(a: GpuFeatureResult, b: GpuFeatureResult) -> list[str]:
    return [n for n, x, y in zip(a.feature_names, a.features, b.features, strict=False)
            if np.float32(x).tobytes() != np.float32(y).tobytes()]


# --------------------------------------------------------------------------- the span


class GraphedExternalSpan(ExternalSpan):
    """An :class:`ExternalSpan` whose reductions replay recorded CUDA graphs.

    Each layer's attention reduction and the ``finish`` reduction are recorded the
    first time a span shape is seen; that span's own result is ``ExternalSpan``'s
    eager result, and the recording is kept only if its replay produced the same
    bytes. Later spans of the shape copy their inputs into the recording's buffers
    and replay it. Attention must arrive in layer order for its graphs to replay;
    a span whose attention arrives otherwise, or whose shape was refused, is reduced
    eagerly. The result, names, lane id and receipt are ``ExternalSpan``'s; the
    receipt adds ``graph_replayed`` (whether the vector came from replayed graphs)
    and ``attention_replayed`` (whether its attention reduction did).
    """

    def __init__(self, lane: GpuFeatureLane, *, start: int, end: int, num_layers: int,
                 cache: SpanGraphCache | None = None):
        super().__init__(lane, start=start, end=end, num_layers=num_layers)
        self.cache = DEFAULT_CACHE if cache is None else cache
        self._graphs = lane.device.type == "cuda" and torch.cuda.is_available()
        self._consume_mode = "pending" if self._graphs else "eager"
        self._consume_key: tuple | None = None
        self._consume: ConsumeGraphs | None = None
        self._consume_began = 0.0
        self._next_layer = 0

    # ------------------------------------------------------------------ attention

    def attention(self, layer: int, weights: Tensor) -> None:
        if self.finished:
            raise ExternalCaptureError("attention after the span was finished")
        if not 0 <= layer < self.num_layers:
            raise ExternalCaptureError(f"attention layer {layer} outside 0..{self.num_layers - 1}")
        w = weights if weights.ndim == 4 else weights.unsqueeze(0)
        if w.ndim != 4 or w.shape[0] != 1 or w.shape[2] < self.end or w.shape[3] < self.end:
            raise ExternalCaptureError(f"attention weights {tuple(weights.shape)} do not cover the span")
        span_weights = w[:, :, self.start:self.end, :self.end].to(self.lane.device)
        mode = self._consume_mode_for(layer, span_weights)
        if mode == "replay":
            self._consume_replay(layer, span_weights)
            return
        self.attention_reducer.consume(layer, span_weights)
        if mode == "record":
            self._consume_record(layer, span_weights)

    def _consume_mode_for(self, layer: int, span_weights: Tensor) -> str:
        if self._consume_mode == "eager":
            return "eager"
        layout = _layout(span_weights)
        if self._consume_mode == "pending":
            if layer != 0 or span_weights.device != self.lane.device:
                return self._consume_eager()
            key = (id(self.lane), self.start, self.end, self.num_layers,
                   torch.cuda.current_stream(self.lane.device).stream_id, layout)
            self._consume_key = key
            self.cache.check_fft_plans(self.lane.device)
            entry = self.cache.get_consume(key)
            if entry is not None:
                if entry.held_by_another(self):
                    return self._consume_eager()
                entry.owner = weakref.ref(self)
                self._consume, self._consume_mode = entry, "replay"
            elif key in self.cache.refused:
                return self._consume_eager()
            else:
                self._consume_mode = "record"
                self._consume_began = time.perf_counter()
        elif layer != self._next_layer or layout != self._consume_key[-1]:
            return self._consume_eager()
        self._next_layer = layer + 1
        return self._consume_mode

    def _consume_eager(self) -> str:
        """The rest of this span's attention reduced eagerly; a recording in progress
        is dropped (this span's order, not the shape, stopped it)."""
        if self._consume_mode == "record":
            self._consume = None
        self._consume_mode = "eager"
        return "eager"

    def _consume_record(self, layer: int, span_weights: Tensor) -> None:
        lane, device = self.lane, self.lane.device
        try:
            if layer == 0:
                replica = _Replica(span_weights)
                collector = FeatureCollector(device)
                self._consume = ConsumeGraphs(
                    lane=lane, pool=torch.cuda.graph_pool_handle(), replica=replica,
                    collector=collector, attention=_attention_reducer(self, collector, []),
                    owner=weakref.ref(self))
            shape = self._consume
            shape.replica.view.copy_(span_weights)
            before_names = len(shape.collector._raw)
            before_bytes = shape.attention.retained_bytes
            before_hfer = len(shape.attention.deferred)
            tape, _ = _record_tape(
                device, shape.pool, {_storage(shape.replica.view)},
                warm=lambda: _attention_reducer(self, FeatureCollector(device), []).consume(
                    layer, shape.replica.view),
                run=lambda: shape.attention.consume(layer, shape.replica.view))
            for deferred in shape.attention.deferred[before_hfer:]:
                tape.steps.append(_HferStep(*deferred))
            tape.replay()
            shape.layers.append(_LayerTape(
                tape, list(shape.collector._raw)[before_names:],
                shape.attention.retained_bytes - before_bytes))
        except Exception as exc:  # any capture failure leaves this shape's attention eager
            self.cache.refuse(self._consume_key, f"attention layer {layer}: {type(exc).__name__}: {exc}")
            warnings.warn(f"span shape c={self.start} t={self.steps}: attention reduced eagerly: {exc}",
                          RuntimeWarning, stacklevel=3)
            torch.cuda.synchronize(device)
            self._consume, self._consume_mode = None, "eager"

    def _consume_replay(self, layer: int, span_weights: Tensor) -> None:
        shape = self._consume
        recorded = shape.layers[layer]
        shape.replica.view.copy_(span_weights)
        recorded.tape.replay()
        raw = self.collector._raw
        for name in recorded.names:
            if name in raw:
                raise ExternalCaptureError(f"duplicate GPU feature {name}")
            raw[name] = shape.collector._raw[name]
        reducer = self.attention_reducer
        if layer in reducer.seen:
            raise ExternalCaptureError(f"attention layer {layer} captured twice in one replay")
        reducer.seen.add(layer)
        if layer in shape.attention.spectral:
            reducer.spectral[layer] = shape.attention.spectral[layer]
        if layer in shape.attention.per_head:
            reducer.per_head[layer] = shape.attention.per_head[layer]
        reducer.retained_bytes += recorded.retained_bytes
        if layer == self.num_layers - 1:
            shape.replays += 1

    # ------------------------------------------------------------------ finish

    @torch.no_grad()
    def finish(
        self,
        *,
        hidden: Tensor,
        keys: Mapping[int, Tensor],
        values: Mapping[int, Tensor],
        queries: Mapping[int, Tensor],
        gates: Mapping[int, Tensor],
        logits: Tensor,
        token_ids: Sequence[int],
        capture_identity: Mapping[str, Any],
        routing: Sequence[RoutingPoint] | None = None,
        committed: Sequence[Tensor] | None = None,
    ) -> GpuFeatureResult:
        call = dict(hidden=hidden, keys=keys, values=values, queries=queries, gates=gates,
                    logits=logits, token_ids=token_ids, capture_identity=capture_identity,
                    routing=routing, committed=committed)
        if self.finished:
            raise ExternalCaptureError("a span is finished once")
        recording = (self._consume if self._consume_mode == "record"
                     and self._next_layer == self.num_layers else None)
        if not self._graphs:
            return self._eager(call)
        self._validate(hidden, keys, values, queries, gates, logits, token_ids, routing, committed)
        if self.attention_reducer.seen != set(range(self.num_layers)):
            return self._eager(call)
        if recording is not None:
            state = _ConsumeState(dict(recording.collector._raw), list(recording.collector._zeroed),
                                  dict(recording.attention.spectral), dict(recording.attention.per_head))
        else:
            state = self._state()
        inputs = self._inputs(call, state)
        tensors = inputs.tensors()
        if any(t.device != self.lane.device for t in tensors):
            return self._eager(call)
        key = self._key(inputs, tensors)
        self.cache.check_fft_plans(self.lane.device)
        entry = self.cache.get_finish(key)
        if entry is None and key in self.cache.refused:
            if recording is not None:
                self.cache.refuse(self._consume_key, "its finish reduction is refused")
            return self._eager(call)
        if entry is not None and recording is None:
            topk = self._topk(entry.reduction, inputs)
            _copy_into(entry.replicas, tensors)
            entry.chosen.copy_(self._chosen(token_ids))
            entry.tape.replay()
            entry.replays += 1
            self.finished = True
            return self._assemble(entry.reduction, topk, call, state, replayed=True)
        return self._verify(key, entry, recording, state, inputs, tensors, call)

    def _verify(self, key: tuple, entry: FinishGraphs | None, recording: ConsumeGraphs | None,
                state: _ConsumeState, inputs: _Inputs, tensors: list[Tensor],
                call: dict) -> GpuFeatureResult:
        """Record what is new (the finish reduction, this span's attention recording or
        both), replay it, and keep it only if the replay is the eager reduction's bytes.
        Returns the eager reduction."""
        began = time.perf_counter()
        new = entry is None
        try:
            if new:
                entry = self._capture_finish(inputs, tensors, call["token_ids"])
            else:
                _copy_into(entry.replicas, tensors)
                entry.chosen.copy_(self._chosen(call["token_ids"]))
            topk = self._topk(entry.reduction, inputs)
            entry.tape.replay()
            graphed = self._assemble(entry.reduction, topk, call, state, replayed=True)
        except Exception as exc:  # any capture failure leaves this shape eager
            reason = f"finish: {type(exc).__name__}: {exc}"
            if new:
                self.cache.refuse(key, reason)
            if recording is not None:
                self.cache.refuse(self._consume_key, reason)
            warnings.warn(f"span shape c={self.start} t={self.steps} reduced eagerly: {exc}",
                          RuntimeWarning, stacklevel=3)
            torch.cuda.synchronize(self.lane.device)
            return self._eager(call)
        eager = self._eager(call)
        if new:
            entry.capture_seconds = time.perf_counter() - began
        if _bitwise_equal(graphed, eager):
            if new:
                self.cache.put_finish(key, entry)
            if recording is not None:
                recording.capture_seconds = time.perf_counter() - self._consume_began
                recording.pool_bytes = _pool_bytes(recording.pool)
                recording.owner = None
                self.cache.put_consume(self._consume_key, recording)
            return eager
        differing = _differing(eager, graphed)
        reason = (f"replay differs from the eager reduction at {len(differing)} coordinates, "
                  f"first {differing[:8]}")
        if new:
            self.cache.refuse(key, reason)
        if recording is not None:
            self.cache.refuse(self._consume_key, reason)
        warnings.warn(f"span shape c={self.start} t={self.steps} reduced eagerly: {reason}",
                      RuntimeWarning, stacklevel=3)
        return eager

    def _eager(self, call: dict) -> GpuFeatureResult:
        result = ExternalSpan.finish(self, **call)
        result.metadata["graph_replayed"] = False
        result.metadata["attention_replayed"] = self._consume_mode == "replay"
        return result

    def _state(self) -> _ConsumeState:
        reducer = self.attention_reducer
        return _ConsumeState(dict(self.collector._raw), list(self.collector._zeroed),
                             dict(reducer.spectral), dict(reducer.per_head))

    def _inputs(self, call: dict, state: _ConsumeState) -> _Inputs:
        routing, committed = call["routing"], call["committed"]
        return _Inputs(call["hidden"], dict(call["keys"]), dict(call["values"]), dict(call["queries"]),
                       dict(call["gates"]), call["logits"],
                       None if routing is None else list(routing),
                       None if committed is None else list(committed),
                       dict(state.spectral), dict(state.per_head))

    def _key(self, inputs: _Inputs, tensors: list[Tensor]) -> tuple:
        return (
            id(self.lane), self.start, self.end, self.num_layers, "finish",
            None if inputs.routing is None else tuple((tag, int(layer)) for tag, layer, _ in inputs.routing),
            None if inputs.committed is None else len(inputs.committed),
            tuple(sorted(inputs.keys)),
            tuple((layer, tuple(steps)) for layer, (steps, _) in sorted(inputs.spectral.items())),
            tuple(sorted(inputs.per_head)),
            torch.cuda.current_stream(self.lane.device).stream_id,
            tuple(_layout(t) for t in tensors),
        )

    def _chosen(self, token_ids: Sequence[int]) -> Tensor:
        return torch.as_tensor(list(token_ids[self.start + 1:self.end]), dtype=torch.long)

    def _topk(self, reduction: _Reduction, inputs: _Inputs) -> list:
        """The deferred gate top-k overlaps, started on the host pool from the bytes the
        eager reducer would read."""
        lane = self.lane
        return [(name, _topk_pool().submit(
                    gate_topk_overlap, inputs.gates[layer].to(lane.device).float().float().cpu().numpy()))
                for name, layer in reduction.deferred_topk]

    def _capture_finish(self, inputs: _Inputs, tensors: list[Tensor],
                        token_ids: Sequence[int]) -> FinishGraphs:
        lane = self.lane
        replicas = [_Replica(t) for t in tensors]
        _copy_into(replicas, tensors)
        static = inputs.rebuilt([r.view for r in replicas])
        pm = np.ascontiguousarray(lane.pm[1:self.num_layers + 1, self.start:self.start + self.steps])
        positional = torch.as_tensor(pm, device=lane.device)
        chosen = torch.empty(self.steps, dtype=torch.long, device=lane.device)
        chosen.copy_(self._chosen(token_ids))
        data = {_storage(r.view) for r in replicas if r.extent} | {_storage(chosen)}
        pool = torch.cuda.graph_pool_handle()
        tape, reduction = _record_tape(
            lane.device, pool, data,
            warm=lambda: _reduce(lane, self, static, positional, chosen),
            run=lambda: _reduce(lane, self, static, positional, chosen))
        return FinishGraphs(
            lane=lane, tape=tape, replicas=replicas, positional=positional, chosen=chosen,
            reduction=reduction, capture_seconds=0.0,
            input_bytes=sum(r.base.numel() * r.base.element_size() for r in replicas),
            pool_bytes=_pool_bytes(pool),
        )

    def _reduce_eagerly(self, call: dict) -> GpuFeatureResult:
        """The restated reduction run without graphs, assembled as a replay is: what a
        replay computes, on any device."""
        if self.finished:
            raise ExternalCaptureError("a span is finished once")
        self._validate(call["hidden"], call["keys"], call["values"], call["queries"], call["gates"],
                       call["logits"], call["token_ids"], call["routing"], call["committed"])
        lane = self.lane
        state = self._state()
        inputs = self._inputs(call, state)
        pm = np.ascontiguousarray(lane.pm[1:self.num_layers + 1, self.start:self.start + self.steps])
        chosen = torch.as_tensor(list(call["token_ids"][self.start + 1:self.end]), device=lane.device,
                                 dtype=torch.long)
        reduction = _reduce(lane, self, inputs, torch.as_tensor(pm, device=lane.device), chosen)
        self.finished = True
        return self._assemble(reduction, self._topk(reduction, inputs), call, state, replayed=False)

    def _assemble(self, reduction: _Reduction, topk: list, call: dict, state: _ConsumeState, *,
                  replayed: bool) -> GpuFeatureResult:
        """The vector, names, mean log-probability and receipt of a reduction, as
        ``ExternalSpan.finish`` assembles them: the attention reduction's features,
        the finish reduction's and the host top-k, in one collector."""
        lane, steps, start, end = self.lane, self.steps, self.start, self.end
        out = FeatureCollector(lane.device)
        graph_raw = reduction.collector._raw
        if state.raw.keys() & graph_raw.keys():
            raise ExternalCaptureError("a feature computed by both the attention and finish reductions")
        out._raw = {**state.raw, **graph_raw}
        out._zeroed = list(state.zeroed) + list(reduction.collector._zeroed)
        for name, overlap in topk:
            out.put(name, overlap.result())
        vector = out.finish(list(lane.names))
        names = list(lane.names)
        if reduction.routed is not None:
            vector = torch.cat([vector, reduction.routed.to(vector.dtype)])
            names += reduction.routed_names
        features = vector.cpu().numpy()
        if not np.isfinite(features).all():
            raise ExternalCaptureError("the reduction produced a nonfinite feature")
        hidden, token_ids = call["hidden"], call["token_ids"]
        knnlm = (hidden[-1][steps - 1].float().cpu().numpy().copy()
                 if lane.config.enable_knnlm_baseline else None)
        attention = self.attention_reducer
        receipt = dict(
            lane_id=external_lane_id(lane, call["capture_identity"]),
            fast_lane_id=lane.lane_id,
            lane_source_sha256=dict(lane.source_sha256),
            readout_source_sha256=dict(READOUT_SOURCE_SHA256),
            capture=dict(call["capture_identity"]),
            replay_id=str(uuid.uuid4()),
            replay_path="external",
            span_start=start,
            span_end=end,
            input_tokens_sha256=hashlib.sha256(
                np.asarray(list(token_ids[:end]), dtype="<i8").tobytes()).hexdigest(),
            feature_schema_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            calibration_sha256=lane.calibration_sha256,
            channel="probe-free",
            certified=False,
            batch_size=1,
            attention_layers_reduced=len(attention.seen),
            retained_attention_summary_bytes=attention.retained_bytes,
            routing_family=call["routing"] is not None,
            graph_replayed=replayed,
            attention_replayed=self._consume_mode == "replay",
        )
        return GpuFeatureResult(features, tuple(names), float(reduction.logprob.cpu()), knnlm, receipt)
