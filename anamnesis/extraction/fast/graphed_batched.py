"""CUDA-graph replay of the batched reducer's families (graph step 3).

A padded bucket group's families (:func:`~anamnesis.extraction.fast.engine_reducer._families`) launch tens of
thousands of small kernels, and the host's launch cost, not the device's arithmetic, sets much of their wall time.
:class:`BatchedGraphCache` records them once per batch shape as CUDA graphs and replays them for every later batch of
that shape. A replayed row is the eager row, byte for byte, and keeps the eager row's lane id.

How a batch becomes replayable
------------------------------
- **Device-only counts.** Under device-only step counts and prompt lengths
  (:meth:`~anamnesis.extraction.fast.batched_stats.Counts.device_only`), no row's count or start is read on the host
  between the batch's checks and its finish, so the launches are a function of the batch's shape alone. Every per-row
  table is gathered on the device from tables built once per bucket.
- **The ladder.** A group's row count varies from batch to batch; it is padded up to the next size of :data:`LADDER`
  with filler rows, copies of its first row, whose results are dropped. The reduction is batchmate-invariant, so the
  fillers move no row's bytes, and a bucket has at most a dozen shapes.
- **Inputs** (the stacked surfaces, the counts, the prompt lengths and the committed blocks) are copied into replicas
  at the caller's strides and address alignment before each replay. Everything else the graph reads (the lane's
  positional means, basis and tables) is captured by address; each entry holds a strong reference to every such
  tensor (the tape's ``keep``) and to the lane, so none is freed and reused under it.
- **The spectral solve** checks its status on the host. It is one operator
  (:data:`~anamnesis.extraction.fast.batched_attention.EIGVALSH_SIZED_OP`), and the capture cuts around it: each key is
  two graph segments with the eager solve between them, reading the first segment's outputs (in the graph's pool) and
  the counts' replica, and writing buffers the second segment reads. It is never a graph calling eager code.
- **Host reads** of a value computed from constants are answered from the eager run, as for the per-row graphs
  (:mod:`~anamnesis.extraction.fast.graphed`); a read of anything derived from the batch's data refuses the capture,
  and so does an upload of host data (:class:`~anamnesis.extraction.fast.batched_stats.HostUploadInCapture`).
- **The finish** (the rows' assembly on the host) and the lens family run eagerly after the replay, on its outputs.

Every new key is verified before it is kept: its first batch is reduced eagerly as well, and a replay that differs in
any byte, or a capture that is refused, leaves the key eager from then on. After that, one batch in ``audit_every``
per key is reduced both ways and compared again, so a stale-input fault cannot stay silent; the stats count the audits
and any mismatch (which refuses the key). The cache holds at most ``max_entries`` keys and ``pool_budget_bytes`` of
graph pools, the least recently used evicted first. It is not thread-safe.
"""

from __future__ import annotations

import time
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from anamnesis.extraction.fast.batched_attention import EIGVALSH_SIZED_OP
from anamnesis.extraction.fast.batched_stats import Counts, device_arch
from anamnesis.extraction.fast.features import GpuFeatureLane
from anamnesis.extraction.fast.graphed import (
    _SOLVERS,
    _copy_into,
    _layout,
    _pool_bytes,
    _record_tape,
    _Replica,
    _storage,
    _Tape,
)

LADDER: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
"""The batch sizes a bucket group is padded to: at most 1.5x rows above eight, so a group's fillers cost at most half
again its own work, and a bucket has twelve shapes."""

SOLVERS = _SOLVERS | {EIGVALSH_SIZED_OP}
"""The operators a capture cuts around: the dense solvers and the sized spectral solve."""


def ladder_size(rows: int) -> int | None:
    """The smallest :data:`LADDER` size that holds ``rows``; ``None`` above the largest (reduced eagerly)."""
    for size in LADDER:
        if rows <= size:
            return size
    return None


def _versions() -> tuple[str, str]:
    try:
        import triton

        triton_version = str(triton.__version__)
    except ImportError:  # no triton: the invariant ops run their torch path
        triton_version = "none"
    return str(torch.__version__), triton_version


@dataclass
class _Inputs:
    """A batch's inputs to the families, in the order their replicas hold them."""

    stacked: dict[str, Tensor]
    counts: Tensor
    starts: Tensor
    committed: list[Tensor]

    def tensors(self) -> list[Tensor]:
        return [self.stacked[n] for n in sorted(self.stacked)] + [self.counts, self.starts] + list(self.committed)

    def rebuilt(self, tensors: Sequence[Tensor]) -> "_Inputs":
        names = sorted(self.stacked)
        k = len(names)
        return _Inputs(dict(zip(names, tensors[:k], strict=True)), tensors[k], tensors[k + 1], list(tensors[k + 2:]))


def _padded(inputs: _Inputs, size: int) -> _Inputs:
    """``inputs`` with filler rows (copies of row 0) up to ``size`` rows."""
    rows = int(inputs.counts.shape[0])
    if rows == size:
        return inputs

    def pad(x: Tensor) -> Tensor:
        return torch.cat([x, x[:1].expand(size - rows, *x.shape[1:])], dim=0)
    return _Inputs({n: pad(x) for n, x in inputs.stacked.items()}, pad(inputs.counts), pad(inputs.starts),
                   [pad(c) for c in inputs.committed])


@dataclass
class _Entry:
    """One batch shape's recorded families."""

    lane: GpuFeatureLane
    tape: _Tape
    replicas: list[_Replica]
    static: _Inputs
    core: Any  # FamilyCore, its tensors in the graph's pool
    rows: int
    pool_bytes: int
    input_bytes: int
    capture_seconds: float
    replays: int = 0
    audits: int = 0


@dataclass
class BatchedGraphStats:
    entries: int
    refused: dict[str, str]
    captures: int
    replays: int
    audits: int
    audit_mismatches: int
    evictions: int
    pool_bytes: int
    input_bytes: int
    capture_seconds: float
    solver_cuts: dict[str, int] = field(default_factory=dict)


def _bitwise(a: Any, b: Any) -> list[str]:
    """Where two :class:`FamilyCore` differ (empty when every name and byte agree)."""
    out = []
    if tuple(a.names) != tuple(b.names) or tuple(a.routing_names) != tuple(b.routing_names):
        out.append("names")
    for name in ("values", "logprob", "routing_values", "knnlm"):
        x, y = getattr(a, name), getattr(b, name)
        if (x is None) != (y is None):
            out.append(name)
        elif x is not None and (x.shape != y.shape or x.dtype != y.dtype or not torch.equal(
                x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))):
            out.append(name)
    ra = a.retained.tolist() if isinstance(a.retained, Tensor) else list(a.retained)
    rb = b.retained.tolist() if isinstance(b.retained, Tensor) else list(b.retained)
    if ra != rb:
        out.append("retained")
    return out


class BatchedGraphCache:
    """Recorded batched families, keyed by batch shape (see the module docstring).

    ``max_entries`` and ``pool_budget_bytes`` bound what it holds (least recently used out first); ``audit_every``
    is how often a key's replay is checked against the eager reduction again (0: only at capture)."""

    def __init__(self, max_entries: int = 16, pool_budget_bytes: int = 8 << 30, audit_every: int = 256):
        if max_entries < 1:
            raise ValueError("a graph cache holds at least one entry")
        if pool_budget_bytes < 0 or audit_every < 0:
            raise ValueError("budgets and audit rates cannot be negative")
        self.max_entries, self.pool_budget_bytes, self.audit_every = max_entries, pool_budget_bytes, audit_every
        self.entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self.refused: dict[tuple, str] = {}
        self.captures = self.evictions = self.audit_mismatches = 0
        self._streams: dict[int, torch.cuda.Stream] = {}

    # ------------------------------------------------------------------ the one entry point

    def reduce(self, lane: GpuFeatureLane, stacked: Mapping[str, Tensor], counts: Counts, starts: Tensor,
               committed: Sequence[Tensor], first: Any):
        """The families of one padded bucket group (device-only ``counts``, device ``starts``), replayed when this
        batch's shape has a verified graph, else captured and verified, else reduced eagerly. Returns a
        :class:`~anamnesis.extraction.fast.engine_reducer.FamilyCore` of the batch's own rows, owned by the caller."""
        from anamnesis.extraction.fast.engine_reducer import _families

        def eager():
            return _families(lane, stacked, counts, starts, committed, first)

        rows = counts.rows
        size = ladder_size(rows)
        if (size is None or not counts.is_device_only or not isinstance(starts, Tensor)
                or lane.device.type != "cuda" or not torch.cuda.is_available()):
            return eager()
        inputs = _padded(_Inputs(dict(stacked), counts.dev, starts, list(committed)), size)
        tensors = inputs.tensors()
        if any(t.device != lane.device for t in tensors):
            return eager()
        key = self._key(lane, counts, size, first, inputs, tensors)
        if key in self.refused:
            return eager()
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            _copy_into(entry.replicas, tensors)
            entry.tape.replay()
            entry.replays += 1
            replayed = entry.core.rows(rows)
            if self.audit_every and entry.replays % self.audit_every == 0:
                entry.audits += 1
                reference = eager()
                differing = _bitwise(replayed, reference)
                if differing:
                    self.audit_mismatches += 1
                    self._refuse(key, f"audit: replay differs from the eager reduction in {differing}")
                    return reference
            return replayed
        return self._capture(key, lane, counts, size, first, inputs, tensors, eager)

    # ------------------------------------------------------------------ capture and verification

    def _capture(self, key, lane, counts, size, first, inputs, tensors, eager):
        from anamnesis.extraction.fast.engine_reducer import _families

        began = time.perf_counter()
        rows = counts.rows
        try:
            replicas = [_Replica(t) for t in tensors]
            _copy_into(replicas, tensors)
            static = inputs.rebuilt([r.view for r in replicas])
            data = {_storage(r.view) for r in replicas if r.extent}
            pool = torch.cuda.graph_pool_handle()

            def run():
                # Fresh counts each run: their memos (masks, per-row gathers) derive from the batch's counts, so the
                # capture must compute them, not reuse what the warm run left.
                static_counts = Counts(None, static.counts, counts.width, True, counts.floor)
                return _families(lane, static.stacked, static_counts, static.starts, static.committed, first)
            tape, core = _record_tape(lane.device, self._stream(lane.device), pool, data, warm=run, run=run,
                                      solvers=SOLVERS)
            tape.replay()
            replayed = core.rows(rows)
        except Exception as exc:  # any capture failure leaves this shape eager
            self._refuse(key, f"capture: {type(exc).__name__}: {exc}")
            torch.cuda.synchronize(lane.device)
            return eager()
        reference = eager()
        differing = _bitwise(replayed, reference)
        if differing:
            self._refuse(key, f"replay differs from the eager reduction in {differing}")
            return reference
        entry = _Entry(lane=lane, tape=tape, replicas=replicas, static=static, core=core, rows=size,
                       pool_bytes=_pool_bytes(pool),
                       input_bytes=sum(r.base.numel() * r.base.element_size() for r in replicas),
                       capture_seconds=time.perf_counter() - began)
        self._put(key, entry)
        return reference

    def _key(self, lane, counts, size, first, inputs, tensors) -> tuple:
        torch_version, triton_version = _versions()
        return (
            id(lane), getattr(lane, "calibration_sha256", None), counts.width, counts.floor, size,
            tuple(sorted(inputs.stacked)), len(inputs.committed),
            tuple(tuple(int(n) for n in pair) for pair in first.route_sources), int(first.final_sources),
            torch.cuda.current_stream(lane.device).stream_id,
            device_arch(lane.device), torch_version, triton_version,
            tuple(_layout(t) for t in tensors),
        )

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in self._streams:
            self._streams[index] = torch.cuda.Stream(device)
        return self._streams[index]

    def _put(self, key: tuple, entry: _Entry) -> None:
        self.entries[key] = entry
        self.entries.move_to_end(key)
        self.captures += 1
        while len(self.entries) > 1 and (len(self.entries) > self.max_entries
                                         or self.pool_bytes() > self.pool_budget_bytes):
            torch.cuda.synchronize()
            self.entries.popitem(last=False)
            self.evictions += 1

    def _refuse(self, key: tuple, reason: str) -> None:
        self.refused[key] = reason
        self.entries.pop(key, None)
        warnings.warn(f"batched families reduced eagerly for this shape: {reason}", RuntimeWarning, stacklevel=4)

    # ------------------------------------------------------------------ bookkeeping

    def pool_bytes(self) -> int:
        return sum(e.pool_bytes for e in self.entries.values())

    def clear(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.entries.clear()
        self.refused.clear()

    def stats(self) -> BatchedGraphStats:
        entries = list(self.entries.values())
        return BatchedGraphStats(
            entries=len(entries), refused={repr(k[2:5]): v for k, v in self.refused.items()},
            captures=self.captures, replays=sum(e.replays for e in entries), audits=sum(e.audits for e in entries),
            audit_mismatches=self.audit_mismatches, evictions=self.evictions, pool_bytes=self.pool_bytes(),
            input_bytes=sum(e.input_bytes for e in entries), capture_seconds=sum(e.capture_seconds for e in entries),
            solver_cuts={repr(k[2:5]): e.tape.solver_steps for k, e in self.entries.items()})


__all__ = ("LADDER", "SOLVERS", "BatchedGraphCache", "BatchedGraphStats", "ladder_size")
