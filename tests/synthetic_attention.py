"""References and synthetic paged batches for the vLLM lane's attention kernels.

The instrumented kernels in :mod:`anamnesis.extraction.vllm.stats_kernel` and
:mod:`anamnesis.extraction.vllm.second_pass` keep online softmax statistics that
nothing in the package recomputes, so the tests hold them to references kept here,
outside the package:

* the **materialized reference** — a float64 softmax over the whole row, its
  entropy and fixed-region probabilities. It is the correctness reference: the
  kernel must match it to accumulation tolerance, and a mismatch is a defect;
* the **rounded reference** — fp32 softmax rounded per entry to the model dtype
  before promotion, which is how the fast lane materializes attention. Its gap to
  the materialized reference is the arithmetic change the rounding introduces,
  reported beside a comparison and never scored as a defect;
* the **online simulation** — the kernel's tile recurrence (running maximum M,
  mass L, entropy moment E and region-mass accumulators, with the guarded
  arithmetic at absent old mass and at masked entries) replayed in a chosen
  dtype, so an algebra error in the update equations is visible apart from score
  formation;
* :func:`finalize_stats`, the completion arithmetic that turns the kernel's
  accumulators into entropy, region probabilities and log-mass.

Beside the references sit the synthetic cases: sequences with random or
axis-controlled scores (exact ties, peaked and extreme magnitudes), packed into
the engine's unified paged KV layout with the per-token region metadata the
kernel reads. Building a batch needs no device; launching one needs Triton and a
GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch import Tensor

from anamnesis.extraction.vllm.stats_kernel import (
    NUM_STATS,
    TILE_SIZE,
    unified_attention_stats,
)

REGION_NAMES = ("sink", "prompt", "recent", "gen_third_1", "gen_third_2", "gen_third_3")
"""The fixed key regions, restated here so the kernel's own order is checked
against an independent statement rather than against itself."""

BLOCK_SIZE = 16

# Kernel-versus-materialized ceilings. 'standard' covers well-scaled scores;
# 'extreme' covers score magnitudes where a half-ulp of fp32 score formation is
# itself of order 1e-3. Region probabilities live in [0, 1]; entropy is in nats;
# log-mass compares relatively with a floor of one. The output ceilings absorb the
# kernel's rounding of P to the model dtype before the PV product.
TOLERANCES = {
    "standard": dict(entropy=5e-3, region=1e-3, log_mass=1e-3),
    "extreme": dict(entropy=2e-2, region=2e-3, log_mass=5e-3),
}
OUTPUT_TOLERANCES = {torch.float16: 2e-3, torch.bfloat16: 1.2e-2}


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


def region_masks(
    *, keys: int, prefix_length: int, row_length: int, recency_cutoff: int, device=None,
) -> dict[str, Tensor]:
    """Fixed key-region masks for one query row, on absolute key positions.

    ``row_length`` keys are valid; the generated region ``[prefix_length,
    row_length)`` splits into thirds by the reference rule (floor division,
    clamped to at least one position, remainder in the final third).
    """
    if not 0 < prefix_length < row_length <= keys:
        raise ValueError("regions require 0 < prefix < row_length <= keys")
    if not 0 <= recency_cutoff <= row_length:
        raise ValueError("recency cutoff must sit inside the row")
    pos = torch.arange(keys, device=device)
    valid = pos < row_length
    third = max((row_length - prefix_length) // 3, 1)
    starts = (prefix_length, prefix_length + third, prefix_length + 2 * third)
    return dict(
        sink=(pos == 0) & valid,
        prompt=(pos < prefix_length) & valid,
        recent=(pos >= recency_cutoff) & valid,
        gen_third_1=(pos >= starts[0]) & (pos < starts[1]) & valid,
        gen_third_2=(pos >= starts[1]) & (pos < starts[2]) & valid,
        gen_third_3=(pos >= starts[2]) & valid,
    )


def _check_row(scores: Tensor, valid: Tensor) -> None:
    if scores.ndim != 1 or scores.shape != valid.shape:
        raise ValueError("one score row with a matching validity mask required")
    if not bool(valid.any()):
        raise ValueError("a fully masked row emits no statistics")
    if not bool(torch.isfinite(scores[valid]).all()):
        raise ValueError("valid scores must be finite")


def _region_probabilities(
    prob: Tensor, valid: Tensor, regions: dict[str, Tensor],
) -> dict[str, float]:
    masses = {}
    for name in REGION_NAMES:
        mask = regions[name]
        if mask.shape != valid.shape or bool((mask & ~valid).any()):
            raise ValueError(f"region {name} must lie inside the valid keys")
        masses[name] = float(prob[mask].sum())
    return masses


def materialized_stats(scores: Tensor, valid: Tensor, regions: dict[str, Tensor]) -> dict:
    """Float64 softmax entropy, region probabilities and log-mass over valid keys."""
    _check_row(scores, valid)
    s = scores.double()
    m = s[valid].max()
    p = torch.where(valid, (s - m).exp(), torch.zeros((), dtype=torch.float64))
    total = p.sum()
    prob = p / total
    entropy = float(-(prob.clamp_min(1e-300).log() * prob)[valid].sum())
    return dict(
        entropy=entropy,
        region_probabilities=_region_probabilities(prob, valid, regions),
        log_mass=float(total.log() + m),
    )


def rounded_reference_stats(
    scores: Tensor, valid: Tensor, regions: dict[str, Tensor], *, dtype: torch.dtype,
) -> dict:
    """Statistics of the row the fast lane materializes: rounded, then promoted.

    Eager attention materializes ``softmax(scores, dtype=float32)`` and rounds it
    to the model dtype before any reducer promotes it, so the fast lane's
    statistics are functions of the rounded row. This reproduces that at row
    level — fp32 softmax, per-entry rounding to ``dtype``, float64 promotion —
    then the materialized reference's entropy and region masses, normalized by the
    rounded row's own total.
    """
    _check_row(scores, valid)
    s = scores.float()
    m = s[valid].max()
    raw = torch.where(valid, (s - m).exp(), torch.zeros((), dtype=torch.float32))
    p = (raw / raw.sum()).to(dtype).double()
    total = p.sum()
    prob = p / total
    entropy = float(-(prob.clamp_min(1e-300).log() * prob)[valid].sum())
    return dict(
        entropy=entropy,
        region_probabilities=_region_probabilities(prob, valid, regions),
        rounded_total=float(total),
    )


def online_stats(
    scores: Tensor, valid: Tensor, regions: dict[str, Tensor], *, tile: int,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Simulate the kernel's tile loop: M, L, E and region accumulators.

    Per tile with running maximum m, alpha = exp(M - m) and P = exp(S - m)::

        E <- alpha * E + (M - m) * alpha * L + sum(P * (S - m))
        A <- alpha * A + sum(P * region)
        L <- alpha * L + sum(P)
        M <- m

    Guarded arithmetic, as the kernel implements it: while no mass has been seen
    (M still -inf) the rescale terms contribute zero rather than NaN, and masked
    entries contribute zero to every sum rather than 0 * (-inf). Completion:
    H = log(L) - E / L and region probability A / L.
    """
    if tile < 1:
        raise ValueError("tile width must be at least one")
    _check_row(scores, valid)
    s = scores.to(dtype)
    zero = torch.zeros((), dtype=dtype)
    running_max = torch.full((), -torch.inf, dtype=dtype)
    mass = zero.clone()
    moment = zero.clone()
    accum = {name: zero.clone() for name in REGION_NAMES}
    for start in range(0, len(s), tile):
        keep = valid[start:start + tile]
        if not bool(keep.any()):
            continue  # an all-masked tile changes nothing, including M
        block = s[start:start + tile]
        new_max = torch.maximum(running_max, block[keep].max())
        seen = bool(torch.isfinite(running_max))
        alpha = (running_max - new_max).exp() if seen else zero
        shifted = block - new_max
        p = torch.where(keep, shifted.exp(), zero)
        contribution = torch.where(keep, p * shifted, zero)
        rescale = alpha * (running_max - new_max) * mass if seen else zero
        moment = alpha * moment + rescale + contribution.sum()
        for name in REGION_NAMES:
            region = regions[name][start:start + tile]
            accum[name] = alpha * accum[name] + p[region & keep].sum()
        mass = alpha * mass + p.sum()
        running_max = new_max
    return dict(
        entropy=float(mass.log() - moment / mass),
        region_probabilities={name: float(accum[name] / mass) for name in REGION_NAMES},
        log_mass=float(mass.log() + running_max),
        tiles=-(-len(s) // tile),
    )


def finalize_stats(stats: Tensor) -> dict[str, Tensor]:
    """Complete the kernel's accumulators: entropy, region probabilities, log-mass.

    Promotion to float64 happens here, after the fp32 accumulators leave the
    kernel: H = log(L) - E/L, region probability A/L, log_mass = log(L) + M. NaN
    rows (never written: padding) stay NaN and are the caller's to exclude.
    """
    if stats.ndim < 1 or stats.shape[-1] != NUM_STATS:
        raise ValueError(f"statistics must carry {NUM_STATS} fields in the last dim")
    s = stats.double()
    running_max, mass, moment = s[..., 0], s[..., 1], s[..., 2]
    result = dict(entropy=mass.log() - moment / mass, log_mass=mass.log() + running_max)
    for i, name in enumerate(REGION_NAMES):
        result[name] = s[..., 3 + i] / mass
    return result


# --------------------------------------------------------------------------- #
# Synthetic sequences and paged batches
# --------------------------------------------------------------------------- #


@dataclass
class SequenceSpec:
    """One sequence: cached keys/values, the query chunk, region metadata."""

    keys: Tensor  # [S, num_kv_heads, head_size], model dtype
    values: Tensor  # [S, num_kv_heads, head_size], model dtype
    queries: Tensor  # [T, num_query_heads, head_size], model dtype
    prompt_length: int
    recent_window: int

    @property
    def context_length(self) -> int:
        return self.keys.shape[0] - self.queries.shape[0]

    def row_length(self, t: int) -> int:
        return self.context_length + t + 1

    def recency_cutoff(self, t: int) -> int:
        return max(self.row_length(t) - self.recent_window, 0)


@dataclass
class PagedBatch:
    """Sequences packed into the unified paged KV layout, with kernel metadata."""

    q: Tensor
    k_cache: Tensor
    v_cache: Tensor
    block_table: Tensor
    cu_seqlens_q: Tensor
    seqused_k: Tensor
    prefix_lengths: Tensor
    recency_cutoffs: Tensor
    seqs: list[SequenceSpec] = field(default_factory=list)


def paged_batch(seqs: list[SequenceSpec], *, device) -> PagedBatch:
    """Pack sequences into the unified paged KV layout plus per-token metadata.

    Block 0 stays zero so padded block-table entries are harmless, and each
    sequence gets two spare blocks so a table row is wider than its content.
    """
    if not seqs:
        raise ValueError("a batch requires at least one sequence")
    num_kv_heads, head_size = seqs[0].keys.shape[1:]
    dtype = seqs[0].keys.dtype
    blocks_per_seq = [-(-s.keys.shape[0] // BLOCK_SIZE) + 2 for s in seqs]
    num_blocks = sum(blocks_per_seq) + 1
    k_cache = torch.zeros(
        (num_blocks, BLOCK_SIZE, num_kv_heads, head_size), dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    table = torch.zeros((len(seqs), max(blocks_per_seq)), dtype=torch.int32, device=device)
    next_block = 1
    prefix: list[int] = []
    cutoffs: list[int] = []
    queries: list[Tensor] = []
    seq_lens: list[int] = []
    for i, seq in enumerate(seqs):
        s, t = seq.keys.shape[0], seq.queries.shape[0]
        if not 0 < seq.prompt_length <= s:
            raise ValueError("prompt boundary must sit inside the sequence")
        blocks = blocks_per_seq[i]
        table[i, :blocks] = torch.arange(next_block, next_block + blocks, dtype=torch.int32)
        flat_k = seq.keys.to(device)
        flat_v = seq.values.to(device)
        for j in range(-(-s // BLOCK_SIZE)):
            chunk = slice(j * BLOCK_SIZE, min((j + 1) * BLOCK_SIZE, s))
            width = chunk.stop - chunk.start
            k_cache[next_block + j, :width] = flat_k[chunk]
            v_cache[next_block + j, :width] = flat_v[chunk]
        next_block += blocks
        queries.append(seq.queries.to(device))
        seq_lens.append(s)
        prefix.extend([seq.prompt_length] * t)
        cutoffs.extend(seq.recency_cutoff(t_row) for t_row in range(t))
    starts = [0]
    for seq in seqs:
        starts.append(starts[-1] + seq.queries.shape[0])
    return PagedBatch(
        q=torch.cat(queries), k_cache=k_cache, v_cache=v_cache, block_table=table,
        cu_seqlens_q=torch.tensor(starts, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        prefix_lengths=torch.tensor(prefix, dtype=torch.int32, device=device),
        recency_cutoffs=torch.tensor(cutoffs, dtype=torch.int32, device=device),
        seqs=list(seqs),
    )


def run_batch(batch: PagedBatch, *, scale: float) -> tuple[Tensor, Tensor]:
    """Launch the instrumented kernel on a batch; return (output, statistics)."""
    out = torch.empty_like(batch.q)
    stats = torch.empty(
        (batch.q.shape[0], batch.q.shape[1], NUM_STATS),
        dtype=torch.float32, device=batch.q.device)
    unified_attention_stats(
        batch.q, batch.k_cache, batch.v_cache, out=out, stats=stats,
        cu_seqlens_q=batch.cu_seqlens_q, seqused_k=batch.seqused_k,
        block_table=batch.block_table, softmax_scale=scale,
        prefix_lengths=batch.prefix_lengths, recency_cutoffs=batch.recency_cutoffs)
    return out, stats


def scored_sequence(
    *, context: int, steps: int, num_q: int, num_kv: int, head_size: int, dtype,
    prompt: int, recent_window: int, seed: int, key_scores: Tensor | None = None,
    query_gain: float = 1.0,
) -> SequenceSpec:
    """Random K/V/Q, optionally with axis-0 score control.

    With ``key_scores`` given, the first head-dimension coordinate of every key is
    set so a first-coordinate query of ``query_gain`` produces exactly those score
    values (up to dtype rounding of the stored key), giving exact ties and
    controlled magnitudes; the remaining coordinates are zeroed so the scores are
    pure axis-0 products.
    """
    generator = torch.Generator().manual_seed(seed)
    total = context + steps
    keys = torch.randn((total, num_kv, head_size), generator=generator)
    values = torch.randn((total, num_kv, head_size), generator=generator)
    queries = torch.randn((steps, num_q, head_size), generator=generator)
    if key_scores is not None:
        if key_scores.shape != (total,):
            raise ValueError("one controlled score per key required")
        keys = torch.zeros_like(keys)
        keys[:, :, 0] = key_scores[:, None] / query_gain
        queries = torch.zeros_like(queries)
        queries[:, :, 0] = query_gain
    return SequenceSpec(
        keys=keys.to(dtype), values=values.to(dtype), queries=queries.to(dtype),
        prompt_length=prompt, recent_window=recent_window)


def reference_rows(
    seq: SequenceSpec, *, scale: float,
) -> Iterator[tuple[int, int, Tensor, dict[str, Tensor]]]:
    """Yield (t, head, float64 scores, regions) for every comparable span row.

    Rows inside the prompt have no generated region and are skipped.
    """
    num_q = seq.queries.shape[1]
    group = num_q // seq.keys.shape[1]
    k64 = seq.keys.double()
    for t in range(seq.queries.shape[0]):
        row_length = seq.row_length(t)
        if row_length <= seq.prompt_length:
            continue
        regions = region_masks(
            keys=row_length, prefix_length=seq.prompt_length, row_length=row_length,
            recency_cutoff=seq.recency_cutoff(t))
        for head in range(num_q):
            q64 = seq.queries[t, head].double()
            yield t, head, scale * (k64[:row_length, head // group] @ q64), regions


def evaluate_case(
    name: str, seqs: list[SequenceSpec], *, scale: float, tolerance_class: str, device,
) -> dict:
    """The kernel against the references over every row and head of a batch.

    Pass requires entropy, region probabilities, log-mass and the attention output
    to sit within the materialized reference's ceilings. The online simulation's
    distance and the rounded reference's gap are reported beside, not scored.
    """
    batch = paged_batch(seqs, device=device)
    out, stats = run_batch(batch, scale=scale)
    if bool(torch.isnan(stats).any()):
        nan_rows = int(torch.isnan(stats).flatten(1).any(1).sum())
        raise ValueError(f"case {name}: {nan_rows} unwritten statistics rows")
    final = finalize_stats(stats.cpu())
    out_cpu = out.cpu().double()
    bounds = TOLERANCES[tolerance_class]
    out_bound = OUTPUT_TOLERANCES[batch.q.dtype]
    errors = dict.fromkeys(("entropy", "region", "log_mass", "output", "online_entropy"), 0.0)
    gaps = dict.fromkeys(("entropy", "region"), 0.0)
    rows = 0
    token_base = 0
    for seq in batch.seqs:
        group = seq.queries.shape[1] // seq.keys.shape[1]
        for t, head, scores, regions in reference_rows(seq, scale=scale):
            token = token_base + t
            valid = torch.ones(scores.shape[0], dtype=torch.bool)
            reference = materialized_stats(scores, valid, regions)
            rounded = rounded_reference_stats(scores, valid, regions, dtype=batch.q.dtype)
            simulated = online_stats(scores, valid, regions, tile=TILE_SIZE, dtype=torch.float32)
            entropy = float(final["entropy"][token, head])
            errors["entropy"] = max(errors["entropy"], abs(entropy - reference["entropy"]))
            errors["online_entropy"] = max(
                errors["online_entropy"], abs(entropy - simulated["entropy"]))
            gaps["entropy"] = max(gaps["entropy"], abs(rounded["entropy"] - reference["entropy"]))
            for region in REGION_NAMES:
                expected = reference["region_probabilities"][region]
                errors["region"] = max(
                    errors["region"], abs(float(final[region][token, head]) - expected))
                gaps["region"] = max(
                    gaps["region"], abs(rounded["region_probabilities"][region] - expected))
            errors["log_mass"] = max(
                errors["log_mass"],
                abs(float(final["log_mass"][token, head]) - reference["log_mass"])
                / max(abs(reference["log_mass"]), 1.0))
            prob = (scores - scores.max()).exp()
            prob = prob / prob.sum()
            reference_out = prob @ seq.values[:scores.shape[0], head // group].double()
            errors["output"] = max(
                errors["output"],
                float((out_cpu[token, head] - reference_out).abs().max()
                      / max(float(reference_out.abs().max()), 1.0)))
            rows += 1
        token_base += seq.queries.shape[0]
    if not rows:
        raise ValueError(f"case {name} compared no rows")
    passed = (errors["entropy"] <= bounds["entropy"]
              and errors["region"] <= bounds["region"]
              and errors["log_mass"] <= bounds["log_mass"]
              and errors["output"] <= out_bound)
    return dict(name=name, rows_compared=rows, errors=errors, rounding_gap=gaps,
                output_bound=out_bound, passed=passed)


def bitwise_case(name: str, stats_a: Tensor, stats_b: Tensor) -> dict:
    """Two runs whose shared statistics must agree bit for bit (NaN pairs agree)."""
    if stats_a.shape != stats_b.shape:
        raise ValueError("bitwise comparison requires identical shapes")
    same = (stats_a == stats_b) | (stats_a.isnan() & stats_b.isnan())
    return dict(name=name, rows_compared=int(stats_a.shape[0]),
                mismatched_entries=int((~same).sum()), passed=bool(same.all()))
