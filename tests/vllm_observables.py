"""Synthetic attention rows and the capture's view of them, for the vLLM lane's tests.

The fast lane's :class:`anamnesis.extraction.fast.attention.AttentionReducer`
reduces a layer's materialized attention weights, shaped the way an eager forward
returns them. The vLLM lane never materializes those weights: its capture holds
the first pass's per-row statistics and the second pass's reduced products. This
module builds both views of the *same* rows on the CPU, so a test can hand one to
the reducer and the other to :class:`anamnesis.extraction.vllm.adapter.AttentionFeatureAdapter`
and ask whether the two agree.

The products come from the package's own product code
(:mod:`anamnesis.extraction.vllm.step_products` and
:func:`anamnesis.extraction.vllm.second_pass.coverage_of_selected_rows`) run on
rows taken from the materialized weights, which is what the second pass writes
on a device. What this cannot show is that the device kernels write those rows:
that needs the instrumented kernel on a GPU.
"""

from __future__ import annotations

import torch

from anamnesis.extraction.vllm.rows import request_row_schema
from anamnesis.extraction.vllm.second_pass import coverage_of_selected_rows
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS, STAT_FIELDS
from anamnesis.extraction.vllm.step_products import (
    agreement_pair,
    entropy_of_rows,
    per_head_summaries,
)


def span_weights(t: int, c: int, heads: int, seed: int) -> torch.Tensor:
    """Eager-shaped ``[1, heads, t+1, c+t+1]`` causal softmax rows.

    ``t`` is the number of predicted positions and ``c`` the prompt length; row
    ``i`` attends to the first ``c+1+i`` keys, and the final row and column are
    the ones the reducer masks.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, heads, t + 1, c + t + 1, generator=g).double()
    pos = torch.arange(c + t + 1)
    lengths = torch.arange(c + 1, c + t + 2)
    valid = pos[None, :] < lengths[:, None]
    logits = logits.masked_fill(~valid[None, None], -torch.inf)
    return torch.softmax(logits, dim=-1).float()


def engine_observables(weights: torch.Tensor, *, c: int) -> dict:
    """One layer's capture fields for the rows in ``weights``, keyed as the adapter takes them.

    The statistics carry a mass of one and each row's entropy moment; the adapter
    reads them for their head count and finiteness, and every attention
    coordinate it emits comes from the products beside them.
    """
    t = weights.shape[2] - 1
    schema = request_row_schema(prompt_length=c, end=c + t + 1)
    a = weights[0, :, :t, :].permute(1, 0, 2).float()
    pos = torch.arange(a.shape[-1])
    lengths = torch.arange(c + 1, c + t + 1)
    valid = pos[None, :] < lengths[:, None]
    a = a * valid[:, None, :]
    p = a.double()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    stats = torch.zeros(t, a.shape[1], NUM_STATS)
    stats[..., STAT_FIELDS.index("mass")] = 1.0
    stats[..., STAT_FIELDS.index("entropy_moment")] = (
        (p * p.clamp_min(1e-300).log()).sum(dim=-1).float())
    h_mean, h_heads = agreement_pair(a[list(schema.agreement)])
    mean32 = a.mean(dim=1)
    width = schema.end - 1
    summaries = per_head_summaries(
        a, lengths=torch.arange(c + 1, c + t + 1, dtype=torch.int32),
        prefixes=torch.full((t,), c, dtype=torch.int32),
        widths=torch.full((t,), c + t + 1, dtype=torch.int32))
    return dict(
        schema=schema,
        stats=stats,
        h_mean=h_mean,
        h_heads=h_heads,
        coverage=coverage_of_selected_rows(a, lengths),
        spectral_rows=mean32[list(schema.spectral)][:, :width].contiguous(),
        decay_rows=mean32[list(schema.decay)][:, :width].contiguous(),
        entropy_rows=entropy_of_rows(
            a[list(schema.entropy)],
            widths=torch.full((len(schema.entropy),), c + t + 1, dtype=torch.int32)),
        span_rows=mean32[:, :width].contiguous(),
        **summaries,
    )


SAMPLED_KEYS = ("coverage", "spectral_rows", "decay_rows", "span_rows",
                "head_ent", "head_sink", "head_prompt", "head_recency")
"""The observables only a sampled layer carries."""


def consume_arguments(observed: dict, *, sampled: bool) -> dict:
    """The keyword arguments ``AttentionFeatureAdapter.consume`` takes for one layer."""
    keys = ("stats", "h_mean", "h_heads", "entropy_rows")
    if sampled:
        keys += SAMPLED_KEYS
    return {key: observed[key] for key in keys}
