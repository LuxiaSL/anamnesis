"""Instrumented unified-attention kernel: online row statistics in-loop.

This module carries the instrumented copy of vLLM's 2D unified-attention
Triton kernel with per-row attention statistics accumulated inside the same
tile loop that computes the attention output. The statistics are the online
softmax quantities: running maximum M, mass L, entropy moment E and the six
fixed region masses of :data:`REGION_NAMES`, per query token and query head,
all in fp32 accumulators. They complete to the row's entropy H = log(L) - E/L
and its region probabilities A/L; the tests hold the update equations and the
compiled kernel to a materialized float64 reference.

Documented delta against the upstream source
--------------------------------------------
Base: the pinned engine's unified-attention Triton module (vLLM 0.16.0,
importable as ``vllm.v1.attention.ops.triton_unified_attention``, source
sha256 ``8e8d393c4d551547de397859462ad7c3750230841458e658d73381dbf3f59005``
= :data:`UPSTREAM_KERNEL_SHA256`), function ``kernel_unified_attention_2d``
and its launch arithmetic. The instrumented kernel keeps the upstream structure, names and
update order exactly, with these changes:

- REMOVED upstream modes outside the lane scope, refused at launch instead of
  silently diverging: sinks, alibi (both variants), query-query bias,
  softcap, sliding window, multimodal prefix ranges, fp8 KV cache and fp8
  output quantization, cascade and encoder paths, and the 3D split-KV kernel
  entirely. The launcher accepts only the causal 16/32-bit decoder
  configuration the lane runs; every removed branch was constexpr-dead
  under that configuration, so the compiled arithmetic of the retained path
  is the upstream arithmetic.
- ADDED per-lane accumulators E and one mass per fixed key region, updated
  with the tile's own m_j and alpha between their computation and the L/M
  update, with guarded arithmetic: the rescale term is zero while no mass has
  been seen (M still -inf) and masked entries contribute exactly zero instead
  of 0 * (-inf). ADDED a statistics epilogue storing (M, L, E, regions) per
  query token and head under the same validity masks as the output store;
  padding lanes are never stored, and the launcher pre-fills the statistics
  tensor with NaN so an unwritten entry can never read as a statistic.
- ADDED per-token metadata inputs: the absolute prompt boundary and the
  recency cutoff, from which the generated-region thirds are derived in-kernel
  by the reference rule (floor division, clamped to at least one position,
  remainder in the final third).

Batch invariance of the statistics follows the kernel's own structure: one
program owns a (query block, kv head) lane and walks tiles sequentially in a
fixed order that depends only on the row itself, so every accumulator's
reduction order is fixed per sequence regardless of batch composition.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:  # Triton is required to launch, not to import the host helpers.
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only without triton
    HAVE_TRITON = False

UPSTREAM_KERNEL_SHA256 = (
    '8e8d393c4d551547de397859462ad7c3750230841458e658d73381dbf3f59005')
"""sha256 of the pinned engine's ``triton_unified_attention.py``, the source this
kernel is an instrumented copy of."""

REGION_NAMES = ('sink', 'prompt', 'recent', 'gen_third_1', 'gen_third_2',
                'gen_third_3')
"""The fixed key regions, in the order their masses are stored: the first key,
the prompt, the recency window, and the generated region's three thirds."""

# Statistics layout per (query token, query head): running max, mass,
# entropy moment, then the six region masses in REGION_NAMES order.
STAT_FIELDS = ('running_max', 'mass', 'entropy_moment') + REGION_NAMES
NUM_STATS = len(STAT_FIELDS)

# The 2D kernel's tile width for 16/32-bit inputs (the upstream prefill
# choice; the invariant configuration always dispatches 2D).
TILE_SIZE = 32

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


if HAVE_TRITON:

    @triton.jit
    def _cdiv_fn(x, y):
        return (x + y - 1) // y

    @triton.jit
    def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs,
                      BLOCK_Q: tl.constexpr):
        left: tl.int32 = 0
        right = num_seqs
        while left < right:
            mid = (left + right) // 2
            val = tl.load(query_start_len_ptr + mid)
            mid_val = val // BLOCK_Q + mid
            if mid_val <= target_idx:
                left = mid + 1
            else:
                right = mid
        return left - 1

    @triton.jit
    def kernel_unified_attention_stats_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        stats_ptr,  # [num_tokens, num_query_heads, NUM_STATS] fp32
        prefix_len_ptr,  # [num_tokens] int32, absolute prompt boundary
        recency_cutoff_ptr,  # [num_tokens] int32, absolute recency cutoff
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        scale,  # float32
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        output_stride_0: tl.int64,
        output_stride_1: tl.int64,
        stats_stride_0: tl.int64,
        stats_stride_1: tl.int64,
        BLOCK_SIZE: tl.constexpr,
        TILE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.constexpr,
        stride_v_cache_0: tl.int64,
        stride_v_cache_1: tl.int64,
        stride_v_cache_2: tl.int64,
        stride_v_cache_3: tl.constexpr,
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr,
        num_seqs: tl.int32,
        BLOCK_M: tl.constexpr,
    ):
        q_block_global_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)

        seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx,
                                num_seqs, BLOCK_Q)
        q_block_start_idx = (tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q
                             + seq_idx)
        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx
                                              + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index
                               - cur_batch_in_all_start_index)

        if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
            return

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        offs_t = tl.arange(0, TILE)
        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

        query_offset_0 = cur_batch_in_all_start_index + query_pos
        query_offset_1 = (kv_head_idx * num_queries_per_kv
                          + offs_m % num_queries_per_kv)
        query_offset = (query_offset_0[:, None] * query_stride_0
                        + query_offset_1[:, None] * query_stride_1
                        + offs_d[None, :])

        dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
        query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1,
                                0).to(tl.int1)
        query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1,
                                0).to(tl.int1)

        # Q : (BLOCK_M, HEAD_SIZE_PADDED)
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None]
            & query_mask_1[:, None],
            other=0.0,
        )

        block_table_offset = seq_idx * block_table_stride

        M = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

        # Statistics accumulators, one per (query row, head) lane.
        E = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_sink = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_prompt = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_recent = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_g1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_g2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        A_g3 = tl.zeros([BLOCK_M], dtype=tl.float32)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        context_len = seq_len - cur_batch_query_len

        # Per-lane region geometry from the token metadata: the generated
        # region [prefix, row_length) splits into thirds by the reference
        # rule (floor division, clamped to at least one, remainder last).
        prefix = tl.load(prefix_len_ptr + query_offset_0, mask=query_mask_0,
                         other=1).to(tl.int32)
        cutoff = tl.load(recency_cutoff_ptr + query_offset_0,
                         mask=query_mask_0, other=0).to(tl.int32)
        row_length = context_len + query_pos + 1
        third = tl.maximum((row_length - prefix) // 3, 1)
        g2_start = prefix + third
        g3_start = prefix + 2 * third

        max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q
                              + (BLOCK_M - 1) // num_queries_per_kv + 1)
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
        num_tiles = _cdiv_fn(max_seq_prefix_len, TILE)

        for j in range(0, num_tiles):
            seq_offset = j * TILE + offs_t
            tile_mask = seq_offset < max_seq_prefix_len

            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset
                + seq_offset // BLOCK_SIZE).to(tl.int64)

            v_offset = (physical_block_idx[:, None] * stride_v_cache_0
                        + kv_head_idx * stride_v_cache_2
                        + offs_d[None, :] * stride_v_cache_3
                        + (seq_offset % BLOCK_SIZE)[:, None]
                        * stride_v_cache_1)
            k_offset = (physical_block_idx[None, :] * stride_k_cache_0
                        + kv_head_idx * stride_k_cache_2
                        + offs_d[:, None] * stride_k_cache_3
                        + (seq_offset % BLOCK_SIZE)[None, :]
                        * stride_k_cache_1)

            # K : (HEAD_SIZE, TILE)
            K = tl.load(key_cache_ptr + k_offset,
                        mask=dim_mask[:, None] & tile_mask[None, :],
                        other=0.0)
            # V : (TILE, HEAD_SIZE)
            V = tl.load(value_cache_ptr + v_offset,
                        mask=dim_mask[None, :] & tile_mask[:, None],
                        other=0.0)

            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = seq_offset[None, :] <= query_abs_pos

            # S : (BLOCK_M, TILE)
            S = tl.zeros(shape=(BLOCK_M, TILE), dtype=tl.float32)
            S += scale * tl.dot(Q, K)
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S,
                float('-inf'))

            # m_j : (BLOCK_M,)
            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float('-inf'), m_j, 0.0)

            # P : (BLOCK_M, TILE)
            P = tl.exp(S - m_j[:, None])
            # l_j : (BLOCK_M,)
            l_j = tl.sum(P, axis=1)
            # alpha : (BLOCK_M,)
            alpha = tl.exp(M - m_j)

            # Statistics updates use the OLD M and L with this tile's alpha.
            # The rescale term is zero while no mass has been seen, and
            # masked entries (P == 0) contribute exactly zero rather than
            # 0 * (-inf).
            shifted = S - m_j[:, None]
            tile_moment = tl.sum(tl.where(P > 0.0, P * shifted, 0.0), axis=1)
            rescale = tl.where(M > float('-inf'), (M - m_j) * alpha * L, 0.0)
            E = alpha * E + rescale + tile_moment

            pos = seq_offset[None, :]
            A_sink = alpha * A_sink + tl.sum(tl.where(pos == 0, P, 0.0),
                                             axis=1)
            A_prompt = alpha * A_prompt + tl.sum(
                tl.where(pos < prefix[:, None], P, 0.0), axis=1)
            A_recent = alpha * A_recent + tl.sum(
                tl.where(pos >= cutoff[:, None], P, 0.0), axis=1)
            A_g1 = alpha * A_g1 + tl.sum(
                tl.where((pos >= prefix[:, None]) & (pos < g2_start[:, None]),
                         P, 0.0), axis=1)
            A_g2 = alpha * A_g2 + tl.sum(
                tl.where((pos >= g2_start[:, None])
                         & (pos < g3_start[:, None]), P, 0.0), axis=1)
            A_g3 = alpha * A_g3 + tl.sum(
                tl.where(pos >= g3_start[:, None], P, 0.0), axis=1)

            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V.dtype), V)

        # epilogue
        acc = acc / L[:, None]

        output_offset = (query_offset_0[:, None] * output_stride_0
                         + query_offset_1[:, None] * output_stride_1
                         + offs_d[None, :])
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None]
            & query_mask_1[:, None],
        )

        stats_mask = query_mask_0 & query_mask_1
        stats_base = (query_offset_0 * stats_stride_0
                      + query_offset_1 * stats_stride_1)
        tl.store(stats_ptr + stats_base + 0, M, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 1, L, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 2, E, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 3, A_sink, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 4, A_prompt, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 5, A_recent, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 6, A_g1, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 7, A_g2, mask=stats_mask)
        tl.store(stats_ptr + stats_base + 8, A_g3, mask=stats_mask)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unified_attention_stats(q: Tensor, k_cache: Tensor, v_cache: Tensor, *,
                            out: Tensor, stats: Tensor, cu_seqlens_q: Tensor,
                            seqused_k: Tensor, block_table: Tensor,
                            softmax_scale: float, prefix_lengths: Tensor,
                            recency_cutoffs: Tensor) -> None:
    """Launch the instrumented 2D kernel on the lane's supported shapes.

    Only the causal decoder configuration the lane runs is accepted:
    16-bit model dtype, unquantized KV cache in the unified layout, no sinks,
    no alibi, no softcap, no sliding window, no bias. Anything else is
    refused here rather than computed differently from the reference.
    ``stats`` is pre-filled with NaN so entries the kernel never writes
    (padding) can never be mistaken for statistics.
    """
    _require(HAVE_TRITON, 'triton is required to launch the kernel')
    _require(q.ndim == 3, 'query must be [num_tokens, num_heads, head_size]')
    _require(q.dtype in SUPPORTED_DTYPES,
             'only the 16-bit model dtypes are supported; other dtypes are '
             'refused rather than silently diverging from the reference')
    num_tokens, num_query_heads, head_size = q.shape
    _require(k_cache.ndim == 4 and v_cache.shape == k_cache.shape,
             'KV cache must be [num_blocks, block_size, num_kv_heads, '
             'head_size] with matching key and value shapes')
    _require(k_cache.dtype == q.dtype and v_cache.dtype == q.dtype,
             'a KV cache quantized away from the model dtype is refused')
    num_kv_heads = k_cache.shape[2]
    _require(k_cache.shape[3] == head_size, 'head size mismatch with cache')
    _require(num_query_heads % num_kv_heads == 0,
             'query heads must be a multiple of KV heads')
    _require(out.shape == q.shape and out.dtype == q.dtype,
             'output must match the query shape and dtype')
    _require(
        stats.shape == (num_tokens, num_query_heads, NUM_STATS)
        and stats.dtype == torch.float32 and stats.stride(-1) == 1,
        'statistics must be fp32 [num_tokens, num_query_heads, '
        f'{NUM_STATS}] with a contiguous last dimension')
    _require(cu_seqlens_q.ndim == 1 and seqused_k.ndim == 1
             and cu_seqlens_q.numel() == seqused_k.numel() + 1,
             'cu_seqlens_q must have one entry more than seqused_k')
    _require(cu_seqlens_q.dtype == torch.int32
             and seqused_k.dtype == torch.int32, 'sequence metadata is int32')
    for name, tensor in (('prefix_lengths', prefix_lengths),
                         ('recency_cutoffs', recency_cutoffs)):
        _require(tensor.shape == (num_tokens,)
                 and tensor.dtype == torch.int32,
                 f'{name} must be int32 with one entry per query token')
    _require(block_table.ndim == 2, 'block table must be 2D')

    num_seqs = seqused_k.numel()
    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv <= 16:
        block_m = 16
    else:
        block_m = triton.next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = num_tokens // block_q + num_seqs

    stats.fill_(float('nan'))

    kernel_unified_attention_stats_2d[(total_num_q_blocks, num_kv_heads)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        stats_ptr=stats,
        prefix_len_ptr=prefix_lengths,
        recency_cutoff_ptr=recency_cutoffs,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        stats_stride_0=stats.stride(0),
        stats_stride_1=stats.stride(1),
        BLOCK_SIZE=k_cache.shape[1],
        TILE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
    )

