"""Bounded second pass: normalized attention rows for selected query rows.

Head-mean coverage and the selected-row consumers need normalized per-head
probabilities, which the online first pass cannot produce (the threshold on
a probability depends on the final softmax denominator). This second pass
walks the SAME tile loop as the instrumented first pass — same prologue,
same masks, same score formation — and normalizes each tile with the FIRST
pass's final running maximum and mass, so its scores are the first pass's
scores by construction of the shared code path. The online invariant that
guards that construction is asserted, not trusted: every selected row's
probabilities must sum to one within the declared tolerance, and a breach
raises instead of renormalizing silently.

Selected rows write their per-head probability rows into a bounded scratch
of shape [selected rows, query heads, keys] — the correctness-first layout:
reduce to the head mean, count coverage, reuse the scratch. The head mean and
the strict threshold are :func:`coverage_reference`, which restates the coverage
expression of :class:`anamnesis.extraction.fast.attention.AttentionReducer` term
for term; ``tests/test_vllm_readout.py`` holds the two lanes' vectors to
identical bytes on the same rows.

The rounding switch: with ``round_to_model_dtype`` each head's normalized
probability is rounded to the model dtype before it is stored (promoted
back to fp32), reproducing the fast lane's per-head rounding of its
materialized attention; without it the probabilities stay plain fp32. The lane
runs with rounding on; the capture record names the setting.
"""
from __future__ import annotations

import torch
from torch import Tensor

from anamnesis.extraction.vllm.stats_kernel import HAVE_TRITON, SUPPORTED_DTYPES

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from anamnesis.extraction.vllm.stats_kernel import _cdiv_fn, _find_seq_idx

# The second pass accumulates each selected row's probability mass in fp32
# over up to the full context; the invariant must hold to accumulation
# error, and anything beyond it means the two passes diverged.
ROW_SUM_TOLERANCE = 1e-3

TILE_SIZE = 32


if HAVE_TRITON:

    @triton.jit
    def kernel_unified_second_pass_2d(
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        stats_ptr,  # [num_tokens, num_query_heads, NUM_STATS] from pass one
        row_slot_ptr,  # [num_tokens] int32, scratch slot or -1
        scratch_ptr,  # [num_slots, num_query_heads, key_width] fp32
        rowsum_ptr,  # [num_tokens, num_query_heads] fp32
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        scale,  # float32
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        stats_stride_0: tl.int64,
        stats_stride_1: tl.int64,
        scratch_stride_0: tl.int64,
        scratch_stride_1: tl.int64,
        rowsum_stride_0: tl.int64,
        BLOCK_SIZE: tl.constexpr,
        TILE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.constexpr,
        query_start_len_ptr,  # [num_seqs+1]
        BLOCK_Q: tl.constexpr,
        num_seqs: tl.int32,
        BLOCK_M: tl.constexpr,
        ROUND_TO_MODEL: tl.constexpr,
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

        slot = tl.load(row_slot_ptr + query_offset_0, mask=query_mask_0,
                       other=-1).to(tl.int32)
        selected = (slot >= 0) & query_mask_0 & query_mask_1
        if tl.max(selected.to(tl.int32)) == 0:
            return

        # Q : (BLOCK_M, HEAD_SIZE_PADDED)
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None]
            & query_mask_1[:, None],
            other=0.0,
        )

        block_table_offset = seq_idx * block_table_stride

        stats_base = (query_offset_0 * stats_stride_0
                      + query_offset_1 * stats_stride_1)
        lane_mask = query_mask_0 & query_mask_1
        M1 = tl.load(stats_ptr + stats_base + 0, mask=lane_mask, other=0.0)
        L1 = tl.load(stats_ptr + stats_base + 1, mask=lane_mask, other=1.0)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        context_len = seq_len - cur_batch_query_len

        rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)

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

            k_offset = (physical_block_idx[None, :] * stride_k_cache_0
                        + kv_head_idx * stride_k_cache_2
                        + offs_d[:, None] * stride_k_cache_3
                        + (seq_offset % BLOCK_SIZE)[None, :]
                        * stride_k_cache_1)

            # K : (HEAD_SIZE, TILE)
            K = tl.load(key_cache_ptr + k_offset,
                        mask=dim_mask[:, None] & tile_mask[None, :],
                        other=0.0)

            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = seq_offset[None, :] <= query_abs_pos

            # S : (BLOCK_M, TILE) — the first pass's score formation.
            S = tl.zeros(shape=(BLOCK_M, TILE), dtype=tl.float32)
            S += scale * tl.dot(Q, K)
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S,
                float('-inf'))

            # P : (BLOCK_M, TILE), normalized by the first pass's M and L.
            # The row-sum invariant guards score reproduction, so it
            # accumulates the unrounded probabilities; the rounding switch
            # applies to what is stored, and the stored row's own total (the
            # rounded total) is recoverable from the scratch.
            P = tl.exp(S - M1[:, None]) / L1[:, None]
            rowsum += tl.sum(P, axis=1)
            if ROUND_TO_MODEL:
                P = P.to(Q.dtype).to(tl.float32)

            scratch_offset = (slot[:, None] * scratch_stride_0
                              + query_offset_1[:, None] * scratch_stride_1
                              + seq_offset[None, :])
            tl.store(scratch_ptr + scratch_offset, P,
                     mask=selected[:, None] & tile_mask[None, :] & seq_mask)

        rowsum_offset = query_offset_0 * rowsum_stride_0 + query_offset_1
        tl.store(rowsum_ptr + rowsum_offset, rowsum, mask=selected)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def second_pass_rows(q: Tensor, k_cache: Tensor, *, stats: Tensor,
                     row_slots: Tensor, num_slots: int, cu_seqlens_q: Tensor,
                     seqused_k: Tensor, block_table: Tensor,
                     softmax_scale: float,
                     round_to_model_dtype: bool) -> tuple[Tensor, Tensor]:
    """Launch the second pass; return (scratch, rowsum) after the invariant.

    ``scratch`` is fp32 [num_slots, num_query_heads, key_width] with zeros
    beyond each row's valid keys; ``rowsum`` is fp32 [num_tokens,
    num_query_heads], NaN where the row was not selected. Every selected
    row-head sum must be one within ``ROW_SUM_TOLERANCE`` or the launch
    raises — a breach means the two passes diverged, and renormalizing it
    away would hide exactly the defect the invariant exists to catch.
    """
    _require(HAVE_TRITON, 'triton is required to launch the kernel')
    _require(q.ndim == 3, 'query must be [num_tokens, num_heads, head_size]')
    _require(q.dtype in SUPPORTED_DTYPES,
             'only the 16-bit model dtypes are supported')
    num_tokens, num_query_heads, head_size = q.shape
    _require(k_cache.ndim == 4 and k_cache.dtype == q.dtype
             and k_cache.shape[3] == head_size,
             'key cache must match the query dtype and head size')
    num_kv_heads = k_cache.shape[2]
    _require(num_query_heads % num_kv_heads == 0,
             'query heads must be a multiple of KV heads')
    _require(stats.ndim == 3 and stats.shape[0] == num_tokens
             and stats.shape[1] == num_query_heads
             and stats.dtype == torch.float32 and stats.stride(-1) == 1,
             'first-pass statistics for every token and head are required')
    _require(row_slots.shape == (num_tokens,)
             and row_slots.dtype == torch.int32,
             'row slots must be int32 with one entry per query token')
    _require(num_slots > 0, 'the second pass requires at least one slot')
    slots = row_slots[row_slots >= 0]
    _require(slots.numel() > 0, 'no rows are selected')
    _require(int(slots.max()) < num_slots, 'a slot exceeds the scratch')
    _require(int(torch.bincount(slots, minlength=num_slots).max()) <= 1,
             'each scratch slot may be assigned to at most one row')
    key_width = int(seqused_k.max())
    scratch = torch.zeros((num_slots, num_query_heads, key_width),
                          dtype=torch.float32, device=q.device)
    rowsum = torch.full((num_tokens, num_query_heads), float('nan'),
                        dtype=torch.float32, device=q.device)

    num_seqs = seqused_k.numel()
    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv <= 16:
        block_m = 16
    else:
        block_m = triton.next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = num_tokens // block_q + num_seqs

    kernel_unified_second_pass_2d[(total_num_q_blocks, num_kv_heads)](
        query_ptr=q,
        key_cache_ptr=k_cache,
        stats_ptr=stats,
        row_slot_ptr=row_slots,
        scratch_ptr=scratch,
        rowsum_ptr=rowsum,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        stats_stride_0=stats.stride(0),
        stats_stride_1=stats.stride(1),
        scratch_stride_0=scratch.stride(0),
        scratch_stride_1=scratch.stride(1),
        rowsum_stride_0=rowsum.stride(0),
        BLOCK_SIZE=k_cache.shape[1],
        TILE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        ROUND_TO_MODEL=round_to_model_dtype,
    )

    selected_sums = rowsum[row_slots >= 0]
    if not bool(torch.isfinite(selected_sums).all()):
        raise RuntimeError('a selected row emitted no finite row sum; the '
                           'second pass never wrote it')
    worst = float((selected_sums - 1.0).abs().max())
    if worst > ROW_SUM_TOLERANCE:
        raise RuntimeError(
            f'second-pass row sums deviate from one by {worst:.3e} '
            f'(tolerance {ROW_SUM_TOLERANCE:.0e}); the passes diverged')
    return scratch, rowsum


def coverage_reference(mean: Tensor, lengths: Tensor, valid: Tensor) -> Tensor:
    """Coverage of head-mean attention rows: the share of each row's valid keys
    whose weight strictly exceeds uniform, 1/row_length, compared in float64.

    ``mean`` is [rows, keys], ``lengths`` the valid key count per row, ``valid``
    the [rows, keys] mask of keys inside each row. Returns float64 [rows].
    """
    return ((mean.double() > 1 / lengths.double()[:, None]) & valid).sum(
        dim=-1) / lengths.double()


def coverage_of_selected_rows(scratch: Tensor, row_lengths: Tensor) -> Tensor:
    """The banked coverage of each selected row's head-mean probability row.

    The head mean is taken in fp32 over all query heads, promoted to double,
    and thresholded strictly against 1/row_length over valid keys, through
    :func:`coverage_reference`.
    """
    _require(scratch.ndim == 3 and scratch.dtype == torch.float32,
             'scratch must be fp32 [rows, heads, keys]')
    _require(row_lengths.shape == (scratch.shape[0],),
             'one row length per selected row is required')
    lengths = row_lengths.to(scratch.device)
    _require(bool((lengths > 0).all())
             and int(lengths.max()) <= scratch.shape[2],
             'row lengths must fit the scratch width')
    mean32 = scratch.mean(dim=1)
    positions = torch.arange(scratch.shape[2], device=scratch.device)
    valid = positions[None, :] < lengths[:, None]
    return coverage_reference(mean32, lengths, valid)
