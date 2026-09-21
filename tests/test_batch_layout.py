"""Batch packing: padding that is invisible to every position the model reads.

Packing ragged spans into one forward has to satisfy two things at once. Physically,
rows share a tensor width and a cache index. Semantically, every row must see exactly
the absolute positions it would see alone, because RoPE and every positional feature
read those. The layout separates the two: prefixes are left-padded so each row's real
prefix ends at the cache boundary, continuations are right-padded, and `position_ids`
carry per-row absolute positions rather than physical offsets.

`attention_slices` is the inverse — the query and key slices that recover one row's
unpadded attention block — and the first test checks it by round-tripping: the tokens
recovered through the slice are the row's own tokens, and every position it selects is
masked in. Getting that slice wrong shifts a row's whole attention feature set without
raising.

The composition digest is identity, not a checksum of the arrays: reordering the rows
changes it (the same rows in a different batch are a different batch), while tokens past
`end` do not, because a span's identity is the text that was read, not the buffer it
arrived in.

The last test runs a real ragged forward and asserts that the batched logits, attention
and hidden states equal the single-row ones through those slices. That is the claim the
packing exists to make, checked against the model rather than against the arithmetic.
"""

import numpy as np
import pytest

from anamnesis.extraction.fast.batch_layout import ReplaySpan, pack_spans


def test_ragged_prefix_and_continuation_alignment():
    spans = (ReplaySpan((1, 2, 3, 4, 5), 2, 5), ReplaySpan((6, 7, 8, 9, 10, 11), 4, 6))
    layout = pack_spans(spans, 20)
    assert layout.prefix_ids.tolist() == [[0, 0, 1, 2], [6, 7, 8, 9]]
    assert layout.prefix_positions.tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
    assert layout.continuation_ids.tolist() == [[3, 4, 5], [10, 11, 0]]
    assert layout.continuation_positions.tolist() == [[2, 3, 4], [4, 5, 0]]
    assert layout.attention_mask.tolist() == [
        [0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
    ]
    for i, span in enumerate(spans):
        q, k = layout.attention_slices(i)
        assert len(range(layout.continuation_width)[q]) == span.end - span.start
        physical = np.concatenate((layout.prefix_ids[i], layout.continuation_ids[i]))
        assert physical[k].tolist() == list(span.tokens[: span.end])
        assert layout.attention_mask[i, k].all()


def test_batch_order_recorded_without_changing_each_rows_tokens():
    spans = (ReplaySpan((1, 2, 3), 1, 3), ReplaySpan((4, 5, 6, 7), 2, 4))
    a, b = pack_spans(spans, 10), pack_spans(spans[::-1], 10)
    assert a.composition_sha256 != b.composition_sha256
    np.testing.assert_array_equal(a.prefix_ids, b.prefix_ids[::-1])
    np.testing.assert_array_equal(
        a.continuation_positions, b.continuation_positions[::-1]
    )


@pytest.mark.parametrize(
    "span",
    [
        ReplaySpan((1, 2), 0, 2),
        ReplaySpan((1, 2), 1, 2),
        ReplaySpan((1, -1, 3), 1, 3),
        ReplaySpan((1, 2, 99), 1, 3),
    ],
)
def test_bad_span_rejected(span):
    with pytest.raises(ValueError):
        pack_spans((span,), 10)


def test_unread_suffix_does_not_change_identity():
    a = pack_spans((ReplaySpan((1, 2, 3), 1, 3),), 10)
    b = pack_spans((ReplaySpan((1, 2, 3, 999), 1, 3),), 10)
    assert a.composition_sha256 == b.composition_sha256


def test_ragged_cached_forward_matches_individual_positions_and_attention():
    import torch
    from test_fast_lane_equivalence import tiny_loaded

    loaded = tiny_loaded()
    loaded.disable_hooks()
    spans = (
        ReplaySpan(tuple(range(1, 20)), 3, 12),
        ReplaySpan(tuple(range(20, 40)), 11, 16),
    )
    layout = pack_spans(spans, 64)
    with torch.no_grad():
        pre = loaded.model(
            torch.from_numpy(layout.prefix_ids),
            attention_mask=torch.from_numpy(layout.prefix_mask),
            position_ids=torch.from_numpy(layout.prefix_positions),
            use_cache=True,
            return_dict=True,
        )
        result = loaded.model(
            torch.from_numpy(layout.continuation_ids),
            attention_mask=torch.from_numpy(layout.attention_mask),
            position_ids=torch.from_numpy(layout.continuation_positions),
            cache_position=torch.arange(
                layout.prefix_width, layout.prefix_width + layout.continuation_width
            ),
            past_key_values=pre.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        for i, span in enumerate(spans):
            prefix = loaded.model(
                torch.tensor([span.tokens[: span.start]]),
                use_cache=True,
                return_dict=True,
            )
            single = loaded.model(
                torch.tensor([span.tokens[span.start : span.end]]),
                past_key_values=prefix.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            q, k = layout.attention_slices(i)
            torch.testing.assert_close(
                result.logits[i, q], single.logits[0], atol=1e-6, rtol=1e-5
            )
            for actual, expected in zip(
                result.attentions, single.attentions, strict=True
            ):
                torch.testing.assert_close(
                    actual[i, :, q, k], expected[0], atol=1e-6, rtol=1e-5
                )
            for actual, expected in zip(
                result.hidden_states, single.hidden_states, strict=True
            ):
                torch.testing.assert_close(
                    actual[i, q], expected[0], atol=1e-6, rtol=1e-5
                )
