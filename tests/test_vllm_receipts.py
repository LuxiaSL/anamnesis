"""The substrate a vLLM capture may hold, and the content receipts written for it.

:func:`anamnesis.extraction.vllm.receipts.assert_substrate_fields` is the gate
between a capture and the files the reduction reads: the fields must be exactly
:data:`~anamnesis.extraction.vllm.receipts.SUBSTRATE_FIELDS`, and any
engine-returned logprob is refused by name, so no feature path can read a sampler
output. The receipts hash native bytes, so they must separate what a precision
conversion would merge. All of this runs on CPU tensors; whether a device capture
produces these bytes is the install check's question.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.vllm.receipts import (
    SUBSTRATE_FIELDS,
    assert_substrate_fields,
    capture_receipt,
    tensor_receipt,
)


def test_substrate_fields_are_the_projections_and_the_attention_products() -> None:
    projections = {"hidden", "keys", "values", "queries", "gates", "logits", "chosen"}
    assert projections <= SUBSTRATE_FIELDS
    rest = SUBSTRATE_FIELDS - projections
    assert rest == {"attn_stats", "attn_coverage", "attn_h_mean", "attn_h_heads",
                    "attn_spectral_rows", "attn_decay_rows", "attn_span_rows",
                    "attn_entropy_rows", "attn_head_ent", "attn_head_sink",
                    "attn_head_prompt", "attn_head_recency"}


def test_substrate_fields_must_be_exact() -> None:
    assert_substrate_fields(dict.fromkeys(SUBSTRATE_FIELDS), context="row 0")
    assert_substrate_fields(sorted(SUBSTRATE_FIELDS), context="row 0")
    for field in ("logits", "attn_stats", "attn_span_rows", "attn_head_ent"):
        with pytest.raises(ValueError, match="missing") as caught:
            assert_substrate_fields(SUBSTRATE_FIELDS - {field}, context="row 0")
        assert field in str(caught.value)
    with pytest.raises(ValueError, match="extra") as caught:
        assert_substrate_fields(SUBSTRATE_FIELDS | {"router"}, context="row 0")
    assert "router" in str(caught.value)


def test_engine_returned_logprobs_are_refused_by_name() -> None:
    for field in ("logprobs", "prompt_logprobs", "LogProbs", "wrapper_logprob"):
        with pytest.raises(ValueError, match="never feature substrate") as caught:
            assert_substrate_fields(SUBSTRATE_FIELDS | {field}, context="row 3")
        assert field in str(caught.value)
        assert "row 3" in str(caught.value)


def test_tensor_receipt_retains_dtype_and_signed_zero() -> None:
    assert tensor_receipt(torch.tensor([0.])) != tensor_receipt(torch.tensor([-0.]))
    assert tensor_receipt(torch.ones(3, dtype=torch.bfloat16))["dtype"] == "torch.bfloat16"
    assert (tensor_receipt(torch.ones(3, dtype=torch.bfloat16))["sha256"]
            != tensor_receipt(torch.ones(3, dtype=torch.float16))["sha256"])
    x = torch.arange(12).reshape(3, 4).T
    assert tensor_receipt(x) == tensor_receipt(x.contiguous())
    assert tensor_receipt(x)["shape"] == [4, 3]


def test_capture_receipt_names_every_layer_and_is_stable() -> None:
    capture = dict(
        chosen=torch.tensor([1, 2]),
        keys={2: torch.ones(2, 3), 0: torch.zeros(2, 3)},
    )
    receipt = capture_receipt(capture)
    assert set(receipt["tensors"]) == {"chosen", "keys/0", "keys/2"}
    assert receipt == capture_receipt(dict(reversed(capture.items())))
    changed = dict(capture, keys={0: torch.zeros(2, 3), 2: torch.full((2, 3), 2.0)})
    assert capture_receipt(changed)["sha256"] != receipt["sha256"]
    assert capture_receipt(changed)["tensors"]["keys/0"] == receipt["tensors"]["keys/0"]
