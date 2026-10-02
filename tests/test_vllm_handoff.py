"""The in-memory hand-off: a captured row's substrate through a shared-memory segment.

:mod:`anamnesis.extraction.vllm.handoff` is plain files, ``mmap`` and torch on the host,
so every property it has is checked here on the CPU: the segment carries every tensor's
native bytes, shape and dtype, so a content receipt over the segment equals the receipt
over the capture it was written from; and the readout side refuses any handle that is not
the one the segment was published for, before it reads a tensor.
"""

from __future__ import annotations

import json

import pytest
import torch

from anamnesis.extraction.vllm import handoff
from anamnesis.extraction.vllm.receipts import SUBSTRATE_FIELDS, capture_receipt

LAYERED = sorted(SUBSTRATE_FIELDS - {"hidden", "logits", "chosen"})


def substrate(seed: int = 0) -> dict:
    """Every substrate field, in the dtypes a capture holds them in, with an empty one."""
    g = torch.Generator().manual_seed(seed)
    capture = dict(
        hidden=torch.randn(3, 5, 8, generator=g).to(torch.bfloat16),
        logits=torch.randn(5, 11, generator=g),
        chosen=torch.randint(0, 11, (5,), generator=g, dtype=torch.int64),
    )
    for name in LAYERED:
        capture[name] = {layer: torch.randn(5, 4, generator=g).to(torch.bfloat16)
                         for layer in (0, 2)}
    capture["attn_spectral_rows"][2] = torch.zeros((0, 4), dtype=torch.float32)
    capture["attn_h_mean"][0] = torch.randn(7, generator=g, dtype=torch.float64)
    return capture


def same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, dict):
            if set(x) != set(y) or not all(
                    x[k].dtype == y[k].dtype and x[k].shape == y[k].shape
                    and torch.equal(x[k].view(torch.uint8) if x[k].numel() else x[k],
                                    y[k].view(torch.uint8) if y[k].numel() else y[k])
                    for k in x):
                return False
        elif x.dtype != y.dtype or not torch.equal(x, y):
            return False
    return True


def test_a_segment_carries_every_tensor_byte_for_byte(tmp_path):
    capture = substrate()
    handle, view = handoff.publish(capture, tmp_path, generation_id=7, receipt_id="r-7")
    assert same(handoff.substrate(view), capture)
    opened = handoff.open_segment(handle, generation_id=7)
    assert same(handoff.substrate(opened), capture)
    handoff.release(opened)
    handoff.release(view)


def test_a_receipt_over_the_segment_is_the_receipt_over_the_capture(tmp_path):
    capture = substrate(3)
    handle, view = handoff.publish(capture, tmp_path, generation_id=3, receipt_id="r-3")
    assert capture_receipt(handoff.substrate(view)) == capture_receipt(capture)
    opened = handoff.open_segment(handle, generation_id=3)
    assert capture_receipt(handoff.substrate(opened)) == capture_receipt(capture)
    handoff.release(opened)
    handoff.release(view)


def test_layer_keys_come_back_as_layer_indices(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=1, receipt_id="r")
    opened = handoff.open_segment(handle, generation_id=1)
    assert all(set(opened[name]) == {0, 2} for name in LAYERED)
    handoff.release(opened)
    handoff.release(view)


def test_every_tensor_is_aligned_and_inside_the_segment(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=1, receipt_id="r")
    path = tmp_path / json.loads(json.dumps(handle))["path"]
    assert path.stat().st_size == handle["total_bytes"]
    for entry in handle["tensors"]:
        assert entry["offset"] % handoff.ALIGNMENT == 0
        assert entry["offset"] + entry["nbytes"] <= handle["total_bytes"]
    handoff.release(view)


def test_the_readout_refuses_a_handle_for_another_row(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=4, receipt_id="r")
    with pytest.raises(handoff.HandoffError, match="names generation 4, not 5"):
        handoff.open_segment(handle, generation_id=5)
    handoff.release(view)


@pytest.mark.parametrize("change", [
    lambda h: h.update(receipt_id="another"),
    lambda h: h["tensors"][0].update(shape=[h["tensors"][0]["shape"][0], 1]
                                     + h["tensors"][0]["shape"][1:]),
    lambda h: h["tensors"][1].update(dtype="torch.float32",
                                     nbytes=h["tensors"][1]["nbytes"]
                                     // torch.empty((), dtype=handoff.DTYPES[
                                         h["tensors"][1]["dtype"]]).element_size() * 4),
])
def test_a_handle_whose_header_differs_from_the_segment_is_refused(tmp_path, change):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=2, receipt_id="r")
    changed = json.loads(json.dumps(handle))
    change(changed)
    with pytest.raises(handoff.HandoffError):
        handoff.open_segment(changed, generation_id=2)
    handoff.release(view)


def test_a_handle_naming_bytes_it_does_not_hold_is_refused(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=2, receipt_id="r")
    changed = json.loads(json.dumps(handle))
    changed["tensors"][0]["nbytes"] += 2
    with pytest.raises(handoff.HandoffError, match="does not describe its own bytes"):
        handoff.open_segment(changed, generation_id=2)
    changed = json.loads(json.dumps(handle))
    changed["total_bytes"] -= handoff.ALIGNMENT * 64
    with pytest.raises(handoff.HandoffError, match="run past its segment"):
        handoff.open_segment(changed, generation_id=2)
    handoff.release(view)


def test_a_handle_for_fields_other_than_the_substrate_is_refused(tmp_path):
    capture = substrate()
    capture["logprobs"] = torch.zeros(5)
    with pytest.raises(handoff.HandoffError):
        handoff.layout(capture, generation_id=1, receipt_id="r")
    capture.pop("logprobs")
    handle, view = handoff.publish(capture, tmp_path, generation_id=1, receipt_id="r")
    changed = json.loads(json.dumps(handle))
    changed["tensors"] = [e for e in changed["tensors"] if e["field"] != "hidden"]
    with pytest.raises(ValueError, match="missing"):
        handoff.open_segment(changed, generation_id=1)
    handoff.release(view)


def test_a_gone_or_resized_segment_is_refused(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=2, receipt_id="r")
    handoff.release(view)
    with open(handle["path"], "ab") as f:
        f.write(b"\0")
    with pytest.raises(handoff.HandoffError, match="not the handle's"):
        handoff.open_segment(handle, generation_id=2)
    handoff.unlink(handle)
    with pytest.raises(handoff.HandoffError, match="is gone"):
        handoff.open_segment(handle, generation_id=2)


def test_a_non_contiguous_tensor_is_refused_not_compacted(tmp_path):
    capture = substrate()
    capture["logits"] = torch.randn(11, 5).t()
    with pytest.raises(handoff.HandoffError, match="not contiguous"):
        handoff.publish(capture, tmp_path, generation_id=1, receipt_id="r")
    assert not list(tmp_path.iterdir())


def test_a_field_keyed_by_anything_but_layer_indices_is_refused(tmp_path):
    capture = substrate()
    capture["keys"] = {"0": capture["keys"][0]}
    with pytest.raises(handoff.HandoffError, match="not a layer index"):
        handoff.publish(capture, tmp_path, generation_id=1, receipt_id="r")


def test_a_segment_is_never_overwritten(tmp_path):
    handle, view = handoff.publish(substrate(), tmp_path, generation_id=1, receipt_id="r")
    with pytest.raises(handoff.HandoffError, match="exists already"):
        handoff.publish(substrate(), tmp_path, generation_id=1, receipt_id="r")
    handoff.release(view)


def test_writing_to_an_opened_capture_never_reaches_the_segment(tmp_path):
    capture = substrate()
    handle, view = handoff.publish(capture, tmp_path, generation_id=1, receipt_id="r")
    opened = handoff.open_segment(handle, generation_id=1)
    opened["logits"].fill_(0)
    again = handoff.open_segment(handle, generation_id=1)
    assert torch.equal(again["logits"], capture["logits"])
    for mapped in (opened, again, view):
        handoff.release(mapped)


def test_outstanding_counts_the_segments_not_yet_taken(tmp_path):
    first, a = handoff.publish(substrate(), tmp_path, generation_id=1, receipt_id="r")
    second, b = handoff.publish(substrate(), tmp_path, generation_id=2, receipt_id="r")
    assert handoff.outstanding(tmp_path) == 2
    handoff.unlink(first)
    handoff.unlink(first)
    assert handoff.outstanding(tmp_path) == 1
    for mapped in (a, b):
        handoff.release(mapped)
