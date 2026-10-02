"""Hand one captured row's substrate from the engine process to the readout process in memory.

The disk path writes each row with ``torch.save``, and the readout loads it back and
re-hashes it against its receipt: about a gigabyte per row for the largest model, three
passes over it, and a file system in between. A resident session hands the same bytes
over in shared host memory instead. The two processes stay two, with their own
environments, and the readout still never imports the engine; only the bytes cross.

A **segment** is one file in a shared-memory directory (a ``tmpfs`` mount), holding a
JSON header and then every tensor of the capture, each at an aligned offset in its native
dtype and layout. The **handle** is that header plus the
segment's path: the generation id, a receipt id the engine assigns, and per tensor its
field, layer, shape, dtype, offset and byte count. The engine publishes the handle
beside the segment; the readout opens the segment only through a handle and refuses
unless the header written inside the segment is the handle it was given and names the
row it asked for. That check replaces the re-hash, because no file crossed: the content
receipt (:func:`anamnesis.extraction.vllm.receipts.capture_receipt`, the sha256 of each
tensor's native bytes) is still computed by the engine, over the very bytes in the
segment, and a row is released only once its receipt is written.

Every tensor is contiguous (a capture's tensors are made so when it finishes), and a
non-contiguous one is refused rather than silently compacted, because a strided view
and its compacted copy are different inputs to the readout's arithmetic.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from anamnesis.extraction.vllm.receipts import assert_substrate_fields

ALIGNMENT = 256
"""Every tensor starts at a multiple of this many bytes from the segment's start."""

HEADER_PREFIX = struct.Struct("<Q")
"""The segment's first eight bytes: the header's length, little-endian."""

HANDLE_FIELDS = ("generation_id", "receipt_id", "total_bytes", "tensors")
"""What a handle states about its segment; the header inside the segment states the same."""

DTYPES: dict[str, torch.dtype] = {
    str(dtype): dtype for dtype in (
        torch.float64, torch.float32, torch.float16, torch.bfloat16,
        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
}
"""The dtypes a segment may hold, by the name ``str(dtype)`` gives."""


class HandoffError(ValueError):
    """A segment or a handle that is not the one the engine published for this row."""


def _flatten(capture: Mapping[str, Any]) -> list[tuple[str, int | None, torch.Tensor]]:
    """Every tensor of ``capture`` with its field and layer, in receipt order."""
    flat = []
    for name, value in sorted(capture.items()):
        if isinstance(value, Mapping):
            for layer, tensor in sorted(value.items()):
                if type(layer) is not int:
                    raise HandoffError(f"field {name} is keyed by {layer!r}, not a layer index")
                flat.append((str(name), layer, tensor))
        else:
            flat.append((str(name), None, value))
    for name, layer, tensor in flat:
        if not isinstance(tensor, torch.Tensor):
            raise HandoffError(f"{name}/{layer} is not a tensor")
        if tensor.device.type != "cpu":
            raise HandoffError(f"{name}/{layer} is on {tensor.device}, not the host")
        if not tensor.is_contiguous():
            raise HandoffError(f"{name}/{layer} is not contiguous")
        if str(tensor.dtype) not in DTYPES:
            raise HandoffError(f"{name}/{layer} has dtype {tensor.dtype}, which a segment "
                               "does not carry")
    return flat


def _aligned(offset: int) -> int:
    return -(-offset // ALIGNMENT) * ALIGNMENT


def layout(capture: Mapping[str, Any], *, generation_id: int, receipt_id: str) -> dict[str, Any]:
    """The handle ``capture`` will have, without its path: one entry per tensor.

    Raises
    ------
    HandoffError
        On fields other than the substrate's, a non-tensor value, a tensor off the
        host, a non-contiguous tensor, an unsupported dtype, or a field keyed by
        anything but layer indices.
    """
    if type(generation_id) is not int or not isinstance(receipt_id, str) or not receipt_id:
        raise HandoffError("a handle needs an integer generation id and a receipt id")
    try:
        assert_substrate_fields(capture, context=f"hand-off of row {generation_id}")
    except ValueError as exc:
        raise HandoffError(str(exc)) from exc
    flat = _flatten(capture)
    entries, cursor = [], 0
    for name, layer, tensor in flat:
        nbytes = tensor.numel() * tensor.element_size()
        entries.append(dict(field=name, layer=layer, shape=list(tensor.shape),
                            dtype=str(tensor.dtype), offset=cursor, nbytes=nbytes))
        cursor = _aligned(cursor + nbytes)
    handle = dict(generation_id=generation_id, receipt_id=receipt_id, total_bytes=0,
                  tensors=entries)
    # The data begins after the header, whose length depends on the offsets: fix the
    # header's size first with room for any data offset, then shift every entry.
    base = _aligned(HEADER_PREFIX.size + len(json.dumps(
        dict(handle, total_bytes=10**15,
             tensors=[dict(e, offset=10**15) for e in entries])).encode()))
    for entry in entries:
        entry["offset"] += base
    handle["total_bytes"] = base + cursor
    return handle


def _header(handle: Mapping[str, Any]) -> dict[str, Any]:
    return {key: handle[key] for key in HANDLE_FIELDS}


def publish(capture: Mapping[str, Any], directory: Path, *, generation_id: int,
            receipt_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write ``capture`` into a new segment under ``directory``.

    Returns ``(handle, view)``: the handle to pass to the readout, and the capture
    rebuilt over the segment's own bytes in this process, which is what a receipt
    is computed over. The segment stays mapped here while the view or any of its
    tensors is held (:func:`release`).

    Raises
    ------
    HandoffError
        As :func:`layout`, or when the segment already exists.
    """
    handle = layout(capture, generation_id=generation_id, receipt_id=receipt_id)
    path = Path(directory) / f"row-{generation_id:05d}-{receipt_id}.seg"
    header = json.dumps(_header(handle), sort_keys=True).encode()
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise HandoffError(f"segment {path.name} exists already") from exc
    try:
        os.ftruncate(fd, handle["total_bytes"])
        region = mmap.mmap(fd, handle["total_bytes"], access=mmap.ACCESS_WRITE)
    finally:
        os.close(fd)
    region[:HEADER_PREFIX.size] = HEADER_PREFIX.pack(len(header))
    region[HEADER_PREFIX.size:HEADER_PREFIX.size + len(header)] = header
    sources = {(name, layer): tensor for name, layer, tensor in _flatten(capture)}
    for entry in handle["tensors"]:
        if entry["nbytes"]:
            target = torch.frombuffer(region, dtype=torch.uint8, count=entry["nbytes"],
                                      offset=entry["offset"])
            target.copy_(sources[(entry["field"], entry["layer"])].reshape(-1)
                         .view(torch.uint8))
            del target
    del sources
    handle["path"] = str(path)
    return handle, _tensors(region, handle)


def _tensors(region: mmap.mmap, handle: Mapping[str, Any]) -> dict[str, Any]:
    capture: dict[str, Any] = {}
    for entry in handle["tensors"]:
        dtype = DTYPES[entry["dtype"]]
        shape = tuple(int(x) for x in entry["shape"])
        if entry["nbytes"]:
            itemsize = torch.empty((), dtype=dtype).element_size()
            tensor = torch.frombuffer(region, dtype=dtype, count=entry["nbytes"] // itemsize,
                                      offset=entry["offset"]).view(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        if entry["layer"] is None:
            capture[entry["field"]] = tensor
        else:
            capture.setdefault(entry["field"], {})[int(entry["layer"])] = tensor
    capture["__region__"] = region
    return capture


def check_handle(handle: Mapping[str, Any], *, generation_id: int) -> None:
    """Refuse a handle that is malformed or names another row, before its segment is opened.

    Raises
    ------
    HandoffError
        On a missing field, another generation id, a tensor entry outside the
        segment or overlapping another, or fields other than the substrate's.
    """
    if any(key not in handle for key in (*HANDLE_FIELDS, "path")):
        raise HandoffError("the handle is missing a field")
    if type(handle["generation_id"]) is not int or handle["generation_id"] != generation_id:
        raise HandoffError(f"the handle names generation {handle['generation_id']!r}, "
                           f"not {generation_id}")
    if not isinstance(handle["receipt_id"], str) or not handle["receipt_id"]:
        raise HandoffError("the handle has no receipt id")
    total, end = int(handle["total_bytes"]), 0
    for entry in sorted(handle["tensors"], key=lambda e: e["offset"]):
        if entry["dtype"] not in DTYPES:
            raise HandoffError(f"the handle names dtype {entry['dtype']!r}")
        itemsize = torch.empty((), dtype=DTYPES[entry["dtype"]]).element_size()
        numel = 1
        for size in entry["shape"]:
            numel *= int(size)
        if entry["nbytes"] != numel * itemsize or entry["offset"] < end:
            raise HandoffError(f"the handle's entry for {entry['field']}/{entry['layer']} "
                               "does not describe its own bytes")
        end = entry["offset"] + entry["nbytes"]
    if end > total:
        raise HandoffError("the handle's tensors run past its segment")
    try:
        assert_substrate_fields({e["field"] for e in handle["tensors"]}, context="hand-off")
    except ValueError as exc:
        raise HandoffError(str(exc)) from exc


def open_segment(handle: Mapping[str, Any], *, generation_id: int) -> dict[str, Any]:
    """The capture ``handle`` names, read in place from its segment.

    The segment's own header must equal the handle (generation id, receipt id, size
    and every tensor's field, layer, shape, dtype and offset), and the segment must be
    exactly the size the handle states. The returned capture keeps the segment mapped
    while it or any of its tensors is held (:func:`release`); it is mapped
    copy-on-write, so nothing written to it reaches the segment.

    Raises
    ------
    HandoffError
        As :func:`check_handle`, or when the segment is missing, has another size, or
        its header differs from the handle.
    """
    check_handle(handle, generation_id=generation_id)
    path = Path(handle["path"])
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError as exc:
        raise HandoffError(f"segment {path.name} is gone") from exc
    try:
        size = os.fstat(fd).st_size
        if size != int(handle["total_bytes"]):
            raise HandoffError(f"segment {path.name} holds {size} bytes, not the handle's "
                               f"{handle['total_bytes']}")
        region = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
    finally:
        os.close(fd)
    (length,) = HEADER_PREFIX.unpack(region[:HEADER_PREFIX.size])
    try:
        header = json.loads(region[HEADER_PREFIX.size:HEADER_PREFIX.size + length])
    except ValueError as exc:
        raise HandoffError(f"segment {path.name} has no readable header") from exc
    if header != json.loads(json.dumps(_header(handle))):
        raise HandoffError(f"segment {path.name} is not the one this handle was published "
                           "for: its header differs")
    return _tensors(region, handle)


def release(capture: dict[str, Any]) -> None:
    """Drop ``capture``'s tensors and its hold on the segment's mapping.

    The mapping is never closed explicitly: every tensor read from it holds a reference
    to it, so it is unmapped when the last of them is collected, and a tensor kept
    elsewhere stays valid.
    """
    capture.pop("__region__", None)
    capture.clear()


def substrate(capture: Mapping[str, Any]) -> dict[str, Any]:
    """``capture`` without its mapping: the fields a readout and a receipt read."""
    return {key: value for key, value in capture.items() if key != "__region__"}


def unlink(handle: Mapping[str, Any]) -> None:
    """Remove ``handle``'s segment name; its memory is freed once no process maps it."""
    try:
        Path(handle["path"]).unlink()
    except FileNotFoundError:
        pass


def outstanding(directory: Path) -> int:
    """How many segments under ``directory`` the readout has not yet taken."""
    return sum(1 for _ in Path(directory).glob("row-*.seg"))
