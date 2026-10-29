"""The digest of a loaded model's weights: what a registry row's ``checkpoint_sha256`` names.

The digest reads the tensors a model holds once it is loaded, never the files they came
from, so one set of weights has one digest whether it was read from safetensors shards,
a ``.pt`` file, or merged from a base and an adapter in memory. For a merged adapter the
digest is of the merged weights: those are what computes. The adapter file's own hash is
provenance and is recorded beside it, not in it.

Definition (version ``weights-v1``):

* every entry of ``model.state_dict()``: parameters and persistent buffers, as torch
  reports them; non-persistent buffers are not entries and are not read;
* keys in Python string order (``sorted``);
* for each entry, one sha256 is updated with, in order: the key in utf-8, ``b"\\0"``,
  ``str(tensor.dtype)`` (``torch.bfloat16``) in utf-8, ``b"\\0"``, the shape as decimal
  sizes joined by ``,`` (empty for a scalar), ``b"\\0"``, and the tensor's raw bytes in
  its loaded dtype, read from ``t.detach().contiguous().cpu()`` viewed as ``uint8``
  (so bfloat16 and every other dtype hash their storage bytes as they are, little-endian
  on every platform this runs on);
* no deduplication: a tensor tied or aliased under two keys is hashed under each;
* the result is the hexdigest.

This module depends on nothing but torch and the standard library, so a consumer that
cannot import this package can carry a copy; ``tests/test_weights_binding.py`` pins the
digest of a small module as a constant any copy can assert against.

Cost: every tensor is copied to host memory once, one tensor at a time, so the peak
extra memory is the largest single tensor, and the hash reads every byte once on one
core. sha256's throughput bounds it: about 0.35 GiB/s end to end on a laptop CPU whose
bare sha256 runs at 0.5 GiB/s, and faster where sha256 is faster. A 1B model in bfloat16
(about 2 GiB) takes a few seconds, and an 8B model (about 15 GiB) tens of seconds. A
caller that checks one model often computes the digest once and passes it in.
"""

from __future__ import annotations

import hashlib

import torch

WEIGHTS_DIGEST_VERSION = "weights-v1"
"""The name of the definition above. A changed definition is a new name."""


def _storage_bytes(key: str, tensor: torch.Tensor) -> memoryview:
    """The tensor's raw bytes in its own dtype, from a contiguous host copy."""
    if tensor.device.type == "meta":
        raise ValueError(
            f"state_dict entry {key!r} is on the meta device and holds no values; "
            "load the weights before taking their digest"
        )
    host = tensor.detach().contiguous().cpu()
    return memoryview(host.reshape(-1).view(torch.uint8).numpy())


def weights_digest(model: torch.nn.Module) -> str:
    """The ``weights-v1`` sha256 of ``model``'s loaded weights, as a hex string.

    Raises
    ------
    TypeError
        When a ``state_dict`` entry is not a tensor (an opaque extra state), since the
        definition says nothing about how to hash one.
    ValueError
        When an entry is on the meta device, so its values are not loaded.
    """
    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"state_dict entry {key!r} is a {type(tensor).__name__}, not a tensor; "
                f"{WEIGHTS_DIGEST_VERSION} hashes tensors only"
            )
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(",".join(map(str, tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_storage_bytes(key, tensor))
    return digest.hexdigest()
