"""A dense Llama held as a layer pipeline: which contiguous layers sit on which GPU.

A checkpoint too large for one GPU can still run each layer's arithmetic on a
single GPU, by placing contiguous runs of decoder layers on successive devices
and handing the residual stream from one to the next. Nothing is summed across
devices; a hand-off is a copy. Whether the placement changes any computed byte
is a property of the machine, measured rather than assumed, so the placement is
part of what a result is about: a :class:`LayerSplit` is always explicit,
validated, recorded beside what it produced, and digested into its identity.
``"auto"`` placement is never accepted, because it can move a boundary between
two loads of the same checkpoint.

The embeddings and the rotary tables sit on the first device; the final norm and
the output projection on the last.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEVICE = re.compile(r"^(cuda:\d+|cpu)$")
_LAYER_PARAMETER = re.compile(r"^model\.layers\.(\d+)\.")
FIRST_MODULES = ("model.embed_tokens", "model.rotary_emb")
LAST_MODULES = ("model.norm", "lm_head")


class LayerSplit(BaseModel):
    """Contiguous decoder-layer ranges, one per device, covering the model once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_layers: int = Field(gt=0)
    devices: tuple[str, ...] = Field(min_length=1, description="Explicit devices, in pipeline order")
    layer_ranges: tuple[tuple[int, int], ...] = Field(
        min_length=1, description="Per device, the half-open range of decoder layers it holds")

    @model_validator(mode="after")
    def _covers_once(self) -> Self:
        bad = [d for d in self.devices if not _DEVICE.match(d)]
        if bad:
            raise ValueError(f"devices must be explicit, like 'cuda:1' or 'cpu'; got {bad}")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("a device appears twice in the split")
        if len(self.layer_ranges) != len(self.devices):
            raise ValueError("one layer range per device is required")
        expected = 0
        for start, end in self.layer_ranges:
            if start != expected or end <= start:
                raise ValueError("layer ranges must be nonempty, contiguous and in order "
                                 f"from layer 0; got {list(self.layer_ranges)}")
            expected = end
        if expected != self.num_layers:
            raise ValueError(f"layer ranges end at {expected}, the model has "
                             f"{self.num_layers} layers")
        return self

    @classmethod
    def single(cls, num_layers: int, device: str) -> LayerSplit:
        """The whole model on ``device``."""
        return cls(num_layers=num_layers, devices=(device,), layer_ranges=((0, num_layers),))

    @classmethod
    def from_boundaries(cls, num_layers: int, devices: Sequence[str],
                        boundaries: Sequence[int]) -> LayerSplit:
        """A split whose device ``k+1`` starts at layer ``boundaries[k]``."""
        edges = [0, *boundaries, num_layers]
        return cls(num_layers=num_layers, devices=tuple(devices),
                   layer_ranges=tuple(zip(edges[:-1], edges[1:])))

    @classmethod
    def balanced(cls, num_layers: int, devices: Sequence[str]) -> LayerSplit:
        """Layers dealt as evenly as they divide, the earlier devices taking any extra."""
        n = len(devices)
        if not n or n > num_layers:
            raise ValueError(f"cannot split {num_layers} layers over {n} devices")
        sizes = [num_layers // n + (1 if k < num_layers % n else 0) for k in range(n)]
        boundaries, total = [], 0
        for size in sizes[:-1]:
            total += size
            boundaries.append(total)
        return cls.from_boundaries(num_layers, devices, boundaries)

    @property
    def is_single(self) -> bool:
        return len(self.devices) == 1

    @property
    def first(self) -> str:
        return self.devices[0]

    @property
    def last(self) -> str:
        return self.devices[-1]

    def layer_device(self, layer: int) -> str:
        for device, (start, end) in zip(self.devices, self.layer_ranges):
            if start <= layer < end:
                return device
        raise ValueError(f"layer {layer} is outside a {self.num_layers}-layer split")

    def hf_device_map(self) -> dict[str, str]:
        """The explicit module → device map a Hugging Face load places the model by."""
        placement = {name: self.first for name in FIRST_MODULES}
        placement.update({f"model.layers.{layer}": self.layer_device(layer)
                          for layer in range(self.num_layers)})
        placement.update({name: self.last for name in LAST_MODULES})
        return placement

    def identity(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def digest(self) -> str:
        """sha256 of the canonical JSON of the split."""
        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def parameter_device(self, name: str) -> str:
        """The device a parameter named ``name`` must sit on."""
        match = _LAYER_PARAMETER.match(name)
        if match:
            return self.layer_device(int(match.group(1)))
        for prefix in FIRST_MODULES:
            if name.startswith(prefix + "."):
                return self.first
        for prefix in LAST_MODULES:
            if name.startswith(prefix + "."):
                return self.last
        raise ValueError(f"parameter {name!r} belongs to no module the split places")

    def check_placement(self, model: Any) -> None:
        """Refuse a model any of whose parameters is off its declared device.

        Raises
        ------
        ValueError
            Naming the first misplaced parameter.
        """
        import torch

        for name, parameter in model.named_parameters():
            want = torch.device(self.parameter_device(name))
            if parameter.device != want:
                raise ValueError(f"{name} is on {parameter.device}; the split places it on "
                                 f"{want}")


def model_devices(model: Any) -> list[Any]:
    """Every device the model's parameters occupy, in first-appearance order."""
    seen: list[Any] = []
    for parameter in model.parameters():
        if parameter.device not in seen:
            seen.append(parameter.device)
    return seen
