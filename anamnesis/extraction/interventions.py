"""Turning a banked intervention spec into an armed hook, for either pass.

An intervention is described as data — a vector bank and a key, a layer, a
magnitude; or a routing perturbation's mode and parameters — because that
description travels: it is written into a cell's ``metadata.json`` when the cell
is generated, and read back when the cell is replayed, so that generation and
replay cannot drift onto different specs. This module is the one place that
description becomes a hook.

Two intervention kinds, each with the same three steps, and both needed on both
sides of the instrument:

* **A residual write** adds a unit vector at one decoder layer, from the first
  generated position on. Free generation and teacher-forced replay attach the
  same write, which is what makes a steered generation and its replay comparable
  at all.
* **A routing perturbation** disturbs expert selection on a mixture-of-experts
  checkpoint, model-wide for the duration of a cell.

Magnitude is absolute here. A dose expressed as a fraction of the median
residual norm at a site is resolved before it reaches this module, and the
fraction rides along as bookkeeping so a recorded absolute alpha stays
reconstructible.

The refusals are deliberate. A missing vector key, an incomplete spec, or a
write whose position gating did not fire are all reported rather than worked
around: an intervention that quietly did not happen produces a cell that looks
like a dose and is a control.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

F32 = NDArray[np.float32]

logger = logging.getLogger(__name__)

INJECTION_METADATA_KEY = "a5_injection"
"""Where a cell's run metadata records the write it was generated under."""

UNIT_NORM_TOLERANCE = 0.01
"""How far a banked vector's norm may sit from one before it is worth saying so.
Banked vectors are unit-normalised on save, so a drifted norm means the bank is
not the one the dose was priced against."""


def load_vector(npz_path: Path | str, key: str) -> F32:
    """One vector from a banked bank, as float32.

    Raises
    ------
    KeyError
        When the bank does not hold that key, listing what it does hold — the
        cheapest way to catch a site name that drifted against the bank.
    """
    bank = np.load(str(npz_path))
    if key not in bank:
        raise KeyError(f"vector key {key!r} not in {npz_path} (has {list(bank.keys())})")
    vector: F32 = bank[key].astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if not (1.0 - UNIT_NORM_TOLERANCE) < norm < (1.0 + UNIT_NORM_TOLERANCE):
        logger.warning(f"vector {key} in {npz_path} has norm {norm:.4f}, expected unit")
    return vector


def injection_fields(
    npz: Any = None,
    key: str | None = None,
    layer: int | None = None,
    alpha: float | None = None,
    alpha_frac: float | None = None,
) -> dict[str, Any]:
    """The five ``inject_*`` fields, keyed the way they travel.

    Command lines, job files and run metadata all carry a write under these exact
    names, so the names are written down once here rather than in every entry
    point that has to spell them.
    """
    return {
        "inject_npz": npz,
        "inject_key": key,
        "inject_layer": layer,
        "inject_alpha": alpha,
        "inject_alpha_frac": alpha_frac,
    }


class InjectionSpec(BaseModel):
    """A residual write, fully determined before a model is loaded."""

    model_config = ConfigDict(extra="forbid")

    npz: Path = Field(description="Vector bank the direction is read from")
    key: str = Field(min_length=1, description="Which direction in the bank")
    layer: int = Field(ge=0, description="Decoder layer the write is applied at")
    alpha: float = Field(description="Absolute magnitude; zero is a no-op rider cell")
    alpha_frac: float | None = Field(
        default=None,
        description="The dose fraction this alpha encodes, recorded so it stays reconstructible",
    )

    def metadata(self) -> dict[str, Any]:
        """The spec as it is recorded beside a generation, keys and all."""
        return {
            "inject_npz": str(self.npz),
            "inject_key": self.key,
            "inject_layer": int(self.layer),
            "inject_alpha": float(self.alpha),
            "inject_alpha_frac": self.alpha_frac,
        }

    @classmethod
    def from_mapping(cls, fields: dict[str, Any]) -> Self | None:
        """A spec from the ``inject_*`` fields a job or a metadata block carries.

        Returns ``None`` when no bank is named, which is how a plain cell says it
        is unsteered.

        Raises
        ------
        SystemExit
            When a bank is named but the site or the magnitude is not. A partial
            spec is a command a caller meant to complete, and guessing the rest
            would bank a cell under a dose it did not receive.
        """
        npz = fields.get("inject_npz")
        if npz is None:
            return None
        key, layer, alpha = (
            fields.get("inject_key"),
            fields.get("inject_layer"),
            fields.get("inject_alpha"),
        )
        if key is None or layer is None or alpha is None:
            raise SystemExit(
                "an injection needs a key, a layer and an absolute alpha; "
                f"got key={key!r} layer={layer!r} alpha={alpha!r}"
            )
        frac = fields.get("inject_alpha_frac")
        return cls(
            npz=Path(str(npz)),
            key=str(key),
            layer=int(layer),
            alpha=float(alpha),
            alpha_frac=None if frac is None else float(frac),
        )

    @classmethod
    def from_run_metadata(cls, run_dir: Path) -> Self:
        """The spec a cell was generated under, read from its own run metadata.

        This is what makes generation and replay provably the same intervention:
        neither side restates the spec.

        Raises
        ------
        SystemExit
            When the run records no injection block, which means the cell is not
            a steered one and the caller asked the wrong question of it.
        """
        path = Path(run_dir) / "metadata.json"
        block = json.loads(path.read_text()).get(INJECTION_METADATA_KEY)
        if not block:
            raise SystemExit(f"no {INJECTION_METADATA_KEY} block in {path}")
        spec = cls.from_mapping(dict(block))
        if spec is None:
            raise SystemExit(f"{INJECTION_METADATA_KEY} in {path} names no vector bank")
        return spec


def resolve_injection(
    run_dir: Path | None,
    *,
    from_metadata: bool = False,
    fields: dict[str, Any] | None = None,
) -> InjectionSpec | None:
    """The write a cell runs under, from its metadata or from explicit fields.

    ``from_metadata`` reads the block the generation pass wrote; otherwise the
    ``inject_*`` fields in ``fields`` are used, and their absence means no write.
    """
    if from_metadata:
        if run_dir is None:
            raise SystemExit("reading an injection from metadata needs the cell's run dir")
        return InjectionSpec.from_run_metadata(run_dir)
    return InjectionSpec.from_mapping(dict(fields or {}))


def attach_injection(model: Any, spec: InjectionSpec | None, label: str) -> Any | None:
    """Arm a residual write on a model; return its handle, or ``None`` for no spec.

    The handle's lifecycle belongs to the caller: a multi-cell pass removes the
    previous cell's write before attaching the next, so writes never stack.
    ``start_pos`` is left unset here and set per generation, because where the
    prompt ends is a property of the sequence and not of the dose.
    """
    if spec is None:
        return None
    import torch

    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

    write = ResidualWriteSpec(
        layer_idx=int(spec.layer),
        vector=torch.from_numpy(load_vector(spec.npz, spec.key)),
        alpha=float(spec.alpha),
        start_pos=None,
        normalize=True,
    )
    logger.info(f"[{label}] residual write armed: {spec.metadata()}")
    return attach_residual_write(model, write)


def check_injection_gating(handle: Any, expected_positions: int, where: str) -> None:
    """Refuse a cell whose write did not fire at exactly the positions it should.

    A write is gated on the decoder's cache position so that it lands on the
    generated span and nowhere else. If the gate never saw a cache position, the
    position semantics are unverifiable; if it fired the wrong number of times,
    they are wrong. Either way the cell is not the dose it claims, so it fails
    here rather than entering a bank.

    A zero-magnitude rider cell is not checked at all: a no-op write is expected
    to do nothing, so the caller skips this rather than asking it to pass.

    Raises
    ------
    RuntimeError
        When gating did not fire, or fired at the wrong count.
    """
    stats = dict(handle.stats)
    positions = int(stats.get("positions", 0))
    if not stats.get("saw_cache_position", False):
        raise RuntimeError(
            f"{where}: the residual write ran without cache-position gating, so its "
            "position semantics are unverifiable"
        )
    if positions != expected_positions:
        raise RuntimeError(
            f"{where}: the residual write landed on {positions} positions, expected "
            f"{expected_positions} (one per generated token)"
        )


class PerturbationSpec(BaseModel):
    """A routing perturbation, as a cell records it."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(min_length=1, description="Which perturbation the router is put under")
    top_k: int | None = Field(default=None, description="Experts selected per token, when overridden")
    eps: float | None = Field(default=None, description="Perturbation scale, where the mode takes one")
    sigma_logit: dict[int, float] | None = Field(
        default=None, description="Per-layer logit noise scale"
    )
    m: int | None = Field(default=None, description="Mode-specific count, where the mode takes one")
    seed: int = Field(default=0, description="Seed, so a perturbed pass is reproducible")


def attach_perturbation(model: Any, fields: dict[str, Any] | None, label: str) -> Any | None:
    """Arm a routing perturbation model-wide; return its handle, or ``None``.

    A dense checkpoint has no router to perturb, which the hook layer reports
    rather than this function. As with a write, the handle is the caller's: a
    multi-cell pass removes the previous cell's perturbation first.
    """
    if not fields:
        return None
    from anamnesis.extraction.model_loader import MoEPerturbSpec, attach_moe_perturbation

    spec = PerturbationSpec.model_validate(
        {**fields, "sigma_logit": (
            {int(k): float(v) for k, v in fields["sigma_logit"].items()}
            if fields.get("sigma_logit") else None
        )}
    )
    logger.info(f"[{label}] routing perturbation armed: {spec.model_dump(exclude_none=True)}")
    return attach_moe_perturbation(
        model,
        MoEPerturbSpec(
            mode=spec.mode,
            top_k=spec.top_k,
            eps=spec.eps,
            sigma_logit=spec.sigma_logit,
            m=spec.m,
            seed=int(spec.seed),
        ),
    )
