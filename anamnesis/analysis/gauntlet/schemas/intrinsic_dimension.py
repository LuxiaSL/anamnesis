"""Section 4 schemas: intrinsic dimension of the signature cloud.

Global and per-mode estimates with their bootstrap intervals, the GRIDE scale
profile, and the convergence check across blocks. The estimators are
third-party (``dadapy``, ``skdim``), so a block that could not be estimated
carries its reason rather than a number.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_serializer, model_validator

from anamnesis.analysis.gauntlet.schemas.base import _FORBID


class BootstrapStats(BaseModel):
    """Per-seed bootstrap distribution of dadapy TwoNN IDs."""

    model_config = _FORBID

    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    n_successful: int


class GlobalBlockIDResult(BaseModel):
    """ID metrics for one block of the full dataset.

    ``dadapy_id`` / ``skdim_id`` fall back to ``"ERROR: ..."`` strings
    when the respective estimator raises. ``dadapy_err`` is set only on
    dadapy success. ``bootstrap_by_seed`` keys are stringified ints.
    """

    model_config = _FORBID

    n_features_clean: int
    dadapy_id: float | str | None = None
    dadapy_err: float | None = None
    skdim_id: float | str | None = None
    bootstrap_by_seed: dict[str, BootstrapStats]


class PerModeIDResult(BaseModel):
    """ID metrics for one mode on the attention-and-cache feature set."""

    model_config = _FORBID

    n_samples: int
    dadapy_id: float | str | None = None
    skdim_id: float | str | None = None
    bootstrap_mean: float | None = None
    bootstrap_std: float | None = None
    bootstrap_ci: list[float] | None = None


class GRIDEResult(BaseModel):
    """Multiscale GRIDE estimator output or an error stub."""

    model_config = _FORBID

    ids: list[float] | None = None
    errors: list[float] | None = None
    error: str | None = None


class BlockConvergenceResult(BaseModel):
    """Whether intrinsic dimension converges across the first three core blocks."""

    model_config = _FORBID

    max_pairwise_diff: float
    converged_within_2: bool
    values: dict[str, float]


class IntrinsicDimensionResult(BaseModel):
    """Section 4 result.

    All fields optional so the top-level "dadapy not installed" error
    stub (``{"error": "..."}``) round-trips cleanly.
    """

    model_config = _FORBID

    global_: dict[str, GlobalBlockIDResult] | None = None
    per_mode: dict[str, PerModeIDResult] | None = None
    gride: GRIDEResult | None = None
    block_convergence: BlockConvergenceResult | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "global" in data:
            out = dict(data)
            out["global_"] = out.pop("global")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.global_ is not None:
            out["global"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.global_.items()
            }
        if self.per_mode is not None:
            out["per_mode"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.per_mode.items()
            }
        if self.gride is not None:
            out["gride"] = self.gride.model_dump(mode="json", exclude_none=True)
        if self.block_convergence is not None:
            out["block_convergence"] = self.block_convergence.model_dump(mode="json")
        if self.error is not None:
            out["error"] = self.error
        return out
