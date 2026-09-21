"""Whether a direction can be injected, and where — the checks before a grid runs.

A screen is an a-priori statistic. It is cheap next to a steering grid and it
answers the two questions a grid cannot: *will this direction break the model*,
and *is this the layer where the thing I mean to steer is separable at all*.

**The covariance screen.** Unit vectors all carry the same injected magnitude at
a matched dose, and they do not do the same damage. The residual stream is
strongly anisotropic: the model is robust along high-variance directions, which
it sees constantly, and fragile along low-variance ones, which are off-manifold.
``vᵀΣ⁻¹v`` and the share of a vector's energy in the bottom eigendirections say
which of those a candidate is, before a single steered generation is produced.
Σ is per site, so vectors are compared only within a shared site.

**Band mass.** How much of a vector survives projection into a covariance
eigenband. A band member built from a source with almost no band component is an
amplified residue rather than a band-confined direction, and the band mass of the
source is what tells those apart.

**Deformation against dose.** ``vᵀΣ⁻¹v`` is dose-trivial — it scales with α².
The graded readout is the Mahalanobis distance of the *induced* deformation per
dose, measured at the layer's **output**: at the injection site's input the
deformation is exactly the injected vector, and the block's nonlinear response to
it is what lives one step later. A deformation growing faster than α² is
off-manifold, which is the signature of a dial being pushed past what the model
can absorb.

**Layer separation.** Before committing a grid to a site, measure where the axis
is linearly separable, by held-out Cohen's d along the mean-difference direction
plus the scale-free centroid ratio. Two substrates read the same way and the
numbers compare: the residual stream, and — for a mixture-of-experts model —
expert-routing space, where a contrast that is weak in the residual stream may
live instead.

Every quantity here is per-model and per-site: Σ, the dose currency and the peak
layer are properties of one checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.steering.vectors import BAND, F64, Spectrum, heldout_cohens_d, unit

logger = logging.getLogger(__name__)

#: Ridge for the Σ⁻¹ regularization, as a fraction of the mean eigenvalue.
DEFAULT_RIDGE_REL = 1e-3

#: Fraction of the spectrum counted as the tail (and, at the other end, the top)
#: when reporting where a vector's energy sits.
DEFAULT_TAIL_FRACTION = 0.25


# ── The covariance screen ─────────────────────────────────────────────────────
def screen_vector(
    v: NDArray[Any], spectrum: Spectrum, tail_fraction: float = DEFAULT_TAIL_FRACTION
) -> dict[str, float]:
    """Where a unit vector sits in a residual covariance: cost, and which end.

    ``mahalanobis`` is ``vᵀ(Σ + ridge·I)⁻¹v`` — high means the vector points into
    the low-variance tail, where the model is fragile. The eigenmass entries
    split that into an answer about *where*: since ``‖v‖² = 1``, each is directly
    the fraction of the vector living in the bottom or top ``tail_fraction`` of
    the spectrum, and ``tail_over_top`` is the ratio a reader compares across
    candidates at one site.
    """
    v_unit = unit(v)
    if v_unit.shape[0] != spectrum.d:
        raise ValueError(f"vector is {v_unit.shape[0]}-dimensional, spectrum is {spectrum.d}")
    coeff = spectrum.evecs.T @ v_unit
    k = max(1, int(tail_fraction * spectrum.d))
    tail_mass = float((coeff[-k:] ** 2).sum())
    top_mass = float((coeff[:k] ** 2).sum())
    return {
        "mahalanobis": float((coeff**2 / (spectrum.evals + spectrum.ridge)).sum()),
        f"bottom_{k}_eigenmass": tail_mass,
        f"top_{k}_eigenmass": top_mass,
        "tail_over_top": float(tail_mass / max(top_mass, 1e-12)),
    }


def screen_bank(
    bank: Mapping[str, NDArray[Any]],
    spectrum: Spectrum,
    site: int,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> dict[str, dict[str, float]]:
    """Screen every vector in a bank that belongs at ``site`` and fits the spectrum.

    Membership is by key: a ``_L<site>`` suffix names the site, and a key with no
    site suffix is site-independent and screened everywhere. A vector whose width
    does not match the spectrum is skipped rather than reshaped, because a width
    mismatch means it belongs to another model.
    """
    rows: dict[str, dict[str, float]] = {}
    for key, value in bank.items():
        array = np.asarray(value)
        if array.ndim != 1 or array.shape[0] != spectrum.d:
            continue
        if "_L" in key and key.rsplit("_L", 1)[1].isdigit():
            if int(key.rsplit("_L", 1)[1]) != int(site):
                continue
        rows[key] = screen_vector(array, spectrum, tail_fraction)
    return rows


def band_mass(v: NDArray[Any], spectrum: Spectrum, band: tuple[int, int] = BAND) -> dict[str, Any]:
    """``‖P[lo:hi] v‖ / ‖v‖`` and the mass profile around the band.

    The health check for a band-passed construction, readable from a banked
    covariance and a banked vector with no rebuild. ``sqrt_band_massfrac_check``
    restates the band mass as the square root of the band's energy fraction; the
    two agree by definition, which is what makes a disagreement a sign that the
    eigenvectors and the vector come from different spaces.
    """
    v64 = np.asarray(v, dtype=np.float64)
    if v64.shape[0] != spectrum.d:
        raise ValueError(f"vector is {v64.shape[0]}-dimensional, spectrum is {spectrum.d}")
    norm = float(np.linalg.norm(v64))
    if norm == 0.0:
        raise ValueError("cannot read the band mass of a zero vector")
    U = spectrum.band_basis(band)
    projected = U @ (U.T @ v64)
    profile = spectrum.mass_profile(v64, band)
    lo, hi = band
    return {
        "band": list(band),
        "band_mass": float(np.linalg.norm(projected) / norm),
        "mass_fractions": profile,
        "sqrt_band_massfrac_check": float(np.sqrt(profile[f"band{lo}_{min(hi, spectrum.d)}"])),
        "vec_norm": norm,
    }


# ── Capture, and the deformation curve ────────────────────────────────────────
def capture_site_inputs(
    model: Any, entries: Mapping[str, Mapping[str, Any]], gen_ids: Sequence[int], site: int
) -> F64:
    """Residual rows entering decoder layer ``site``, pooled over generated positions.

    The rows Σ is estimated from. Read at the layer *input* through a forward
    pre-hook rather than from ``hidden_states``, so the same capture path serves
    the steered case, where the input is what an injection modifies.
    """
    import torch

    from anamnesis.extraction.model_loader import decoder_layers

    layers = decoder_layers(model)
    grabbed: dict[str, Any] = {}

    def pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        grabbed["h"] = hidden.detach()
        return None

    handle = layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True)
    rows: list[NDArray[Any]] = []
    try:
        device = next(model.parameters()).device
        for gen_id in gen_ids:
            entry = entries[str(gen_id)]
            ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
            prompt_length = int(entry["prompt_length"])
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            rows.append(grabbed["h"][0, prompt_length:, :].float().cpu().numpy())
            grabbed.clear()
    finally:
        handle.remove()
    return np.concatenate(rows, axis=0).astype(np.float64)


def capture_site_outputs(
    model: Any,
    entries: Mapping[str, Mapping[str, Any]],
    gen_ids: Sequence[int],
    site: int,
    inject_vector: NDArray[Any] | None = None,
    inject_alpha: float = 0.0,
) -> F64:
    """Rows leaving decoder layer ``site``, optionally with an injection at its input.

    Injection uses the package's one residual-write mechanism, positioned at the
    generated span only (``start_pos = prompt_length``), which is the semantics
    production steering runs under: the prompt is never steered.
    """
    import torch

    from anamnesis.extraction.model_loader import (
        ResidualWriteSpec,
        attach_residual_write,
        decoder_layers,
    )

    layers = decoder_layers(model)
    grabbed: dict[str, Any] = {}

    def out_hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
        grabbed["o"] = (output[0] if isinstance(output, tuple) else output).detach()

    out_handle = layers[site].register_forward_hook(out_hook)
    write_handle = None
    if inject_vector is not None:
        spec = ResidualWriteSpec(
            layer_idx=int(site),
            vector=torch.from_numpy(np.asarray(inject_vector, dtype=np.float32)),
            alpha=float(inject_alpha),
            normalize=True,
        )
        write_handle = attach_residual_write(model, spec)
    rows: list[NDArray[Any]] = []
    try:
        device = next(model.parameters()).device
        for gen_id in gen_ids:
            entry = entries[str(gen_id)]
            ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
            prompt_length = int(entry["prompt_length"])
            if write_handle is not None:
                write_handle.spec.start_pos = prompt_length
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            rows.append(grabbed["o"][0, prompt_length:, :].float().cpu().numpy())
            grabbed.clear()
    finally:
        out_handle.remove()
        if write_handle is not None:
            write_handle.remove()
    return np.concatenate(rows, axis=0).astype(np.float64)


def deformation_curve(
    unsteered_outputs: NDArray[Any],
    steered_outputs_by_dose: Mapping[float, NDArray[Any]],
    median_site_input_norm: float,
    ridge_rel: float = DEFAULT_RIDGE_REL,
) -> dict[str, Any]:
    """Mahalanobis of the induced output deformation, per dose, against α².

    The metric is the *unsteered* output covariance: the question is how far the
    steered state sits from where the model's own outputs live. A ratio to α²
    that rises with dose is off-manifold graded Goodhart; a flat one is a linear,
    on-manifold response.
    """
    baseline = np.asarray(unsteered_outputs, dtype=np.float64)
    spectrum = Spectrum.from_rows(baseline, ridge_rel=ridge_rel)
    mean_baseline = baseline.mean(axis=0)
    by_dose: dict[str, dict[str, float | None]] = {}
    for alpha, steered in sorted(steered_outputs_by_dose.items()):
        delta = np.asarray(steered, dtype=np.float64).mean(axis=0) - mean_baseline
        coeff = spectrum.evecs.T @ delta
        maha = float((coeff**2 / (spectrum.evals + spectrum.ridge)).sum())
        by_dose[str(alpha)] = {
            "deformation_maha": maha,
            "maha_over_alpha2": maha / (alpha * alpha) if alpha > 0 else None,
            "delta_l2": float(np.linalg.norm(delta)),
        }
    return {
        "n_positions": int(len(baseline)),
        "median_site_input_norm": float(median_site_input_norm),
        "ridge": spectrum.ridge,
        "doses": sorted(steered_outputs_by_dose),
        "by_dose": by_dose,
        "readout": (
            "deformation_maha growing faster than α² (maha_over_alpha2 rising with dose) is "
            "off-manifold graded Goodhart; flat is a linear, on-manifold response"
        ),
    }


# ── Layer separation ──────────────────────────────────────────────────────────
def two_fold_heldout_d(a: NDArray[Any], b: NDArray[Any], min_direction_norm: float = 1e-9) -> float:
    """Held-out Cohen's d averaged over the two even/odd folds, both directions.

    A deterministic two-fold split with no seed, so a layer curve is reproducible
    from the corpora alone. A fold too small to carry a variance contributes
    nothing rather than a zero, which would drag the mean toward no separation.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    folds = (
        ((A[::2], A[1::2]), (B[::2], B[1::2])),
        ((A[1::2], A[::2]), (B[1::2], B[::2])),
    )
    values: list[float] = []
    for (a_fit, a_eval), (b_fit, b_eval) in folds:
        if min(len(a_fit), len(a_eval), len(b_fit), len(b_eval)) < 2:
            continue
        values.append(
            heldout_cohens_d(
                a_fit, a_eval, b_fit, b_eval,
                min_direction_norm=min_direction_norm, sd_floor=None,
            )
        )
    return float(np.mean(values)) if values else 0.0


def centroid_ratio(a: NDArray[Any], b: NDArray[Any]) -> float:
    """Centroid distance over within-class rms distance: separation, scale-free.

    Reported beside the held-out d because the two fail differently. This needs
    no split and no variance along a fitted direction, so it stays readable where
    a sample is too thin for d to mean much.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    separation = float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
    wa = float(np.sqrt(((A - A.mean(axis=0)) ** 2).sum(axis=1).mean()))
    wb = float(np.sqrt(((B - B.mean(axis=0)) ** 2).sum(axis=1).mean()))
    within = 0.5 * (wa + wb)
    return separation / within if within > 0 else 0.0


def axis_separation_rows(
    pos: NDArray[Any],
    neg: NDArray[Any],
    layer_indices: Sequence[int],
    n_layers: int,
    min_direction_norm: float = 1e-9,
) -> list[dict[str, float | int]]:
    """One row per measured layer: held-out d, centroid ratio and the counts behind them.

    Inputs are ``[n_gens, n_measured_layers, width]``. ``layer_indices`` names the
    decoder layer each slice came from — the residual substrate measures every
    layer, the routing substrate only the mixture layers — so a row always says
    which layer it is and what depth that is, and the two substrates' curves are
    read against each other by layer rather than by position.

    Rows carrying a non-finite value are dropped per layer, which is how a
    generation with no span is excluded without removing the whole layer.
    """
    A = np.asarray(pos, dtype=np.float64)
    B = np.asarray(neg, dtype=np.float64)
    if A.shape[1] != len(layer_indices) or B.shape[1] != len(layer_indices):
        raise ValueError(f"{A.shape[1]}/{B.shape[1]} measured layers against {len(layer_indices)} indices")
    rows: list[dict[str, float | int]] = []
    for slot, layer in enumerate(layer_indices):
        a_slice = A[:, slot, :]
        b_slice = B[:, slot, :]
        a_slice = a_slice[np.isfinite(a_slice).all(axis=1)]
        b_slice = b_slice[np.isfinite(b_slice).all(axis=1)]
        rows.append({
            "layer": int(layer),
            "depth_pct": round(100.0 * int(layer) / int(n_layers), 1),
            "cohen_d": round(two_fold_heldout_d(a_slice, b_slice, min_direction_norm), 4),
            "centroid_ratio": round(centroid_ratio(a_slice, b_slice), 4),
            "n_pos": int(len(a_slice)),
            "n_neg": int(len(b_slice)),
        })
    return rows


def peak_layer(rows: Sequence[Mapping[str, Any]], key: str = "cohen_d") -> dict[str, Any]:
    """The row with the largest value of ``key``, reduced to layer, depth and value."""
    if not rows:
        raise ValueError("no separation rows to take a peak from")
    best = max(rows, key=lambda row: row[key])
    return {"layer": best["layer"], "depth_pct": best["depth_pct"], key: best[key]}


def capture_layer_means(
    model: Any, manifest: Path | str, n_layers: int, limit: int | None = None, log_every: int = 50
) -> F64:
    """``[n_gens, n_layers, d]`` mean residuals at every layer input over a corpus.

    Layer ``s`` is ``hidden_states[s]``, ``s`` running from 1, so a slot's index
    in the returned array is one less than the layer it names. A generation with
    no generated span contributes a row of NaN rather than being dropped here, so
    the array stays aligned with the manifest and the per-layer filter decides.
    """
    import torch

    from anamnesis.steering.vectors import replay_entries

    entries = replay_entries(manifest)
    keys = sorted(entries, key=lambda k: int(k))
    if limit is not None:
        keys = keys[:limit]
    device = next(model.parameters()).device
    captured: list[NDArray[Any]] = []
    for i, key in enumerate(keys):
        entry = entries[key]
        ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
        prompt_length = int(entry["prompt_length"])
        if ids.shape[1] - prompt_length < 1:
            continue
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        rows = []
        for s in range(1, int(n_layers) + 1):
            h = out.hidden_states[s][0, prompt_length:]
            width = out.hidden_states[s].shape[-1]
            rows.append(
                np.full(width, np.nan, dtype=np.float32)
                if h.shape[0] == 0
                else h.float().mean(dim=0).cpu().numpy().astype(np.float32)
            )
        captured.append(np.stack(rows))
        if log_every and (i + 1) % log_every == 0:
            logger.info(f"captured {i + 1}/{len(keys)} generations")
    if not captured:
        raise ValueError(f"no usable generations in {manifest}")
    return np.stack(captured).astype(np.float64)


def moe_router_layers(model: Any) -> list[tuple[int, Any]]:
    """``(decoder layer index, module)`` for every layer whose MLP exposes a router gate."""
    from anamnesis.extraction.model_loader import decoder_layers

    found: list[tuple[int, Any]] = []
    for index, layer in enumerate(decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            found.append((index, mlp))
    return found


def capture_routing_means(
    model: Any,
    manifest: Path | str,
    router_layers: Sequence[tuple[int, Any]],
    limit: int | None = None,
    log_every: int = 50,
) -> F64:
    """``[n_gens, n_router_layers, n_experts]`` mean routing mass over a corpus.

    The dense pre-top-k router softmax, recomputed exactly as the extraction hook
    computes it — ``softmax(linear(h.float(), gate.weight.float()))`` — so a
    routing separation read here and a routing feature read from an extracted
    signature are the same quantity rather than two nearby ones.
    """
    import torch
    import torch.nn.functional as F

    from anamnesis.steering.vectors import replay_entries

    store: dict[int, Any] = {}
    handles = []

    def make_hook(layer_index: int) -> Any:
        def pre(module: Any, args: tuple[Any, ...]) -> None:
            logits = F.linear(args[0].to(torch.float32), module.gate.weight.to(torch.float32))
            store[layer_index] = logits.softmax(dim=-1).detach()
        return pre

    for layer_index, module in router_layers:
        handles.append(module.register_forward_pre_hook(make_hook(layer_index)))

    entries = replay_entries(manifest)
    keys = sorted(entries, key=lambda k: int(k))
    if limit is not None:
        keys = keys[:limit]
    device = next(model.parameters()).device
    captured: list[NDArray[Any]] = []
    try:
        for i, key in enumerate(keys):
            entry = entries[key]
            ids = torch.tensor([entry["input_ids"]], dtype=torch.long, device=device)
            prompt_length = int(entry["prompt_length"])
            if ids.shape[1] - prompt_length < 1:
                continue
            store.clear()
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            captured.append(np.stack([
                store[layer_index][0, prompt_length:, :].mean(0).cpu().numpy().astype(np.float32)
                for layer_index, _ in router_layers
            ]))
            if log_every and (i + 1) % log_every == 0:
                logger.info(f"captured {i + 1}/{len(keys)} generations")
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise ValueError(f"no usable generations in {manifest}")
    return np.stack(captured).astype(np.float64)
