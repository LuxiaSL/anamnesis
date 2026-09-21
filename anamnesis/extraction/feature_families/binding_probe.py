"""Binding-probe features: span-resolved, per-head attention structure.

Every other family in this package pools over *prompt position*. The finest
prompt-side cut available anywhere in the suite is the four-way
``[prompt | early_gen | mid_gen | recent]`` region split (``attention_flow``)
or the whole-prompt cache-read scalars (``cache_lookback_ratio`` and
friends): they record how much mass landed on the prompt, never *which*
prompt token it landed on. Binding — which attribute went with which entity —
is therefore invisible to the standard suite by construction, not by accident.

This family reads the same banked raw tensors at full positional resolution,
restricted to caller-declared spans.

CONTRAST-TIME FAMILY — NOT A SUITE MEMBER
-----------------------------------------
Its output is only meaningful relative to stimulus metadata (the span table),
so it is deliberately **not** registered in ``FeaturePipelineConfig`` and does
not belong to any suite version. Experiment drivers import and call it
directly with their own spans. Nothing here changes the dimensionality of v1,
v2 or v3 signatures.

Aggregation discipline (the constraints this family is built to satisfy)
------------------------------------------------------------------------
- **No whole-generation means.** Per-step series reduce through the house
  windowed operators, never a single mean/std collapse.
- **Per-head throughout.** Head-averaging is the documented signal killer;
  couplings are emitted per head, with head-level summaries *beside* them
  rather than instead of them.
- **Second-order is the binding quantity.** Which span carries mass is a
  first moment and mostly reflects content; whether two spans' mass series
  *covary* is the relational quantity that flips under a binding swap.
- **Common mode is removed, not assumed away.** Raw span-pair couplings are
  dominated by how much a head is attending to the prompt *at all* — measured
  at up to r = 0.975 on banked 3B generations, which leaves almost no dynamic
  range for binding structure to express itself in. Partial couplings, which
  residualise both series against total prompt density before correlating,
  ship beside the raw ones. Which of the two carries a contrast is itself a
  finding, so both are emitted and the driver selects.
- **Per-step and per-span-length normalisation**, so a longer span cannot win
  on size alone (the span-length degeneracy floor).
- **The diagnostic span is searched, not assumed.** The family emits every
  declared span and every pairing and leaves selection to the driver's
  analysis, which is the correction that overturned an earlier binding result
  on a different substrate.

STFT operators default OFF here: they were measured to be the weakest
operator class and partly artifactual under length-varying ``nperseg``.
"""

from __future__ import annotations

import itertools
import logging
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData, _safe_entropy

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

#: Half-open ``[start, end)`` token range in ABSOLUTE prompt coordinates.
Span = tuple[int, int]

FAMILY_NAME = "binding_probe"

#: Per-span scalar summaries, in emission order.
_SPAN_SCALARS = ("head_entropy", "head_top1_share", "head_argmax_frac", "peak_step_frac", "peak_value")

#: Per-pair head-level summaries, in emission order.
_PAIR_SCALARS = ("head_mean", "head_std", "head_max", "head_min", "head_absmax")


def extract_binding_probe(
    data: RawGenerationData,
    spans: dict[str, Span],
    sampled_layers: list[int] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    n_windows: int = 4,
    include_stft: bool = False,
    per_head_windowed: bool = False,
    include_partial_coupling: bool = True,
    include_path_level2: bool = True,
    increment_permute_seed: int | None = None,
) -> FeatureFamilyResult:
    """Span-resolved attention features for a single generation.

    Parameters
    ----------
    data
        Per-step tensors from one generation. ``data.attentions[t][layer]`` is
        ``[n_heads, cols_at_step_t]`` and its columns span the whole visible
        causal context, prompt included — which is what makes prompt-position
        resolution recoverable from banked raw tensors with no new extraction.
    spans
        ``label -> (start, end)`` half-open ranges in absolute prompt token
        coordinates. Labels should be **semantic** (``ent0``, ``attr_of_ent0``)
        rather than positional, so that a hold/swap pair whose spans sit at
        different offsets still produces aligned, subtractable vectors.
    sampled_layers
        Attention layer indices. Defaults to the 8B preset's sampled layers;
        pass the model's own preset for anything else.
    pairs
        Span-label pairs to score for coupling. Defaults to every unordered
        pair, which is the "search, don't assume" default.
    n_windows
        Windows for the temporal operators applied to each density series.
    include_stft
        Adds the five STFT operators per series. Off by default.
    per_head_windowed
        Also emit windowed operators for *every head* of every span. Large
        (``n_layers x n_heads x n_spans x n_windows x 3``); off by default.
    include_partial_coupling
        Emit partial couplings alongside the raw ones: both series are
        residualised against that head's total prompt density before
        correlating, which strips the "is this head reading the prompt at
        all" common mode that otherwise saturates the raw coupling. On by
        default — the raw coupling alone has too little dynamic range to be
        trusted as the binding readout.
    increment_permute_seed
        THE ORDER NULL. When set, the level-2 block is computed on paths whose
        increments have been jointly permuted in time and re-cumulated from
        ``path[0]``. That leaves level-1 numerically identical (the increment
        multiset is unchanged) while destroying temporal order, so subtracting
        this arm from the real one removes the shuffle-invariant magnitude
        component that rides inside every signed area: ``|A_ij|`` grows with the
        path's quadratic variation whether or not anything is ordered. Report
        real-minus-null; a raw level-2 number is not an order claim.
        Everything outside the level-2 block is unaffected.

    Returns
    -------
    FeatureFamilyResult
        Flat vector plus names. Name lists are deterministic given
        ``(sampled_layers, spans, pairs, n_windows, include_stft,
        per_head_windowed, include_partial_coupling, include_path_level2,
        n_heads)`` so that hold/swap vectors align exactly.

    Raises
    ------
    ValueError
        If the span table is empty, malformed, or reaches past the prompt.
    """
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    span_items = _validate_spans(spans, data.prompt_length)
    span_labels = tuple(label for label, _ in span_items)
    pair_labels = _resolve_pairs(pairs, span_labels)

    n_steps = len(data.attentions)
    n_heads = int(data.attentions[0].shape[1]) if n_steps > 0 else 0
    n_attn_layers = int(data.attentions[0].shape[0]) if n_steps > 0 else 0

    features: list[float] = []
    names: list[str] = []

    def _emit_zeros(layer_idx: int) -> None:
        layer_names = _binding_probe_names(
            layer_idx, span_labels, pair_labels, n_windows,
            include_stft, n_heads, per_head_windowed, include_partial_coupling,
            include_path_level2,
        )
        features.extend([0.0] * len(layer_names))
        names.extend(layer_names)

    # Too short to have a second-order structure at all: emit an aligned zero
    # vector rather than a ragged one, matching every other family's contract.
    if n_steps < 2 or n_heads == 0:
        logger.warning(
            "%s: generation has %d step(s) and %d head(s) — emitting aligned zeros",
            FAMILY_NAME, n_steps, n_heads,
        )
        for layer_idx in sampled_layers:
            _emit_zeros(layer_idx)
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name=FAMILY_NAME,
        )

    for layer_idx in sampled_layers:
        if layer_idx >= n_attn_layers:
            logger.warning(
                "%s: layer %d >= %d captured attention layers — emitting aligned zeros",
                FAMILY_NAME, layer_idx, n_attn_layers,
            )
            _emit_zeros(layer_idx)
            continue

        try:
            layer_feats, layer_names = _extract_one_layer(
                data=data,
                layer_idx=layer_idx,
                span_items=span_items,
                pair_labels=pair_labels,
                n_steps=n_steps,
                n_heads=n_heads,
                n_windows=n_windows,
                include_stft=include_stft,
                per_head_windowed=per_head_windowed,
                include_partial_coupling=include_partial_coupling,
                include_path_level2=include_path_level2,
                increment_permute_seed=increment_permute_seed,
            )
        except Exception:  # pragma: no cover — defensive; keeps the vector aligned
            logger.exception(
                "%s: layer %d failed; emitting aligned zeros for it", FAMILY_NAME, layer_idx
            )
            _emit_zeros(layer_idx)
            continue

        expected = _binding_probe_names(
            layer_idx, span_labels, pair_labels, n_windows,
            include_stft, n_heads, per_head_windowed, include_partial_coupling,
            include_path_level2,
        )
        if tuple(layer_names) != expected:
            # A silent name drift would desynchronise hold/swap deltas, which is
            # the one failure this family cannot be allowed to have.
            raise AssertionError(
                f"{FAMILY_NAME}: layer {layer_idx} emitted {len(layer_names)} names that do "
                f"not match the declared {len(expected)}-name contract — refusing to emit a "
                "vector whose coordinates cannot be trusted to align across conditions"
            )

        features.extend(layer_feats)
        names.extend(layer_names)

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name=FAMILY_NAME,
    )


def _extract_one_layer(
    data: RawGenerationData,
    layer_idx: int,
    span_items: list[tuple[str, Span]],
    pair_labels: tuple[tuple[str, str], ...],
    n_steps: int,
    n_heads: int,
    n_windows: int,
    include_stft: bool,
    per_head_windowed: bool,
    include_partial_coupling: bool,
    include_path_level2: bool,
    increment_permute_seed: int | None,
) -> tuple[list[float], list[str]]:
    """One layer's features. Emission order must match :func:`_binding_probe_names`."""
    prefix = f"bind_L{layer_idx}"
    prompt_len = max(int(data.prompt_length), 1)

    # ── density[label] = [T, n_heads], per-head, per-step, length-normalised ──
    density: dict[str, F64] = {
        label: np.zeros((n_steps, n_heads), dtype=np.float64) for label, _ in span_items
    }
    # Common-mode control: total prompt density, same normalisation as the spans.
    control = np.zeros((n_steps, n_heads), dtype=np.float64)

    for t in range(n_steps):
        attn = np.asarray(data.attentions[t][layer_idx], dtype=np.float64)  # [H, cols]
        if attn.ndim != 2 or attn.shape[1] == 0:
            continue
        # Attention rows are already a distribution over visible columns, but the
        # row total is recomputed rather than assumed: eager attention under a
        # padded/truncated capture is not guaranteed to sum to exactly 1.
        row_total = np.maximum(attn.sum(axis=1), 1e-12)  # [H]
        n_cols = attn.shape[1]
        prompt_hi = min(prompt_len, n_cols)
        if prompt_hi > 0:
            control[t] = attn[:, :prompt_hi].sum(axis=1) / row_total / float(prompt_hi)
        for label, (start, end) in span_items:
            hi = min(end, n_cols)
            if hi <= start:
                continue
            span_len = float(hi - start)
            density[label][t] = attn[:, start:hi].sum(axis=1) / row_total / span_len

    features: list[float] = []
    names: list[str] = []

    # ── A. head-mean density series → windowed operators (one per span) ──
    for label, _ in span_items:
        series = density[label].mean(axis=1).astype(np.float32)  # [T]
        op_f, op_n = apply_operators(
            series,
            prefix=f"{prefix}_dens_{label}",
            n_windows=n_windows,
            include_stft=include_stft,
        )
        features.extend(float(x) for x in op_f)
        names.extend(op_n)

    # ── B. per-span scalar summaries (head selectivity + peak localisation) ──
    for label, _ in span_items:
        dens = density[label]  # [T, H]
        head_mean = dens.mean(axis=0)  # [H] — mean over steps of a per-head quantity
        total = float(head_mean.sum())
        if total > 1e-12:
            head_dist = head_mean / total
            entropy = _safe_entropy(head_dist.astype(np.float32))
            top1 = float(head_dist.max())
            argmax_frac = float(int(np.argmax(head_dist)) / max(n_heads - 1, 1))
        else:
            entropy, top1, argmax_frac = 0.0, 0.0, 0.0

        series = dens.mean(axis=1)  # [T]
        if series.size > 0 and float(np.abs(series).max()) > 1e-12:
            peak_idx = int(np.argmax(series))
            peak_step_frac = float(peak_idx / max(n_steps - 1, 1))
            peak_value = float(series[peak_idx])
        else:
            peak_step_frac, peak_value = 0.0, 0.0

        for stat_name, value in zip(
            _SPAN_SCALARS, (entropy, top1, argmax_frac, peak_step_frac, peak_value)
        ):
            features.append(value)
            names.append(f"{prefix}_span_{label}_{stat_name}")

    # ── C. per-head cross-span coupling — the binding quantity ──
    for label_a, label_b in pair_labels:
        dens_a = density[label_a]
        dens_b = density[label_b]
        per_head = np.array(
            [_corr(dens_a[:, h], dens_b[:, h]) for h in range(n_heads)], dtype=np.float64
        )
        for h in range(n_heads):
            features.append(float(per_head[h]))
            names.append(f"{prefix}_coup_{label_a}__{label_b}_h{h}")
        summaries = (
            float(per_head.mean()),
            float(per_head.std()),
            float(per_head.max()),
            float(per_head.min()),
            float(np.abs(per_head).max()),
        )
        for stat_name, value in zip(_PAIR_SCALARS, summaries):
            features.append(value)
            names.append(f"{prefix}_coup_{label_a}__{label_b}_{stat_name}")

    # ── C2. common-mode-removed coupling: the same quantity with "is this head
    #        reading the prompt at all" partialled out ──
    if include_partial_coupling:
        for label_a, label_b in pair_labels:
            dens_a = density[label_a]
            dens_b = density[label_b]
            per_head = np.array(
                [
                    _partial_corr(dens_a[:, h], dens_b[:, h], control[:, h])
                    for h in range(n_heads)
                ],
                dtype=np.float64,
            )
            for h in range(n_heads):
                features.append(float(per_head[h]))
                names.append(f"{prefix}_pcoup_{label_a}__{label_b}_h{h}")
            summaries = (
                float(per_head.mean()),
                float(per_head.std()),
                float(per_head.max()),
                float(per_head.min()),
                float(np.abs(per_head).max()),
            )
            for stat_name, value in zip(_PAIR_SCALARS, summaries):
                features.append(value)
                names.append(f"{prefix}_pcoup_{label_a}__{label_b}_{stat_name}")

    # ── C3. level-2 signed areas: JOINT order, not marginal order ──
    #
    # Everything above (windowed stats, correlations) is invariant to a joint
    # permutation of the time axis: corr(a, b) does not know whether a led b.
    # That is the same marginal-order limitation the whole engineered suite has,
    # reproduced inside a family built to fix a different blindness. The Lévy
    # area is the antisymmetric part that carries lead-lag:
    #
    #     A_ij = 1/2 sum_t ( Xc_i[t] dX_j[t] - Xc_j[t] dX_i[t] ),  Xc = X - X[0]
    #
    # Left-endpoint quadrature, centred — the same convention the registered
    # path_signature family uses, written out here over the span-density paths
    # this family builds, so a contrast-time family carries no dependency on the
    # registered suite's shape.
    #
    # The clock column is NORMALISED t/(T-1): this measures pacing, not duration
    # (length is already a suite feature and already residualised), so the area
    # against the clock says WHEN a span's mass arrived, not for how long.
    if include_path_level2:
        span_names = [label for label, _ in span_items]
        dim_names = [*span_names, "clock"]
        clock = (
            np.arange(n_steps, dtype=np.float64) / max(n_steps - 1, 1)
        )[:, None]
        for h in range(n_heads):
            path = np.concatenate(
                [np.stack([density[label][:, h] for label in span_names], axis=1), clock],
                axis=1,
            )  # [T, S+1]
            if increment_permute_seed is not None:
                rng = np.random.default_rng(increment_permute_seed + h)
                incs = np.diff(path, axis=0)
                # ONE permutation applied to every coordinate, so each
                # coordinate keeps its increment multiset (level-1 exactly
                # invariant) and only the joint timing is destroyed.
                incs = incs[rng.permutation(incs.shape[0])]
                path = np.concatenate(
                    [path[:1], path[:1] + np.cumsum(incs, axis=0)], axis=0
                )
            level1 = path[-1] - path[0]
            for name, value in zip(dim_names, level1):
                features.append(float(value))
                names.append(f"{prefix}_psig1_{name}_h{h}")
            diffs = np.diff(path, axis=0)
            centered = path[:-1] - path[0][None, :]
            gram = centered.T @ diffs
            area = 0.5 * (gram - gram.T)
            for i, j in itertools.combinations(range(len(dim_names)), 2):
                features.append(float(area[i, j]))
                names.append(f"{prefix}_psig2_{dim_names[i]}__{dim_names[j]}_h{h}")

    # ── D. optional per-head windowed density ──
    if per_head_windowed:
        for label, _ in span_items:
            for h in range(n_heads):
                op_f, op_n = apply_operators(
                    density[label][:, h].astype(np.float32),
                    prefix=f"{prefix}_dens_{label}_h{h}",
                    n_windows=n_windows,
                    include_stft=include_stft,
                )
                features.extend(float(x) for x in op_f)
                names.extend(op_n)

    return features, names


def binding_contrast(
    hold: FeatureFamilyResult,
    swap: FeatureFamilyResult,
    label: str = "hold-swap",
) -> tuple[F32, list[str]]:
    """Within-pair delta ``swap - hold``, with a hard name-alignment check.

    The within-pair matched contrast is the construction that recovers signal
    where naive across-condition comparison does not, so it is provided here
    rather than left to each driver to re-derive. Name alignment is asserted,
    not assumed: a silent misalignment would produce a plausible-looking delta
    between coordinates that are not the same coordinate.

    Raises
    ------
    AssertionError
        If the two results' feature-name lists differ in any position.
    """
    if list(hold.feature_names) != list(swap.feature_names):
        n_a, n_b = len(hold.feature_names), len(swap.feature_names)
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(hold.feature_names, swap.feature_names)) if a != b),
            None,
        )
        raise AssertionError(
            f"[{label}] binding_contrast alignment failed: lengths {n_a} vs {n_b}, first "
            + (
                f"mismatch at index {first_bad} "
                f"({hold.feature_names[first_bad]!r} vs {swap.feature_names[first_bad]!r})"
                if first_bad is not None
                else "difference in length"
            )
        )
    delta = (
        np.asarray(swap.features, dtype=np.float64)
        - np.asarray(hold.features, dtype=np.float64)
    )
    return delta.astype(np.float32), list(hold.feature_names)


def _corr(a: F64, b: F64) -> float:
    """Pearson correlation over the generation-step axis; 0.0 when degenerate."""
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    std_a = float(a.std())
    std_b = float(b.std())
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    cov = float(((a - a.mean()) * (b - b.mean())).mean())
    return float(np.clip(cov / (std_a * std_b), -1.0, 1.0))


def _partial_corr(a: F64, b: F64, control: F64) -> float:
    """Correlation of ``a`` and ``b`` with ``control`` partialled out.

    Uses the closed form ``(r_ab - r_ac*r_bc) / sqrt((1-r_ac^2)(1-r_bc^2))``.
    Returns 0.0 whenever the control explains essentially all of either series
    (no residual left to correlate) or any series is degenerate — the same
    quiet-zero convention the raw coupling uses, so the two are comparable.
    """
    r_ab = _corr(a, b)
    r_ac = _corr(a, control)
    r_bc = _corr(b, control)
    denom_a = 1.0 - r_ac * r_ac
    denom_b = 1.0 - r_bc * r_bc
    if denom_a < 1e-12 or denom_b < 1e-12:
        return 0.0
    return float(np.clip((r_ab - r_ac * r_bc) / np.sqrt(denom_a * denom_b), -1.0, 1.0))


def _validate_spans(spans: dict[str, Span], prompt_length: int) -> list[tuple[str, Span]]:
    """Sort spans by label (determinism) and reject anything unusable."""
    if not spans:
        raise ValueError(
            f"{FAMILY_NAME}: empty span table — this family is meaningless without "
            "caller-declared spans; pass at least one labelled prompt range"
        )
    items: list[tuple[str, Span]] = []
    for label in sorted(spans):
        span = spans[label]
        if (
            not isinstance(span, (tuple, list))
            or len(span) != 2
            or not all(isinstance(v, (int, np.integer)) for v in span)
        ):
            raise ValueError(
                f"{FAMILY_NAME}: span {label!r} must be a (start, end) pair of ints, got {span!r}"
            )
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise ValueError(
                f"{FAMILY_NAME}: span {label!r} = ({start}, {end}) is empty or negative; "
                "ranges are half-open [start, end) with end > start >= 0"
            )
        if end > prompt_length:
            raise ValueError(
                f"{FAMILY_NAME}: span {label!r} = ({start}, {end}) reaches past the prompt "
                f"(prompt_length={prompt_length}). Spans index PROMPT tokens in absolute "
                "coordinates; generated positions are not declarable spans"
            )
        items.append((label, (start, end)))
    return items


def _resolve_pairs(
    pairs: list[tuple[str, str]] | None, span_labels: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    """Default to every unordered pair; validate any explicit list."""
    if pairs is None:
        return tuple(itertools.combinations(span_labels, 2))
    known = set(span_labels)
    resolved: list[tuple[str, str]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"{FAMILY_NAME}: coupling pair {pair!r} must have exactly 2 labels")
        label_a, label_b = pair
        missing = {label_a, label_b} - known
        if missing:
            raise ValueError(
                f"{FAMILY_NAME}: coupling pair {pair!r} references undeclared span(s) "
                f"{sorted(missing)}; declared spans are {sorted(known)}"
            )
        if label_a == label_b:
            raise ValueError(f"{FAMILY_NAME}: coupling pair {pair!r} is a self-pair")
        resolved.append((label_a, label_b))
    return tuple(resolved)


@lru_cache(maxsize=None)
def _binding_probe_names(
    layer_idx: int,
    span_labels: tuple[str, ...],
    pair_labels: tuple[tuple[str, str], ...],
    n_windows: int,
    include_stft: bool,
    n_heads: int,
    per_head_windowed: bool,
    include_partial_coupling: bool,
    include_path_level2: bool,
) -> tuple[str, ...]:
    """Declared name contract for one layer (cached). Order matches emission."""
    prefix = f"bind_L{layer_idx}"
    names: list[str] = []

    for label in span_labels:
        names.extend(_operator_names(f"{prefix}_dens_{label}", n_windows, include_stft))

    for label in span_labels:
        for stat_name in _SPAN_SCALARS:
            names.append(f"{prefix}_span_{label}_{stat_name}")

    for label_a, label_b in pair_labels:
        for h in range(n_heads):
            names.append(f"{prefix}_coup_{label_a}__{label_b}_h{h}")
        for stat_name in _PAIR_SCALARS:
            names.append(f"{prefix}_coup_{label_a}__{label_b}_{stat_name}")

    if include_partial_coupling:
        for label_a, label_b in pair_labels:
            for h in range(n_heads):
                names.append(f"{prefix}_pcoup_{label_a}__{label_b}_h{h}")
            for stat_name in _PAIR_SCALARS:
                names.append(f"{prefix}_pcoup_{label_a}__{label_b}_{stat_name}")

    if include_path_level2:
        dim_names = [*span_labels, "clock"]
        for h in range(n_heads):
            for name in dim_names:
                names.append(f"{prefix}_psig1_{name}_h{h}")
            for i, j in itertools.combinations(range(len(dim_names)), 2):
                names.append(f"{prefix}_psig2_{dim_names[i]}__{dim_names[j]}_h{h}")

    if per_head_windowed:
        for label in span_labels:
            for h in range(n_heads):
                names.extend(
                    _operator_names(f"{prefix}_dens_{label}_h{h}", n_windows, include_stft)
                )

    return tuple(names)


def _operator_names(prefix: str, n_windows: int, include_stft: bool) -> list[str]:
    """Mirror of ``apply_operators``' naming, for the declared-name contract."""
    names = [
        f"{prefix}_w{wi}_{stat}"
        for wi in range(n_windows)
        for stat in ("mean", "std", "slope")
    ]
    if include_stft:
        names.extend(
            f"{prefix}_{feat}"
            for feat in (
                "spectral_centroid",
                "bandwidth",
                "low_band_energy",
                "mid_band_energy",
                "high_band_energy",
            )
        )
    return names
