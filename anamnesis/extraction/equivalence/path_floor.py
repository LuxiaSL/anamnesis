"""Cheap sufficient certificates for the per-row path-floor inequality.

For standardized coordinate errors x, ||x||_2 >= max_j |x_j| for any
coordinate subset. First-position activation-norm coordinates depend only on
the first incremental step, so later steps need not run if this lower bound
already exceeds the candidate's full-vector cross-lane distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.state_extractor import RawGenerationData, extract_norms_and_output_stats

if TYPE_CHECKING:
    from anamnesis.extraction.model_loader import LoadedModel


def first_position_coordinates(
    hidden: NDArray[np.float32],
    positional_means: NDArray[np.float32],
    prefix_length: int,
    config: ExtractionConfig,
) -> tuple[NDArray[np.float32], list[str]]:
    """Use the canonical NumPy operator, retaining ONLY activation trajectory-0.

    Dummy logits satisfy the reference container contract; no logit-derived
    value is returned or used as evidence. First trajectory indices are zero
    for both a one-step container and the full continuation.
    """
    if hidden.ndim != 2 or hidden.dtype != np.float32 or not np.isfinite(hidden).all():
        raise ValueError("expected finite float32 [layers+embedding, hidden]")
    raw = RawGenerationData(
        hidden_states=[hidden],
        attentions=[],
        logits=[np.zeros(5, dtype=np.float32)],
        chosen_token_ids=np.zeros(1, dtype=np.float32),
        pre_rope_keys={},
        prompt_length=prefix_length,
        positional_means=positional_means,
    )
    features, names = extract_norms_and_output_stats(raw, config)
    indices = [
        i for i, name in enumerate(names) if name.startswith("activation_norm_traj0_L")
    ]
    expected = [
        f"activation_norm_traj0_L{layer}" for layer in range(hidden.shape[0] - 1)
    ]
    if [names[i] for i in indices] != expected:
        raise ValueError("first-position coordinate schema mismatch")
    return features[indices], expected


def first_incremental_coordinates(
    loaded: LoadedModel,
    token_ids: list[int],
    start: int,
    end: int,
    positional_means: NDArray[np.float32],
    config: ExtractionConfig,
) -> tuple[NDArray[np.float32], list[str]]:
    """Fresh full-prefix prefill + exactly the original first incremental call."""
    import torch

    if not 0 < start < end <= len(token_ids) or end - start < 2:
        raise ValueError("invalid continuation")
    device = next(loaded.model.parameters()).device
    ids = torch.tensor(token_ids[:end], device=device).unsqueeze(0)
    loaded.clear_hook_state()
    loaded.disable_hooks()
    try:
        with torch.no_grad():
            pre = loaded.model(ids[:, :start], use_cache=True, return_dict=True)
        cache = pre.past_key_values
        del pre
        loaded.clear_hook_state()
        loaded.enable_hooks()
        with torch.no_grad():
            out = loaded.model(
                ids[:, start : start + 1],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                position_ids=torch.tensor([[start]], device=device),
                cache_position=torch.tensor([start], device=device),
            )
        hidden = np.stack([h[0, 0].float().cpu().numpy() for h in out.hidden_states])
        return first_position_coordinates(hidden, positional_means, start, config)
    finally:
        loaded.clear_hook_state()


def coordinate_lower_bound(
    incremental: NDArray[np.float32],
    reference: NDArray[np.float32],
    sigma: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> dict[str, float]:
    """Max-coordinate lower bound, conservatively reduced for FP64 roundoff.

    This returns a bound, never an estimate of the complete path distance.
    An insufficient bound requires more work; it is not a fidelity failure.
    """
    if not (incremental.shape == reference.shape == sigma.shape == weights.shape):
        raise ValueError("coordinate/ruler dimensions differ")
    if not all(np.isfinite(v).all() for v in (incremental, reference, sigma, weights)):
        raise ValueError("nonfinite lower-bound input")
    if not len(incremental) or not (sigma > 0).all() or not (weights > 0).all():
        raise ValueError("empty coordinates or invalid ruler")
    delta = (
        (incremental.astype(np.float64) - reference.astype(np.float64))
        * weights
        / sigma
    )
    raw = float(np.max(np.abs(delta)))
    guard = 1e-12 * max(1.0, raw)
    lower = max(0.0, float(np.nextafter(raw - guard, 0.0)))
    return dict(
        floor_b_lower_bound=lower,
        max_coordinate_distance=raw,
        subset_l2_diagnostic=float(np.linalg.norm(delta)),
        roundoff_guard=guard,
    )
