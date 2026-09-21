"""Path-floor certificates: a bound that is a bound, computed from one step.

A fidelity verdict compares a candidate vector to the anchor's under a standardizing
ruler, and the per-row bound it is compared against costs a full replay to measure.
This module is the shortcut, and the shortcut has to be sound in one direction only:
because ||x||_2 >= max_j |x_j| for any coordinate subset, a bound computed from the
first incremental step alone can already exceed a candidate's full-vector distance,
in which case the remaining steps carry no information and need not run. An
insufficient bound means more work, never a failure.

Two claims, and they are different kinds. The first is an identity: the coordinates
the shortcut reads must be exactly the anchor's `activation_norm_traj0_L*` values,
byte-for-byte, not a re-derivation that happens to agree — asserted over hidden widths
from 128 to 8192 and spans from one step to 512, since the trajectory index and the
positional-correction lookup both depend on those. The second is the inequality
itself, checked against the full-vector distance over a hundred random draws plus the
degenerate identical-input case, where the bound must be exactly zero.

Pure numpy over the anchor; no model, no device.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.equivalence.path_floor import (
    coordinate_lower_bound,
    first_position_coordinates,
)
from anamnesis.extraction.state_extractor import RawGenerationData, extract_norms_and_output_stats


@pytest.mark.parametrize(
    "steps,width", [(1, 128), (2, 3072), (127, 4096), (127, 8192), (512, 128)]
)
def test_first_coordinates_exactly_match_full_reference(steps, width):
    rng = np.random.default_rng(14)
    hidden = rng.normal(size=(steps, 4, width)).astype(np.float32)
    pm = rng.normal(0, 0.01, size=(4, steps + 20, width)).astype(np.float32)
    cfg = ExtractionConfig(
        sampled_layers=[0, 1, 2], pca_layers=[0, 1],
        early_layer_cutoff=8, late_layer_cutoff=24,
    )
    raw = RawGenerationData(
        hidden_states=list(hidden),
        attentions=[],
        logits=[np.zeros(5, dtype=np.float32) for _ in range(steps)],
        chosen_token_ids=np.zeros(steps, dtype=np.float32),
        pre_rope_keys={},
        prompt_length=11,
        positional_means=pm,
    )
    full, names = extract_norms_and_output_stats(raw, cfg)
    values, selected = first_position_coordinates(hidden[0], pm, 11, cfg)
    expected = full[[names.index(n) for n in selected]]
    assert values.tobytes() == expected.tobytes()


def test_lower_bound_never_exceeds_full_distance():
    rng = np.random.default_rng(19)
    for _ in range(100):
        a = rng.normal(size=4086).astype(np.float32)
        b = rng.normal(size=4086).astype(np.float32)
        sigma = np.exp(rng.normal(size=4086))
        weights = np.ones(4086)
        indices = rng.choice(4086, size=80, replace=False)
        bound = coordinate_lower_bound(
            a[indices], b[indices], sigma[indices], weights[indices]
        )
        assert (
            0
            <= bound["floor_b_lower_bound"]
            <= np.linalg.norm((a.astype(float) - b) * weights / sigma)
        )
    bound = coordinate_lower_bound(a, a, np.ones(4086), weights)
    assert bound["floor_b_lower_bound"] == 0
