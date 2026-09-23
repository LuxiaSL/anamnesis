"""Depth bands: the one axis of the taxonomy that needs a fact from outside the name.

Source, method and the static/dynamic wrapper are readable from a feature name
alone. The band is not: `res_norm_L16_mean` is mid-depth in a 32-layer model and
late in a 20-layer one, so the band is the name divided by a layer count the map
has to be told. That makes the layer count the one place in
`anamnesis/feature_map.py` where a wrong input produces a complete, plausible,
wrong answer — every early/mid/late label shifted, and a localization claim read
off it that names the wrong depths.

So the layer count is never guessed. A run whose model the map does not know
refuses, and a caller who knows the count states it. These tests pin both halves,
and the arithmetic of the band cut in between, because a refusal on the unknown
case is worth nothing if the known case is off by one band.
"""

from __future__ import annotations

import pytest

from anamnesis.config.models import (
    layer_counts_by_run_prefix,
    load_registry,
    resolve_preset,
)
from anamnesis.feature_map import (
    Band,
    FeatureMap,
    UnknownDepthError,
    classify,
    layers_for_run,
)

NAMES = ["res_norm_L2_mean", "attn_entropy_L16_mean", "gate_sparsity_L30_mean"]


def test_a_known_run_takes_its_layer_count_from_the_model_it_names():
    assert layers_for_run("8b_fat_01") == resolve_preset("8b").num_layers
    assert layers_for_run("3b_v2") == resolve_preset("3b").num_layers


def test_a_depth_is_the_registry_row_and_is_not_restated_beside_it():
    """The prefix map reads each row's own count, so there is nothing to disagree."""
    registry = load_registry()
    counts = layer_counts_by_run_prefix()
    for key, row in registry.presets.items():
        assert counts[key] == row.num_layers, key
        for prefix in row.run_prefixes:
            assert counts[prefix] == row.num_layers, prefix
    for alias, key in registry.aliases.items():
        assert counts[alias] == registry.presets[key].num_layers, alias
    for prefix, count in registry.run_depths.items():
        assert prefix not in registry.presets, prefix
        assert counts[prefix] == count, prefix
    assert set(counts) == (
        set(registry.presets)
        | set(registry.aliases)
        | set(registry.run_depths)
        | {p for row in registry.presets.values() for p in row.run_prefixes}
    )


def test_the_longest_matching_prefix_answers():
    """Nested keys: a `dsv2_lite` run must not be answered by the `dsv2` row.

    Both happen to carry 27 layers, so the assertion is on which key was used
    rather than on the number, which is what would catch the next nested pair.
    """
    nested = [k for k in layer_counts_by_run_prefix() if k.startswith("dsv2")]
    assert len(nested) > 1, "the nesting this rule exists for is gone; drop the rule too"
    assert layers_for_run("dsv2_lite_m6") == resolve_preset("dsv2_lite").num_layers


def test_an_unnamed_model_refuses_rather_than_assuming_a_depth():
    with pytest.raises(UnknownDepthError, match="no layer count known"):
        layers_for_run("mistral-12b_run_01")
    with pytest.raises(UnknownDepthError, match="known prefixes"):
        layers_for_run("")


def test_a_caller_that_knows_the_count_states_it():
    """The refusal is a demand for the fact, not a wall: bands still come out."""
    fm = FeatureMap(NAMES, 40)
    assert fm.n_layers == 40
    assert [t.band for t in fm.tags] == [Band.early, Band.mid, Band.late]


def test_the_same_names_band_differently_under_a_different_depth():
    bands = {n: classify(n, 32).band for n in NAMES}
    assert bands["attn_entropy_L16_mean"] is Band.mid
    assert classify("attn_entropy_L16_mean", 20).band is Band.late


@pytest.mark.parametrize("n_layers", [0, -1])
def test_a_non_positive_layer_count_is_refused_at_both_entry_points(n_layers):
    """Zero would label every layer late, which is a map with no early or mid in it."""
    with pytest.raises(UnknownDepthError):
        FeatureMap(NAMES, n_layers)
    with pytest.raises(UnknownDepthError):
        classify(NAMES[0], n_layers)
