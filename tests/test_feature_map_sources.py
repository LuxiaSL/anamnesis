"""The substrate axis: which read a feature name is credited to, and how it fails.

`anamnesis/feature_map.py` classifies a banked feature name by parsing it, and the
last thing it tries is a scan for operator keywords. That scan is the hazard on this
axis. A name with no prefix rule of its own reaches it, and words like "entropy",
"top" and "prob" appear in reads of every substrate — so a prefix nobody wrote a rule
for does not come out unplaced, it comes out confidently wrong. `unclassified()` does
not list it, no total moves, and a table of which substrate carries a signal is off by
however many features fell through.

These cases pin the prefixes where that has teeth: the families whose names are the
name of the signal they wrap, the short spellings a bank may use instead of the long
ones, and the prefixes that belong to no extractor here but that banked corpora carry.
The read-side rules exist for exactly those corpora, which is why a retired family's
prefix is a rule to keep rather than a rule to delete.

The names below are banked spellings, held here as data. CPU only; no banked file is
read, because a test that needed one would only run where that file is.
"""

from __future__ import annotations

import pytest

from anamnesis.feature_map import FeatureMap, Method, Source, classify

N_LAYERS = 32

# The windowed family's five signals, the three engineered short spellings, and the
# cross-layer key cosines: one name per (prefix, substrate) pair a bank can hold.
BANKED_SOURCES: tuple[tuple[str, Source], ...] = (
    ("td_L16_attn_entropy_w0_mean", Source.attention),
    ("td_L16_attn_entropy_spectral_centroid", Source.attention),
    ("td_L0_head_agreement_w3_slope", Source.attention),
    ("td_L24_lookback_ratio_w1_std", Source.attention),
    ("td_L16_key_drift_w1_mean", Source.keys),
    ("td_L8_key_novelty_bandwidth", Source.keys),
    ("cp_L16_t3_d07", Source.residual),
    ("af_L8_recency_bias", Source.attention),
    ("af_L16_prompt_mass", Source.attention),
    ("gf_L16_sparsity_mean", Source.gate),
    ("rt_L16_velocity_mean", Source.residual),
    ("cross_layer_early_late_agreement", Source.keys),
    ("cross_layer_adjacent_agreement", Source.keys),
    ("cross_layer_overall_coherence", Source.keys),
)


@pytest.mark.parametrize(("name", "source"), BANKED_SOURCES)
def test_a_banked_name_is_credited_to_the_substrate_it_reads(name: str, source: Source) -> None:
    assert classify(name, N_LAYERS).source is source


def test_a_windowed_attention_read_is_not_credited_to_the_output_source() -> None:
    """The case the keyword scan gets wrong, stated on its own.

    A windowed read's substrate is the substrate of the signal it windows. Attention
    entropy is a read of the attention weights; the word "entropy" in it is the
    operator, not the source, and crediting it to the token distribution moves a
    quarter of the family onto a substrate it never touches.
    """
    tag = classify("td_L16_attn_entropy_w0_mean", N_LAYERS)
    assert tag.source is not Source.output
    assert tag.source is Source.attention
    assert tag.method is Method.distributional


def test_a_windowed_signal_the_rules_do_not_name_is_reported_unplaced() -> None:
    """The refusal that keeps the rule above from becoming the next wrong answer.

    The five signals this family wraps are stated, not pattern-matched. A sixth one is
    something nobody has classified, and saying so is the only answer that does not
    invent a substrate for it.
    """
    tag = classify("td_L16_a_signal_no_rule_names_w0_mean", N_LAYERS)
    assert tag.source is Source.unknown
    assert FeatureMap([tag.name], N_LAYERS).unclassified() == [tag.name]


def test_every_pinned_name_is_placed_on_all_four_axes() -> None:
    """A source without a method or a wrapper is still an unplaced feature."""
    fm = FeatureMap([name for name, _ in BANKED_SOURCES], N_LAYERS)
    assert fm.unclassified() == []


def test_a_retired_familys_prefix_still_carries_its_own_family_label() -> None:
    """Corpora banked before a family was retired are read through these rules.

    The family label is what per-family numbers are keyed by, so the two retired
    prefixes resolve to their own families rather than to the block a name nothing
    claims falls into.
    """
    assert classify("td_L16_key_drift_w1_mean", N_LAYERS).family == "temporal_dynamics"
    assert classify("cp_L16_t3_d07", N_LAYERS).family == "contrastive_projection"


def test_the_gate_familys_own_cross_layer_features_stay_with_the_gate() -> None:
    """One live family spells `cross_layer` inside its own names.

    Its prefix answers first, which is what keeps the rule for the banked cross-layer
    key cosines from reaching across into a gate read.
    """
    assert classify("gate_cross_layer_agreement", N_LAYERS).source is Source.gate
    assert classify("gate_cross_layer_sparsity_diversity", N_LAYERS).source is Source.gate
