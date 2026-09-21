"""The binding probe: a contrast-time family, and the alignment it lives or dies by.

This family is not in any suite version, and it cannot be: its second argument is
a table of labelled prompt spans, which is a property of the stimulus. Its output
is only ever read as a difference between two conditions built from that table —
so the property it must have above all others is that two conditions' vectors are
the same vector, coordinate for coordinate. Every test here is about that or about
what the family measures that the registered suite cannot:

  * the emitted names match the declared contract exactly, and are a function of
    the call's arguments alone — so a hold and a swap subtract cleanly;
  * a span table that is empty, malformed, or reaching past the prompt is refused
    rather than silently clipped, because a span quietly moved is a wrong answer
    that looks right;
  * a generation too short to have second-order structure, and a layer the bank
    did not capture, emit ALIGNED ZEROS rather than a shorter vector;
  * couplings are per head, and the partial coupling removes the "is this head
    reading the prompt at all" common mode the raw coupling saturates on;
  * the level-2 signed areas carry joint order: the increment-permutation null
    leaves level 1 numerically identical and destroys level 2, which is the only
    thing that proves a signed area is an order statistic and not a magnitude;
  * ``binding_contrast`` refuses a misaligned pair loudly.

CPU only; no model, no GPU, no banked data.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.binding_probe import (
    FAMILY_NAME,
    binding_contrast,
    extract_binding_probe,
)
from anamnesis.extraction.state_extractor import RawGenerationData

PROMPT_LENGTH = 12
N_STEPS = 9
N_LAYERS = 4
N_HEADS = 3
HIDDEN = 8
LAYERS = [0, 2]
SPANS = {"ent0": (1, 4), "attr_of_ent0": (5, 9)}
#: ``bind_L0_coup_<a>__<b>_h<head>`` — the per-head coupling names, pair captured.
_COUP_HEAD_RE = re.compile(r"_coup_(.+?__.+?)_h\d+$")


def make_data(
    *,
    n_steps: int = N_STEPS,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    prompt_length: int = PROMPT_LENGTH,
    seed: int = 20260920,
    weight_span: str | None = None,
) -> RawGenerationData:
    """Per-step attention over a growing context, rows normalised as a distribution.

    ``weight_span`` concentrates extra mass on one declared span, which is how a
    hold/swap pair is simulated without a model.
    """
    rng = np.random.default_rng(seed)
    attentions: list[np.ndarray] = []
    for t in range(n_steps):
        visible = prompt_length + t + 1
        row = rng.random((n_layers, n_heads, visible)) + 0.05
        if weight_span is not None:
            start, end = SPANS[weight_span]
            row[:, :, start:end] *= 6.0
        row /= row.sum(axis=-1, keepdims=True)
        attentions.append(row.astype(np.float32))
    return RawGenerationData(
        hidden_states=[rng.standard_normal((n_layers + 1, HIDDEN)).astype(np.float32)
                       for _ in range(n_steps)],
        attentions=attentions,
        logits=[],
        chosen_token_ids=np.arange(n_steps, dtype=np.float32),
        pre_rope_keys={},
        prompt_length=prompt_length,
    )


def extract(data: RawGenerationData, **kwargs: object) -> FeatureFamilyResult:
    params: dict = dict(spans=SPANS, sampled_layers=LAYERS, n_windows=2)
    params.update(kwargs)
    return extract_binding_probe(data, **params)  # type: ignore[arg-type]


# ── The name contract ─────────────────────────────────────────────────────────
def test_the_vector_and_its_names_are_the_same_length_and_the_family_says_its_name() -> None:
    result = extract(make_data())
    assert isinstance(result, FeatureFamilyResult)
    assert result.family_name == FAMILY_NAME == "binding_probe"
    assert len(result.features) == len(result.feature_names) == len(result)
    assert result.features.dtype == np.float32
    assert len(set(result.feature_names)) == len(result.feature_names), "no duplicate names"
    assert np.all(np.isfinite(result.features))


def test_the_names_depend_on_the_arguments_and_nothing_else() -> None:
    """Two different generations, same call: identical names. That is what makes a
    within-pair delta a delta between the same coordinates."""
    a = extract(make_data(seed=1))
    b = extract(make_data(seed=2))
    assert a.feature_names == b.feature_names
    assert not np.allclose(a.features, b.features), "different data, different values"

    # Every emitted name is scoped to its layer, its spans and its pairing.
    for layer in LAYERS:
        assert any(n.startswith(f"bind_L{layer}_") for n in a.feature_names)
    assert any("_coup_attr_of_ent0__ent0_h0" in n or "_coup_ent0__attr_of_ent0_h0" in n
               for n in a.feature_names)
    assert all(f"_h{N_HEADS}" not in n for n in a.feature_names), "heads are 0-indexed"


def test_each_optional_block_changes_the_width_it_declares() -> None:
    base = extract(make_data(), include_partial_coupling=False, include_path_level2=False)
    partial = extract(make_data(), include_partial_coupling=True, include_path_level2=False)
    level2 = extract(make_data(), include_partial_coupling=False, include_path_level2=True)
    per_head = extract(make_data(), include_partial_coupling=False,
                       include_path_level2=False, per_head_windowed=True)
    stft = extract(make_data(), include_partial_coupling=False,
                   include_path_level2=False, include_stft=True)
    widths = {len(base), len(partial), len(level2), len(per_head), len(stft)}
    assert len(widths) == 5, "each block is visible in the width"
    assert len(base) < min(len(partial), len(level2), len(per_head), len(stft))
    assert all(n.startswith("bind_") for n in base.feature_names)
    assert any("_pcoup_" in n for n in partial.feature_names)
    assert any("_psig2_" in n for n in level2.feature_names)
    assert any("spectral_centroid" in n for n in stft.feature_names)


def test_the_defaults_are_every_unordered_pair_and_an_8b_layer_plan() -> None:
    searched = extract_binding_probe(make_data(n_layers=32), spans=SPANS, n_windows=2)
    assert any("bind_L31_" in n for n in searched.feature_names), "the 8B preset's layers"
    three = {"a": (0, 2), "b": (3, 5), "c": (6, 8)}
    result = extract_binding_probe(make_data(), spans=three, sampled_layers=[0], n_windows=2)
    pairs = {m.group(1) for m in (_COUP_HEAD_RE.search(n) for n in result.feature_names)
             if m is not None}
    assert len(pairs) == 3, "three spans give three unordered pairs, searched not assumed"


def test_an_explicit_pair_list_narrows_the_search() -> None:
    result = extract(make_data(), pairs=[("ent0", "attr_of_ent0")])
    coup_names = [n for n in result.feature_names if "_coup_" in n and "_pcoup_" not in n]
    assert coup_names, "the requested pair is emitted"
    assert all("ent0__attr_of_ent0" in n for n in coup_names)


# ── Refusals ──────────────────────────────────────────────────────────────────
def test_an_unusable_span_table_is_refused_not_clipped() -> None:
    data = make_data()
    with pytest.raises(ValueError, match="empty span table"):
        extract_binding_probe(data, spans={}, sampled_layers=LAYERS)
    with pytest.raises(ValueError, match="must be a \\(start, end\\) pair"):
        extract_binding_probe(data, spans={"bad": (1, 2, 3)},  # type: ignore[dict-item]
                              sampled_layers=LAYERS)
    with pytest.raises(ValueError, match="empty or negative"):
        extract_binding_probe(data, spans={"bad": (4, 4)}, sampled_layers=LAYERS)
    with pytest.raises(ValueError, match="empty or negative"):
        extract_binding_probe(data, spans={"bad": (-1, 3)}, sampled_layers=LAYERS)
    with pytest.raises(ValueError, match="reaches past the prompt"):
        extract_binding_probe(data, spans={"bad": (1, PROMPT_LENGTH + 5)},
                              sampled_layers=LAYERS)


def test_a_coupling_pair_must_name_declared_spans() -> None:
    data = make_data()
    with pytest.raises(ValueError, match="undeclared span"):
        extract_binding_probe(data, spans=SPANS, sampled_layers=LAYERS,
                              pairs=[("ent0", "nobody")])
    with pytest.raises(ValueError, match="self-pair"):
        extract_binding_probe(data, spans=SPANS, sampled_layers=LAYERS,
                              pairs=[("ent0", "ent0")])
    with pytest.raises(ValueError, match="exactly 2 labels"):
        extract_binding_probe(data, spans=SPANS, sampled_layers=LAYERS,
                              pairs=[("ent0",)])  # type: ignore[list-item]


# ── Aligned zeros ─────────────────────────────────────────────────────────────
def test_a_generation_too_short_for_second_order_structure_emits_aligned_zeros() -> None:
    full = extract(make_data())
    short = extract(make_data(n_steps=1))
    assert len(short) == len(full), "a short generation is zero-filled, never truncated"
    assert short.feature_names == full.feature_names
    assert np.all(short.features == 0.0)


def test_a_layer_the_bank_did_not_capture_is_zero_filled_in_place() -> None:
    data = make_data(n_layers=2)
    result = extract_binding_probe(data, spans=SPANS, sampled_layers=[0, 31], n_windows=2)
    present = [i for i, n in enumerate(result.feature_names) if n.startswith("bind_L0_")]
    absent = [i for i, n in enumerate(result.feature_names) if n.startswith("bind_L31_")]
    assert present and absent
    assert len(present) == len(absent), "the missing layer occupies the same coordinates"
    assert np.all(result.features[absent] == 0.0)
    assert np.any(result.features[present] != 0.0)


# ── What it measures ──────────────────────────────────────────────────────────
def test_the_partial_coupling_removes_the_common_mode_the_raw_one_saturates_on() -> None:
    result = extract(make_data(), include_path_level2=False)
    names = result.feature_names
    raw = np.array([result.features[i] for i, n in enumerate(names)
                    if "_coup_" in n and "_pcoup_" not in n and n.endswith("_head_mean")])
    partial = np.array([result.features[i] for i, n in enumerate(names)
                        if "_pcoup_" in n and n.endswith("_head_mean")])
    assert raw.size == partial.size == len(LAYERS)
    assert np.all(np.abs(raw) <= 1.0) and np.all(np.abs(partial) <= 1.0)
    assert not np.allclose(raw, partial), "partialling out the control changes the number"


def test_couplings_are_emitted_per_head_with_summaries_beside_them() -> None:
    result = extract(make_data(), include_path_level2=False,
                     include_partial_coupling=False)
    for layer in LAYERS:
        per_head = [result.features[i] for i, n in enumerate(result.feature_names)
                    if n.startswith(f"bind_L{layer}_coup_") and _COUP_HEAD_RE.search(n)]
        assert len(per_head) == N_HEADS, "one coupling per head, not a head average"
        summary = {n.rsplit("_", 1)[-1] for n in result.feature_names
                   if n.startswith(f"bind_L{layer}_coup_")}
        assert {"mean", "std", "max", "min", "absmax"} <= summary, (
            "the head-level summaries ship beside the per-head values, not instead"
        )


def test_a_span_carrying_more_mass_shows_it_in_its_density() -> None:
    plain = extract(make_data(weight_span=None), include_path_level2=False)
    weighted = extract(make_data(weight_span="ent0"), include_path_level2=False)
    idx = [i for i, n in enumerate(plain.feature_names)
           if n == "bind_L0_span_ent0_peak_value"]
    assert len(idx) == 1
    assert weighted.features[idx[0]] > plain.features[idx[0]]


def test_the_order_null_leaves_level_one_and_destroys_level_two() -> None:
    """The control that makes a signed area an order claim. Permuting the path's
    increments keeps each coordinate's increment multiset — so the endpoint, and
    therefore level 1, is numerically identical — while destroying joint timing."""
    data = make_data()
    real = extract(data, include_partial_coupling=False, include_path_level2=True)
    null = extract(data, include_partial_coupling=False, include_path_level2=True,
                   increment_permute_seed=7)
    assert real.feature_names == null.feature_names

    level1 = [i for i, n in enumerate(real.feature_names) if "_psig1_" in n]
    level2 = [i for i, n in enumerate(real.feature_names) if "_psig2_" in n]
    assert level1 and level2
    assert np.allclose(real.features[level1], null.features[level1], atol=1e-5), \
        "level 1 is invariant under an increment permutation"
    assert not np.allclose(real.features[level2], null.features[level2], atol=1e-6), \
        "level 2 is not — which is what makes it an order statistic"

    # And the null is a null: a different seed gives different level-2 values.
    other = extract(data, include_partial_coupling=False, include_path_level2=True,
                    increment_permute_seed=8)
    assert not np.allclose(null.features[level2], other.features[level2], atol=1e-8)


# ── The within-pair contrast ───────────────────────────────────────────────────
def test_the_within_pair_contrast_is_a_difference_of_the_same_coordinates() -> None:
    hold = extract(make_data(seed=11))
    swap = extract(make_data(seed=12))
    delta, names = binding_contrast(hold, swap, label="ent0 hold-swap")
    assert names == hold.feature_names
    assert delta.dtype == np.float32
    assert np.allclose(delta, swap.features - hold.features, atol=1e-6)
    zero, _ = binding_contrast(hold, hold)
    assert np.all(zero == 0.0), "a condition against itself is exactly zero"


def test_a_misaligned_pair_is_refused_and_says_where() -> None:
    hold = extract(make_data(seed=11))
    narrower = extract(make_data(seed=12), include_path_level2=False)
    with pytest.raises(AssertionError, match="alignment failed"):
        binding_contrast(hold, narrower, label="widths differ")

    renamed = FeatureFamilyResult(
        features=hold.features.copy(),
        feature_names=["renamed"] + list(hold.feature_names[1:]),
        family_name=FAMILY_NAME,
    )
    with pytest.raises(AssertionError, match="mismatch at index 0"):
        binding_contrast(hold, renamed, label="names differ")


# ── The registry boundary ─────────────────────────────────────────────────────
def test_the_family_stays_out_of_the_registered_suite() -> None:
    """A registry entry would promise a vector a run over unlabelled prompts
    cannot produce, so the pipeline config must not know this family's name."""
    from anamnesis.config.experiment import FeaturePipelineConfig

    fields = set(FeaturePipelineConfig.model_fields)
    assert not any(FAMILY_NAME in field for field in fields)
    assert FAMILY_NAME not in fields
