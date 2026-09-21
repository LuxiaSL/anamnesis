"""How a banking pass is configured, where its records land, and what it records.

A banked corpus outlives the command line that produced it, and every number
computed over it is conditioned on how the text was sampled. So three things have to
be right in the library rather than in a launcher: the policy a pass runs under, the
filename its records land on, and the block a run's metadata carries about both.
None of the three needs a device, which is why they are pinned here rather than
inferred from a run that happened to work.

The record path is the case worth stating twice. Four callers read it — the loop that
writes a record, the resume filter that skips a written one, a fan-out deciding which
specs are left, and the shortfall accounting — and they must agree exactly. A
fan-out counting differently from its workers spawns a worker with nothing to do, or
skips work no worker did, and the second failure is invisible: the pass reports
success over a corpus with a hole in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anamnesis.config import resolve_preset
from anamnesis.extraction.interventions import INJECTION_METADATA_KEY, injection_fields
from anamnesis.extraction.token_generation import (
    NEUTRAL_REPETITION_PENALTY,
    DecodePolicy,
    generation_passthrough,
    record_on_disk,
    record_path,
)

MODEL = "8b"


def _policy(**overrides: object) -> DecodePolicy:
    settings: dict[str, object] = dict(top_p=0.9, max_new_tokens=512)
    settings.update(overrides)
    return DecodePolicy.from_preset(resolve_preset(MODEL), **settings)  # type: ignore[arg-type]


def test_the_stop_tokens_come_from_the_row_and_are_not_an_argument() -> None:
    """A pass that assumed the wrong stop tokens banks generations that run past their end."""
    policy = _policy()
    preset = resolve_preset(MODEL)
    assert policy.eos_token_ids == tuple(preset.eos_token_ids)
    with pytest.raises(TypeError):
        DecodePolicy.from_preset(  # type: ignore[call-arg]
            preset, top_p=0.9, max_new_tokens=512, eos_token_ids=(1,)
        )


def test_an_unset_temperature_is_the_rows_own() -> None:
    preset = resolve_preset(MODEL)
    assert _policy().temperature == preset.temperature
    assert _policy(temperature=0.95).temperature == 0.95


@pytest.mark.parametrize("model", ["3b", "8b", "gemma3-27b", "dsv2-lite"])
def test_a_policy_over_any_row_carries_that_rows_stop_tokens(model: str) -> None:
    preset = resolve_preset(model)
    policy = DecodePolicy.from_preset(preset, top_p=preset.top_p, max_new_tokens=64)
    assert policy.eos_token_ids == tuple(preset.eos_token_ids)
    assert policy.top_p == preset.top_p


def test_a_neutral_penalty_is_withheld_from_the_sampler() -> None:
    assert _policy().repetition_penalty == NEUTRAL_REPETITION_PENALTY
    assert _policy().sampler_extras() == {}
    assert _policy(repetition_penalty=1.2).sampler_extras() == {"repetition_penalty": 1.2}


def test_a_record_is_named_by_its_generation_and_padded_to_sort(tmp_path: Path) -> None:
    assert record_path(tmp_path, 7).name == "gen_007.json"
    assert record_path(tmp_path, 123).name == "gen_123.json"
    assert not record_on_disk(tmp_path, 7)
    record_path(tmp_path, 7).write_text("{}")
    assert record_on_disk(tmp_path, 7)
    assert not record_on_disk(tmp_path, 8)


def test_the_resume_predicate_and_the_writer_are_the_same_path(tmp_path: Path) -> None:
    """A resume that looked somewhere else would recompute a record already banked."""
    written = record_path(tmp_path, 3)
    written.write_text("{}")
    assert record_on_disk(tmp_path, 3)
    assert record_path(tmp_path, 3) == written


def test_the_metadata_block_records_the_policy_the_pass_resolved_to() -> None:
    preset = resolve_preset(MODEL)
    block = generation_passthrough(preset, _policy(temperature=0.6))
    assert block["model"]["model_id"] == preset.model_id
    assert block["model"]["num_layers"] == preset.num_layers
    assert block["generation_config"]["temperature"] == 0.6
    assert block["generation_config"]["eos_token_ids"] == list(preset.eos_token_ids)
    assert block["generation_config"]["do_sample"] is True
    assert "template_date_string" not in block


def test_a_pinned_template_date_is_recorded_because_it_changes_the_prompt() -> None:
    block = generation_passthrough(
        resolve_preset(MODEL), _policy(date_string="12 Jul 2026")
    )
    assert block["template_date_string"] == "12 Jul 2026"


def test_an_unsteered_run_records_no_intervention_block() -> None:
    """Absent rather than empty, so a reader never has to guess which it means."""
    preset = resolve_preset(MODEL)
    assert INJECTION_METADATA_KEY not in generation_passthrough(preset, _policy())
    assert INJECTION_METADATA_KEY not in generation_passthrough(
        preset, _policy(), injection=injection_fields()
    )


def test_a_steered_run_records_the_write_under_the_key_replay_reads_it_by() -> None:
    """The same key :func:`resolve_injection` reads back, so the two cannot drift."""
    block = generation_passthrough(
        resolve_preset(MODEL),
        _policy(),
        injection=injection_fields("/bank.npz", "V3", 16, 2.5, 0.1),
    )
    recorded = block[INJECTION_METADATA_KEY]
    assert recorded["inject_key"] == "V3"
    assert recorded["inject_layer"] == 16
    assert recorded["inject_alpha"] == 2.5
    assert recorded["inject_alpha_frac"] == 0.1
