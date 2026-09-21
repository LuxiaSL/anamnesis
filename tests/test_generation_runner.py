"""One pass end to end: the spec plan, the alignment of the generate path, the artifacts.

The generate path's alignment is the counterpart to replay's, and the two have to
agree or features from a generated run and a replayed one are not comparable.
`_convert_outputs_to_raw` is where the generate side of that contract lives, and its
three indexing rules are all off-by-one hazards that raise nothing when broken:

* `hidden_states[t][l]` has the embedding output at `l = 0`, so a transformer layer
  `l` is at `l + 1`;
* `hidden_states[0]` and `attentions[0]` are the prefill, and the banked per-step
  lists start after them, while `logits` is already prefill-free;
* `chosen_token_ids` drops the first generated token, because no banked state
  produced it — the prefill did.

The artifacts are the other half: on-disk metadata is what every banked run carries,
and the replay manifest is what makes this run replayable. The realized token ids
travel between the two in memory and land only in the manifest.

No checkpoint: see `synthetic_runtime`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anamnesis.config import ExperimentConfig
from anamnesis.extraction.generation_runner import (
    _convert_outputs_to_raw,
    _convert_streaming_to_raw,
    router_fields_from_hooks,
    build_generation_specs,
    find_completed_ids,
    format_prompt,
    lean_metadata,
    make_seed,
    manifest_entry,
    save_generation,
    save_metadata_index,
    save_replay_manifest,
)
from anamnesis.extraction.replay.manifest import (
    INPUT_IDS_KEY,
    entry_from_ids,
    load_replay_manifest,
)
from anamnesis.extraction.state_extractor import ExtractionResult
from anamnesis.extraction.streaming_generate import StreamingOutput
from anamnesis.modes.run4_modes import RUN4_MODES
from synthetic_runtime import HookPlan, loaded_tiny_model

N_LAYERS = 3
HIDDEN = 16
HEADS = 4
VOCAB = 32


class FakeHookState:
    """The generate path reads hook captures through these five accessors only."""

    def __init__(self, keys: dict[int, list[Any]] | None = None,
                 gates: dict[int, list[Any]] | None = None) -> None:
        self._keys = keys or {}
        self._gates = gates or {}
        self.gate_activations = self._gates
        self.router_dist: dict[int, list[Any]] = {}
        self.router_shared_norm: dict[int, list[Any]] = {}
        self.router_routed_norm: dict[int, list[Any]] = {}
        self.router_logit_norm: dict[int, list[Any]] = {}
        self.router_shared_vec: dict[int, list[Any]] = {}
        self.router_routed_vec: dict[int, list[Any]] = {}

    def get_generation_keys(self, layer: int) -> list[Any]:
        captured = self._keys.get(layer, [])
        return captured[1:] if len(captured) > 1 else []

    def get_generation_gates(self, layer: int) -> list[Any]:
        captured = self._gates.get(layer, [])
        return captured[1:] if len(captured) > 1 else []


def generate_outputs(prompt_length: int, n_generated: int, seed: int = 0) -> Any:
    """Outputs shaped as `model.generate(return_dict_in_generate=True)` returns them.

    There is one entry per generation STEP, and step zero is the prefill: its
    hidden-state entry spans the whole prompt, and its logits are the ones that
    produced the first generated token. So an N-token generation has N entries in each
    list and N-1 of them are banked. The per-step tensors carry their step index as
    their value, so a slice that took the wrong step shows up in the number rather
    than only in the shape.
    """
    import torch

    torch.manual_seed(seed)
    total = prompt_length + n_generated

    hidden_states: list[tuple[Any, ...]] = [
        tuple(torch.full((1, prompt_length, HIDDEN), -1.0) for _ in range(N_LAYERS + 1))
    ]
    attentions: list[tuple[Any, ...]] = [
        tuple(torch.zeros(1, HEADS, prompt_length, prompt_length) for _ in range(N_LAYERS))
    ]
    logits: list[Any] = [torch.full((1, VOCAB), -1.0)]
    for step in range(1, n_generated):
        hidden_states.append(tuple(
            torch.full((1, 1, HIDDEN), float(step * 10 + layer))
            for layer in range(N_LAYERS + 1)
        ))
        attentions.append(tuple(
            torch.full((1, HEADS, 1, prompt_length + step), float(step))
            for _ in range(N_LAYERS)
        ))
        logits.append(torch.full((1, VOCAB), float(step)))

    sequences = torch.arange(total, dtype=torch.long).reshape(1, total)
    return type("Outputs", (), {
        "hidden_states": hidden_states, "attentions": attentions,
        "logits": logits, "sequences": sequences,
    })()


# ── the spec plan ─────────────────────────────────────────────────────────────


def test_a_seed_is_a_function_of_its_coordinates_and_its_namespace() -> None:
    assert make_seed(3, 1, 0) == make_seed(3, 1, 0)
    assert make_seed(3, 1, 0) != make_seed(3, 1, 1)
    # The prompt-set prefix is a namespace, so an alternate run's seeds do not collide
    # with the baseline's even at identical coordinates.
    assert make_seed(3, 1, 0, prompt_set="8B") != make_seed(3, 1, 0, prompt_set="3B")


def test_the_spec_plan_is_topics_by_modes_by_repetitions(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({
        "topics": {"set_a": ["alpha", "beta"], "set_b": ["gamma"]},
        "user_prompt_template": "Write about: {topic}",
        "num_repetitions": 2,
    }))
    config = ExperimentConfig.from_preset("8b", outputs_dir=tmp_path / "run", prompts_path=prompts)
    specs = build_generation_specs(config)

    assert len(specs) == 3 * len(RUN4_MODES) * 2
    assert [s.generation_id for s in specs] == list(range(len(specs)))
    assert {s.mode for s in specs} == set(RUN4_MODES)
    # Each spec carries the mode's system prompt verbatim: the label in banked data
    # means this text.
    assert all(s.system_prompt == RUN4_MODES[s.mode] for s in specs)
    assert specs[0].user_prompt == "Write about: alpha"


def test_the_spec_plan_takes_an_alternate_mode_set_and_topic_list(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"topics": {"set_a": ["x"], "set_b": []}}))
    config = ExperimentConfig.from_preset("8b", outputs_dir=tmp_path / "run", prompts_path=prompts)
    specs = build_generation_specs(
        config, mode_dict={"only": "be brief"}, topics=["one", "two"], num_reps=1,
        prompt_set="8B_r2",
    )
    assert [s.mode for s in specs] == ["only", "only"]
    assert [s.topic for s in specs] == ["one", "two"]
    assert all(s.prompt_set == "8B_r2" for s in specs)


def test_a_prompt_is_formatted_through_the_chat_template() -> None:
    loaded, _ = loaded_tiny_model(HookPlan(key_layers=[0]))
    input_ids, prompt_length = format_prompt(loaded, "system text", "user text")
    assert input_ids.shape == (1, prompt_length)
    assert prompt_length == len("system text") + len("user text") + 2


# ── the generate path's alignment ─────────────────────────────────────────────


def test_the_banked_lists_start_after_the_prefill() -> None:
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    # Four generated tokens, four steps, and the prefill step is not banked.
    assert len(raw.hidden_states) == 3
    assert len(raw.attentions) == 3
    assert len(raw.logits) == 3


def test_the_layer_axis_keeps_the_embedding_output_at_index_zero() -> None:
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    first = raw.hidden_states[0]
    assert first.shape == (N_LAYERS + 1, HIDDEN)
    # Step 1 of the fixture writes value `step * 10 + layer`; the banked first entry is
    # generation step 1, so the layer axis reads 10, 11, 12, 13 in order.
    assert first[:, 0].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_each_banked_attention_row_is_the_last_query_position_of_its_step() -> None:
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    widths = [row.shape[-1] for row in raw.attentions]
    assert widths == [6, 7, 8]                  # prefill skipped: steps 1, 2, 3
    assert raw.attentions[0].shape[:2] == (N_LAYERS, HEADS)


def test_the_first_generated_token_is_not_among_the_banked_chosen_ids() -> None:
    """No banked state produced the first generated token — the prefill did — so the
    per-step lists and the chosen ids line up only once it is dropped."""
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    assert raw.chosen_token_ids.tolist() == [6.0, 7.0, 8.0]
    assert len(raw.chosen_token_ids) == len(raw.hidden_states)


def test_captured_keys_reach_the_raw_data_with_the_prefill_dropped() -> None:
    import torch

    keys = {1: [torch.zeros(1, 2, 5, 4)] + [torch.full((1, 2, 1, 4), float(s)) for s in range(1, 4)]}
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(keys=keys), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    assert sorted(raw.pre_rope_keys) == [1]
    assert len(raw.pre_rope_keys[1]) == 3
    # A per-step capture is [num_kv_heads, 1, head_dim] and is squeezed to [heads, dim].
    assert raw.pre_rope_keys[1][0].shape == (2, 4)


def test_gate_captures_are_absent_rather_than_zero_filled_when_no_hook_ran() -> None:
    raw = _convert_outputs_to_raw(
        outputs=generate_outputs(prompt_length=5, n_generated=4),
        prompt_length=5, hook_state=FakeHookState(), sampled_layers=[0, 1, 2],
        positional_means=None,
    )
    assert raw.gate_activations is None


def test_a_dense_model_yields_no_router_fields() -> None:
    """A dense checkpoint populates no router captures, so the expert-routing family is
    absent from the vector rather than reading zeros."""
    assert router_fields_from_hooks(FakeHookState(), [0, 1, 2]) == (None, None, None)


def streaming_output(n_generated: int = 3, prompt_length: int = 4) -> StreamingOutput:
    """A `StreamingOutput` in the alignment the streaming loop produces.

    Its per-step lists already exclude the prefill — the loop never appends on the
    first step — so there are `n_generated - 1` of them for `n_generated` tokens, and
    the generated ids are the full set.
    """
    import torch

    steps = max(n_generated - 1, 0)
    return StreamingOutput(
        sequences=torch.arange(prompt_length + n_generated, dtype=torch.long).reshape(1, -1),
        hidden_states=[np.zeros((N_LAYERS + 1, HIDDEN), dtype=np.float32) for _ in range(steps)],
        attentions=[
            np.zeros((N_LAYERS, HEADS, prompt_length + i + 1), dtype=np.float32)
            for i in range(steps)
        ],
        logits=[np.zeros(VOCAB, dtype=np.float32) for _ in range(steps)],
        generated_token_ids=[10 + i for i in range(n_generated)],
        prompt_length=prompt_length,
    )


def test_the_streaming_conversion_shares_the_alignment_contract() -> None:
    """The streaming loop already produces the post-prefill alignment, so its conversion
    only has to drop the first generated token from the chosen ids."""
    raw = _convert_streaming_to_raw(
        stream_out=streaming_output(), hook_state=FakeHookState(),
        sampled_layers=[0, 1, 2], positional_means=None,
    )
    assert raw.chosen_token_ids.tolist() == [11.0, 12.0]
    assert len(raw.hidden_states) == 2
    assert len(raw.chosen_token_ids) == len(raw.hidden_states)
    assert raw.prompt_length == 4


def test_the_streaming_conversion_of_a_single_token_generation_is_empty() -> None:
    raw = _convert_streaming_to_raw(
        stream_out=streaming_output(n_generated=1), hook_state=FakeHookState(),
        sampled_layers=[0, 1, 2], positional_means=None,
    )
    assert raw.chosen_token_ids.tolist() == []
    assert raw.hidden_states == []


# ── the artifacts ─────────────────────────────────────────────────────────────


def extraction_result(n_features: int = 6) -> ExtractionResult:
    features = np.arange(n_features, dtype=np.float32)
    return ExtractionResult(
        features=features,
        feature_names=[f"f{i}" for i in range(n_features)],
        block_slices={"tier1": (0, 3), "tier2": (3, n_features)},
        knnlm_baseline=None,
    )


def metadata_row(gen_id: int, input_ids: list[int], prompt_length: int) -> dict[str, Any]:
    return {
        "generation_id": gen_id,
        "prompt_set": "8B",
        "topic": "alpha",
        "topic_idx": 0,
        "mode": "linear",
        "mode_idx": 0,
        "system_prompt": RUN4_MODES["linear"],
        "user_prompt": "Write about: alpha",
        "seed": 7,
        "repetition": 0,
        "generated_text": "text",
        "num_generated_tokens": len(input_ids) - prompt_length,
        "prompt_length": prompt_length,
        "num_features": 6,
        "tier_slices": {"tier1": (0, 3), "tier2": (3, 6)},
        INPUT_IDS_KEY: input_ids,
        "timing": {"generation_seconds": 1.0, "extraction_seconds": 1.0, "total_seconds": 2.0},
    }


def test_the_realized_ids_do_not_reach_the_per_generation_metadata(tmp_path: Path) -> None:
    """Both on-disk metadata schemas are exactly what every banked run carries; the ids
    belong to the manifest, and two files answering one question is how they drift."""
    signatures = tmp_path / "signatures"
    signatures.mkdir()
    row = metadata_row(0, [1, 2, 3, 4, 5], prompt_length=2)
    npz_path, json_path = save_generation(0, extraction_result(), row, signatures)

    written = json.loads(json_path.read_text())
    assert INPUT_IDS_KEY not in written
    assert written["prompt_length"] == 2
    assert written["tier_slices"] == {"tier1": [0, 3], "tier2": [3, 6]}
    with np.load(npz_path) as bundle:
        assert bundle["features"].tolist() == [0, 1, 2, 3, 4, 5]
        assert bundle["features_tier1"].tolist() == [0, 1, 2]


def test_the_realized_ids_do_not_reach_the_run_metadata(tmp_path: Path) -> None:
    config = ExperimentConfig.from_preset("8b", outputs_dir=tmp_path / "run")
    config.ensure_dirs()
    rows = [metadata_row(0, [1, 2, 3, 4], 2), metadata_row(1, [5, 6, 7, 8, 9], 3)]
    save_metadata_index(rows, [4], config)

    written = json.loads(config.metadata_path.read_text())
    assert written["total_generations"] == 2
    assert written["failed_ids"] == [4]
    assert all(INPUT_IDS_KEY not in gen for gen in written["generations"])
    # The in-memory rows are untouched: the strip is on the way to disk.
    assert all(INPUT_IDS_KEY in row for row in rows)


def test_a_lean_copy_keeps_everything_but_the_ids() -> None:
    row = metadata_row(0, [1, 2, 3, 4], 2)
    lean = lean_metadata(row)
    assert set(row) - set(lean) == {INPUT_IDS_KEY}
    assert lean["generation_id"] == 0


def test_a_manifest_entry_is_built_from_the_realized_ids() -> None:
    row = metadata_row(0, [1, 2, 3, 4, 5], prompt_length=2)
    assert manifest_entry(row) == entry_from_ids([1, 2, 3, 4, 5], 2)


def test_metadata_read_back_from_disk_cannot_produce_a_manifest_entry() -> None:
    """Which is why the manifest is written incrementally beside the metadata rather
    than derived from it at the end of a run."""
    row = lean_metadata(metadata_row(0, [1, 2, 3, 4], 2))
    with pytest.raises(KeyError):
        manifest_entry(row)


def test_the_run_writes_a_manifest_beside_its_metadata(tmp_path: Path) -> None:
    config = ExperimentConfig.from_preset("8b", outputs_dir=tmp_path / "run")
    config.ensure_dirs()
    entries = {
        "0": entry_from_ids([1, 2, 3, 4], 2),
        "1": entry_from_ids([5, 6, 7, 8, 9], 3),
    }
    written = save_replay_manifest(entries, config)

    assert written == config.outputs_dir / "replay_manifest.json"
    manifest = load_replay_manifest(config.outputs_dir)
    assert manifest.gen_ids() == (0, 1)
    assert manifest.n_flagged == 0          # realized ids, so nothing to flag
    assert manifest.entry(1).n_gen == 2


# ── resume ────────────────────────────────────────────────────────────────────


def test_a_generation_counts_as_complete_only_with_both_of_its_files(tmp_path: Path) -> None:
    signatures = tmp_path / "signatures"
    signatures.mkdir()
    for gen_id in (0, 1, 2):
        np.savez(signatures / f"gen_{gen_id:03d}.npz", features=np.zeros(3))
    (signatures / "gen_000.json").write_text("{}")
    (signatures / "gen_002.json").write_text("{}")
    assert find_completed_ids(signatures) == {0, 2}


def test_an_absent_signatures_directory_has_nothing_completed(tmp_path: Path) -> None:
    assert find_completed_ids(tmp_path / "nothing") == set()


def test_an_unparsable_filename_is_skipped_rather_than_crashing_the_scan(tmp_path: Path) -> None:
    signatures = tmp_path / "signatures"
    signatures.mkdir()
    np.savez(signatures / "gen_notanumber.npz", features=np.zeros(3))
    (signatures / "gen_notanumber.json").write_text("{}")
    assert find_completed_ids(signatures) == set()
