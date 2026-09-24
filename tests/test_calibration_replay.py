"""A calibration samples tokens, then replays them: the same text, the same positions.

The means and the basis are taken over the states a replay reads, one forward pass
per sampled sequence, rather than over the states a step-by-step decode produced
while sampling. The cases below pin what makes the two interchangeable, against a
real decoder with random weights and the one-pass decode written out as the
reference:

* **the same text** — sampling with no states captured draws the same tokens the
  capturing decode drew under the same seed;
* **the same positions** — the replay reads the prompt and one state per generated
  token but the last, the positions a decode computes, so ``steps[i]`` is absolute
  position ``prompt_length + i`` in both;
* **the same states** — up to floating-point reduction order, since a full forward
  and a cached step reduce in different orders;
* **the same fit** — means and basis over replayed states agree with those over
  decoded ones.

It also pins the two sizing rules: the position a bank requires, and the budget at
which the ruler reaches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME
from anamnesis.extraction.calibration_fit import (
    PromptStates,
    budget_to_reach,
    build_receipt,
    fit_calibration,
    generate_calibration_tokens,
    generate_prompt_states,
    generation_settings,
    replay_prompt_states,
    required_position,
    tokens_path,
    write_calibration,
)
from anamnesis.extraction.replay.manifest import (
    ReplayManifest,
    entry_from_ids,
    manifest_from_entries,
)
from anamnesis.scripts import run_calibration

PROMPTS = ("one", "two", "three")
PROMPT_IDS = {
    "one": [3, 9, 4],
    "two": [5, 1, 7, 2, 8],
    "three": [6, 6, 1, 3],
    "four": [2, 4, 8],
    "five": [9, 9, 9, 1],
    "six": [7, 3, 5],
    "seven": [1, 2, 3, 4, 5, 6],
}
RULER = tuple(PROMPT_IDS)
"""Seven prompts, so every position the shortest one reaches is over the count floor."""
STEPS = 12


class _BareTokenizer:
    """A tokenizer with no chat template, giving each prompt fixed, distinct ids."""

    chat_template = None

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([PROMPT_IDS[text]])}


def _loaded() -> Any:
    loaded = tiny_loaded()
    loaded.tokenizer = _BareTokenizer()
    loaded.disable_hooks()
    return loaded


def _settings() -> Any:
    return generation_settings("3b", max_new_tokens=STEPS)


def _decoded_states(loaded: Any, prompts: tuple[str, ...], settings: Any) -> list[PromptStates]:
    """The reference: one capturing decode per prompt, seeded by its index."""
    out_states = []
    for index, text in enumerate(prompts):
        input_ids = torch.tensor([PROMPT_IDS[text]])
        torch.manual_seed(index)
        with torch.no_grad():
            out = loaded.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=True,
                eos_token_id=list(settings.eos_token_ids),
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
        hidden = out.hidden_states
        out_states.append(
            (
                out.sequences[0].tolist(),
                PromptStates(
                    prompt_length=int(input_ids.shape[1]),
                    prefill=np.stack([layer[0].float().numpy() for layer in hidden[0]]),
                    steps=tuple(
                        np.stack([layer[0, -1].float().numpy() for layer in step])
                        for step in hidden[1:]
                    ),
                ),
            )
        )
    return out_states


def test_sampling_without_capture_draws_the_text_a_capturing_decode_draws() -> None:
    loaded, settings = _loaded(), _settings()
    manifest = generate_calibration_tokens(loaded, PROMPTS, settings)
    reference = _decoded_states(loaded, PROMPTS, settings)
    assert manifest.gen_ids() == (0, 1, 2), "keyed by prompt index"
    for index, (sequence, states) in enumerate(reference):
        entry = manifest.entry(index)
        assert entry.input_ids == sequence
        assert entry.prompt_length == states.prompt_length


def test_the_replay_reads_the_positions_and_states_a_decode_computes() -> None:
    loaded, settings = _loaded(), _settings()
    replayed = list(generate_prompt_states(loaded, PROMPTS, settings))
    reference = [states for _, states in _decoded_states(loaded, PROMPTS, settings)]
    assert len(replayed) == len(reference)
    for got, want in zip(replayed, reference):
        assert got.prompt_length == want.prompt_length
        assert got.prefill.shape == want.prefill.shape
        assert len(got.steps) == len(want.steps) == STEPS - 1
        np.testing.assert_allclose(got.prefill, want.prefill, rtol=1e-5, atol=1e-5)
        for step_got, step_want in zip(got.steps, want.steps):
            np.testing.assert_allclose(step_got, step_want, rtol=1e-5, atol=1e-5)


def test_a_fit_over_replayed_states_matches_one_over_decoded_states() -> None:
    loaded, settings = _loaded(), _settings()
    prompts = RULER
    preset = run_calibration.resolve_preset("3b")
    tiny = preset.model_copy(
        update={"num_layers": 3, "hidden_dim": 32, "pca_layers": [0, 1, 2]}
    )
    replayed = fit_calibration(
        generate_prompt_states(loaded, prompts, settings),
        preset=tiny, settings=settings, n_components=2, max_positions=24,
    )
    decoded = fit_calibration(
        (states for _, states in _decoded_states(loaded, prompts, settings)),
        preset=tiny, settings=settings, n_components=2, max_positions=24,
    )
    assert np.array_equal(replayed.position_counts, decoded.position_counts)
    assert replayed.positions_calibrated == decoded.positions_calibrated > 0
    for layer in tiny.pca_layers:
        np.testing.assert_allclose(
            np.abs(replayed.basis[layer]["components"]),
            np.abs(decoded.basis[layer]["components"]),
            rtol=1e-3, atol=1e-4,
        )
    np.testing.assert_allclose(
        replayed.positional_means, decoded.positional_means, rtol=1e-5, atol=1e-5
    )


def test_a_replay_over_banked_ids_does_not_sample() -> None:
    loaded = _loaded()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a replay sampled")

    loaded.model.generate = refuse
    manifest = manifest_from_entries({0: entry_from_ids([3, 9, 4, 11, 12, 13], 3)})
    (states,) = list(replay_prompt_states(loaded, manifest))
    assert states.prompt_length == 3
    assert states.prefill.shape[:2] == (4, 3)
    assert len(states.steps) == 2, "positions 3 and 4; the final token's state is not read"


# ── sizing ─────────────────────────────────────────────────────────────────────


def test_a_bank_requires_the_last_position_its_replay_reads() -> None:
    bank = manifest_from_entries(
        {0: entry_from_ids(list(range(1, 11)), 4), 7: entry_from_ids(list(range(1, 31)), 6)}
    )
    assert required_position(bank) == 28
    with pytest.raises(ValueError, match="no sequences"):
        required_position(manifest_from_entries({}))


def test_the_budget_lets_the_shortest_prompt_reach_the_required_position() -> None:
    lengths = [43, 51, 47]
    budget = budget_to_reach(638, lengths)
    assert 43 + budget - 2 == 638
    assert all(length + budget - 2 >= 638 for length in lengths)
    with pytest.raises(ValueError, match="no prompts"):
        budget_to_reach(10, [])


# ── the sequences travel with the calibration ─────────────────────────────────


def test_the_sequences_are_written_beside_the_basis_and_digested(tmp_path: Path) -> None:
    loaded, settings = _loaded(), _settings()
    tokens = generate_calibration_tokens(loaded, PROMPTS, settings)
    preset = run_calibration.resolve_preset("3b").model_copy(
        update={"num_layers": 3, "hidden_dim": 32, "pca_layers": [0, 1, 2]}
    )
    fit = fit_calibration(
        replay_prompt_states(loaded, tokens),
        preset=preset, settings=settings, n_components=2, pooled=True, max_positions=24,
    )
    means_path, basis_path = tmp_path / POSITIONAL_MEANS_NAME, tmp_path / PCA_MODEL_NAME
    write_calibration(fit, means_path, basis_path, tokens=tokens)
    written = ReplayManifest.model_validate(json.loads(tokens_path(basis_path).read_text()))
    assert written == tokens
    common = dict(
        model="3b", model_id="tiny", prompts=PROMPTS, settings=settings,
        suppress_eos=False, chat_template=True, pooled=True, n_components=2,
        means_path=means_path, basis_path=basis_path,
    )
    receipt = build_receipt(fit, tokens_written=True, **common)
    assert receipt.tokens_sha256 is not None and len(receipt.tokens_sha256) == 64
    assert build_receipt(fit, **common).tokens_sha256 is None, "a file left by another pass is not read"


# ── the command ────────────────────────────────────────────────────────────────


def test_the_dry_run_reads_the_required_position_off_a_bank(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bank = tmp_path / "replay_manifest.json"
    bank.write_text(
        json.dumps(manifest_from_entries({0: entry_from_ids(list(range(1, 641)), 130)}).model_dump())
    )
    run_calibration.main(
        ["--model", "8b", "--out-dir", str(tmp_path), "--dry-run", "--reach-from", str(bank)]
    )
    out = capsys.readouterr().out
    assert "required through position: 638" in out
    assert "means table: 712 positions" in out


def test_a_position_named_twice_is_refused(tmp_path: Path) -> None:
    bank = tmp_path / "replay_manifest.json"
    bank.write_text(json.dumps(manifest_from_entries({0: entry_from_ids([1, 2, 3], 1)}).model_dump()))
    with pytest.raises(SystemExit, match="not both"):
        run_calibration.main(
            ["--model", "8b", "--out-dir", str(tmp_path), "--dry-run",
             "--reach-from", str(bank), "--required-through", "5"]
        )
    with pytest.raises(SystemExit, match="reach-from"):
        run_calibration.main(
            ["--model", "8b", "--out-dir", str(tmp_path), "--dry-run",
             "--reach-from", str(tmp_path / "absent.json")]
        )
