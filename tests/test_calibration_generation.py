"""How a calibration pass drives the checkpoint: the prompt encoding and the cache.

Both are exercised against a real decoder with random weights, because both are
arguments the model runtime reads rather than arithmetic this package does.

* **The key-value cache is asked for explicitly.** A checkpoint whose configuration
  turns the cache off otherwise recomputes its whole prefix at every step, and a
  calibration long enough to reach late positions becomes quadratic in its length.
  The case below turns it off in the checkpoint and checks the call still asks for
  it.
* **The chat template can be declined.** A base checkpoint can ship a tokenizer
  carrying a template it was never trained on, and its calibration must match the
  bare prompts its generations use.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from anamnesis.extraction.calibration_fit import generate_prompt_states, generation_settings
from anamnesis.scripts import run_calibration

BARE = [1, 2, 3]
TEMPLATED = [5, 6, 1, 2, 3, 7, 8]


class _ChatTokenizer:
    """A tokenizer with a chat template, whose two encodings differ in length."""

    chat_template = "{{ messages }}"

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([BARE])}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor([TEMPLATED])


def _loaded_with_recorder() -> tuple[Any, list[dict[str, Any]]]:
    loaded = tiny_loaded()
    loaded.tokenizer = _ChatTokenizer()
    loaded.disable_hooks()
    loaded.model.config.use_cache = False
    loaded.model.generation_config.use_cache = False
    calls: list[dict[str, Any]] = []
    generate = loaded.model.generate

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return generate(*args, **kwargs)

    loaded.model.generate = recording
    return loaded, calls


def test_generation_asks_for_the_cache_whatever_the_checkpoint_says() -> None:
    loaded, calls = _loaded_with_recorder()
    list(generate_prompt_states(loaded, ["prompt"], generation_settings("3b", max_new_tokens=3)))
    assert calls and all(call["use_cache"] is True for call in calls)


@pytest.mark.parametrize(
    ("chat_template", "expected"), [(True, TEMPLATED), (False, BARE)], ids=["chat", "bare"]
)
def test_the_chat_template_is_applied_unless_declined(
    chat_template: bool, expected: list[int]
) -> None:
    loaded, _ = _loaded_with_recorder()
    (states,) = list(
        generate_prompt_states(
            loaded,
            ["prompt"],
            generation_settings("3b", max_new_tokens=3),
            chat_template=chat_template,
        )
    )
    assert states.prompt_length == len(expected)
    assert states.prefill.shape[1] == len(expected)


def test_the_dry_run_says_how_prompts_are_encoded(
    capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    run_calibration.main(["--model", "8b", "--out-dir", str(tmp_path), "--dry-run"])
    assert "prompt encoding: chat template where the tokenizer has one" in capsys.readouterr().out
    run_calibration.main(
        ["--model", "8b", "--out-dir", str(tmp_path), "--dry-run", "--no-chat-template"]
    )
    assert "prompt encoding: bare" in capsys.readouterr().out
