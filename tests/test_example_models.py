"""The example registry files load, and every row in them is a valid preset.

An example is what a newcomer copies first, so one that no longer validates is the
first thing they would hit. Each file is loaded through ``ANAMNESIS_MODELS`` exactly as
the README's recipe loads it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anamnesis.config import preset_names, resolve_preset

EXAMPLES = sorted((Path(__file__).parent.parent / "examples" / "models").glob("*.json"))


def test_there_is_an_example_to_copy() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_an_example_registry_file_adds_valid_rows(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANAMNESIS_MODELS", str(path))
    rows = json.loads(path.read_text())["presets"]
    for name in rows:
        assert name in preset_names()
        preset = resolve_preset(name)
        assert preset.hidden_dim == preset.num_attention_heads * preset.head_dim
