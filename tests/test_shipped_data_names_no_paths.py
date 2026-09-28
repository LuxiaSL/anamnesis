"""No data file the package ships names a filesystem location.

Package data reaches every installation, where a path on the machine that wrote it
resolves to nothing and says where that machine kept its files. The documentation
checkers read comments, docstrings and messages in Python source; this reads every
JSON file under ``anamnesis/`` instead, key and value, and refuses any string that
is an absolute path (POSIX, home-relative or a Windows drive). A record a shipped
file depends on is named by its role and its digest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import anamnesis

ABSOLUTE_PATH = re.compile(r"^(/|~/|[A-Za-z]:[\\/])[^\s/]")
"""An absolute POSIX path, a home-relative path or a Windows drive path."""

SHIPPED_JSON = sorted(Path(anamnesis.__file__).parent.rglob("*.json"))


def _paths_in(value) -> list[str]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _paths_in(k) + _paths_in(v)]
    if isinstance(value, list):
        return [p for v in value for p in _paths_in(v)]
    return [value] if isinstance(value, str) and ABSOLUTE_PATH.match(value) else []


def test_the_package_ships_json_to_check():
    assert SHIPPED_JSON


@pytest.mark.parametrize("path", SHIPPED_JSON,
                         ids=lambda p: str(p.relative_to(Path(anamnesis.__file__).parent)))
def test_a_shipped_json_file_names_no_filesystem_path(path):
    found = _paths_in(json.loads(path.read_text()))
    assert not found, f"{path.name} names filesystem paths: {found[:3]}"


@pytest.mark.parametrize("text", ["/models/run/report.json", "~/data/x.json",
                                  "C:\\\\data\\\\x.json", "D:/x.json"])
def test_the_rule_sees_each_kind_of_absolute_path(text):
    assert _paths_in({"k": [text]}) == [text]


@pytest.mark.parametrize("text", ["hf-vllm-comparison/feature-manifest",
                                  "meta-llama/Llama-3.1-8B-Instruct", "a / b", "/", "1/2"])
def test_the_rule_stays_quiet_on_names_that_are_not_paths(text):
    assert _paths_in({text: text}) == []
