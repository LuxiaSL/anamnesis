"""The semantic section is measured against sentence embeddings, and has no stand-in for them.

Every "orthogonal to content" claim the section makes is a comparison with what a
sentence-embedding model says the text is about. When that model is unavailable — the
``semantic`` extra not installed, or its weights not downloadable — the section used to
carry on: its embedding rows erred quietly, the text-to-compute regression ran on
TF-IDF under the same name, and the pass counted the section as produced. Now the
section returns an error stub naming what is missing, which the gauntlet's section
accounting counts short, the way the intrinsic-dimension section refuses without its
estimators.
"""

from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from anamnesis.analysis.gauntlet import is_error_stub, semantic
from anamnesis.analysis.gauntlet.semantic import YardstickUnavailable, run_semantic

TEXTS = [f"generation {i} about topic {i % 3}" for i in range(12)]


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        generated_texts=TEXTS,
        modes=np.array(["a", "b", "c"] * 4),
        topics=np.array([f"t{i % 3}" for i in range(12)]),
    )


@pytest.fixture(autouse=True)
def no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each case decides whether an embedding was cached."""
    monkeypatch.setattr(semantic, "_cached_embedding_load", lambda key: None)
    monkeypatch.setattr(semantic, "_cached_embedding_save", lambda key, out: None)


def _without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def refuse(name: str, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)


def test_without_the_package_the_section_is_an_error_stub_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_the_package(monkeypatch)
    result = run_semantic(_data())
    assert is_error_stub(result)
    assert "`semantic` extra" in result.error


def test_a_model_that_cannot_load_is_a_stub_that_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unreachable:
        def __init__(self, name: str) -> None:
            raise OSError("could not reach the model hub")

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = Unreachable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    result = run_semantic(_data())
    assert is_error_stub(result)
    assert "could not reach the model hub" in result.error
    assert "network" in result.error


def test_a_cached_embedding_needs_neither_the_package_nor_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_the_package(monkeypatch)
    cached = np.ones((len(TEXTS), 4), dtype=np.float32)
    monkeypatch.setattr(semantic, "_cached_embedding_load", lambda key: cached)
    assert semantic._embed_texts_sbert(TEXTS) is cached


def test_the_embedder_raises_rather_than_returning_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_the_package(monkeypatch)
    with pytest.raises(YardstickUnavailable):
        semantic._embed_texts_sbert(TEXTS)
