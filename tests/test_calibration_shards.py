"""Calibration sampling split over shards, and the merge that refuses a mixed set.

The claim a shard rests on is that each prompt is seeded by its own index, so its
sequence does not depend on which other prompts were sampled or in which process.
It is checked against a real decoder with random weights: the shards merged are the
one-pass manifest, byte for byte. The basis half of the reproducibility claim is
checked on its own: two fits over the same samples give the same basis.
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import Any

import numpy as np
import pytest
import torch

from test_fast_lane_equivalence import tiny_loaded
from anamnesis.extraction import calibration_fit as cf
from anamnesis.scripts import run_calibration

PROMPTS = ["alpha", "bravo charlie", "delta", "echo foxtrot golf", "hotel", "india juliet"]


class _BareTokenizer:
    """A template-less tokenizer whose ids differ by prompt, inside the tiny vocabulary."""

    chat_template = None

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        ids = [3 + (ord(ch) % 50) for ch in text if ch != " "][:6]
        return {"input_ids": torch.tensor([ids])}


def _loaded() -> Any:
    loaded = tiny_loaded()
    loaded.tokenizer = _BareTokenizer()
    loaded.disable_hooks()
    return loaded


def _settings() -> Any:
    return cf.generation_settings("3b", max_new_tokens=5)


def _identity(**overrides: Any) -> cf.ShardIdentity:
    fields: dict[str, Any] = dict(model="3b", model_id="tiny", prompts=PROMPTS,
                                  settings=_settings(), suppress_eos=True, chat_template=False)
    fields.update(overrides)
    return cf.shard_identity(**fields)


def _write_shards(loaded: Any, tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        manifest = cf.generate_calibration_tokens(
            loaded, PROMPTS, _settings(), suppress_eos=True, chat_template=False,
            indices=cf.shard_range(len(PROMPTS), index, count),
        )
        shard = cf.TokenShard(identity=_identity(), index=index, count=count, manifest=manifest)
        paths.append(cf.write_token_shard(
            cf.token_shard_path(tmp_path / "pca_model.pkl", index, count), shard))
    return paths


@pytest.mark.parametrize("n_prompts", [1, 2, 7, 200])
@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_shards_partition_the_prompts(n_prompts: int, count: int) -> None:
    if count > n_prompts:
        with pytest.raises(ValueError, match="at least one"):
            cf.shard_range(n_prompts, 0, count)
        return
    runs = [cf.shard_range(n_prompts, index, count) for index in range(count)]
    assert [i for run in runs for i in run] == list(range(n_prompts))
    assert max(len(r) for r in runs) - min(len(r) for r in runs) <= 1


def test_a_shard_index_outside_its_count_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        cf.shard_range(10, 3, 3)


@pytest.mark.parametrize("count", [2, 3])
def test_merged_shards_are_the_one_pass_manifest(tmp_path: Path, count: int) -> None:
    loaded = _loaded()
    whole = cf.generate_calibration_tokens(
        loaded, PROMPTS, _settings(), suppress_eos=True, chat_template=False)
    merged = cf.merge_token_shards(_write_shards(loaded, tmp_path, count), _identity())
    assert merged.model_dump() == whole.model_dump()


def test_shard_files_are_named_beside_the_basis_and_leave_no_partial(tmp_path: Path) -> None:
    paths = _write_shards(_loaded(), tmp_path, 2)
    assert [p.name for p in paths] == ["pca_model.tokens.shard-1-of-2.json",
                                      "pca_model.tokens.shard-2-of-2.json"]
    assert not list(tmp_path.glob("*.partial"))


def test_a_missing_shard_is_refused(tmp_path: Path) -> None:
    paths = _write_shards(_loaded(), tmp_path, 3)
    with pytest.raises(cf.CalibrationFitError, match=r"missing \[2\]"):
        cf.merge_token_shards([paths[0], paths[2]], _identity())


def test_a_doubled_shard_is_refused(tmp_path: Path) -> None:
    paths = _write_shards(_loaded(), tmp_path, 2)
    with pytest.raises(cf.CalibrationFitError, match=r"present twice \[1\]"):
        cf.merge_token_shards([paths[0], paths[0]], _identity())


@pytest.mark.parametrize(
    ("override", "field"),
    [({"model_id": "another"}, "model_id"),
     ({"prompts": PROMPTS[::-1]}, "prompt_set_sha256"),
     ({"settings": cf.generation_settings("3b", max_new_tokens=6)}, "settings"),
     ({"suppress_eos": False}, "suppress_eos")],
)
def test_a_shard_from_another_calibration_is_refused(
    tmp_path: Path, override: dict[str, Any], field: str
) -> None:
    paths = _write_shards(_loaded(), tmp_path, 2)
    with pytest.raises(cf.CalibrationFitError, match=field):
        cf.merge_token_shards(paths, _identity(**override))


def test_a_shard_holding_the_wrong_prompts_does_not_load(tmp_path: Path) -> None:
    (path, _) = _write_shards(_loaded(), tmp_path, 2)
    body = json.loads(path.read_text())
    body["index"] = 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="must hold prompts"):
        cf.merge_token_shards([path], _identity())


def test_no_shards_is_refused() -> None:
    with pytest.raises(cf.CalibrationFitError, match="no sampling shards"):
        cf.merge_token_shards([], _identity())


def test_an_index_outside_the_prompts_is_refused() -> None:
    with pytest.raises(IndexError, match="outside"):
        cf.generate_calibration_tokens(_loaded(), PROMPTS, _settings(), indices=[len(PROMPTS)])


def test_the_cli_parses_a_shard_and_refuses_a_bad_one() -> None:
    assert run_calibration.parse_shard("2/6") == (1, 6)
    for bad in ("0/6", "7/6", "two/6", "6"):
        with pytest.raises(argparse.ArgumentTypeError):
            run_calibration.parse_shard(bad)


def test_the_basis_is_the_same_from_the_same_samples() -> None:
    """Near-tied variances at the end of the basis: a randomized solver resolves them
    differently on each fit; the exact decomposition does not."""
    rng = np.random.default_rng(3)
    scales = np.r_[np.linspace(10, 2, 40), np.full(560, 1.0) + rng.normal(0, 1e-4, 560)]
    # this shape is one scikit-learn's automatic solver choice resolves randomized
    matrix = rng.normal(size=(2000, 600)) * scales
    first, second = cf._fit_one(matrix, 50), cf._fit_one(matrix, 50)
    assert np.array_equal(first["components"], second["components"])
    assert np.array_equal(first["explained_variance_ratio"], second["explained_variance_ratio"])


def test_one_entry_point_samples_shards_then_fits_their_merge(tmp_path: Path) -> None:
    loaded, basis = _loaded(), tmp_path / "pca_model.pkl"
    kwargs = dict(suppress_eos=True, chat_template=False)
    for index in range(3):
        assert cf.calibration_tokens(loaded, PROMPTS, _settings(), _identity(), basis,
                                     shard=(index, 3), **kwargs) is None
    assert [p.name for p in cf.token_shard_paths(basis)] == [
        f"pca_model.tokens.shard-{i}-of-3.json" for i in (1, 2, 3)]
    merged = cf.calibration_tokens(loaded, PROMPTS, _settings(), _identity(), basis,
                                   from_shards=True, **kwargs)
    whole = cf.calibration_tokens(loaded, PROMPTS, _settings(), _identity(), basis, **kwargs)
    assert merged is not None and whole is not None
    assert merged.model_dump() == whole.model_dump()
    with pytest.raises(cf.CalibrationFitError, match="not both"):
        cf.calibration_tokens(loaded, PROMPTS, _settings(), _identity(), basis,
                              shard=(0, 3), from_shards=True, **kwargs)


def test_the_cli_takes_one_sharding_option_at_a_time() -> None:
    with pytest.raises(SystemExit):
        run_calibration.parser().parse_args(["--model", "3b", "--shard", "1/2", "--from-shards"])
