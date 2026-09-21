"""Banked trajectories to design matrices: the packing, the drops, and the null.

The marshaller adds no numbers of its own, so what is tested is everything around them:

  * the ragged packing round-trips — offsets describe the rows they claim to, a path read
    back is the path written, and a bank whose offsets disagree with its rows is refused
    rather than read as a shorter bank;
  * a path too short for the level asked for is **dropped and counted**, and the kept
    indices say which rows of the matrix belong to which generation — a zero-filled row
    would be a claim about a trajectory that could not be measured;
  * the rank is bounded by the bank: asking for more coordinates than were projected is a
    refusal, not a silent truncation;
  * the increment-permutation null leaves the level-1 terms alone and changes the level-2
    ones, which is the whole reason it is the control for a level-2 result;
  * the null takes at least three seeds, because one shuffle is a draw.

CPU only; the trajectories are synthetic and the family does the arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anamnesis.extraction.path_banks import (
    MIN_NULL_SEEDS,
    PathBank,
    identity_basis,
    null_matrices,
    signature_matrix,
)


def synthetic_paths(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((10 + 2 * i, 4)).astype(np.float32) for i in range(4)]


def test_the_packing_round_trips(tmp_path: Path) -> None:
    paths = synthetic_paths()
    bank = PathBank.from_paths(paths, [3, 5, 7, 11], prompt_lengths=[2, 2, 3, 3])
    assert bank.n == 4 and bank.k_max == 4
    assert bank.lengths.tolist() == [len(p) for p in paths]
    for index, original in enumerate(paths):
        assert np.allclose(bank.path(index), original)

    npz = tmp_path / "paths.npz"
    np.savez(
        npz,
        paths_pc=np.asarray(bank.paths),
        offsets=np.asarray(bank.offsets),
        gen_ids=np.asarray(bank.gen_ids),
        lengths=np.asarray(bank.lengths),
        prompt_lengths=np.asarray(bank.prompt_lengths),
    )
    loaded = PathBank.load(npz)
    assert loaded.n == bank.n and np.allclose(loaded.path(2), bank.path(2))
    assert loaded.source == str(npz)


def test_a_bank_whose_offsets_disagree_with_its_rows_is_refused(tmp_path: Path) -> None:
    bank = PathBank.from_paths(synthetic_paths(), [0, 1, 2, 3])
    npz = tmp_path / "broken.npz"
    offsets = np.asarray(bank.offsets).copy()
    offsets[-1] -= 3
    np.savez(
        npz,
        paths_pc=np.asarray(bank.paths),
        offsets=offsets,
        gen_ids=np.asarray(bank.gen_ids),
        lengths=np.asarray(bank.lengths),
        prompt_lengths=np.asarray(bank.prompt_lengths),
    )
    with pytest.raises(ValueError, match="offsets end at"):
        PathBank.load(npz)


def test_a_bank_missing_the_variant_asked_for_names_what_it_holds(tmp_path: Path) -> None:
    bank = PathBank.from_paths(synthetic_paths(), [0, 1, 2, 3])
    npz = tmp_path / "pc_only.npz"
    np.savez(
        npz,
        paths_pc=np.asarray(bank.paths),
        offsets=np.asarray(bank.offsets),
        gen_ids=np.asarray(bank.gen_ids),
        lengths=np.asarray(bank.lengths),
        prompt_lengths=np.asarray(bank.prompt_lengths),
    )
    with pytest.raises(KeyError, match="paths_nopc"):
        PathBank.load(npz, variant="nopc")
    with pytest.raises(FileNotFoundError):
        PathBank.load(tmp_path / "absent.npz")


def test_a_path_too_short_for_the_level_is_dropped_and_counted() -> None:
    paths = synthetic_paths() + [np.zeros((1, 4), dtype=np.float32)]
    bank = PathBank.from_paths(paths, [0, 1, 2, 3, 99], source="synthetic")
    design = signature_matrix(bank, layer=3, k=3, level=2)
    assert design.n_dropped == 1
    assert design.kept == [0, 1, 2, 3], "the dropped path is named by its absence"
    assert design.X.shape == (4, len(design.names))
    assert np.asarray(bank.gen_ids)[design.kept].tolist() == [0, 1, 2, 3]


def test_asking_for_more_coordinates_than_were_banked_is_refused() -> None:
    bank = PathBank.from_paths(synthetic_paths(), [0, 1, 2, 3], source="synthetic")
    with pytest.raises(ValueError, match="exceeds the banked rank"):
        signature_matrix(bank, layer=3, k=9, level=2)


def test_every_path_failing_is_an_error_rather_than_an_empty_matrix() -> None:
    bank = PathBank.from_paths(
        [np.zeros((1, 4), dtype=np.float32)], [0], source="degenerate"
    )
    with pytest.raises(RuntimeError, match="too short for level"):
        signature_matrix(bank, layer=3, k=2, level=2)


def test_the_permutation_null_keeps_level_one_and_moves_level_two() -> None:
    bank = PathBank.from_paths(synthetic_paths(1), [0, 1, 2, 3], source="synthetic")
    observed = signature_matrix(bank, layer=3, k=3, level=2)
    nulls = null_matrices(bank, layer=3, k=3, level=2, seeds=(1, 2, 3))
    assert len(nulls) == 3
    level_one = [i for i, name in enumerate(observed.names) if "_lvl1_" in name]
    level_two = [i for i, name in enumerate(observed.names) if "_lvl2_" in name]
    assert level_one and level_two, "the names carry both levels"
    for null in nulls:
        assert null.kept == observed.kept
        assert np.allclose(
            null.X[:, level_one], observed.X[:, level_one], atol=1e-5
        ), "shuffling increments preserves the endpoint, so level 1 is untouched"
        assert not np.allclose(null.X[:, level_two], observed.X[:, level_two]), (
            "level 2 is about the order, which the shuffle is destroying"
        )


def test_the_null_needs_more_than_one_shuffle() -> None:
    bank = PathBank.from_paths(synthetic_paths(), [0, 1, 2, 3], source="synthetic")
    with pytest.raises(ValueError, match=f"at least {MIN_NULL_SEEDS}"):
        null_matrices(bank, layer=3, k=3, level=2, seeds=(1, 2))


def test_the_basis_is_the_identity_on_already_projected_coordinates() -> None:
    basis = identity_basis(5)
    assert np.allclose(np.asarray(basis.components), np.eye(5))
    assert basis.label == "pcaA", "the label names the real basis the bank was built with"
