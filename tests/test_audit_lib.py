"""The audit library: the controls, not the conclusions.

Each function here is one step of a protocol that a wrong result would pass
silently, so each test checks the property the step exists for rather than a
number it happens to produce:

  * residualization removes exactly the covariate-explained part, and the split
    version fits on TRAIN only — the leak that would make a length control a
    length reader;
  * folds hold out whole topics, and shrinking the training set drops topics
    rather than rows;
  * the signature matrix pins feature order across runs and fills a missing
    feature with zero rather than shifting a column;
  * a sampled position set is the same size whatever the generation length, which
    is what makes per-generation surface vectors stackable;
  * the Gram reduction preserves inner products up to the global scale, which is
    the claim that makes it lossless for a linear readout;
  * the readout pair is two architectures and nothing between them: the linear one
    is fitted by a convex solver, so it is a floor rather than one fit's opinion,
    and the training accuracy comes back beside the test accuracy because a floor
    at chance means one thing when the fit converged and another when it did not.

CPU only. ``preprocess_fold_gpu`` runs on the CPU device here — the reduction is
device-agnostic and the name records why the path exists, not a requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anamnesis.analysis.audit_lib import (
    ALL_SURFACES,
    ATTN_BINS,
    DEEP,
    DEEP_HIDDEN,
    GEN_BINS,
    HARD,
    LOGIT,
    N_POS,
    PROMPT_BINS,
    SIMPLE,
    SignatureMatrix,
    attention_vector,
    gen_metadata_by_id,
    leak_free_folds,
    load_merged_signature_matrix,
    preprocess_fold_gpu,
    make_encoder,
    residualize,
    residualize_all,
    sample_positions,
    subsample_topics,
    surface_vector,
    train_eval,
    unwrap_generations,
)

torch = pytest.importorskip("torch", reason="preprocess_fold_gpu is a torch path")


def test_the_hard_mode_set_is_the_five_of_record() -> None:
    assert HARD == frozenset(
        {"linear", "socratic", "contrastive", "dialectical", "analogical"}
    )


def test_residualize_all_removes_what_the_covariates_explain() -> None:
    rng = np.random.default_rng(7)
    C = rng.standard_normal((60, 2))
    beta = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]])
    F = C @ beta + 4.0                       # exactly explained + an intercept
    resid = residualize_all(F, C)
    assert np.allclose(resid, 0.0, atol=1e-9), "a fully explained matrix should residualize to zero"

    noise = rng.standard_normal((60, 3))
    resid_noisy = residualize_all(F + noise, C)
    # What survives is uncorrelated with the covariates, which is the property
    # downstream classification depends on.
    for j in range(3):
        for k in range(2):
            assert abs(np.corrcoef(resid_noisy[:, j], C[:, k])[0, 1]) < 1e-8


def test_split_residualize_fits_on_train_only() -> None:
    rng = np.random.default_rng(11)
    Ctr, Cte = rng.standard_normal((40, 2)), rng.standard_normal((10, 2))
    beta = rng.standard_normal((2, 3))
    Ftr, Fte = Ctr @ beta, Cte @ beta
    rtr, rte = residualize(Ftr, Fte, Ctr, Cte)
    assert np.allclose(rtr, 0.0, atol=1e-9)
    assert np.allclose(rte, 0.0, atol=1e-9), "the train-fitted coefficients transfer"

    # A test set whose relationship differs is NOT re-fitted to: the residual
    # keeps the difference, which is the whole point of fitting on train.
    Fte_shifted = Cte @ beta + 5.0
    _, rte_shifted = residualize(Ftr, Fte_shifted, Ctr, Cte)
    assert np.allclose(rte_shifted, 5.0, atol=1e-8)


def test_folds_hold_out_whole_topics() -> None:
    topic = np.repeat(np.arange(10), 4)
    y = np.tile(np.array(["a", "b", "c", "d"]), 10)
    X = np.zeros((40, 3))
    folds = list(leak_free_folds(X, y, topic, n_splits=5))
    assert len(folds) == 5
    seen_test: set[int] = set()
    for _seed, tr, te in folds:
        train_topics = set(topic[tr].tolist())
        test_topics = set(topic[te].tolist())
        assert not (train_topics & test_topics), "a topic straddled the boundary"
        seen_test |= test_topics
    assert seen_test == set(range(10)), "every topic is held out exactly once"


def test_more_seeds_redraw_the_training_subsample_not_the_folds() -> None:
    topic = np.repeat(np.arange(12), 3)
    y = np.tile(np.array(["a", "b", "c"]), 12)
    X = np.zeros((36, 2))
    full = list(leak_free_folds(X, y, topic, n_splits=4, seeds=3, frac=1.0))
    assert len(full) == 12
    # At frac=1.0 the seed changes nothing, so the three passes are identical.
    layouts = [tuple(sorted(te.tolist())) for _s, _tr, te in full]
    assert layouts[:4] == layouts[4:8] == layouts[8:]

    shrunk = list(leak_free_folds(X, y, topic, n_splits=4, seeds=2, frac=0.5))
    sizes = {len(tr) for _s, tr, _te in shrunk}
    assert max(sizes) < 27, "frac<1 must actually shrink the training set"
    for _s, tr, te in shrunk:
        assert not (set(topic[tr].tolist()) & set(topic[te].tolist()))


def test_more_folds_than_topics_is_refused_rather_than_guessed_at() -> None:
    topic = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="fold cannot hold out a topic"):
        list(leak_free_folds(np.zeros((4, 2)), np.array(list("abab")), topic, n_splits=5))


def test_subsample_topics_drops_topics_and_keeps_at_least_two() -> None:
    topic = np.repeat(np.arange(8), 2)
    tr = np.arange(16)
    assert subsample_topics(tr, topic, 1.0, seed=0) is tr
    kept = subsample_topics(tr, topic, 0.5, seed=3)
    assert len(set(topic[kept].tolist())) == 4
    # Whole topics, so every surviving row's topic is fully present.
    for t in set(topic[kept].tolist()):
        assert (topic[kept] == t).sum() == 2
    tiny = subsample_topics(tr, topic, 0.01, seed=3)
    assert len(set(topic[tiny].tolist())) == 2, "the floor is two topics, not zero"


def write_signature_run(
    root: Path, run: str, features: dict[int, dict[str, float]], *, mode: str = "linear"
) -> None:
    """A run directory with metadata.json and one npz per generation."""
    run_dir = root / run
    sig_dir = run_dir / "signatures_v3"
    sig_dir.mkdir(parents=True)
    gens = []
    for gid, feats in features.items():
        np.savez(
            sig_dir / f"gen_{gid:03d}.npz",
            feature_names=np.array(list(feats)),
            features=np.array(list(feats.values()), dtype=np.float64),
        )
        gens.append({
            "generation_id": gid, "mode": mode, "topic_idx": gid % 3,
            "prompt_length": 40 + gid, "num_generated_tokens": 100 + gid,
        })
    (run_dir / "metadata.json").write_text(json.dumps({"generations": gens}))


def test_metadata_unwrapping_accepts_both_shapes(tmp_path: Path) -> None:
    rows = [{"generation_id": 3, "mode": "linear"}]
    assert unwrap_generations({"generations": rows}) == rows
    assert unwrap_generations(rows) == rows
    (tmp_path / "metadata.json").write_text(json.dumps({"generations": rows}))
    assert gen_metadata_by_id(tmp_path / "metadata.json")[3]["mode"] == "linear"


def test_signature_matrix_pins_feature_order_and_zero_fills(tmp_path: Path) -> None:
    write_signature_run(tmp_path, "run_a", {
        0: {"f_a": 1.0, "f_b": 2.0},
        1: {"f_b": 20.0, "f_a": 10.0},      # same features, different order
        2: {"f_a": 100.0},                  # f_b missing entirely
    })
    sm = load_merged_signature_matrix(["run_a", "absent_run"], tmp_path)
    assert isinstance(sm, SignatureMatrix)
    assert list(sm.names) == ["f_a", "f_b"]
    assert sm.X.shape == (3, 2)
    assert sm.X[1].tolist() == [10.0, 20.0], "order came from the name map, not the file"
    assert sm.X[2].tolist() == [100.0, 0.0], "a missing feature fills with zero"
    assert sm.C.shape == (3, 2)
    assert sm.C[0].tolist() == [40.0, 100.0]
    assert list(sm.topic) == [0, 1, 2]


def test_signature_matrix_filters_by_mode_and_survives_an_empty_selection(
    tmp_path: Path,
) -> None:
    write_signature_run(tmp_path, "run_a", {0: {"f": 1.0}}, mode="rhetorical")
    empty = load_merged_signature_matrix(["run_a"], tmp_path)          # default modes=HARD
    assert empty.X.size == 0 and empty.names.size == 0
    kept = load_merged_signature_matrix(["run_a"], tmp_path, modes={"rhetorical"})
    assert kept.X.shape == (1, 1)


def test_sampled_positions_are_a_fixed_count_at_any_length() -> None:
    for T in (1, 2, 7, 512):
        pos = sample_positions(T)
        assert pos.shape == (N_POS,)
        assert pos.min() >= 0 and pos.max() <= T - 1
    assert sample_positions(0).tolist() == [0] * N_POS, "a degenerate length still aligns"
    assert sample_positions(9, n=3).tolist() == [0, 4, 8]


def write_raw(path: Path, *, T: int = 6, layers: int = 2, heads: int = 2,
              prompt_length: int = 4, hidden: int = 5) -> Path:
    """A raw-tensor npz with one simple surface and variable-length attention rows."""
    rng = np.random.default_rng(5)
    max_seq = prompt_length + T
    attn = np.zeros((T, layers, heads, max_seq), dtype=np.float16)
    lengths = np.zeros(T, dtype=np.int64)
    for t in range(T):
        visible = prompt_length + t + 1
        lengths[t] = visible
        row = rng.random((layers, heads, visible))
        attn[t, :, :, :visible] = (row / row.sum(axis=-1, keepdims=True)).astype(np.float16)
    np.savez(
        path,
        attentions=attn,
        actual_lengths=lengths,
        prompt_length=np.array(prompt_length),
        hidden_states=rng.standard_normal((T, layers + 1, hidden)).astype(np.float32),
    )
    return path


def test_surface_vectors_are_fixed_width_per_surface(tmp_path: Path) -> None:
    z = np.load(write_raw(tmp_path / "gen_000.npz"), allow_pickle=True)
    pos = sample_positions(6)
    residual = surface_vector(z, "residual", pos, T=6)
    assert residual is not None
    assert residual.shape == (N_POS * 3 * 5,)
    assert residual.dtype == np.float16

    attention = surface_vector(z, "attention", pos, T=6)
    assert attention is not None
    assert attention.shape == (N_POS * 2 * 2 * ATTN_BINS,)
    assert ATTN_BINS == PROMPT_BINS + GEN_BINS
    assert attention_vector(z, pos) is not None

    # A surface the bank does not carry is a skip, not a zero vector.
    assert surface_vector(z, "values", pos, T=6) is None
    assert set(ALL_SURFACES) == set(SIMPLE) | {"attention"}


def test_attention_resampling_keeps_the_prompt_and_generated_parts_apart(
    tmp_path: Path,
) -> None:
    """The prompt/generated split has to survive the reduction, because the length
    control is applied to the two parts separately downstream."""
    T, prompt_length = 4, 6
    max_seq = prompt_length + T
    attn = np.zeros((T, 1, 1, max_seq), dtype=np.float16)
    lengths = np.zeros(T, dtype=np.int64)
    for t in range(T):
        visible = prompt_length + t + 1
        lengths[t] = visible
        attn[t, 0, 0, :prompt_length] = 1.0        # all mass on the prompt
    np.savez(tmp_path / "g.npz", attentions=attn, actual_lengths=lengths,
             prompt_length=np.array(prompt_length))
    z = np.load(tmp_path / "g.npz", allow_pickle=True)
    vec = attention_vector(z, sample_positions(T)).reshape(N_POS, ATTN_BINS)
    assert np.all(vec[:, :PROMPT_BINS] > 0.0), "prompt bins carry the prompt mass"
    assert np.all(vec[:, PROMPT_BINS:] == 0.0), "generated bins stay empty"


def test_a_bank_without_attention_yields_no_attention_vector(tmp_path: Path) -> None:
    np.savez(tmp_path / "g.npz", hidden_states=np.zeros((3, 2, 4), dtype=np.float32))
    z = np.load(tmp_path / "g.npz", allow_pickle=True)
    assert attention_vector(z, sample_positions(3)) is None


def test_the_gram_reduction_preserves_the_geometry_it_claims_to() -> None:
    rng = np.random.default_rng(20260920)
    n_tr, n_te, P = 24, 6, 400
    Xtr = rng.standard_normal((n_tr, P))
    Xte = rng.standard_normal((n_te, P))
    Ctr, Cte = rng.standard_normal((n_tr, 2)), rng.standard_normal((n_te, 2))

    Ztr, Zte = preprocess_fold_gpu(Xtr, Xte, Ctr, Cte, resid=False, device="cpu")
    assert Ztr.shape[0] == n_tr and Zte.shape[0] == n_te
    assert Ztr.shape[1] <= n_tr, "the reduction lands in the train row space"
    assert Ztr.shape[1] == Zte.shape[1]

    # Standardize by hand and compare Gram matrices: the reduced coordinates
    # reproduce the standardized inner products up to one global scale.
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    Str = (Xtr - mu) / np.maximum(sd, 1e-8)
    scale = (Ztr @ Ztr.T)[0, 0] / (Str @ Str.T)[0, 0]
    assert np.allclose(Ztr @ Ztr.T, scale * (Str @ Str.T), rtol=1e-3, atol=1e-3)

    # With the length control on, the covariate-explained part is gone before the
    # reduction, so the coordinates differ.
    Rtr, _ = preprocess_fold_gpu(Xtr, Xte, Ctr, Cte, resid=True, device="cpu")
    assert Rtr.shape[0] == n_tr
    assert not np.allclose(Rtr[:, :1], Ztr[:, :1])


# ── The readout pair ──────────────────────────────────────────────────────────
def test_the_two_architectures_are_the_only_two_and_a_third_is_refused() -> None:
    linear = make_encoder(6, LOGIT, nclass=3)
    assert linear.in_features == 6 and linear.out_features == 3
    deep = make_encoder(6, DEEP, nclass=3, k=8)
    widths = [module.out_features for module in deep if hasattr(module, "out_features")]
    assert widths == [DEEP_HIDDEN, 8, 3], "the nonlinear readout narrows through its bottleneck"
    with pytest.raises(ValueError, match="unknown arch"):
        make_encoder(6, "svm")


def test_a_separable_problem_is_solved_by_the_floor_alone() -> None:
    """A linearly separable planted signal needs no nonlinearity, and says so."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    X[:, 0] += 4.0 * y
    train, test = np.arange(40), np.arange(40, 60)
    floor_test, floor_train = train_eval(
        X[train], y[train], X[test], y[test], LOGIT, 0, "cpu", nclass=2
    )
    assert floor_test > 0.9, "the convex solver reaches the separating hyperplane"
    assert floor_train > 0.9


def test_the_readout_is_deterministic_under_its_seed() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 5))
    y = (X[:, 1] > 0).astype(np.int64)
    args = (X[:30], y[:30], X[30:], y[30:], DEEP, 7, "cpu")
    first = train_eval(*args, deep_epochs=30, nclass=2, k=4)
    again = train_eval(*args, deep_epochs=30, nclass=2, k=4)
    assert first == again, "same seed, same fold, same numbers"


def test_a_fit_that_did_not_converge_is_visible_in_the_training_accuracy() -> None:
    """Chance on the test rows is read differently depending on the train rows."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 5))
    y = rng.integers(0, 2, size=40)
    test_acc, train_acc = train_eval(
        X[:30], y[:30], X[30:], y[30:], DEEP, 3, "cpu", deep_epochs=1, nclass=2, k=4
    )
    assert 0.0 <= test_acc <= 1.0
    assert 0.0 <= train_acc <= 1.0
    assert train_acc < 1.0, "one epoch has not fitted the training rows either"
