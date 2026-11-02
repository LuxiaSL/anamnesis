"""Column C's FLOOR reductions and schema identities (anamnesis.extraction.column_c)."""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.extraction import column_c as cc

GEO = cc.Geometry("selftest", n_points=9, n_read_layers=8,
                  route_sites=(cc.RouteSite("pre_mlp.L0", ("emb", "self"), (1.0, 0.0)),))


def _row(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    path = rng.normal(size=(9, 32)) * scale
    read = dict(head_ent=np.full((8, 4), 0.5), mean_ent=np.full(8, 1.5), sink=np.full((8, 4), 0.2),
                recency=np.full((8, 4), 0.3), prompt=np.full((8, 4), 0.9))
    return cc.floor_vector(GEO, path, read, [np.array([0.25, 0.75])])


def test_vector_matches_schema_and_is_finite():
    v = _row(np.random.default_rng(0))
    assert len(v) == len(cc.floor_names(GEO)) == 113
    assert np.isfinite(v).all()


def test_digests_are_pinned():
    # Banked rows are matched to their schema by these digests; they change only with the names.
    assert cc.floor_digest(GEO).startswith("482d9cd3472f")
    assert cc.floor_v1_digest(GEO).startswith("0173a4f01b35")


def test_scale_moves_only_magnitude_features():
    names = cc.floor_names(GEO)
    a = cc.floor_vector(GEO, np.random.default_rng(1).normal(size=(9, 32)),
                        dict(head_ent=np.ones((8, 4)), mean_ent=np.ones(8) * 2, sink=np.ones((8, 4)) * .1,
                             recency=np.ones((8, 4)) * .2, prompt=np.ones((8, 4))), [np.array([0.5, 0.5])])
    b = cc.floor_vector(GEO, np.random.default_rng(1).normal(size=(9, 32)) * 3.0,
                        dict(head_ent=np.ones((8, 4)), mean_ent=np.ones(8) * 2, sink=np.ones((8, 4)) * .1,
                             recency=np.ones((8, 4)) * .2, prompt=np.ones((8, 4))), [np.array([0.5, 0.5])])
    for n, x, y in zip(names, a, b):
        if not n.startswith(("depth.lognorm", "depth.dlognorm", "depth.area")):
            assert abs(x - y) < 1e-9, n


def test_v1_is_v0_without_band_and_area():
    rows = np.stack([_row(np.random.default_rng(k)) for k in range(3)])
    x1, n1 = cc.to_v1(rows, cc.floor_names(GEO))
    assert n1 == cc.floor_v1_names(GEO)
    assert x1.shape == (3, len(n1)) == (3, 101)
    assert not any(n.startswith(cc.V1_DROPPED_PREFIXES) for n in n1)
    v0 = cc.floor_names(GEO)
    np.testing.assert_array_equal(x1[:, 0], rows[:, v0.index(n1[0])])
    np.testing.assert_array_equal(x1[:, -1], rows[:, v0.index(n1[-1])])


def test_to_v1_refuses_a_foreign_name_list():
    with pytest.raises(ValueError):
        cc.to_v1(np.zeros((1, 2)), ["read.ent.L0", "read.ent.L1"])


def test_geometry_mismatch_is_refused():
    with pytest.raises(ValueError):
        cc.floor_vector(GEO, np.zeros((8, 32)) + 1.0)
    with pytest.raises(ValueError):
        cc.floor_vector(GEO, np.ones((9, 32)))   # geometry declares a read-profile; none given


def test_route_weights_must_match_the_site():
    with pytest.raises(ValueError):
        cc.route_features([np.array([0.2, 0.3, 0.5])], GEO.route_sites)
