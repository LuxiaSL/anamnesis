"""The lens family: a block-routed model's final read, as if its stack ended at a layer.

A small synthetic block-routed forward (boundaries, committed blocks, every layer's partial)
is read by :func:`lens_features` and by an independent float64 reference of the same final
read. The last layer must be the model's own read (agreement 1, KL 0, the step's entropy);
every layer's features must match the reference; a row's bytes must not depend on its
batchmates or on where the chunks fall; and the reducer must carry the family opt-in, under
its own lane id, after every other family.
"""

from __future__ import annotations

import pytest
import torch

from anamnesis.extraction.fast.engine_reducer import EngineReducer
from anamnesis.extraction.fast.lens import (
    CHUNK,
    QUANTITIES,
    LensError,
    LensHead,
    committed_ends,
    lens_features,
    lens_names,
    visible_blocks,
)

HIDDEN, VOCAB, LAYERS = 8, 33, 7
BOUNDARIES = (0, 2, 5)
WEIGHTS = "w" * 64


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def plan(boundaries=BOUNDARIES, layers=LAYERS):
    """Each layer's (attention sources, MLP sources), as a block-routed decode routes them:
    a layer with ``c`` committed blocks before it reads ``c + 1`` sources before attention
    (none when ``c == 0``) and ``c + 2`` before the MLP at a boundary, else ``c + 1``."""
    out, c = [], 0
    for layer in range(layers):
        boundary = layer in boundaries
        out.append(((c + 1) if c else 0, (c + 2) if boundary else (c + 1)))
        c += boundary
    return tuple(out)


def head(seed=0, weights=WEIGHTS):
    g = torch.Generator().manual_seed(seed)
    return LensHead(final_query=torch.randn(HIDDEN, generator=g),
                    final_key_gain=1.0 + 0.1 * torch.randn(HIDDEN, generator=g),
                    final_key_eps=1e-5, norm_gain=1.0 + 0.1 * torch.randn(HIDDEN, generator=g),
                    norm_eps=1e-5, unembedding=torch.randn(VOCAB, HIDDEN, generator=g),
                    weights=weights)


def forward(steps, seed):
    """(depth [steps, layers, hidden], committed blocks) of one row: the embedding, then each
    block's last partial."""
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(steps, HIDDEN, generator=g)
    depth = torch.randn(steps, LAYERS, HIDDEN, generator=g) * 2.0
    ends = committed_ends(plan())
    committed = [embed] + [depth[:, e] for e in ends[1:]]
    return depth, committed


def reference_logprobs(h: LensHead, depth, committed, layer):
    """The final read at ``layer`` in float64, position by position."""
    ends = committed_ends(plan())
    qw = (h.final_query * h.final_key_gain).double()
    out = []
    for t in range(depth.shape[0]):
        srcs = [committed[c][t].double() for c in range(len(ends)) if ends[c] < layer]
        srcs.append(depth[t, layer].double())
        x = torch.stack(srcs)
        logits = (x @ qw) / torch.sqrt(x.pow(2).mean(-1) + h.final_key_eps)
        w = torch.softmax(logits, 0)
        mix = (w[:, None] * x).sum(0)
        mix = mix / torch.sqrt(mix.pow(2).mean() + h.norm_eps) * h.norm_gain.double()
        out.append(torch.log_softmax(h.unembedding.double() @ mix, -1))
    return torch.stack(out)


def batch_of(rows_steps, width=None, seeds=None):
    seeds = seeds or list(range(len(rows_steps)))
    width = width or max(rows_steps)
    depth = torch.zeros(len(rows_steps), width, LAYERS, HIDDEN)
    committed = [torch.zeros(len(rows_steps), width, HIDDEN) for _ in committed_ends(plan())]
    chosen = torch.zeros(len(rows_steps), width, dtype=torch.long)
    entropy = torch.zeros(len(rows_steps), width, dtype=torch.float64)
    h = head()
    for b, (n, seed) in enumerate(zip(rows_steps, seeds)):
        d, c = forward(n, seed)
        depth[b, :n] = d
        for i, block in enumerate(c):
            committed[i][b, :n] = block
        final = reference_logprobs(h, d, c, LAYERS - 1)
        entropy[b, :n] = -(final.exp() * final).sum(-1)
        chosen[b, :n] = torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))
    lengths = torch.as_tensor(rows_steps)
    return h, depth, committed, chosen, entropy, lengths


def test_committed_ends_follow_the_routing_plan():
    ends = committed_ends(plan())
    assert ends == [-1, 1, 4]
    assert visible_blocks(ends, 0) == [0]
    assert visible_blocks(ends, 1) == [0]  # layer 1 ends block 1: it is the partial, not committed
    assert visible_blocks(ends, 2) == [0, 1]
    assert visible_blocks(ends, LAYERS - 1) == [0, 1, 2]


def test_the_last_layer_is_the_models_own_read():
    layers = (1, 3, LAYERS - 1)
    h, depth, committed, chosen, entropy, lengths = batch_of([9])
    got = lens_features(h, depth, committed, plan(), layers, chosen, entropy, None)
    v = dict(zip(got.names, got.values[0].tolist()))
    last = LAYERS - 1
    assert v[f"lens_L{last}_agree_final_mean"] == 1.0
    assert v[f"lens_L{last}_kl_final_mean"] == 0.0
    assert float(got.final_check[0]) < 1e-5


@pytest.mark.parametrize("layer", [0, 1, 2, 4, LAYERS - 1])
def test_every_layer_matches_the_float64_reference(layer):
    h, depth, committed, chosen, entropy, lengths = batch_of([11])
    got = lens_features(h, depth, committed, plan(), (layer,), chosen, entropy, None)
    v = dict(zip(got.names, got.values[0].tolist()))
    lp = reference_logprobs(h, depth[0], [c[0] for c in committed], layer)
    final = reference_logprobs(h, depth[0], [c[0] for c in committed], LAYERS - 1)
    ent = -(lp.exp() * lp).sum(-1)
    kl = (lp.exp() * (lp - final)).sum(-1)
    assert v[f"lens_L{layer}_entropy_mean"] == pytest.approx(float(ent.mean()), abs=1e-5)
    assert v[f"lens_L{layer}_entropy_std"] == pytest.approx(float(ent.std(correction=0)), abs=1e-5)
    assert v[f"lens_L{layer}_kl_final_mean"] == pytest.approx(float(kl.mean()), abs=1e-5)
    assert v[f"lens_L{layer}_agree_final_mean"] == pytest.approx(
        float((lp.argmax(-1) == final.argmax(-1)).double().mean()))
    assert v[f"lens_L{layer}_agree_chosen_mean"] == pytest.approx(
        float((lp.argmax(-1) == chosen[0]).double().mean()))


def _read(batch, layers):
    h, depth, committed, chosen, entropy, lengths = batch
    return lens_features(h, depth, committed, plan(), layers, chosen, entropy, lengths)


def test_a_rows_bytes_do_not_depend_on_its_batchmates_or_the_chunks():
    layers = (0, 2, 4, LAYERS - 1)
    alone = _read(batch_of([37], width=64, seeds=[5]), layers)
    # six rows of 64 padded steps: 384 positions, so the row's steps fall in another chunk
    many = _read(batch_of([20, 37, 64, 3, 64, 64], width=64, seeds=[1, 5, 2, 3, 4, 6]), layers)
    assert 6 * 64 > CHUNK
    assert alone.values[0].numpy().tobytes() == many.values[1].numpy().tobytes()
    assert alone.final_check[0] == many.final_check[1]


def test_padded_steps_do_not_reach_the_row():
    layers = (2, LAYERS - 1)
    h, depth, committed, chosen, entropy, lengths = batch_of([5], width=5, seeds=[9])
    exact = lens_features(h, depth, committed, plan(), layers, chosen, entropy, None)
    h, depth, committed, chosen, entropy, lengths = batch_of([5], width=16, seeds=[9])
    depth[:, 5:] = 7.0  # garbage past the row's steps
    padded = lens_features(h, depth, committed, plan(), layers, chosen, entropy, lengths)
    torch.testing.assert_close(padded.values, exact.values, rtol=0, atol=1e-6)


def test_lens_depth_is_where_agreement_holds_for_the_rest():
    h, depth, committed, chosen, entropy, lengths = batch_of([12])
    layers = tuple(range(LAYERS))
    got = lens_features(h, depth, committed, plan(), layers, chosen, entropy, None)
    v = dict(zip(got.names, got.values[0].tolist()))
    agree = [v[f"lens_L{layer}_agree_final_mean"] for layer in layers]
    first = LAYERS - 1
    for layer in reversed(layers):
        if agree[layer] <= 0.5:
            break
        first = layer
    assert v["lens_depth"] == pytest.approx(first / (LAYERS - 1))
    assert got.names == lens_names(layers)
    assert len(got.names) == len(layers) * len(QUANTITIES) + 1


def test_a_head_names_its_weights_and_its_digest_follows_its_tensors():
    with pytest.raises(LensError, match="weights digest"):
        head(weights="")
    assert head(seed=0).sha256 == head(seed=0).sha256
    assert head(seed=0).sha256 != head(seed=1).sha256
    with pytest.raises(LensError, match="must be"):
        LensHead(final_query=torch.zeros(3), final_key_gain=torch.ones(HIDDEN), final_key_eps=1e-5,
                 norm_gain=torch.ones(HIDDEN), norm_eps=1e-5,
                 unembedding=torch.zeros(VOCAB, HIDDEN), weights=WEIGHTS)


def test_a_head_reads_a_state_dict_tied_or_compiled():
    h = head()
    state = {"_orig_mod.final_res_query": h.final_query,
             "_orig_mod.final_res_norm.weight": h.final_key_gain,
             "_orig_mod.norm.weight": h.norm_gain,
             "_orig_mod.embed_tokens.weight": h.unembedding,
             "_orig_mod.layers.0.attn.q_proj.weight": torch.zeros(2, 2)}
    got = LensHead.from_state_dict(state, weights=WEIGHTS, final_key_eps=1e-5, norm_eps=1e-5)
    assert got.sha256 == h.sha256
    with pytest.raises(LensError, match="final_res_query"):
        LensHead.from_state_dict({"norm.weight": h.norm_gain}, weights=WEIGHTS,
                                 final_key_eps=1e-5, norm_eps=1e-5)


def test_a_head_reads_a_checkpoint(tmp_path):
    h = head()
    path = tmp_path / "step.pt"
    torch.save({"model": {"final_res_query": h.final_query, "final_res_norm.weight": h.final_key_gain,
                          "norm.weight": h.norm_gain, "lm_head.weight": h.unembedding},
                "step": 3}, path)
    got = LensHead.from_checkpoint(path, weights=WEIGHTS, dtype=None)
    assert got.sha256 == h.sha256
    assert LensHead.from_checkpoint(path, weights=WEIGHTS).unembedding.dtype == torch.bfloat16


def test_inputs_that_disagree_are_refused():
    h, depth, committed, chosen, entropy, lengths = batch_of([4])
    with pytest.raises(LensError, match="commits"):
        lens_features(h, depth, committed[:-1], plan(), (1,), chosen, entropy, None)
    with pytest.raises(LensError, match="outside"):
        lens_features(h, depth, committed, plan(), (LAYERS,), chosen, entropy, None)
    with pytest.raises(LensError, match="routing entries"):
        lens_features(h, depth, committed, plan()[:-1], (1,), chosen, entropy, None)


# --- the reducer carries it -------------------------------------------------------------

def signed(steps=12, seed=0):
    from test_engine_reducer import signed_example

    lane, row, token_ids, _, _ = signed_example(steps=steps, seed=seed)
    depth = row.surfaces["depth"]
    ends = committed_ends(row.route_sources)
    row.surfaces["committed"] = [torch.randn(steps, depth.shape[2])] + [depth[:, e] for e in ends[1:]] \
        if ends and ends[0] == -1 else [depth[:, e] for e in ends]
    return lane, row, token_ids


def engine_head(width):
    g = torch.Generator().manual_seed(3)
    return LensHead(final_query=torch.randn(width, generator=g), final_key_gain=torch.ones(width),
                    final_key_eps=1e-5, norm_gain=torch.ones(width), norm_eps=1e-5,
                    unembedding=torch.randn(40, width, generator=g), weights=WEIGHTS)


def test_the_reducer_carries_the_lens_opt_in_under_its_own_lane():
    from test_engine_reducer import ENGINE

    lane, row, token_ids = signed()
    h = engine_head(row.surfaces["depth"].shape[2])
    plain = EngineReducer(lane_for=lambda s: lane, expected_weights=WEIGHTS).reduce(
        [row], token_ids=[token_ids], engine=ENGINE)[0]
    lensed = EngineReducer(lane_for=lambda s: lane, expected_weights=WEIGHTS, lens=h).reduce(
        [row], token_ids=[token_ids], engine=ENGINE)[0]
    sampled = tuple(lane.config.sampled_layers)
    n = len(plain.meta["feature_names"])
    assert lensed.meta["feature_names"][:n] == plain.meta["feature_names"]
    assert tuple(lensed.meta["feature_names"][n:]) == lens_names(sampled)
    assert lensed.features[:n].tobytes() == plain.features.tobytes()
    assert lensed.lane_id != plain.lane_id
    assert lensed.meta["reducer_identity"] != plain.meta["reducer_identity"]
    assert lensed.meta["receipt"]["reductions"]["lens"] == "batched"
    assert lensed.meta["receipt"]["capture"]["lens"]["head_sha256"] == h.sha256
    assert lensed.names_digest != plain.names_digest


def test_the_reducer_refuses_a_foreign_head_or_the_per_row_path():
    lane, row, _ = signed()
    h = engine_head(row.surfaces["depth"].shape[2])
    with pytest.raises(ValueError, match="belongs to weights"):
        EngineReducer(lane_for=lambda s: lane, expected_weights="v" * 64, lens=h)
    with pytest.raises(ValueError, match="batched"):
        EngineReducer(lane_for=lambda s: lane, expected_weights=WEIGHTS, lens=h, batched=False)
    with pytest.raises(ValueError, match="expected_weights"):
        EngineReducer(lane_for=lambda s: lane, lens=h)
