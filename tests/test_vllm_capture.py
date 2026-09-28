"""LaneCapture against a stand-in model runner: no vLLM import, no model, no device.

:class:`anamnesis.extraction.vllm.capture.LaneCapture` touches the engine only
through the runner object it is handed, so these cases hand it a stand-in with the
same private surface (``_prepare_inputs``, ``_get_prompt_logprobs_dict``,
``sample_tokens``, the model's ``compute_logits``, and a Llama-shaped module
graph whose projections are mutated in place after the hooks read them). The
stand-in scheduler can split a prompt across steps in any order, so capture
reassembly is checked against a whole-prompt reference, and every refusal the
capture makes about the schedule, the hooks and the prompt-logit views is
exercised by breaking exactly one of them.

The attention statistics are driven through the real collector and the real
product reductions of :mod:`anamnesis.extraction.vllm.backend` and
:mod:`anamnesis.extraction.vllm.step_products`; only the device kernel is
replaced, by a deterministic scratch whose rows depend on the row length alone.
Whether the instrumented kernel itself computes those rows is covered where a
device and the engine are present.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from anamnesis.extraction.vllm import backend
from anamnesis.extraction.vllm.backend import (
    StatsCollector,
    current_collector,
    install_collector,
    uninstall_collector,
)
from anamnesis.extraction.vllm.capture import LaneCapture
from anamnesis.extraction.vllm.receipts import (
    SUBSTRATE_FIELDS,
    assert_substrate_fields,
    capture_receipt,
)
from anamnesis.extraction.vllm.rows import request_row_schema
from anamnesis.extraction.vllm.second_pass import coverage_of_selected_rows
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS
from anamnesis.extraction.vllm.step_products import (
    agreement_pair,
    entropy_of_rows,
    per_head_summaries,
    series_products,
    span_products,
)

HEADS = 2

ATTENTION_KEYS = ("attn_stats", "attn_coverage", "attn_h_mean", "attn_h_heads",
                  "attn_spectral_rows", "attn_decay_rows", "attn_span_rows",
                  "attn_entropy_rows", "attn_head_ent", "attn_head_sink",
                  "attn_head_prompt", "attn_head_recency")


def fill_stats(positions):
    base = positions.float()[:, None, None]
    return (base + torch.arange(HEADS)[None, :, None] / 10
            + torch.arange(NUM_STATS)[None, None, :] / 100)


def fake_scratch(lengths):
    """Deterministic normalized rows, a pure function of the row length."""
    width = int(lengths.max())
    k = torch.arange(width)
    rows = []
    for length in lengths.tolist():
        scores = (k[:length] % 3).float() + 0.05 * length
        p = torch.softmax(
            torch.stack([scores + h * 0.2 for h in range(HEADS)]), dim=-1)
        row = torch.zeros(HEADS, width)
        row[:, :length] = p
        rows.append(row)
    return torch.stack(rows)


def emulate_backend_layer(name, positions):
    """What the instrumented impl does for one layer of one step."""
    collector = current_collector()
    if collector is None or not collector.armed:
        return
    stats = collector.stats_buffer(
        layer_name=name, num_tokens=len(positions), num_query_heads=HEADS,
        device="cpu")
    stats[:] = fill_stats(positions)
    selection = collector.second_pass_for(name)
    if selection is None:
        return
    scratch = fake_scratch(selection.lengths)
    rowsum = torch.where(
        (selection.slots >= 0)[:, None],
        torch.ones(len(selection.slots), HEADS),
        torch.full((len(selection.slots), HEADS), float("nan")))
    if selection.kind == "span":
        products = span_products(
            scratch, lengths=selection.lengths,
            agreement_index=selection.agreement_index,
            spectral_index=selection.spectral_index,
            decay_index=selection.decay_index,
            entropy_index=selection.entropy_index,
            prefixes=selection.prefixes,
            widths=selection.widths,
            rowsum=rowsum, slots=selection.slots)
    else:
        products = series_products(
            scratch, agreement_index=selection.agreement_index,
            entropy_index=selection.entropy_index,
            widths=selection.widths,
            rowsum=rowsum, slots=selection.slots)
    collector.record_second(name, products)


class Projection(nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, x):
        return x.repeat(1, self.repeats), None


class Norm(nn.Module):
    def forward(self, x, residual=None):
        value = x if residual is None else x + residual
        return value * 0.5, value.clone()


class Block(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.input_layernorm = Norm()
        self.self_attn = nn.Module()
        self.self_attn.q_size = 4
        self.self_attn.kv_size = 2
        self.self_attn.head_dim = 2
        self.self_attn.qkv_proj = Projection(2)
        self.self_attn.attn = SimpleNamespace(
            layer_name=f"model.layers.{index}.self_attn.attn")
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = Projection(3)

    def forward(self, positions, hidden_states, residual):
        h, r = self.input_layernorm(hidden_states, residual)
        qkv, _ = self.self_attn.qkv_proj(h)
        gate, _ = self.mlp.gate_up_proj(h)
        # The engine reuses these buffers; the capture must have copied them.
        qkv.add_(1000)
        gate.zero_()
        emulate_backend_layer(self.self_attn.attn.layer_name, positions)
        return h + 1, r


class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=31, hidden_size=4, intermediate_size=6)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(0), Block(1), Block(2)])
        self.model.norm = Norm()

    def forward(self, input_ids, positions):
        h = (
            input_ids[:, None] + positions[:, None] / 16 + torch.arange(4)[None, :] / 4
        ).to(torch.bfloat16)
        r = None
        for block in self.model.layers:
            h, r = block(positions, h, r)
        return self.model.norm(h, r)[0]

    def compute_logits(self, h):
        return h[:, 0:1] + torch.arange(31)[None, :] / 32


class Runner:
    """A model runner with the private surface LaneCapture wraps.

    ``step(counts)`` schedules ``counts[request]`` tokens of each request, in the
    dict's order, the way the engine's scheduler hands a step to the runner.
    """

    def __init__(self, specs):
        self.model = Graph()
        self.device = "cpu"
        self.specs = specs
        self.requests = {
            r: SimpleNamespace(prompt_token_ids=s["input_ids"], num_computed_tokens=0)
            for r, s in specs.items()
        }
        self.num_prompt_logprobs = {r: 0 for r in specs}

    def _prepare_inputs(self, scheduler, counts):
        ids = list(scheduler.num_scheduled_tokens)
        offsets = np.r_[0, np.cumsum(counts)]
        self.input_batch = SimpleNamespace(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            num_computed_tokens_cpu=np.array(
                [self.requests[r].num_computed_tokens for r in ids]
            ),
        )
        self.query_start_loc = SimpleNamespace(np=offsets)
        return torch.tensor(offsets[1:] - 1), None

    def _get_prompt_logprobs_dict(self, hidden, counts):
        completed = []
        for r in self.num_prompt_logprobs:
            if r not in counts:
                continue
            state = self.requests[r]
            remaining = len(state.prompt_token_ids) - state.num_computed_tokens - 1
            n = min(counts[r], remaining)
            if counts[r] > remaining:
                completed.append(r)
            if n <= 0:
                continue
            offset = self.query_start_loc.np[self.input_batch.req_id_to_index[r]]
            self.model.compute_logits(hidden[offset : offset + n])
        for r in completed:
            del self.num_prompt_logprobs[r]
        return {}

    def sample_tokens(self, _grammar=None):
        out = self._get_prompt_logprobs_dict(
            self.hidden, self.scheduler.num_scheduled_tokens
        )
        for r, n in self.scheduler.num_scheduled_tokens.items():
            self.requests[r].num_computed_tokens += n
        return out

    def step(self, counts, corrupt=None):
        self.scheduler = SimpleNamespace(
            num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()),
            scheduled_spec_decode_tokens={},
            preempted_req_ids=set(),
        )
        indices, _ = self._prepare_inputs(
            self.scheduler, np.asarray(list(counts.values()))
        )
        tokens = []
        positions = []
        for r, n in counts.items():
            s = self.requests[r].num_computed_tokens
            tokens.extend(self.specs[r]["input_ids"][s : s + n])
            positions.extend(range(s, s + n))
        tokens = torch.tensor(tokens, dtype=torch.int32)
        positions = torch.tensor(positions)
        if corrupt == "positions":
            positions[0] += 1
        if corrupt == "tokens":
            tokens[0] += 1
        self.hidden = self.model(input_ids=tokens, positions=positions)
        self.model.compute_logits(self.hidden[indices])
        return self.sample_tokens()


def capture(runner, specs, layers, rounding=True):
    return LaneCapture(runner, specs, layers, attention_rounding=rounding)


def assert_no_hooks_left(runner):
    assert current_collector() is None
    assert not any(
        m._forward_hooks or m._forward_pre_hooks for m in runner.model.modules()
    )
    for name in ("_prepare_inputs", "_get_prompt_logprobs_dict", "sample_tokens"):
        assert name not in vars(runner), name
    assert "compute_logits" not in vars(runner.model)


@pytest.fixture
def specs():
    return {
        "rA": dict(input_ids=[1, 2, 3, 4, 5], start=1, end=5),
        "rB": dict(input_ids=[6, 7, 8, 9], start=0, end=4),
        "neighbor": dict(input_ids=[1, 2, 3, 4, 5], start=1, end=5, retain=False),
    }


def attention_specs():
    ids_long = [i % 30 + 1 for i in range(24)]
    ids_short = [i % 29 + 1 for i in range(15)]
    return {
        "rA": dict(input_ids=ids_long, start=6, end=24),
        "rB": dict(input_ids=ids_short, start=3, end=15),
        "neighbor": dict(input_ids=ids_long, start=6, end=24, retain=False),
    }


def solo_specs():
    return {"solo": dict(input_ids=[i % 30 + 1 for i in range(24)], start=6, end=24)}


def chunked_and_full(specs, full_counts, chunked_steps, layers=(0, 2)):
    baseline = Runner(specs)
    with capture(baseline, specs, list(layers)) as base:
        baseline.step(full_counts)
    runner = Runner(specs)
    with capture(runner, specs, list(layers)) as tap:
        for counts in chunked_steps:
            runner.step(counts)
    return base.finish(), tap, runner


# --- the projections, the logits and the schedule -----------------------------------


def test_interleaved_reordered_one_token_chunks_equal_full_reference(specs):
    expected, tap, runner = chunked_and_full(
        specs, {"rA": 5, "rB": 4, "neighbor": 5},
        [{"rB": 1, "rA": 1, "neighbor": 1},
         {"rA": 2, "neighbor": 2, "rB": 1},
         {"rB": 1, "neighbor": 1, "rA": 1},
         {"neighbor": 1, "rB": 1, "rA": 1}])
    actual = tap.finish()
    assert set(actual) == {"rA", "rB"}
    assert {r: capture_receipt(x) for r, x in actual.items()} == {
        r: capture_receipt(x) for r, x in expected.items()
    }
    assert [x["total_tokens"] for x in tap.schedule] == [3, 5, 3, 3]
    assert [x["role"] for x in tap.schedule[-1]["logit_roles"]] == ["sampling"]
    # Every native input norm and projection was copied before the in-place mutation.
    assert actual["rA"]["queries"][0].max() < 100
    assert actual["rA"]["gates"][0].abs().sum() > 0
    assert_no_hooks_left(runner)


def test_a_capture_holds_exactly_the_substrate(specs):
    runner = Runner(specs)
    with capture(runner, specs, [0, 2]) as tap:
        runner.step({"rA": 5, "rB": 4, "neighbor": 5})
    out = tap.finish()
    for request, fields in out.items():
        assert_substrate_fields(fields, context=request)
        assert set(fields) == SUBSTRATE_FIELDS
    rA = out["rA"]
    assert rA["hidden"].shape == (3, 3, 4)
    assert set(rA["keys"]) == {0, 2}
    assert rA["keys"][0].shape == (3, 1, 2) and rA["queries"][0].shape == (3, 2, 2)
    assert rA["gates"][2].shape == (3, 6)
    assert rA["logits"].shape == (3, 31)
    assert rA["chosen"].tolist() == [3, 4, 5]


def test_one_request_one_token_logit_calls_are_explicit_roles():
    specs = {"same": dict(input_ids=[1, 2, 3], start=0, end=3)}
    runner = Runner(specs)
    with capture(runner, specs, [0]) as tap:
        for _ in range(3):
            runner.step({"same": 1})
    out = tap.finish()["same"]
    assert out["logits"].shape == (2, 31)
    assert [x["role"] for x in tap.schedule[0]["logit_roles"]] == ["sampling", "prompt"]
    assert [x["role"] for x in tap.schedule[-1]["logit_roles"]] == ["sampling"]


@pytest.mark.parametrize("corrupt", ["tokens", "positions"])
def test_wrong_packed_inputs_fail_closed(specs, corrupt):
    runner = Runner(specs)
    with pytest.raises(ValueError, match="actual packed"):
        with capture(runner, specs, [0]) as tap:
            runner.step({"rA": 1}, corrupt)
    with pytest.raises(RuntimeError, match="failed"):
        tap.finish()
    assert_no_hooks_left(runner)


def test_missing_positions_refused(specs):
    runner = Runner(specs)
    with capture(runner, specs, [0]) as tap:
        runner.step({"rA": 2})
    with pytest.raises(RuntimeError, match="missing request"):
        tap.finish()


def test_duplicate_chunk_refused(specs):
    runner = Runner(specs)
    with pytest.raises(ValueError, match="missing/duplicate"):
        with capture(runner, specs, [0]):
            runner.step({"rA": 1})
            runner.requests["rA"].num_computed_tokens = 0
            runner.step({"rA": 1})


def test_stale_step_refused(specs):
    runner = Runner(specs)
    sched = SimpleNamespace(
        num_scheduled_tokens={"rA": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        preempted_req_ids=set(),
    )
    with pytest.raises(RuntimeError, match="stale scheduler"):
        with capture(runner, specs, [0]):
            runner._prepare_inputs(sched, np.array([1]))
            runner._prepare_inputs(sched, np.array([1]))


def test_speculation_or_preemption_refused(specs):
    runner = Runner(specs)
    sched = SimpleNamespace(
        num_scheduled_tokens={"rA": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        preempted_req_ids={"rB"},
    )
    with pytest.raises(ValueError, match="speculation or preemption"):
        with capture(runner, specs, [0]):
            runner._prepare_inputs(sched, np.array([1]))


def test_missing_prompt_projection_and_restoration(specs):
    runner = Runner(specs)
    runner._get_prompt_logprobs_dict = lambda *a: {}
    prior = runner._get_prompt_logprobs_dict
    with pytest.raises(RuntimeError, match="missing prompt"):
        with capture(runner, specs, [0]):
            runner.step({"rA": 1})
    assert runner._get_prompt_logprobs_dict is prior


def test_wrong_prompt_view_is_rejected(specs):
    runner = Runner(specs)
    runner._get_prompt_logprobs_dict = lambda h, c: runner.model.compute_logits(
        h.clone()
    )
    with pytest.raises(ValueError, match="input view differs"):
        with capture(runner, specs, [0]):
            runner.step({"rA": 1})


def test_missing_native_hook_refused(specs):
    runner = Runner(specs)
    with pytest.raises(RuntimeError, match="incomplete native hooks"):
        with capture(runner, specs, [0]):
            runner.model.model.layers[0].mlp.gate_up_proj.forward = lambda x: (
                x.repeat(1, 3),
                None,
            )
            runner.model.model.layers[0].mlp.gate_up_proj._forward_hooks.clear()
            runner.step({"rA": 1})


def test_duplicate_native_hook_refused(specs):
    runner = Runner(specs)
    block = runner.model.model.layers[0]
    original = block.forward

    def duplicated(*args, **kwargs):
        result = original(*args, **kwargs)
        block.self_attn.qkv_proj(torch.zeros(len(args[0]), 4))
        return result

    block.forward = duplicated
    with pytest.raises(RuntimeError, match="duplicate hook queries"):
        with capture(runner, specs, [0]):
            runner.step({"rA": 1})


def test_unmapped_neighbor_refused(specs):
    runner = Runner(specs)
    targets = {"rA": specs["rA"]}
    with pytest.raises(ValueError, match="unmapped"):
        with capture(runner, targets, [0]):
            runner.step({"rA": 1, "neighbor": 1})


@pytest.mark.parametrize("layers", [[], [0, 0], [3]])
def test_invalid_sampled_layers_refused(specs, layers):
    with pytest.raises(ValueError, match="invalid sampled layers"):
        capture(Runner(specs), specs, layers)


def test_a_capture_context_is_not_reused(specs):
    runner = Runner(specs)
    tap = capture(runner, specs, [0])
    with tap:
        runner.step({"rA": 5, "rB": 4, "neighbor": 5})
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with tap:
            pass  # pragma: no cover - the enter itself refuses
    tap.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        tap.finish()


def test_native_fragment_memory_accounting(specs):
    runner = Runner(specs)
    with capture(runner, specs, [0]) as tap:
        runner.step({"rA": 5, "rB": 4, "neighbor": 5})
    before = tap.retained_bytes
    out = tap.finish()

    def size(x):
        return (
            sum(size(v) for v in x.values())
            if isinstance(x, dict)
            else x.numel() * x.element_size()
        )

    expected = size(out) - sum(
        x["chosen"].numel() * x["chosen"].element_size() for x in out.values()
    )
    assert before == tap.peak_fragment_bytes == expected
    assert tap.retained_bytes == 0


# --- the attention statistics and products -----------------------------------------


def test_attention_capture_chunked_equals_full():
    specs = attention_specs()
    expected, tap, runner = chunked_and_full(
        specs, {"rA": 24, "rB": 15, "neighbor": 24},
        [{"rA": 5, "rB": 7, "neighbor": 24},
         {"rA": 12, "rB": 1},
         {"rB": 7, "rA": 7}])
    actual = tap.finish()
    for r in ("rA", "rB"):
        for key in ATTENTION_KEYS:
            assert set(actual[r][key]) == set(expected[r][key]), (r, key)
            for layer, value in expected[r][key].items():
                assert torch.equal(actual[r][key][layer], value), (r, key, layer)
        steps = specs[r]["end"] - 1 - specs[r]["start"]
        assert set(actual[r]["attn_span_rows"]) == {0, 2}
        for value in actual[r]["attn_span_rows"].values():
            assert value.shape == (steps, specs[r]["end"] - 1)
    for step in tap.schedule:
        assert set(step["row_sum_worst"]) and all(
            v == 0.0 for v in step["row_sum_worst"].values())
    assert_no_hooks_left(runner)


def test_attention_capture_values_and_structure():
    specs = solo_specs()
    runner = Runner(specs)
    with capture(runner, specs, [0], rounding=False) as tap:
        assert current_collector() is tap.stats_collector
        runner.step({"solo": 24})
    assert current_collector() is None
    out = tap.finish()["solo"]
    steps = 24 - 1 - 6
    positions = torch.arange(6, 23)
    for layer in range(3):
        assert out["attn_stats"][layer].shape == (steps, HEADS, NUM_STATS)
        assert torch.equal(out["attn_stats"][layer], fill_stats(positions))
    # The products equal the reductions of the deterministic scratch directly.
    lengths = torch.arange(7, 24, dtype=torch.int32)
    scratch = fake_scratch(lengths)
    assert set(out["attn_coverage"]) == {0}
    assert torch.equal(out["attn_coverage"][0],
                       coverage_of_selected_rows(scratch, lengths))
    pair = agreement_pair(scratch)  # agreement selects every span row here
    for layer in range(3):
        assert torch.equal(out["attn_h_mean"][layer], pair[0])
        assert torch.equal(out["attn_h_heads"][layer], pair[1])
    mean = scratch.mean(dim=1)
    assert torch.equal(out["attn_spectral_rows"][0], mean)
    assert torch.equal(out["attn_decay_rows"][0],
                       mean[list(range(steps // 4, steps, 1))[:10]])
    assert set(out["attn_span_rows"]) == {0}
    assert torch.equal(out["attn_span_rows"][0], mean)


def test_attention_family_values():
    specs = solo_specs()
    runner = Runner(specs)
    with capture(runner, specs, [0], rounding=False) as tap:
        runner.step({"solo": 24})
    out = tap.finish()["solo"]
    schema = request_row_schema(prompt_length=6, end=24)
    lengths = torch.arange(7, 24, dtype=torch.int32)
    scratch = fake_scratch(lengths)
    widths = torch.full((len(lengths),), 24, dtype=torch.int32)
    assert set(out["attn_entropy_rows"]) == {0, 1, 2}
    for layer in range(3):
        assert torch.equal(
            out["attn_entropy_rows"][layer],
            entropy_of_rows(scratch[list(schema.entropy)],
                            widths=widths[list(schema.entropy)]))
    summaries = per_head_summaries(
        scratch, lengths=lengths,
        prefixes=torch.full((len(lengths),), 6, dtype=torch.int32),
        widths=widths)
    for key in ("head_ent", "head_sink", "head_prompt", "head_recency"):
        assert set(out[f"attn_{key}"]) == {0}
        assert torch.equal(out[f"attn_{key}"][0], summaries[key]), key


def test_attention_failure_aborts_and_uninstalls():
    specs = attention_specs()
    runner = Runner(specs)
    with pytest.raises(ValueError, match="actual packed"):
        with capture(runner, specs, [0]) as tap:
            runner.step({"rA": 1}, "tokens")
    assert current_collector() is None
    assert not tap.stats_collector.armed
    with pytest.raises(RuntimeError, match="failed"):
        tap.finish()


def test_attention_arming_validation():
    specs = attention_specs()
    with pytest.raises(ValueError, match="rounding switch"):
        LaneCapture(Runner(specs), specs, [0], attention_rounding=1)
    runner = Runner(specs)
    foreign = StatsCollector()
    install_collector(foreign)
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            with capture(runner, specs, [0]):
                pass  # pragma: no cover - the enter itself refuses
    finally:
        assert uninstall_collector() is foreign
    assert current_collector() is None
    assert "_prepare_inputs" not in vars(runner)


def test_attention_layers_require_unique_names(specs):
    runner = Runner(specs)
    runner.model.model.layers[1].self_attn.attn.layer_name = (
        runner.model.model.layers[0].self_attn.attn.layer_name)
    with pytest.raises(ValueError, match="unique names"):
        capture(runner, specs, [0])


def test_a_missing_attention_layer_fails_the_step(specs, monkeypatch):
    """Every attention layer must draw its statistics every step; a layer that
    does not is refused when the step closes, not reassembled around."""
    runner = Runner(specs)
    silent = runner.model.model.layers[1].self_attn.attn.layer_name
    real = emulate_backend_layer

    def skipping(name, positions):
        if name != silent:
            real(name, positions)

    monkeypatch.setitem(globals(), "emulate_backend_layer", skipping)
    with pytest.raises(RuntimeError, match="layers missing"):
        with capture(runner, specs, [0]):
            runner.step({"rA": 5, "rB": 4, "neighbor": 5})
    assert backend.current_collector() is None
