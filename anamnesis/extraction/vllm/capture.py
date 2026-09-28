"""Capture the substrate of every scheduled request from inside a vLLM engine.

:class:`LaneCapture` wraps one engine's model runner for the duration of one
group of requests. It verifies every scheduler step against the requests it was
given (the packed token ids, positions, offsets and chunk boundaries), hooks
the model's hidden states, the sampled layers' queries, keys, values and gate
pre-activations, and the prompt logits, and drives the instrumented attention
backend: it installs the statistics collector, arms each verified step with
the per-token region metadata and the second-pass row plan, closes the step
after sampling, and reassembles per-request statistics and selected-row
products in :meth:`LaneCapture.finish`.

Request ids are the engine's internal ids. The caller queues one group before
stepping the engine and releases captures between groups. GPU fragments are
cloned before the engine can reuse their memory and move to the CPU only at
finish. The module does not import vLLM itself (the backend imports it when it is
installed and tolerates its absence), and touches the engine only through the
runner object it is handed, so it is tested against a stand-in runner.

The runner methods it wraps (``_prepare_inputs``, ``_get_prompt_logprobs_dict``,
``sample_tokens``, and the model's ``compute_logits``) are private to the
engine release :data:`anamnesis.extraction.vllm.envelope.PINNED_PACKAGES`
names; another release is refused before an engine is built.
"""

from __future__ import annotations

from collections import deque
from types import MethodType

import numpy as np
import torch

from anamnesis.extraction.vllm import backend
from anamnesis.extraction.vllm.rows import request_row_schema, step_row_plan


class LaneCapture:
    """Hooks one engine's model runner and collects each request's substrate.

    ``requests`` maps an engine request id to ``input_ids``, ``start`` (the
    prompt length), ``end`` (the full length) and ``retain`` (False for a
    filler that keeps a batch full and is never read back). ``sampled_layers``
    are the layer indices whose queries, keys, values, gates and span
    attention are kept. ``attention_rounding`` is the second pass's rounding
    switch. Use as a context manager, step the engine, then call
    :meth:`finish`.
    """

    def __init__(self, runner, requests, sampled_layers, *, attention_rounding: bool):
        self.runner, self.model = runner, runner.model
        self.blocks = tuple(self.model.model.layers)
        self.layers = tuple(sampled_layers)
        if (
            not self.blocks
            or not self.layers
            or len(set(self.layers)) != len(self.layers)
            or not set(self.layers).issubset(range(len(self.blocks)))
        ):
            raise ValueError("invalid sampled layers")
        self.specs = {}
        vocab = int(self.model.config.vocab_size)
        for req_id, spec in requests.items():
            ids = tuple(spec["input_ids"])
            start, end = spec["start"], spec["end"]
            if (
                not isinstance(req_id, str)
                or not req_id
                or not 0 <= start < end - 1
                or end != len(ids)
            ):
                raise ValueError("invalid request identity/span")
            if any(type(x) is not int or not 0 <= x < vocab for x in ids):
                raise ValueError("invalid banked IDs")
            self.specs[req_id] = dict(
                input_ids=ids, start=start, end=end, retain=spec.get("retain", True)
            )
        if not self.specs:
            raise ValueError("requests required")
        self.next_position = {r: 0 for r in self.specs}
        self.fragments = {r: {} for r, s in self.specs.items() if s["retain"]}
        self.schedule = []
        self.retained_bytes = self.peak_fragment_bytes = 0
        self.step = None
        self.prompt_queue = None
        self.handles, self.restorations = [], []
        self.entered = self.failed = self.finished = False
        self.expected_keys = {f"hidden/{i}" for i in range(len(self.blocks))} | {
            f"{k}/{i}"
            for i in self.layers
            for k in ("keys", "values", "queries", "gates")
        }
        if type(attention_rounding) is not bool:
            raise ValueError("the attention rounding switch must be bool")
        self.attention_rounding = attention_rounding
        self.layer_names = tuple(
            block.self_attn.attn.layer_name for block in self.blocks
        )
        if len(set(self.layer_names)) != len(self.layer_names) or not all(
            isinstance(n, str) and n for n in self.layer_names
        ):
            raise ValueError("attention layers require unique names")
        self.sampled_layer_names = frozenset(
            self.layer_names[i] for i in self.layers
        )
        self.schemas = {
            r: request_row_schema(prompt_length=s["start"], end=s["end"])
            for r, s in self.specs.items()
        }
        self.stats_collector = backend.StatsCollector(
            expected_layers=self.layer_names, retain_steps=False
        )
        self.selected = {r: {} for r in self.fragments}
        self.attention_keys = {
            f"attn_stats/{i}" for i in range(len(self.blocks))
        } | {f"attn_coverage/{i}" for i in self.layers}
        self.installed_collector = False

    def _guard(self, fn):
        def guarded(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except BaseException:
                self.failed = True
                raise

        return guarded

    def _patch(self, obj, name, factory):
        owned = name in vars(obj)
        prior = vars(obj).get(name)
        original = getattr(obj, name)
        wrapped = self._guard(factory(original))
        setattr(obj, name, MethodType(lambda _self, *a, **kw: wrapped(*a, **kw), obj))
        self.restorations.append((obj, name, owned, prior))

    def _once(self, key):
        if self.step is None:
            raise RuntimeError("stale or absent scheduler context")
        if key in self.step["seen"]:
            raise RuntimeError(f"duplicate hook {key}")
        self.step["seen"].add(key)

    def _prepare_wrapper(self, original):
        def prepare(scheduler_output, num_scheduled_tokens):
            if self.step is not None or self.prompt_queue is not None:
                raise RuntimeError("stale scheduler context before next prepare")
            result = original(scheduler_output, num_scheduled_tokens)
            logits_indices, speculative = result
            if (
                speculative is not None
                or scheduler_output.scheduled_spec_decode_tokens
                or scheduler_output.preempted_req_ids
            ):
                raise ValueError("speculation or preemption is outside the lane's envelope")
            batch = self.runner.input_batch
            ids = tuple(batch.req_ids)
            counts = np.asarray(num_scheduled_tokens).copy()
            offsets = self.runner.query_start_loc.np[: len(ids) + 1].copy()
            starts = batch.num_computed_tokens_cpu[: len(ids)].copy()
            if (
                not ids
                or len(set(ids)) != len(ids)
                or len(counts) != len(ids)
                or (counts <= 0).any()
            ):
                raise ValueError("invalid scheduled batch")
            if (
                offsets[0] != 0
                or not np.array_equal(np.diff(offsets), counts)
                or int(offsets[-1]) != scheduler_output.total_num_scheduled_tokens
            ):
                raise ValueError("scheduled counts/offsets disagree")
            if set(scheduler_output.num_scheduled_tokens) != set(ids):
                raise ValueError("scheduler and runner request sets disagree")
            expected_logits = torch.tensor(offsets[1:] - 1, dtype=torch.int64)
            if not torch.equal(logits_indices.detach().cpu(), expected_logits):
                raise ValueError("unexpected sampling logits indices")
            items = []
            for i, r in enumerate(ids):
                if r not in self.specs or batch.req_id_to_index[r] != i:
                    raise ValueError("unmapped or misindexed request")
                s, n = int(starts[i]), int(counts[i])
                spec = self.specs[r]
                state = self.runner.requests[r]
                if (
                    state.num_computed_tokens != s
                    or tuple(state.prompt_token_ids) != spec["input_ids"]
                ):
                    raise ValueError("runner request metadata differs from bank")
                if (
                    s != self.next_position[r]
                    or s + n > spec["end"]
                    or scheduler_output.num_scheduled_tokens[r] != n
                ):
                    raise ValueError("missing/duplicate/out-of-range request chunk")
                items.append(
                    dict(request_id=r, start=s, count=n, offset=int(offsets[i]))
                )
            self.step = dict(
                items=items,
                total=int(offsets[-1]),
                seen=set(),
                logit_roles=[],
                prompt_calls=0,
            )
            self._arm_attention(items)
            return result

        return prepare

    def _arm_attention(self, items):
        device = self.runner.device
        plan = step_row_plan(items, self.schemas, selected=frozenset(self.fragments))

        def moved(array):
            return torch.from_numpy(array).to(device)

        second = backend.StepSecondPass(
            round_to_model_dtype=self.attention_rounding,
            sampled_layers=self.sampled_layer_names,
            span_slots=moved(plan.span_slots),
            span_lengths=moved(plan.span_lengths),
            span_agreement=moved(plan.span_agreement),
            span_spectral=moved(plan.span_spectral),
            span_decay=moved(plan.span_decay),
            span_entropy=moved(plan.span_entropy),
            span_prefixes=moved(plan.span_prefixes),
            span_widths=moved(plan.span_widths),
            series_slots=moved(plan.series_slots),
            series_lengths=moved(plan.series_lengths),
            series_agreement=moved(plan.series_agreement),
            series_entropy=moved(plan.series_entropy),
            series_widths=moved(plan.series_widths),
        )
        self.stats_collector.arm_step(
            prefix_lengths=moved(plan.prefix_lengths),
            recency_cutoffs=moved(plan.recency_cutoffs),
            second_pass=second,
        )
        self.step["row_plan"] = plan

    def _input(self, module, args, kwargs):
        self._once("input")
        tokens = kwargs.get("input_ids", args[0] if args else None)
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        if kwargs.get("inputs_embeds") is not None or (
            len(args) > 3 and args[3] is not None
        ):
            raise ValueError("embedding substitution outside envelope")
        expected_ids = []
        expected_pos = []
        for item in self.step["items"]:
            r, s, n = item["request_id"], item["start"], item["count"]
            expected_ids.extend(self.specs[r]["input_ids"][s : s + n])
            expected_pos.extend(range(s, s + n))
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.dtype not in (torch.int32, torch.int64)
            or not torch.equal(tokens.detach().cpu(), torch.tensor(expected_ids))
        ):
            raise ValueError("actual packed token IDs disagree with scheduler snapshot")
        if (
            not isinstance(pos, torch.Tensor)
            or pos.dtype != torch.int64
            or not torch.equal(pos.detach().cpu(), torch.tensor(expected_pos))
        ):
            raise ValueError("actual packed positions disagree with scheduler snapshot")
        self.step["positions_signature"] = (pos.data_ptr(), pos.device, pos.stride())

    def _decoder(self, layer):
        def observe(module, args, kwargs):
            self._once(f"decoder/{layer}")
            positions = kwargs.get("positions", args[0] if args else None)
            expected = [
                p
                for x in self.step["items"]
                for p in range(x["start"], x["start"] + x["count"])
            ]
            # Same positions buffer is shared by native layers; checking identity
            # avoids per-layer GPU synchronization after model input verification.
            if (
                not isinstance(positions, torch.Tensor)
                or positions.shape != (len(expected),)
                or positions.dtype != torch.int64
                or (positions.data_ptr(), positions.device, positions.stride())
                != self.step.get("positions_signature")
            ):
                raise ValueError("decoder positions shape/dtype mismatch")

        return observe

    def _store(self, request_id, key, absolute_start, tensor):
        spec = self.specs[request_id]
        if not spec["retain"]:
            return
        lo = max(absolute_start, spec["start"])
        hi = min(absolute_start + len(tensor), spec["end"] - 1)
        if lo < hi:
            fragment = (
                tensor[lo - absolute_start : hi - absolute_start]
                .detach()
                .clone()
                .contiguous()
            )
            self.fragments[request_id].setdefault(key, []).append((lo, fragment))
            self.retained_bytes += fragment.numel() * fragment.element_size()
            self.peak_fragment_bytes = max(
                self.peak_fragment_bytes, self.retained_bytes
            )

    def _packed(self, key, tensor, width, head_dim=None):
        self._once(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != (self.step["total"], width)
            or not tensor.is_floating_point()
        ):
            raise ValueError(f"packed observation shape mismatch {key}")
        for item in self.step["items"]:
            x = tensor[item["offset"] : item["offset"] + item["count"]]
            if head_dim is not None:
                x = x.reshape(len(x), -1, head_dim)
            self._store(item["request_id"], key, item["start"], x)

    def _norm(self, index, final=False):
        def observe(module, args, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("native norm tuple required")
            self._packed(
                f"hidden/{index}",
                output[0 if final else 1],
                int(self.model.config.hidden_size),
            )

        return observe

    def _qkv(self, layer):
        def observe(module, args, output):
            attn = self.blocks[layer].self_attn
            q, k, d = int(attn.q_size), int(attn.kv_size), int(attn.head_dim)
            if q != self.model.config.hidden_size or q % d or k % d or q % k:
                raise ValueError("the packed QKV projection does not match a single-GPU Llama")
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or output[0].shape != (self.step["total"], q + 2 * k)
            ):
                raise ValueError("native packed QKV shape mismatch")
            for kind, value in zip(
                ("queries", "keys", "values"),
                output[0].split([q, k, k], dim=-1),
                strict=True,
            ):
                self._packed(f"{kind}/{layer}", value, value.shape[-1], d)

        return observe

    def _gate(self, layer):
        def observe(module, args, output):
            width = int(self.model.config.intermediate_size)
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or output[0].shape != (self.step["total"], 2 * width)
            ):
                raise ValueError("native packed gate shape mismatch")
            self._packed(f"gates/{layer}", output[0][:, :width], width)

        return observe

    def _prompt_wrapper(self, original):
        def prompt(hidden_states, num_scheduled_tokens):
            self._once("prompt_method")
            if self.prompt_queue is not None:
                raise RuntimeError("nested prompt-logit context")
            items = {x["request_id"]: x for x in self.step["items"]}
            if num_scheduled_tokens != {r: x["count"] for r, x in items.items()}:
                raise ValueError("prompt scheduled counts differ from snapshot")
            queue = deque()
            for r in self.runner.num_prompt_logprobs:
                if r not in items:
                    continue
                item = items[r]
                s = item["start"]
                L = self.specs[r]["end"]
                if self.runner.requests[r].num_computed_tokens != s:
                    raise ValueError("prompt start changed since prepare")
                n = min(item["count"], L - s - 1)
                if n > 0:
                    expected = hidden_states[item["offset"] : item["offset"] + n]
                    queue.append((r, s, expected))
            if set(items) - set(self.runner.num_prompt_logprobs):
                raise ValueError("scheduled request lacks prompt-logprob registration")
            self.prompt_queue = queue
            try:
                result = original(hidden_states, num_scheduled_tokens)
                if queue:
                    raise RuntimeError("missing prompt-logit projection calls")
                return result
            finally:
                self.prompt_queue = None

        return prompt

    def _compute_wrapper(self, original):
        def compute(hidden_states):
            if self.step is None:
                raise RuntimeError("logits outside scheduler context")
            if self.prompt_queue is None:
                self._once("sampling_logits")
                expected_rows = len(self.step["items"])
                role = dict(
                    role="sampling",
                    positions=[
                        dict(
                            request_id=x["request_id"],
                            position=x["start"] + x["count"] - 1,
                        )
                        for x in self.step["items"]
                    ],
                )
                request_id = None
            else:
                if not self.prompt_queue:
                    raise RuntimeError("extra prompt-logit projection")
                request_id, start, expected = self.prompt_queue.popleft()
                if (
                    hidden_states.shape != expected.shape
                    or hidden_states.dtype != expected.dtype
                    or hidden_states.device != expected.device
                    or hidden_states.data_ptr() != expected.data_ptr()
                    or hidden_states.stride() != expected.stride()
                ):
                    raise ValueError(
                        "prompt-logit input view differs from native mapped slice"
                    )
                expected_rows = len(expected)
                role = dict(
                    role="prompt",
                    request_id=request_id,
                    start=start,
                    count=expected_rows,
                )
            if hidden_states.shape != (expected_rows, self.model.config.hidden_size):
                raise ValueError("logit hidden shape mismatch")
            result = original(hidden_states)
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != (expected_rows, self.model.config.vocab_size)
                or not result.is_floating_point()
            ):
                raise ValueError("native logits shape/vocabulary mismatch")
            if request_id is not None:
                self._store(request_id, "logits", start, result)
            self.step["logit_roles"].append(role)
            return result

        return compute

    def _sample_wrapper(self, original):
        def sample(*args, **kwargs):
            if self.step is None:
                raise RuntimeError("sample_tokens outside active capture step")
            result = original(*args, **kwargs)
            required = (
                self.expected_keys
                | {"input", "sampling_logits", "prompt_method"}
                | {f"decoder/{i}" for i in range(len(self.blocks))}
            )
            if self.step["seen"] != required or self.prompt_queue is not None:
                raise RuntimeError(
                    f"incomplete native hooks: {sorted(required - self.step['seen'])}"
                )
            entry = dict(
                step=len(self.schedule),
                total_tokens=self.step["total"],
                requests=self.step["items"],
                logit_roles=self.step["logit_roles"],
            )
            entry["row_sum_worst"] = self._close_attention()
            for item in self.step["items"]:
                self.next_position[item["request_id"]] = item["start"] + item["count"]
            self.schedule.append(entry)
            self.step = None
            return result

        return sample

    def _select_runs(self, key, members, values, *, pad_rows=False):
        """Store slot-ordered selected values as per-request row runs."""
        i = 0
        while i < len(members):
            request_id = members[i][0]
            j = i
            while j < len(members) and members[j][0] == request_id:
                j += 1
            rows = tuple(row for _, row in members[i:j])
            value = values[i:j].detach().clone()
            if pad_rows:
                width = self.specs[request_id]["end"] - 1
                if value.shape[-1] >= width:
                    value = value[..., :width]
                else:
                    value = torch.cat(
                        [value, value.new_zeros(len(value),
                                                width - value.shape[-1])],
                        dim=-1,
                    )
            value = value.contiguous()
            self.selected[request_id].setdefault(key, []).append((rows, value))
            self.retained_bytes += value.numel() * value.element_size()
            self.peak_fragment_bytes = max(
                self.peak_fragment_bytes, self.retained_bytes
            )
            i = j

    def _close_attention(self):
        record = self.stats_collector.close_step()
        plan = self.step["row_plan"]
        evidence = {}
        for index, name in enumerate(self.layer_names):
            stats = record.stats[name]
            for item in self.step["items"]:
                self._store(
                    item["request_id"],
                    f"attn_stats/{index}",
                    item["start"],
                    stats[item["offset"] : item["offset"] + item["count"]],
                )
            products = record.second.get(name)
            if products is None:
                continue
            evidence[name] = float(products["row_sum_worst"])
            entropy_members = tuple(
                (plan.span_members if name in self.sampled_layer_names
                 else plan.series_members)[s]
                for s in (plan.span_entropy
                          if name in self.sampled_layer_names
                          else plan.series_entropy)
            )
            self._select_runs(f"attn_entropy_rows/{index}",
                              entropy_members,
                              products["entropy_rows"])
            if name not in self.sampled_layer_names:
                self._select_runs(f"attn_h_mean/{index}",
                                  plan.agreement_members,
                                  products["h_mean"])
                self._select_runs(f"attn_h_heads/{index}",
                                  plan.agreement_members,
                                  products["h_heads"])
                continue
            cursor = 0
            for item in self.step["items"]:
                request_id = item["request_id"]
                if request_id not in self.fragments:
                    continue
                schema = self.schemas[request_id]
                lo = max(item["start"], schema.prompt_length)
                hi = min(item["start"] + item["count"], schema.end - 1)
                if lo < hi:
                    self._store(
                        request_id,
                        f"attn_coverage/{index}",
                        lo,
                        products["coverage"][cursor : cursor + hi - lo],
                    )
                    cursor += hi - lo
            if cursor != len(plan.span_members):
                raise RuntimeError("span slots and coverage rows disagree")
            self._select_runs(f"attn_h_mean/{index}", plan.agreement_members,
                              products["h_mean"])
            self._select_runs(f"attn_h_heads/{index}", plan.agreement_members,
                              products["h_heads"])
            spectral_members = tuple(
                plan.span_members[s] for s in plan.span_spectral
            )
            self._select_runs(f"attn_spectral_rows/{index}", spectral_members,
                              products["spectral_rows"], pad_rows=True)
            decay_members = tuple(
                plan.span_members[s] for s in plan.span_decay
            )
            self._select_runs(f"attn_decay_rows/{index}", decay_members,
                              products["decay_rows"], pad_rows=True)
            for key in ("head_ent", "head_sink", "head_prompt", "head_recency"):
                self._select_runs(f"attn_{key}/{index}",
                                  plan.span_members, products[key])
            self._select_runs(f"attn_span_rows/{index}",
                              plan.span_members,
                              products["span_rows"], pad_rows=True)
        return evidence

    def __enter__(self):
        if self.entered:
            raise RuntimeError("capture context cannot be reused")
        self.entered = True
        try:
            self._patch(self.runner, "_prepare_inputs", self._prepare_wrapper)
            self._patch(self.runner, "_get_prompt_logprobs_dict", self._prompt_wrapper)
            self._patch(self.model, "compute_logits", self._compute_wrapper)
            self._patch(self.runner, "sample_tokens", self._sample_wrapper)
            self.handles.append(
                self.model.register_forward_pre_hook(
                    self._guard(self._input), with_kwargs=True
                )
            )
            for layer, block in enumerate(self.blocks):
                self.handles.append(
                    block.register_forward_pre_hook(
                        self._guard(self._decoder(layer)), with_kwargs=True
                    )
                )
                if layer:
                    self.handles.append(
                        block.input_layernorm.register_forward_hook(
                            self._guard(self._norm(layer - 1))
                        )
                    )
            self.handles.append(
                self.model.model.norm.register_forward_hook(
                    self._guard(self._norm(len(self.blocks) - 1, True))
                )
            )
            backend.install_collector(self.stats_collector)
            self.installed_collector = True
            for layer in self.layers:
                self.handles.append(
                    self.blocks[layer].self_attn.qkv_proj.register_forward_hook(
                        self._guard(self._qkv(layer))
                    )
                )
                self.handles.append(
                    self.blocks[layer].mlp.gate_up_proj.register_forward_hook(
                        self._guard(self._gate(layer))
                    )
                )
        except BaseException:
            self.failed = True
            self.__exit__(None, None, None)
            raise
        return self

    def finish(self):
        if (
            not self.entered
            or self.failed
            or self.finished
            or self.step is not None
            or self.prompt_queue is not None
        ):
            raise RuntimeError("capture incomplete, failed, or already finished")
        if any(self.next_position[r] != s["end"] for r, s in self.specs.items()):
            raise RuntimeError("missing request positions")
        outputs = {}
        for r, fragments in self.fragments.items():
            spec = self.specs[r]
            flat = {}
            if set(fragments) != self.expected_keys | {"logits"} | (
                self.attention_keys
            ):
                raise RuntimeError("missing substrate fields")
            for key, pieces in fragments.items():
                cursor = spec["start"]
                dtype = None
                shape = None
                for pos, x in pieces:
                    if pos != cursor or (
                        dtype is not None and (dtype != x.dtype or shape != x.shape[1:])
                    ):
                        raise ValueError("duplicate/missing/mixed substrate fragments")
                    cursor += len(x)
                    dtype = x.dtype
                    shape = x.shape[1:]
                if cursor != spec["end"] - 1:
                    raise ValueError("incomplete substrate positions")
                flat[key] = torch.cat([x for _, x in pieces], dim=0).to(
                    "cpu", copy=True
                )
            hidden = [flat[f"hidden/{i}"] for i in range(len(self.blocks))]
            if len({x.dtype for x in hidden}) != 1:
                raise ValueError("hidden native dtype mismatch")
            outputs[r] = {
                kind: {i: flat[f"{kind}/{i}"] for i in self.layers}
                for kind in ("keys", "values", "queries", "gates")
            }
            outputs[r].update(
                hidden=torch.stack(hidden),
                logits=flat["logits"],
                chosen=torch.tensor(
                    spec["input_ids"][spec["start"] + 1 : spec["end"]],
                    dtype=torch.int64,
                ),
            )
            outputs[r].update(self._finish_attention(r, flat))
        self.finished = True
        self.fragments.clear()
        self.retained_bytes = 0
        return outputs

    def _finish_attention(self, r, flat):
        schema = self.schemas[r]
        selected = self.selected[r]
        width = self.specs[r]["end"] - 1

        def assemble(key, expected_rows, row_width=None):
            pieces = selected.get(key, [])
            rows = [row for run, _ in pieces for row in run]
            if rows != list(expected_rows):
                raise ValueError(f"selected rows incomplete for {key}")
            if pieces:
                return torch.cat([v for _, v in pieces], dim=0).to(
                    "cpu", copy=True
                )
            if row_width is not None:
                return torch.zeros((0, row_width), dtype=torch.float32)
            return torch.zeros((0,), dtype=torch.float64)

        result = dict(
            attn_stats={
                i: flat[f"attn_stats/{i}"] for i in range(len(self.blocks))
            },
            attn_coverage={i: flat[f"attn_coverage/{i}"] for i in self.layers},
            attn_h_mean={
                i: assemble(f"attn_h_mean/{i}", schema.agreement)
                for i in range(len(self.blocks))
            },
            attn_h_heads={
                i: assemble(f"attn_h_heads/{i}", schema.agreement)
                for i in range(len(self.blocks))
            },
            attn_spectral_rows={
                i: assemble(f"attn_spectral_rows/{i}", schema.spectral, width)
                for i in self.layers
            },
            attn_decay_rows={
                i: assemble(f"attn_decay_rows/{i}", schema.decay, width)
                for i in self.layers
            },
        )
        result["attn_span_rows"] = {
            i: assemble(f"attn_span_rows/{i}", range(schema.steps), width)
            for i in self.layers
        }
        result["attn_entropy_rows"] = {
            i: assemble(f"attn_entropy_rows/{i}", schema.entropy)
            for i in range(len(self.blocks))
        }
        for key in ("head_ent", "head_sink", "head_prompt", "head_recency"):
            result[f"attn_{key}"] = {
                i: assemble(f"attn_{key}/{i}", range(schema.steps))
                for i in self.layers
            }
        selected.clear()
        return result

    def __exit__(self, exc_type, exc, traceback):
        self.failed |= exc_type is not None
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for obj, name, owned, prior in reversed(self.restorations):
            if owned:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)
        self.restorations.clear()
        if self.stats_collector.armed:
            self.stats_collector.abort_step()
        if self.installed_collector:
            if backend.current_collector() is self.stats_collector:
                backend.uninstall_collector()
            self.installed_collector = False
