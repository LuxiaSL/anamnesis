"""The lane entry point: one eager forward, reduced on the device it ran on.

:mod:`anamnesis.extraction.state_extractor` defines what every feature means.
This lane emits that same named vector without materialising the raw tensors on
the host: attention weights are reduced inside each layer's forward hook, so the
all-layer attention tensor never exists at once, and only small per-layer
summaries survive the pass.

Scope is the dense Llama probe-free battery, in full. The lane refuses a partial
battery, a non-eager attention implementation, a model split across devices, and
any arithmetic setting that changed after construction — a feature vector is
only comparable to another vector computed the same way, so the way is pinned
into `lane_id` and checked before every span.

A receipt is provenance, not a certification. Whether this box's arithmetic
agrees with the anchor is the business of
:mod:`anamnesis.extraction.equivalence`, and it is a property of a box rather
than of the code.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import transformers
from torch import Tensor

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.fast.batch_layout import ReplaySpan, pack_spans
from anamnesis.extraction.fast.attention import AttentionReducer
from anamnesis.extraction.fast.families import FamilyReducer
from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    rowcos,
    std,
    trajectory_indices,
)
from anamnesis.extraction.model_loader import LoadedModel, decoder_layers


@dataclass(frozen=True)
class GpuFeatureResult:
    features: np.ndarray
    feature_names: tuple[str, ...]
    mean_logprob: float
    knnlm_baseline: np.ndarray | None
    metadata: dict


class GpuFeatureLane:
    def __init__(
        self,
        extraction: ExtractionConfig,
        families: FeaturePipelineConfig,
        feature_names: list[str],
        positional_means: np.ndarray,
        pca_components: np.ndarray | dict[int, np.ndarray],
        pca_mean: np.ndarray | dict[int, np.ndarray],
        *,
        device: str | torch.device,
        calibration_sha256: str,
        replay_path: Literal["cached", "full"] = "cached",
    ):
        if replay_path not in ("cached", "full"):
            raise ValueError("replay_path must be cached or full")
        self.replay_path = replay_path
        required = (
            "include_core_blocks",
            "enable_residual_trajectory",
            "enable_attention_flow",
            "enable_gate_features",
            "enable_per_head",
            "enable_value_geometry",
            "enable_qk_geometry",
            "enable_kv_cka",
        )
        if not all(getattr(families, name) for name in required):
            raise ValueError("GPU lane requires the complete declared battery")
        if any(
            getattr(families, name)
            for name in (
                "enable_temporal_dynamics",
                "enable_contrastive_projection",
                "enable_path_signature",
            )
        ):
            raise ValueError("GPU lane does not support additional feature families")
        if not all(
            (
                extraction.enable_norms_and_output_stats,
                extraction.enable_attention_and_deltas,
                extraction.enable_cache_and_keys,
                extraction.enable_residual_pca,
            )
        ):
            raise ValueError("GPU lane cannot drop any of the four core blocks")
        if len(calibration_sha256) != 64:
            raise ValueError("calibration SHA256 is mandatory")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.config = extraction
        self.families = families
        self.names = tuple(feature_names)
        self.pm = positional_means
        if isinstance(pca_components, dict):
            if (
                not isinstance(pca_mean, dict)
                or pca_components.keys() != pca_mean.keys()
            ):
                raise ValueError(
                    "per-layer PCA components and means must have matching layers"
                )
            if not set(extraction.pca_layers).issubset(pca_components):
                raise ValueError("per-layer PCA is missing a configured layer")
            self.pca = {
                layer: torch.as_tensor(value, device=self.device, dtype=torch.float64)
                for layer, value in pca_components.items()
            }
            self.pca_mean = {
                layer: torch.as_tensor(value, device=self.device, dtype=torch.float64)
                for layer, value in pca_mean.items()
            }
        else:
            if isinstance(pca_mean, dict):
                raise ValueError("pooled PCA requires a pooled mean")
            self.pca = torch.as_tensor(
                pca_components, device=self.device, dtype=torch.float64
            )
            self.pca_mean = torch.as_tensor(
                pca_mean, device=self.device, dtype=torch.float64
            )
        self.calibration_sha256 = calibration_sha256
        sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("ops.py"),
                Path(__file__).with_name("attention.py"),
                Path(__file__).with_name("families.py"),
                Path(__file__).with_name("batch_layout.py"),
            ]
        }
        identity = dict(
            sources=sources,
            extraction=extraction.model_dump(mode="json"),
            families=families.model_dump(mode="json"),
            calibration=calibration_sha256,
            device_type=self.device.type,
            replay_path=replay_path,
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
            cuda_runtime=torch.version.cuda,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            feature_schema_sha256=hashlib.sha256(
                json.dumps(feature_names).encode()
            ).hexdigest(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            tf32=torch.backends.cuda.matmul.allow_tf32,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
        )
        self.identity = identity
        self.lane_id = (
            "torch-eager-reduce-v1-"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
                :20
            ]
        )

    def _outputs(self, out: FeatureCollector, logits: Tensor, chosen: Tensor) -> Tensor:
        entropy, top1, top5, ranks, prob, logprob = [], [], [], [], [], []
        for first in range(0, len(logits), 16):
            x = logits[first : first + 16].float().double()
            x = x - x.max(dim=1, keepdim=True).values
            exp = x.exp()
            z = exp.sum(dim=1)
            logz = z.log()
            cid = chosen[first : first + 16, None]
            xc = x.gather(1, cid).squeeze(1)
            entropy.append((logz - (x * exp).sum(dim=1) / z).float())
            top1.append((exp.max(dim=1).values / z).float())
            top5.append((exp.topk(5, dim=1).values.sum(dim=1) / z).float())
            ranks.append((x > xc[:, None]).sum(dim=1).float())
            prob.append((exp.gather(1, cid).squeeze(1) / z).float())
            logprob.append(xc - logz)
        ent, top, topfive, rank, chosen_prob = [
            torch.cat(v) for v in (entropy, top1, top5, ranks, prob)
        ]
        surprise = -chosen_prob.clamp_min(1e-10).log()
        for name, values in (("logit_entropy", ent), ("top1_prob", top)):
            out.moments(name, values)
            for i, t in enumerate(
                trajectory_indices(len(logits), self.config.trajectory_points)
            ):
                out.put(f"{name}_traj{i}", values[t])
        out.put("top5_mass_mean", topfive.mean())
        for name, values in (("chosen_rank", rank), ("surprise", surprise)):
            out.put("mean_" + name, values.mean())
            out.put("std_" + name, std(values))
        for i, t in enumerate(
            trajectory_indices(len(logits), self.config.trajectory_points)
        ):
            out.put(f"surprise_traj{i}", surprise[t])
        window = self.config.surprise_window
        if len(logits) >= window:
            values = surprise.unfold(0, window, 1)
            running_mean = values.double().mean(dim=1)
            running_std = std(values, dim=1).float()
            threshold = (
                running_mean
                + self.config.surprise_threshold_sigma
                * running_std.double().clamp_min(1e-6)
            )
            out.put(
                "surprise_boundary_count",
                (surprise[window - 1 :].double() > threshold).sum(),
            )
        else:
            out.put("surprise_boundary_count", 0.0)
        return torch.cat(logprob).mean()

    @torch.no_grad()
    def replay_span(
        self, loaded: LoadedModel, token_ids: list[int], start: int, end: int
    ) -> GpuFeatureResult:
        self._validate_span(loaded, token_ids, start, end)
        steps = end - start - 1
        collector = FeatureCollector(self.device)
        attention = AttentionReducer(
            collector,
            steps=steps,
            prefix_length=start,
            sampled_layers=self.config.sampled_layers,
            spectral_stride=self.config.spectral_subsample_step,
            n_windows=self.families.temporal_n_windows,
            include_stft=self.families.enable_stft,
        )
        reducer = FamilyReducer(collector, self.config, self.families)
        ids = torch.tensor(
            token_ids[:end], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        loaded.clear_hook_state()
        loaded.disable_hooks()
        offset = start if self.replay_path == "full" else 0
        cache = None
        if self.replay_path == "cached":
            pre = loaded.model(ids[:, :start], use_cache=True, return_dict=True)
            cache = pre.past_key_values
            del pre
        handles = []
        try:
            layers = decoder_layers(loaded.model)
            for layer, module in enumerate(layers):

                def hook(_module, _args, result, layer=layer):
                    if (
                        not isinstance(result, tuple)
                        or len(result) < 2
                        or result[1] is None
                    ):
                        raise RuntimeError(
                            "installed eager attention does not expose weights to hook"
                        )
                    attention.consume(
                        layer, result[1][:, :, offset : offset + steps + 1, :]
                    )

                handles.append(module.self_attn.register_forward_hook(hook))
            loaded.clear_hook_state()
            loaded.enable_hooks()
            if self.replay_path == "full":
                # Preserve replay_extract's one-pass, no-cache arithmetic path.
                result = loaded.model(
                    ids,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            else:
                result = loaded.model(
                    ids[:, start:end],
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                    position_ids=torch.arange(start, end, device=self.device).unsqueeze(
                        0
                    ),
                    cache_position=torch.arange(start, end, device=self.device),
                )
            if attention.seen != set(range(len(layers))):
                raise RuntimeError("not every eager attention layer was reduced")
            if getattr(result, "attentions", None):
                raise RuntimeError("GPU lane unexpectedly retained attention outputs")
            return self._reduce_capture(
                loaded,
                result,
                ids,
                token_ids,
                start,
                end,
                offset,
                collector,
                attention,
                reducer,
            )
        finally:
            for handle in handles:
                handle.remove()
            loaded.clear_hook_state()

    @torch.no_grad()
    def replay_batch(
        self,
        loaded: LoadedModel,
        spans: Sequence[ReplaySpan],
        *,
        prefill_policy: Literal["batched", "independent"] = "batched",
    ) -> list[GpuFeatureResult]:
        """One batched prefill and continuation; independent per-row reductions.

        Experimental lane: no qualification is inherited from batch size one.
        Prefix padding is removed before attention features see the tensor.
        """
        if self.replay_path != "cached":
            raise ValueError("batched replay currently requires the cached path")
        layout = pack_spans(tuple(spans), loaded.model.config.vocab_size)
        if prefill_policy not in ("batched", "independent"):
            raise ValueError("unknown prefill policy")
        if (
            prefill_policy == "independent"
            and len({s.start for s in layout.spans}) != 1
        ):
            raise ValueError(
                "independent-prefix cache stacking requires equal prefix lengths"
            )
        for span in layout.spans:
            self._validate_span(loaded, span.tokens, span.start, span.end)
        batch_identity = dict(
            parent=self.identity,
            batch_size=len(layout.spans),
            padding_policy="left-prefix/right-continuation-v1",
            prefill_policy=prefill_policy,
        )
        batch_lane = (
            "torch-eager-batch-v1-"
            + hashlib.sha256(
                json.dumps(batch_identity, sort_keys=True).encode()
            ).hexdigest()[:20]
        )
        collectors = [FeatureCollector(self.device) for _ in layout.spans]
        attentions = [
            AttentionReducer(
                c,
                steps=s.end - s.start - 1,
                prefix_length=s.start,
                sampled_layers=self.config.sampled_layers,
                spectral_stride=self.config.spectral_subsample_step,
                n_windows=self.families.temporal_n_windows,
                include_stft=self.families.enable_stft,
            )
            for c, s in zip(collectors, layout.spans, strict=True)
        ]
        reducers = [FamilyReducer(c, self.config, self.families) for c in collectors]
        arrays = (
            layout.prefix_ids,
            layout.prefix_mask,
            layout.prefix_positions,
            layout.continuation_ids,
            layout.attention_mask,
            layout.continuation_positions,
        )
        prefix, prefix_mask, prefix_pos, continuation, mask, positions = [
            torch.as_tensor(a, device=self.device) for a in arrays
        ]
        handles = []
        loaded.clear_hook_state()
        loaded.disable_hooks()
        try:
            if prefill_policy == "batched":
                pre = loaded.model(
                    prefix,
                    attention_mask=prefix_mask,
                    position_ids=prefix_pos,
                    use_cache=True,
                    return_dict=True,
                )
                cache = pre.past_key_values
                del pre
            else:
                from transformers.cache_utils import DynamicCache

                caches = []
                for i in range(len(layout.spans)):
                    # Match the original unpadded single-prefix forward exactly.
                    pre = loaded.model(
                        prefix[i : i + 1], use_cache=True, return_dict=True
                    )
                    caches.append(pre.past_key_values)
                    del pre
                pairs = [
                    (
                        torch.cat([c.layers[layer].keys for c in caches], dim=0),
                        torch.cat([c.layers[layer].values for c in caches], dim=0),
                    )
                    for layer in range(len(caches[0].layers))
                ]
                cache = DynamicCache(ddp_cache_data=pairs, config=loaded.model.config)
                del caches, pairs
            layers = decoder_layers(loaded.model)
            for layer, module in enumerate(layers):

                def hook(_module, _args, result, layer=layer):
                    if (
                        not isinstance(result, tuple)
                        or len(result) < 2
                        or result[1] is None
                    ):
                        raise RuntimeError("eager attention weights unavailable")
                    for i, attention in enumerate(attentions):
                        q, k = layout.attention_slices(i)
                        attention.consume(layer, result[1][i : i + 1, :, q, k])

                handles.append(module.self_attn.register_forward_hook(hook))
            loaded.clear_hook_state()
            loaded.enable_hooks()
            result = loaded.model(
                continuation,
                attention_mask=mask,
                position_ids=positions,
                cache_position=torch.arange(
                    layout.prefix_width,
                    layout.prefix_width + layout.continuation_width,
                    device=self.device,
                ),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            if getattr(result, "attentions", None):
                raise RuntimeError("batch unexpectedly retained attention outputs")
            outputs = []
            invocation = str(uuid.uuid4())
            for i, (span, collector, attention, reducer) in enumerate(
                zip(layout.spans, collectors, attentions, reducers, strict=True)
            ):
                if attention.seen != set(range(len(layers))):
                    raise RuntimeError("not every batch attention layer was reduced")
                ids = torch.tensor(
                    [span.tokens[: span.end]], device=self.device, dtype=torch.long
                )
                output = self._reduce_capture(
                    loaded,
                    result,
                    ids,
                    span.tokens,
                    span.start,
                    span.end,
                    0,
                    collector,
                    attention,
                    reducer,
                    batch_index=i,
                )
                output.metadata.update(
                    lane_id=batch_lane,
                    batch_size=len(layout.spans),
                    batch_invocation_id=invocation,
                    batch_row=i,
                    batch_composition_sha256=layout.composition_sha256,
                    padding_policy=batch_identity["padding_policy"],
                    prefill_policy=prefill_policy,
                    padded_prefix_width=layout.prefix_width,
                    padded_continuation_width=layout.continuation_width,
                    shared_batch_input_h2d_bytes=sum(a.nbytes for a in arrays),
                )
                outputs.append(output)
            return outputs
        finally:
            for handle in handles:
                handle.remove()
            loaded.clear_hook_state()

    def _validate_span(self, loaded, token_ids, start, end):
        if (
            torch.are_deterministic_algorithms_enabled()
            != self.identity["deterministic"]
            or torch.backends.cuda.matmul.allow_tf32 != self.identity["tf32"]
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != self.identity["cublas_workspace_config"]
            or self.replay_path != self.identity["replay_path"]
            or self.config.model_dump(mode="json") != self.identity["extraction"]
            or self.families.model_dump(mode="json") != self.identity["families"]
            or str(torch.backends.cuda.preferred_blas_library())
            != self.identity["preferred_blas_library"]
        ):
            raise ValueError("arithmetic settings changed after lane construction")
        if not 0 < start < end <= len(token_ids) or end - start < 2:
            raise ValueError(
                "invalid span; a nonempty full prefix and >=2 tokens are required"
            )
        if loaded.model.training or loaded.model.config.model_type != "llama":
            raise ValueError("GPU lane requires an eval-mode dense Llama")
        if loaded.model.config._attn_implementation != "eager":
            raise ValueError("GPU lane requires eager attention")
        if any(p.device != self.device for p in loaded.model.parameters()):
            raise ValueError("v1 requires a model entirely on the declared device")
        if end - 2 >= self.pm.shape[1]:
            raise ValueError("positional calibration does not cover span")
        if (
            min(token_ids[:end]) < 0
            or max(token_ids[:end]) >= loaded.model.config.vocab_size
        ):
            raise ValueError("token outside model vocabulary")

    def _reduce_capture(
        self,
        loaded,
        result,
        ids,
        token_ids,
        start,
        end,
        offset,
        collector,
        attention,
        reducer,
        batch_index=0,
    ):
        steps = end - start - 1
        layers = decoder_layers(loaded.model)
        h2d_bytes = 0
        previous = None
        for layer in range(len(layers)):
            h = result.hidden_states[layer + 1][
                batch_index, offset : offset + steps
            ].float()
            pm = np.ascontiguousarray(self.pm[layer + 1, start : start + steps])
            h2d_bytes += pm.nbytes
            corrected = h - torch.as_tensor(pm, device=self.device)
            norms = corrected.norm(dim=-1).float()
            collector.put(f"activation_norm_mean_L{layer}", norms.mean())
            collector.put(f"activation_norm_std_L{layer}", std(norms))
            for i, t in enumerate(
                trajectory_indices(steps, self.config.trajectory_points)
            ):
                collector.put(f"activation_norm_traj{i}_L{layer}", norms[t])
            if previous is not None:
                delta = corrected - previous
                dn = delta.norm(dim=-1).float()
                dc = rowcos(delta.double(), previous.double()).float()
                collector.put(f"delta_norm_mean_L{layer - 1}", dn.mean())
                collector.put(f"delta_norm_std_L{layer - 1}", std(dn))
                collector.put(f"delta_cosine_mean_L{layer - 1}", dc.mean())
            previous = corrected
            if layer in self.config.sampled_layers:
                attention.smoothness(layer, corrected)
            if layer in self.families.trajectory_layers:
                reducer.residual_trajectory(layer, corrected, h[0].norm())
            if layer in self.config.pca_layers:
                components = self.pca[layer] if isinstance(self.pca, dict) else self.pca
                mean = (
                    self.pca_mean[layer]
                    if isinstance(self.pca_mean, dict)
                    else self.pca_mean
                )
                indices = trajectory_indices(steps, self.config.pca_temporal_samples)
                projections = (corrected[indices].double() - mean) @ components[
                    : self.config.pca_components
                ].T
                for ti, projection in enumerate(projections):
                    for ci, value in enumerate(projection):
                        collector.put(f"pca_L{layer}_t{ti}_c{ci}", value)
        logprob = self._outputs(
            collector,
            result.logits[batch_index, offset : offset + steps],
            ids[0, start + 1 : end],
        )
        for layer in self.config.sampled_layers:

            def head(name):
                values = getattr(loaded.hook_state, name).get(layer)
                if values is None or len(values) != 1:
                    raise RuntimeError(f"missing/duplicate {name} layer {layer}")
                return (
                    values[0][batch_index, :, offset : offset + steps, :]
                    .permute(1, 0, 2)
                    .float()
                )

            keys = head("pre_rope_keys")
            reducer.key(layer, keys)
            attention.head_features(layer, keys)
            reducer.value(layer, head("v_proj_values"))
            reducer.query(layer, head("queries"))
            gate = loaded.hook_state.gate_activations.get(layer)
            if gate is None or len(gate) != 1:
                raise RuntimeError(f"missing gate layer {layer}")
            reducer.gate(layer, gate[0][batch_index, offset : offset + steps].float())
        reducer.finish()
        vector = collector.finish(list(self.names)).cpu().numpy()
        if not np.isfinite(vector).all():
            raise RuntimeError("nonfinite GPU feature")
        knnlm = (
            result.hidden_states[-1][batch_index, offset + steps - 1]
            .float()
            .cpu()
            .numpy()
            .copy()
            if self.config.enable_knnlm_baseline
            else None
        )
        ancillary = 8 + (knnlm.nbytes if knnlm is not None else 0)
        receipt = dict(
            lane_id=self.lane_id,
            replay_id=str(uuid.uuid4()),
            fresh_cache=True,
            replay_path=self.replay_path,
            cache_policy="none" if self.replay_path == "full" else "fresh-full-prefix",
            span_start=start,
            span_end=end,
            input_tokens_sha256=hashlib.sha256(
                np.asarray(token_ids[:end], dtype="<i8").tobytes()
            ).hexdigest(),
            feature_schema_sha256=self.identity["feature_schema_sha256"],
            calibration_sha256=self.calibration_sha256,
            channel="probe-free",
            certified=False,
            batch_size=1,
            attention_layers_reduced=len(attention.seen),
            retained_attention_summary_bytes=attention.retained_bytes,
            gate_tie_exception_d2h_bytes=reducer.gate_tie_exception_d2h_bytes,
            explicit_d2h_bytes=vector.nbytes
            + ancillary
            + reducer.gate_tie_exception_d2h_bytes,
            positional_calibration_h2d_bytes=h2d_bytes,
            input_token_h2d_bytes=ids.numel() * ids.element_size(),
        )
        return GpuFeatureResult(
            vector, self.names, float(logprob.cpu()), knnlm, receipt
        )
