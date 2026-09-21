"""First contact with a new model: does the capture path work on it at all?

Adding a model means adding a preset row, and a preset row is a set of claims — how
many decoder layers, how many query heads, which modules the hooks attach to, which
token ids end a generation. Every one of them can be wrong, and most of them fail
*quietly*: an attention implementation that returns no weights yields a feature
vector of the right width full of nothing, and a layer plan off by one indexes a
different part of the network than it names.

So a new preset is validated before any calibration or floor pass is spent on it,
in four steps that each fail loudly:

1. **Load** the checkpoint eagerly, through the same loader the instrument uses. A
   multimodal wrapper whose text decoder nests inside a submodule is the normal
   case, not an exception, and the loader is what knows where to look.
2. **Resolve the layers and the hook targets.** The decoder layer count must be the
   preset's, and the modules the hooks attach to must exist on the layers the plan
   samples — which differ by architecture, and by whether a layer is dense or routed.
3. **Generate with the capture surface armed** and check what came back: attention
   weights at the preset's query-head count (the one check that catches a fused
   kernel silently returning none), hidden states at depth plus one, and hooks that
   fired.
4. **Extract one feature vector** and confirm it is finite. The full battery vector
   is a replay-side object; what this establishes is that the path from a prompt to
   a signature is unbroken.

A routed checkpoint gets a fifth step, because its capture surface has parts a dense
one does not: the router pre-hook has to fire with the expert count the config
declares, the compressed key-value hook has to fire, and the routing family's
features have to come out finite and fully classified by the taxonomy.

Every check returns a named refusal rather than an assertion, because an assertion
disappears under optimization and this is the one pass whose whole output is its
refusals.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.config import (
    ExperimentConfig,
    ExtractionConfig,
    GenerationConfig,
    GenerationSpec,
    ModelConfig,
    ModelPreset,
)

logger = logging.getLogger(__name__)

PROMPT = "Explain how a rainbow forms."
"""The one prompt the smoke runs on. Content is irrelevant here — what is being
checked is the machinery — so it is fixed rather than sampled."""

SMOKE_MAX_NEW_TOKENS = 8
EXTRACT_MAX_NEW_TOKENS = 12
ROUTING_FEATURES_PER_LAYER = 10
"""Features the routing family emits per routed layer. The count is what catches a
family that silently produced a partial vector."""

DENSE_ATTENTION_MODULES = ("k_proj", "q_proj", "o_proj")
MLA_ATTENTION_MODULES = ("kv_a_proj_with_mqa", "kv_b_proj", "q_proj", "o_proj")
MOE_MLP_MODULES = ("gate", "experts", "shared_experts")


class OnboardingError(RuntimeError):
    """A preset claim the checkpoint does not satisfy."""


class OnboardingReport(BaseModel):
    """What the smoke established, in the order it established it."""

    model_config = ConfigDict(extra="forbid")

    model: str
    model_id: str
    model_class: str
    n_decoder_layers: int = Field(gt=0)
    probed_layers: list[int]
    architecture: str = Field(description="`dense` or the routed architecture's own name")
    attention_shape: list[int]
    n_hidden_state_layers: int = Field(gt=0)
    n_features: int = Field(gt=0)
    n_generated_tokens: int = Field(ge=0)
    features_finite: bool
    sample_text: str
    routing: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        """The smoke passes when the vector it produced is a vector, not a shape."""
        return self.features_finite and self.n_features > 0

    def lines(self) -> list[str]:
        """The report as a reader sees it, one line per step."""
        out = [
            f"[1] loaded {self.model_id} as {self.model_class}",
            f"[2] {self.n_decoder_layers} decoder layers; hook targets present on "
            f"{self.probed_layers} (architecture {self.architecture})",
            f"[3] attention at step 1 layer 0 has shape {self.attention_shape}; "
            f"hidden states {self.n_hidden_state_layers} layers; hooks fired",
        ]
        if self.routing is not None:
            out.append(
                f"[3b] router fired on {self.routing['n_layers_fired']}/"
                f"{self.routing['n_routed_layers']} routed layers, distribution width "
                f"{self.routing['distribution_width']}, compressed keys captured; "
                f"{self.routing['n_features']} routing features, "
                f"unclassified {self.routing['n_unclassified']}"
            )
        out.append(
            f"[4] extraction produced {self.n_features} features over "
            f"{self.n_generated_tokens} generated tokens, finite={self.features_finite}"
        )
        out.append(f"    sample text: {self.sample_text!r}")
        out.append("SMOKE PASS" if self.passed else "SMOKE FAIL")
        return out


def probe_layers(preset: ModelPreset) -> list[int]:
    """Three layers worth checking: the first sampled, the middle one, the last.

    Checking every layer would be slower and would say nothing more — a hook target
    missing from one layer of an architecture is missing from all of them, and the
    three positions are where an off-by-one in the plan shows up.
    """
    sampled = list(preset.sampled_layers)
    return sorted({sampled[0], sampled[len(sampled) // 2], sampled[-1]})


def check_hook_targets(model: Any, preset: ModelPreset, layers: list[int]) -> str:
    """Confirm the modules the hooks attach to exist; name the architecture found.

    Two architecture classes are told apart here because their attention surfaces
    are different objects: an ordinary attention block projects keys and values
    directly, while a latent-attention block projects a compressed joint
    representation and expands it, so the module the key hook attaches to has a
    different name. A routed layer's feed-forward is checked as a router plus
    experts plus the shared branch, and a dense layer of the same checkpoint is
    still checked as a plain gate projection.
    """
    from anamnesis.extraction.model_loader import decoder_layers

    layer_modules = decoder_layers(model)
    if len(layer_modules) != preset.num_layers:
        raise OnboardingError(
            f"checkpoint has {len(layer_modules)} decoder layers, the preset claims "
            f"{preset.num_layers} — every layer-indexed feature would name the wrong depth"
        )

    model_type = str(getattr(model.config, "model_type", ""))
    latent_attention = "deepseek_v2" in model_type
    first_dense = int(getattr(model.config, "first_k_dense_replace", 0))

    for index in layers:
        attention = layer_modules[index].self_attn
        wanted = MLA_ATTENTION_MODULES if latent_attention else DENSE_ATTENTION_MODULES
        missing = [name for name in wanted if not hasattr(attention, name)]
        if missing:
            raise OnboardingError(
                f"layer {index}: attention block has no {missing} — the key, query or "
                f"output hooks have nothing to attach to"
            )
        mlp = layer_modules[index].mlp
        if latent_attention and index >= first_dense:
            missing = [name for name in MOE_MLP_MODULES if not hasattr(mlp, name)]
            if missing:
                raise OnboardingError(f"layer {index}: routed feed-forward has no {missing}")
            if not hasattr(mlp.shared_experts, "gate_proj"):
                raise OnboardingError(
                    f"layer {index}: the shared expert branch has no gate projection, so "
                    f"the gate features would read nothing"
                )
        elif not hasattr(mlp, "gate_proj"):
            raise OnboardingError(
                f"layer {index}: feed-forward has no gate projection, so the gate "
                f"features would read nothing"
            )
    return model_type if latent_attention else "dense"


def check_capture_surface(loaded: Any, preset: ModelPreset) -> tuple[list[int], int]:
    """Generate a few tokens with everything armed; return the attention shape and depth.

    The attention check is the one that earns this whole pass: a fused attention
    kernel returns no weights at all, and every attention feature computed against a
    missing tensor is a zero that reads as a measurement.
    """
    import torch

    tokenizer = loaded.tokenizer
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (
        encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    ).to(loaded.model.device)

    loaded.clear_hook_state()
    loaded.enable_hooks()
    with torch.no_grad():
        out = loaded.model.generate(
            input_ids,
            max_new_tokens=SMOKE_MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=preset.eos_token_ids,
            pad_token_id=tokenizer.pad_token_id or preset.eos_token_ids[0],
            output_attentions=True,
            output_hidden_states=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
    if out.attentions is None or len(out.attentions) <= 1:
        raise OnboardingError(
            "generate returned no attention weights — the attention implementation is "
            "not eager, and every attention feature would be computed from nothing"
        )
    first_step = out.attentions[1][0]
    if first_step.shape[1] != preset.num_attention_heads:
        raise OnboardingError(
            f"attention has {first_step.shape[1]} heads, the preset claims "
            f"{preset.num_attention_heads} — per-head features would be misaligned"
        )
    depth = len(out.hidden_states[0])
    if depth != preset.num_layers + 1:
        raise OnboardingError(
            f"hidden states carry {depth} layers, the preset's depth plus the embedding "
            f"output is {preset.num_layers + 1}"
        )
    loaded.flush_hooks_to_cpu()
    return [int(d) for d in first_step.shape], depth


def check_routing_surface(loaded: Any, preset: ModelPreset) -> dict[str, Any]:
    """The routed checkpoint's extra surface: the router, the compressed keys, the family.

    The width check is the substantive one — a router distribution narrower than the
    checkpoint's expert count means the hook caught the wrong tensor — and the family
    is then run over the captured distributions so its own arity and finiteness are
    established here rather than discovered during a battery pass.
    """
    from anamnesis.extraction.feature_families.expert_routing import (
        extract_expert_routing_features,
    )
    from anamnesis.extraction.generation_runner import router_fields_from_hooks
    from anamnesis.extraction.state_extractor import RawGenerationData
    from anamnesis.feature_map import FeatureMap

    first_dense = int(getattr(loaded.model.config, "first_k_dense_replace", 0))
    routed_layers = [layer for layer in preset.sampled_layers if layer >= first_dense]
    distributions = loaded.hook_state.router_dist
    fired = [layer for layer in routed_layers if distributions.get(layer)]
    if not fired:
        raise OnboardingError(
            "the router pre-hook fired on no routed layer — routing features would be absent"
        )
    width = int(distributions[fired[0]][-1].shape[-1])
    declared = int(loaded.model.config.n_routed_experts)
    if width != declared:
        raise OnboardingError(
            f"router distribution is {width} wide, the checkpoint declares {declared} "
            f"experts — the hook captured the wrong tensor"
        )
    if not loaded.hook_state.pre_rope_keys.get(fired[0]):
        raise OnboardingError(
            "the compressed key-value hook did not fire, so the key geometry features "
            "would read nothing on this architecture"
        )

    router_dist, branch_norms, logit_norms = router_fields_from_hooks(
        loaded.hook_state, list(preset.sampled_layers)
    )
    placeholder = RawGenerationData(
        hidden_states=[np.zeros((preset.num_layers + 1, preset.hidden_dim), dtype=np.float32)],
        attentions=[],
        logits=[],
        chosen_token_ids=np.zeros(1),
        pre_rope_keys={},
        prompt_length=1,
        router_dist=router_dist,
        router_branch_norms=branch_norms,
        router_logit_norms=logit_norms,
    )
    result = extract_expert_routing_features(
        placeholder,
        sampled_layers=routed_layers,
        top_k=int(loaded.model.config.num_experts_per_tok),
    )
    expected = ROUTING_FEATURES_PER_LAYER * len(routed_layers)
    if len(result.features) != expected:
        raise OnboardingError(
            f"the routing family emitted {len(result.features)} features, expected "
            f"{expected} ({ROUTING_FEATURES_PER_LAYER} per routed layer)"
        )
    unclassified = FeatureMap(result.feature_names, preset.num_layers).unclassified()
    if unclassified:
        raise OnboardingError(
            f"{len(unclassified)} routing features are unclassified by the taxonomy: "
            f"{unclassified[:5]}"
        )
    if not bool(np.isfinite(result.features).all()):
        raise OnboardingError("routing features are not all finite")
    return {
        "n_routed_layers": len(routed_layers),
        "n_layers_fired": len(fired),
        "distribution_width": width,
        "n_features": int(len(result.features)),
        "n_unclassified": 0,
    }


def onboard_model(preset: ModelPreset, model_path: str | None = None) -> OnboardingReport:
    """Run the whole smoke over one preset; raise on the first claim that fails.

    ``model_path`` overrides the preset's identifier, because a first-contact check
    is usually run against a local snapshot whose bytes are what will be used.
    """
    from anamnesis.extraction.generation_runner import run_single_generation
    from anamnesis.extraction.model_loader import load_model

    model_id = model_path or preset.model_id
    config = ModelConfig.from_preset(preset, model_id=model_id)
    logger.info(f"[1] loading {model_id} eager/{preset.torch_dtype}")
    loaded = load_model(
        config,
        sampled_layers=list(preset.sampled_layers),
        register_gate_hooks=True,
        key_layers=list(preset.sampled_layers),
        value_layers=list(preset.sampled_layers),
        query_layers=list(preset.sampled_layers),
        attn_output_layers=list(preset.sampled_layers),
    )

    layers = probe_layers(preset)
    architecture = check_hook_targets(loaded.model, preset, layers)
    attention_shape, depth = check_capture_surface(loaded, preset)
    routing = check_routing_surface(loaded, preset) if architecture != "dense" else None

    experiment = ExperimentConfig(
        model=config,
        generation=GenerationConfig(
            max_new_tokens=EXTRACT_MAX_NEW_TOKENS,
            temperature=preset.temperature,
            top_p=0.95,
            eos_token_ids=list(preset.eos_token_ids),
            do_sample=True,
        ),
        extraction=ExtractionConfig.from_preset(preset),
    )
    spec = GenerationSpec(
        generation_id=0,
        prompt_set="onboard",
        topic="rainbow",
        topic_idx=0,
        mode="",
        mode_idx=0,
        system_prompt="",
        user_prompt=PROMPT,
        seed=1234,
        repetition=0,
    )
    result, metadata = run_single_generation(loaded, spec, experiment, positional_means=None)

    return OnboardingReport(
        model=preset.name,
        model_id=model_id,
        model_class=type(loaded.model).__name__,
        n_decoder_layers=preset.num_layers,
        probed_layers=layers,
        architecture=architecture,
        attention_shape=attention_shape,
        n_hidden_state_layers=depth,
        n_features=int(metadata["num_features"]),
        n_generated_tokens=int(metadata["num_generated_tokens"]),
        features_finite=bool(np.isfinite(result.features).all()),
        sample_text=str(metadata["generated_text"])[:120],
        routing=routing,
    )


__all__ = [
    "DENSE_ATTENTION_MODULES",
    "MLA_ATTENTION_MODULES",
    "MOE_MLP_MODULES",
    "OnboardingError",
    "OnboardingReport",
    "PROMPT",
    "ROUTING_FEATURES_PER_LAYER",
    "check_capture_surface",
    "check_hook_targets",
    "check_routing_surface",
    "onboard_model",
    "probe_layers",
]
