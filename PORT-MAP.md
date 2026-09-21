# PORT MAP

Where a name from the extraction repository lives in this one. The instrument was
refounded as an allow-list: a capability is here because it was carried over
deliberately, and everything else stays in the frozen repository
`anamnesis-pl`, which remains the citation target for every path it holds. The
research record cites files by their old paths, so this table is how a reader
holding a citation finds the living code — or learns that the code they are
holding is the record's own copy and has no living successor.

One row per name. **Old path** is relative to `pipeline/` in `anamnesis-pl`;
**new home** is relative to this repository's root. A row marked *record* is not
a gap: it is a decision that the name belongs to the history of the program
rather than to its instrument.

## Configuration

`config.py` held two descriptions of a model — a preset registry and a loader
config — and no bridge between them, so callers projected one onto the other
field by field. The port keeps both roles and adds the bridge: every
configuration class is built by a `from_preset` constructor over one registry
row, so an architecture, a layer plan, a decode policy and a calibration path
cannot disagree about which model they describe.

| old path | new home | notes |
|---|---|---|
| `anamnesis/config.py` | `anamnesis/config/` | Split four ways; the sections below say which part went where. |
| `config.py` · `ModelPreset`, `MODEL_PRESETS` | `anamnesis/config/models.py` | The registry of per-model facts, with the decode policy and the calibration location now part of the row. |
| `config.py` · `ModelConfig` | `anamnesis/config/models.py` | The loader contract, built by `ModelConfig.from_preset()`. Architecture fields are required, so a config cannot describe a model other than the one being loaded. |
| `config.py` · `GenerationConfig`, `ExtractionConfig`, `FeaturePipelineConfig`, `CalibrationConfig`, `GenerationSpec`, `ExperimentConfig`, `ProcessingMode` | `anamnesis/config/experiment.py` | Each class carries `from_preset`; `ExperimentConfig.from_preset()` derives all four section configs together. |
| `config.py` · `RUNS`, `RunSpec` | `anamnesis/config/runs.py` + `anamnesis/config/runs.json` | The registry is data beside the module that reads it. A row is a root token plus relative paths; `resolve_run()` turns it into directories on this machine. |
| `config.py` · `PROJECT_ROOT`, `OUTPUTS_BASE`, `LEGACY_DATA_ROOT`, `CALIBRATION_DIR`, `OUTPUTS_DIR`, `SIGNATURES_DIR`, `FIGURES_DIR`, `PROMPTS_PATH`, `RUN_NAME` | `anamnesis/config/paths.py` | The data roots, read at call time rather than at import. `ANAMNESIS_OUTPUTS` names the outputs root and `ANAMNESIS_LEGACY_DATA` the Phase-0 tree; both the preset registry and the run registry resolve against them, which is why the roots are their own module rather than the property of either. |
| `config.py` · `PROCESSING_MODES`, `MODE_INDEX` | `anamnesis/modes/run4_modes.py` · `RUN4_MODES`, `RUN4_MODE_INDEX` | The re-export is gone: configuration does not import the mode prompts. A caller that needs the prompts names the modes package. |

## Modes and prompts

The mode prompts are the protocol: a label in banked data means the exact text in
this package. They port unchanged, and the tests pin their digests.

| old path | new home | notes |
|---|---|---|
| `anamnesis/modes/run4_modes.py` | `anamnesis/modes/run4_modes.py` | The five format-controlled modes, byte-identical. The format constraint is public here, since the eight-mode set shares it rather than restating it. |
| `anamnesis/modes/extended_modes.py` | `anamnesis/modes/extended_modes.py` | The eight-mode set, byte-identical. |
| `anamnesis/modes/prompt_swap.py` | `anamnesis/modes/prompt_swap.py` | The three swap pairs, byte-identical. |
| `anamnesis/modes/__init__.py` | `anamnesis/modes/__init__.py` | Reworked: it exports the three ported sets and nothing else. |
| `anamnesis/modes/run3_original_modes.py` | *record* | Not ported, by ruling: a frozen Phase-0 historical record, with its one script consumer staying beside it in `anamnesis-pl`. The old package re-exported it from `modes/__init__.py` and `config.py` reached that `__init__` on import, which made a record-only module a hard dependency of the whole package; the reworked `__init__` and a configuration package that imports no modes are what let the ruling hold. Tests assert the module is absent and that neither coupling has returned. |
| `anamnesis/prompts/prompt_sets.json` | `anamnesis/prompts/prompt_sets.json` | Byte-identical; digest pinned in the tests. |
| `anamnesis/prompts/prompt_sets_narrative.json` | `anamnesis/prompts/prompt_sets_narrative.json` | Byte-identical; digest pinned in the tests. |

## The feature taxonomy

`feature_map.py` is the executable `source × method × depth` taxonomy, and the
second-most-cited name in the research record. It was filed under `analysis/`,
which inverted the dependency it actually has: extraction emits the names, and
the taxonomy is what both extraction and analysis read them through. It moves to
the package root so neither layer has to reach across the other to name what a
feature is.

| old path | new home | notes |
|---|---|---|
| `anamnesis/analysis/feature_map.py` | `anamnesis/feature_map.py` | Promoted to the package root; module name unchanged, by ruling. The classification rules, the `Source`/`Method`/`Band` values and every emitted tag are unchanged, so a name classifies exactly as it did. The validation CLI is `python -m anamnesis.feature_map <run>`. |

## Extraction — the specification layer

Extraction is three things in one directory, and the names say which is which: a
**specification** of what a signature's numbers are, an **implementation** that
computes them quickly, and a **proof that the two agree**. This section covers the
specification and the recompute-from-raw path over banked tensors; the model-facing
capture layer and the fast lane are their own rows.

The specification is pure numpy. That is enforced by
`tests/test_extraction_purity.py` — statically over the import graph and at runtime
in a subprocess — rather than by a comment, which is the red-flag fix the manifest
ruled: a constraint the whole lane is defined against has to be something a test can
fail on.

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/__init__.py` | `anamnesis/extraction/__init__.py` | Reworked: it documents the three layers and imports none of them, so addressing one submodule never loads another's dependencies and the anchor's purity stays assertable. |
| `anamnesis/extraction/state_extractor.py` | `anamnesis/extraction/state_extractor.py` | The numeric anchor, logic-identical. `RawGenerationData`, `ExtractionResult`, the four baseline blocks and the kNN-LM baseline keep their names, their feature names and their `tier_slices` keys, because those keys are on disk in every banked run. |
| `anamnesis/extraction/state_extractor_reference.py` | `anamnesis/extraction/state_extractor_reference.py` | The golden master, logic-identical. It ports with its equivalence test, under the "together or not at all" rule: one file is the claim and the other is the proof. |
| `anamnesis/extraction/raw_saver.py` | `anamnesis/extraction/raw_saver.py` | The on-disk form of a capture, logic-identical: both npz schemas, the surface names and the lean-load parameters unchanged. It is what lets a feature set be revised without re-running a model, and the recompute CLI cannot be read without it. |
| `anamnesis/extraction/feature_pipeline.py` | `anamnesis/extraction/feature_pipeline.py` | The family orchestrator and the recompute-from-raw CLI, logic-identical. `--model` is now required: the layer plan comes from one preset row through `ExtractionConfig.from_preset()` and `FeaturePipelineConfig.from_preset()`, and a layer index means a different fraction of the network in each model, so there is no correct default to fall back to. |

### Feature families

Each family is a self-contained extractor over banked tensors. The filenames are
cited in the record and the taxonomy already lives in `feature_map`, so they port at
their own names rather than being regrouped by substrate.

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/feature_families/__init__.py` | `anamnesis/extraction/feature_families/__init__.py` | The `FeatureFamilyResult` contract, unchanged. The module list it documents now names every family and says which substrate each reads. |
| `anamnesis/extraction/feature_families/operators.py` | `anamnesis/extraction/feature_families/operators.py` | Shared temporal operators — windowing, drift, STFT — byte-identical. |
| `anamnesis/extraction/feature_families/_helpers.py` | `anamnesis/extraction/feature_families/_helpers.py` | Byte-identical: the math utilities the families share, re-exported from the anchor so it stays their one source. |
| `anamnesis/extraction/feature_families/attention_flow.py` | `anamnesis/extraction/feature_families/attention_flow.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/temporal_dynamics.py` | `anamnesis/extraction/feature_families/temporal_dynamics.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/per_head.py` | `anamnesis/extraction/feature_families/per_head.py` | Logic-identical; one docstring sentence rephrased in the present tense for G3. |
| `anamnesis/extraction/feature_families/residual_stream.py` | `anamnesis/extraction/feature_families/residual_stream.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/contrastive_projection.py` | `anamnesis/extraction/feature_families/contrastive_projection.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/path_signature.py` | `anamnesis/extraction/feature_families/path_signature.py` | Logic-identical; its `feature_map` import follows that module to the package root. The in-module `selftest()` battery ports with it and runs under pytest. |
| `anamnesis/extraction/feature_families/gate_features.py` | `anamnesis/extraction/feature_families/gate_features.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/value_geometry.py` | `anamnesis/extraction/feature_families/value_geometry.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/qk_geometry.py` | `anamnesis/extraction/feature_families/qk_geometry.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/key_cka.py` | `anamnesis/extraction/feature_families/key_cka.py` | Byte-identical. |
| `anamnesis/extraction/feature_families/expert_routing.py` | `anamnesis/extraction/feature_families/expert_routing.py` | Byte-identical. A mixture-of-experts checkpoint supplies a router distribution and a dense one does not, so the family follows the optional-hook pattern `attn_res` documents. |
| `anamnesis/extraction/feature_families/attn_res.py` | `anamnesis/extraction/feature_families/attn_res.py` | **Ported with rework, by ruling.** The math, the feature names, the arity and the constants are identical; what changed is that the capability is stated as an architecture class rather than as one model's feature. Cross-block attention-residual routing is what an architecture has when it commits intermediate blocks and attends over them, and the module is now the reference example of an **optional per-architecture hook**: the substrate arrives as optional fields on `RawGenerationData`, the orchestrator gates on those fields being present rather than on a model's name, and a family with nothing to read is absent from the vector instead of zero-filled. The same naming shed applies to the substrate's declaration on `RawGenerationData`, the orchestrator's gate, and the taxonomy's `routing` source. |
| `anamnesis/extraction/feature_families/binding_probe.py` | *pending* | Not in this port: an unregistered contrast-time family with no home in the family contract. Its admission is ruled conditional on a dedupe against existing capability and on a test, both of which belong to the analysis port. It stays in `anamnesis-pl` until then. |

### Tests

| old path | new home | notes |
|---|---|---|
| `tests/test_extraction_equivalence.py` | `tests/test_extraction_equivalence.py` | The same four shapes over the same synthetic captures, now stated as pytest cases so the suite runs them; the script's `--benchmark` timing read is kept. One case is added: names, vector length and slice bounds have to agree, which is the arithmetic that catches a block emitting the wrong number of zeros on a short generation. |
| `tests/test_path_signature.py` | `tests/test_path_signature.py` | All 92 cases, unchanged but for the `feature_map` import path. |
| `tests/test_lean_loading.py` | `tests/test_lean_loading.py` | Unchanged but for the two per-model layer fields the configuration now requires; no enabled family reads them. |
| `tests/test_dsv2_moe_pipeline.py` | `tests/test_dsv2_moe_pipeline.py` | Unchanged; the pipeline call moved into a helper so the test asserts and returns nothing. |
| — | `tests/test_extraction_purity.py` | New. The purity guard the manifest ruled: the anchor's third-party surface is numpy, scipy and pydantic, inside the package it reaches configuration and nothing else, and no family's import closure reaches a model runtime. |

## Extraction — the capture layer

The specification says what a signature's numbers are; this is what produces the
tensors they are computed from. It is the only part of the instrument that needs a
model runtime, which is why `torch` and `transformers` become dependencies here and
why the numeric anchor's purity guard becomes load-bearing rather than incidental:
torch is now installed, so a probe finding it absent from the anchor's import graph
found a property of the graph.

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/model_loader.py` | `anamnesis/extraction/model_loader.py` | Logic-identical. Two adaptations. `load_model()` now requires its `ModelConfig` — the old default constructed an empty one, which the configuration split no longer permits, since architecture fields have no defaults. And `sampled_layers` defaults through the new `default_sampled_layers()`, which reads the layer plan off the config's preset row instead of the 3B layer set the old default named; a config with no preset is asked for the layers explicitly rather than given another model's. The attention-implementation requirement is read from the config and asserted nowhere else: the configuration refuses a kernel that returns no attention weights, and a test pins that the loader has no literal of its own. |
| `anamnesis/extraction/streaming_generate.py` | `anamnesis/extraction/streaming_generate.py` | Byte-identical. |
| `anamnesis/extraction/generation_runner.py` | `anamnesis/extraction/generation_runner.py` | Logic-identical through the generate path and its alignment. Two changes. The mode prompts arrive as `RUN4_MODES` from `anamnesis.modes.run4_modes`, since configuration no longer re-exports them. And the pass now writes the run's `replay_manifest.json` beside its `metadata.json`, through `replay/manifest.py` — see the replay section for why. The realized token ids ride the in-memory metadata to the manifest writer and are stripped on the way to disk, so both metadata schemas are byte-for-byte what every banked run already carries. |

### The replay core

⚑ SEAM 1, resolved as the desk leant: the four determinism-and-intervention modules
group under `replay/`. They are one capability read in one order — what a run can be
replayed over, the plain teacher-forced pass, exact edits to a cache, and a pass
against an edited one — and the flat names said so only by prefix.

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/replay_extract.py` | `anamnesis/extraction/replay/extract.py` | Byte-identical but for one docstring cross-reference that follows the module's own move. |
| `anamnesis/extraction/replay_cached.py` | `anamnesis/extraction/replay/cached.py` | Byte-identical but for its `cache_surgery` import path and one docstring cross-reference. |
| `anamnesis/extraction/cache_surgery.py` | `anamnesis/extraction/replay/cache_surgery.py` | Logic-identical, same line count. One comment lost a trailing edit date, which is a date on an edit rather than on evidence. |
| `anamnesis/extraction/replay_manifest.py` | `anamnesis/extraction/replay/manifest.py` | **Ported, and given the consumer it had lost.** The module had zero Python import sites in the frozen record: every grep hit was the `replay_manifest.json` file it writes. It is not dead — reconstruction from a text-only bank is a capability nothing else has, and the steering golden path names it — but its *schema* was written inline in two identical script bodies and read as a raw dictionary in about forty more. So the schema is now typed and homed here (`ReplayEntry`, `FlaggedGeneration`, `ReplayManifest`, plus `entry_from_ids` / `write_replay_manifest` / `load_replay_manifest`), and `generation_runner` writes every run's manifest through it. `build_replay_manifest` returns the typed manifest instead of a bare dictionary; the JSON it writes is unchanged, key for key and order for order, and the schema validates against all 10,140 entries of the 29 banked manifests. Wiring the writer into the generation pass also closes the hole reconstruction exists to patch: a run made here records its realized token ids, so it never needs recovering. |
| — | `anamnesis/extraction/replay/__init__.py` | New. Documents the four modules and imports none of them, so the manifest stays readable where no model runtime is installed. |

### Tests

Everything in the capture layer that a synthetic decoder can establish is established
with one: `tests/synthetic_runtime.py` is a small working transformer with real
projection modules, so the hooks under test are the hooks that run, the grouped-query
reshape is exercised at widths where the query and key/value head counts differ, and
the replay alignment is checked against a forward pass rather than against a mock. Its
weights are random and it makes no numerical claim — the numbers belong to the anchor,
and the agreement of a real model's two paths belongs to
`tests/test_runtime_on_a_real_checkpoint.py`, whose cases skip with a named reason
until a checkpoint is reachable.

| old path | new home | notes |
|---|---|---|
| `tests/test_cache_surgery_turnkeep.py` | `tests/test_cache_surgery_turnkeep.py` | Byte-identical but for the import path. |
| `tests/test_rope_theta_14e.py` | `tests/test_rope_theta_14e.py` | The four cases unchanged but for the import path, plus the two wrapper-aware `inv_freq_from_config` cases absorbed from `tests/test_tool_fixes_2026_07_18.py`. |
| `tests/test_tool_fixes_2026_07_18.py` | *split* | Two of its six cases test `inv_freq_from_config` and moved to `tests/test_rope_theta_14e.py`, where the rest of that gate is tested. The other four pin analysis scripts that are not part of this port and stay in `anamnesis-pl` with them. |
| `tests/test_gpu_replay.py` · `tests/test_gpu_batch.py` | *pending* | Lane tests: they port with the GPU lane, under the "together or not at all" rule. |
| — | `tests/synthetic_runtime.py` | New. The tiny transformer the capture-layer tests run against. |
| — | `tests/test_model_loader.py` | New. Layer resolution across architectures including the multimodal nesting, the hook reshapes under grouped-query attention, the prefill skip and the hook lifecycle, and the activation-write path: a zero dose reproduces the unperturbed forward exactly, positional gating injects at exactly the named absolute positions, and a bounded write on an ambiguous single-token step raises rather than guessing. |
| — | `tests/test_streaming_generate.py` | New. The loop's alignment contract, its determinism under a seed, end-of-sequence handling, and that hooks registered on the modules fire through it — which is what lets it replace the framework's loop without touching the capture surface. The calibration variant's accumulation is checked per absolute position. |
| — | `tests/test_generation_runner.py` | New. The spec plan and the seed namespace; the generate path's three off-by-one hazards asserted by value, not by shape; and the artifacts — that the realized ids reach the manifest and neither metadata schema. |
| — | `tests/test_replay_extract.py` | New. The replay alignment contract end to end on a running forward, both plain and against an injected cache, including over a surgically evicted one; the splits replay cannot serve; and that two replays of one sequence agree exactly. |
| — | `tests/test_replay_manifest.py` | New. The schema and its round trip, and every way reconstruction declines to cover a generation: template drift, an empty text, a round trip that will not close, a token count that disagrees with the bank, absent tensors. A declined generation is named with a reason, never dropped. |
| — | `tests/test_cache_surgery_snapshot.py` | New. The rotation's exactness and the check that refuses a scheme where it would not hold; that values are position-free and never touched; the keep geometry's protections; and the value gate on the frequency table, which catches the wrong table where the homomorphism check cannot. |
| — | `tests/test_runtime_on_a_real_checkpoint.py` | New. What only real weights can show: a checkpoint loading with its hooks on the module names its architecture uses, its own rotary buffer matching the reconstruction, and a replay of a generation reproducing it. Skipped with a named reason unless `ANAMNESIS_TEST_MODEL` points at a checkpoint; one case additionally needs an accelerator. The skips are the standing statement of what a box must have. |
