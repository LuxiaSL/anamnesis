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

## Extraction — the primary implementation, and the proof that it agrees

The refounded `extraction/` is three things a reader should be able to tell apart: a
**specification** (`state_extractor`, pure numpy, no torch and no model), a **primary
implementation** (the fast lane), and a **proof that the two agree** on a given
machine. The old tree had all three as flat `gpu_*` siblings of the specification they
are defined against, which said nothing about which was which.

So the lane groups under `fast/` and the harness under `equivalence/`. The names drop
the `gpu_` prefix because the prefix named a device rather than a role, and the lane
runs on whatever device it is handed — a CPU included, which is what lets its agreement
with the specification be a test rather than a claim. Nothing in the grouping is a
fork: the lane imports the families' naming and slicing and derives its whole output
schema from the canonical pipeline, so the families are load-bearing for the primary
path.

`replay_config` sits above both lanes rather than beside either. It declares which
features a replay computes, and an arithmetic backend is a way of computing features,
not a licence to redefine which exist.

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/gpu_features.py` | `anamnesis/extraction/fast/features.py` | The lane entry point, logic-identical. `GpuFeatureLane`, `GpuFeatureResult` and `replay_span` / `replay_batch` keep their names, which the record cites. Two adaptations: the module docstring now states the scope and the refusals in the present tense, and the self-hashing source list names the four renamed siblings. That list feeds `lane_id`, so a lane id computed here differs from one computed in the frozen repository — as it should, since a lane id is a digest of the exact code that produced a vector. |
| `anamnesis/extraction/gpu_ops.py` | `anamnesis/extraction/fast/ops.py` | Byte-identical but for import paths. The float32→float64→float32 boundaries and the population standard deviation are the specification's, deliberately, and `FeatureCollector.finish` refuses a vector whose names are not exactly the declared schema. |
| `anamnesis/extraction/gpu_attention.py` | `anamnesis/extraction/fast/attention.py` | Byte-identical but for import paths. |
| `anamnesis/extraction/gpu_families.py` | `anamnesis/extraction/fast/families.py` | Byte-identical but for import paths, including the named host-transfer exception for the top-k Jaccard feature, whose definition includes numpy's argsort tie order. |
| `anamnesis/extraction/gpu_schema.py` | `anamnesis/extraction/fast/schema.py` | Byte-identical but for import paths. `resolve_gpu_schema` keeps its name. |
| `anamnesis/extraction/batch_layout.py` | `anamnesis/extraction/fast/batch_layout.py` | Byte-identical. It moves under `fast/` because ragged packing exists for the lane's throughput; nothing else reads it. |
| `anamnesis/extraction/fidelity.py` | `anamnesis/extraction/equivalence/fidelity.py` | Byte-identical but for one docstring sentence that carried the date of the ruling rather than the ruling. |
| `anamnesis/extraction/path_floor.py` | `anamnesis/extraction/equivalence/path_floor.py` | Byte-identical but for import paths. |
| `anamnesis/extraction/replay_config.py` | `anamnesis/extraction/replay_config.py` | Byte-identical, at the same path. |
| `anamnesis/analysis/lane_guard.py` | `anamnesis/analysis/lane_guard.py` | Byte-identical, at the same path, and it stays in `analysis/`: lane guarding is a read-side gate on scientific inputs. It ports with the lane under the "together or not at all" rule — a primary lane shipped without the guard that keeps its outputs from being joined to another lane's ships the confound. Its golden-path consumer is the gauntlet's signature loader, which arrives with the gauntlet. |
| — | `anamnesis/extraction/fast/__init__.py` | New. Documents the six modules, states that the lane reuses the family definitions rather than copying them, and imports none of them so the layout arithmetic is readable without torch. |
| — | `anamnesis/extraction/equivalence/__init__.py` | New. Documents the two modules and states the doctrine they exist for: agreement is a property of a box, different hardware gives different numbers, and outputs from different boxes or lanes must not be joined inside one contrast. |
| — | `anamnesis/analysis/__init__.py` | New. Documents why the lane guard is analysis rather than extraction. |

### The entry points

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/run_gpu_replay.py` | `anamnesis/scripts/run_gpu_replay.py` | **Ported as qualified-primary.** The `--experimental-feature-only` flag is gone: the configuration it demanded acknowledgement for — dense 3B/8B Llama, one full teacher-forced pass, the complete battery, one device — is the configuration the lane tests and the equivalence suite cover, so a covered command line runs without ceremony. The fence stays exactly where the qualification stops: adapters, activation interventions and batched submission are not arguments here, and a command line naming one is rejected rather than reinterpreted as the covered case. The docstring states that boundary in the present tense, as what is and is not covered, and names `qualify_box.py` as where agreement is measured. It also gains a local `load_calibration`, because the calibration reader it imported belongs to a script that ports later; when that script lands the two must converge on one definition rather than drift. |
| — | `anamnesis/scripts/qualify_box.py` | New, and the golden path the equivalence harness existed without. It replays a small sample of a banked run's spans twice through the lane and once through the anchor on *this* machine, prices each row's path bound from the first incremental step, hands all of it to `fidelity.verify_vectors`, and prints three things a user can act on: whether the box reproduces its own vectors, whether its two paths agree, and the lane identity its outputs will carry. The ruler is unit-scaled and says so, because a standardizing ruler belongs to a cohort and a cohort is a deployment's own. A pass is not a certification: certification belongs to deployments, never to this repository. This closes the two proof modules' zero-consumer orphaning — `fidelity` and `path_floor` now have an in-package caller. |

### Tests

Eleven lane tests were candidates. Nine port, and every one of them runs on a CPU with
no checkpoint and no accelerator — including the whole agreement proof, which is the
point: the claim that the two extraction paths compute the same features is now a test
that runs wherever the suite runs, on a real `LlamaForCausalLM` with random weights.
The hand-built decoder in `tests/synthetic_runtime.py` cannot serve here, because the
lane reads eager attention weights out of `self_attn`'s forward hook and refuses a
model whose `model_type` is not `llama`.

Where a case's reference came from the measurement campaign in `anamnesis-pl`'s
top-level `scripts/`, the case does not port. The campaign subset is empty by ruling,
and a test whose subject is a frozen measurement harness belongs beside it.

Every ported test grew, by its header: a test in this repository states what the claim
is and why the fixture can establish it. Configuration constructions also gained the
layer plan and band cutoffs explicitly, which the refounded configuration classes
require.

| old path | new home | notes |
|---|---|---|
| `tests/test_gpu_replay.py` | `tests/test_fast_lane_equivalence.py` | Renamed for what it proves, and paired with `test_extraction_equivalence.py`: one checks the specification against its golden master, this one checks the specification against the primary implementation. All 22 cases run on a CPU. It also homes `tiny_loaded`, the real hooked Llama the other lane tests import. |
| `tests/test_gpu_batch.py` | `tests/test_fast_batch.py` | Unchanged but for import paths, configuration fields and its header. |
| `tests/test_gpu_attention.py` | `tests/test_fast_attention.py` | Same. |
| `tests/test_gpu_families.py` | `tests/test_fast_families.py` | Same. |
| `tests/test_batch_layout.py` | `tests/test_batch_layout.py` | Same. |
| `tests/test_fidelity.py` | `tests/test_fidelity.py` | Same; pure numpy, no lane and no device. |
| `tests/test_replay_config.py` | `tests/test_replay_config.py` | Same. |
| `tests/test_path_floor.py` | `tests/test_path_floor.py` | **Two of three cases.** The coordinate-identity and lower-bound cases port. `test_first_forward_matches_full_incremental_replay` does not: its reference is `incremental_raw` from the campaign's `extraction_perf_phase0.py`, loaded by file path. The capability it checked — that a fresh-prefix single incremental forward reproduces the anchor's first-position coordinates — is exercised instead by `qualify_box.py`, which calls `first_incremental_coordinates` on a real model. |
| `tests/test_lane_guard.py` | `tests/test_lane_guard.py` | **Two of six cases.** The two that test the guard port now, with the guard. The four that drive `geometric_trio/data_loader.load_run4` test the loader's use of it, and arrive with the loader. |
| `tests/test_gpu_replay_cli.py` | `tests/test_run_gpu_replay.py` | Renamed after its subject. The acknowledgement-flag case is replaced by its inverse — a covered command line parses — and the three refusal cases are unchanged, so the qualification boundary is still pinned by test rather than by prose. |
| `tests/test_extraction_perf_phase0.py` | *record* | Not ported. Classified as a lane test by module import, but its single case loads `scripts/extraction_perf_phase0.py` by file path and exercises that module's `incremental_raw`. Its subject is the campaign hub, not the lane. |
| `tests/test_verify_batch_schedule.py` | *record* | Not ported. Every case calls `validate_batch_schedule` from the campaign; `batch_layout.pack_spans` appears only inside the fixture, as the digest the campaign's receipts are checked against. There are no batch-layout assertions to separate out. |
| — | `tests/test_qualify_box.py` | New. What the golden path asks for, the refusals that stop a meaningless measurement before a model loads, and the verdict's three distinguishable states with the exit status that follows them. The sentence about not mixing boxes is asserted, not merely written. |

## Analysis — the gauntlet

`unified_runner` named a mechanism (a runner that unifies) rather than the thing it
does. It runs eleven standing analyses over one set of signatures, because a claim
about signatures is usually a claim about several of them agreeing: an accuracy
means one thing beside a clean orthogonality result and another beside a length-only
baseline reaching the same number. It is `gauntlet/` here, by ruling.

Two names in that package both said "load". `geometric_trio/data_loader.py` held
`Run4Data` and the join rules; `unified_runner/data_loading.py` was a 138-line
wrapper that added the generated text and delegated everything else. Side by side
under one package, a stranger could not tell which to call — and the trio's own
mathematics left that package long ago, so its directory dissolves. **They are one
module, `signature_io.py`.** `Run4Data` and `AnalysisData` both live there, the
lane guard runs inside it, and the merged module is smaller than its two donors
(603 lines against 608). Nothing about the merge changes what a load returns.

| old path | new home | notes |
|---|---|---|
| `anamnesis/analysis/unified_runner/__init__.py` | `anamnesis/analysis/gauntlet/__init__.py` | The eleven-section registry, the `importlib` dispatch, the checkpoint and the resume, logic-identical. Three adaptations: the default output directory is `outputs_root()` from the configuration package rather than a path computed from `__file__`, the module docstring says what the gauntlet is for rather than listing its features, and the printed banner names the gauntlet rather than the runner it used to be filed as. `run_full_analysis`, `SECTIONS`, `SECTION_KEYS`, `SECTION_NAMES` and `SECTION_MODELS` keep their names. |
| `anamnesis/analysis/geometric_trio/data_loader.py` + `anamnesis/analysis/unified_runner/data_loading.py` | `anamnesis/analysis/gauntlet/signature_io.py` | **Merged, the F7 resolution.** Every name both donors exported survives: `Run4Data`, `SampleMeta`, `AnalysisData`, `load_run4`, `load_analysis_data`, `check_data_quality`, `TIER_KEYS`, `TIER_GROUPS`, `BASELINE_TIERS`, `ENGINEERED_TIERS`. The `SIGNATURE_DIR` module constant becomes `default_signature_dir()`, so the Phase-0 root is read when asked rather than frozen at import — the same call-time rule the configuration package holds to, and what lets a test redirect the root. `load_run4`'s first argument defaults to `None` and resolves to that function. The lane-guard import moves from inside the function to the module header, unchanged in effect. |
| `anamnesis/analysis/geometric_trio/__init__.py` | *record* | The directory dissolves: an empty `__init__` for a package whose mathematics went elsewhere. |
| `anamnesis/analysis/unified_runner/results_schema.py` | `anamnesis/analysis/gauntlet/schemas/` | **Split per section**, one module each for the eleven sections plus the composite, class bodies unchanged. `base.py` holds the one shared `model_config` and the two rules that follow from `extra="forbid"`; the package `__init__` re-exports all 81 models, so `from ...schemas import X` reaches every name the flat module exported. The split costs lines against a single file (1,731 across fourteen files, against 1,409) and buys the property the section layout already had everywhere else: a section's schema, its runner and its `--skip` number move together. |
| `anamnesis/analysis/unified_runner/classification.py` | `anamnesis/analysis/gauntlet/classification.py` | Import paths only, plus one comment that carried the date of a decision rather than of evidence. |
| `anamnesis/analysis/unified_runner/tier_ablation.py` | `anamnesis/analysis/gauntlet/tier_ablation.py` | Import paths, plus one necessary adaptation: `LogisticRegression(multi_class="multinomial")` no longer constructs under scikit-learn 1.7 and later, which removed the argument after making multinomial the only multiclass fit its solvers perform. The argument is dropped and the behaviour stated in a comment; the fit is the same fit. The file keeps its name — the tier-vocabulary rename is a later sweep. |
| `anamnesis/analysis/unified_runner/{geometry,clustering,contrastive,semantic,integrity,scorecard,utils}.py` | `anamnesis/analysis/gauntlet/` | Import paths only. |
| — | `anamnesis/analysis/gauntlet/schemas/base.py` | New. The shared `_FORBID`, and the two backward-compatibility rules that follow from it, stated once where every section module reads them. |

### The battery

The metrology layer, which answers the question prior to every arm: how large a
difference has to be before it counts, and how many samples it would take to see
one. It ports whole and as-is — twelve modules, one adaptation, which is that
`feature_map` now lives at the package root.

| old path | new home | notes |
|---|---|---|
| `anamnesis/analysis/battery/floors.py` | `anamnesis/analysis/battery/floors.py` | The floors and the n-min law. Byte-identical but for the `feature_map` import path and one comment reflowed so its "observed" sits on the line with its date. |
| `anamnesis/analysis/battery/{__init__,manifest,stats,deltas,magnitude,channel,decomp,dissoc,gates,report,text_decode}.py` | same paths under `anamnesis/analysis/battery/` | Byte-identical. `decomp.decompose` and `dissoc.dissociation_row` still raise `NotImplementedError`; their containers are typed and their contracts are tested, which is what a Wave-1 stub is. |

### Consolidations

| old path | new home | notes |
|---|---|---|
| `v3_audit/_common.py` · `v3_audit/build_surface_caches.py` · `v3_audit/surface_encoder_floor.py` | `anamnesis/analysis/audit_lib.py` (**C5**) | 648 donor lines to 353. Extracted: `residualize` / `residualize_all`, `subsample_topics`, `load_signature_matrix` with `SignatureMatrix` / `gen_metadata_by_id` / `unwrap_generations`, `sample_positions` / `surface_vector` / `attention_vector` with the bin constants they need, and `preprocess_fold_gpu`. Added: `leak_free_folds`, which is the `GroupKFold(5)`-by-topic-plus-`subsample_topics` idiom the suite repeated in eight files, composed so the folds are identical to that idiom's and a fold count exceeding the topic count is refused rather than guessed at. `SignatureMatrix` is pydantic unconditionally — the donor's dataclass fallback existed for a node environment without pydantic, and this package depends on it. The three donors stay frozen, with their seven import sites, in `anamnesis-pl`. |
| `scripts/vmb_arm_a5_analyze.py` · `scripts/vmb_c3_entropy_replay.py` · `scripts/vmb_14n_hedging_index.py` | `anamnesis/analysis/text_stats.py` (**C4**) | 826 donor lines to 269, ported as capability rather than as a consolidation (ruling 5). The text channel beside the signature: `text_stats` (length, type-token ratio, trigram repetition — the degeneracy check), `entropy_and_nll_over_generation` (donor `_ent_nll_over_gen`), the hedge and definitive lexicons with `HEDGE_RE` and `DEF_RE` under their own names, `marker_rate` / `texts_by_prompt_group` / `group_marker_rates` / `placebo_marker_floor` (donors `_rate` / `_cell_groups` / `_group_rates` / `_placebo_floor`, promoted out of privacy because a library's capability should be callable), and `add_null_ratios` with its zero-denominator guard. The donors' analyzers are superseded and stay in the record; `parse_cell_dir` and `rekey_topic`, which several record-side scripts import from `vmb_arm_a5_analyze`, are cell-directory parsing rather than text statistics and stay with them. |

### Admitted under ruling 7

| old path | new home | notes |
|---|---|---|
| `anamnesis/extraction/feature_families/binding_probe.py` (untracked in `anamnesis-pl`) | `anamnesis/extraction/feature_families/binding_probe.py` | **Admitted.** It duplicates nothing: no other family cuts the prompt finer than a four-way region split, and binding — which attribute went with which entity — is invisible to the rest of the suite by construction. It satisfies the family contract (a `FeatureFamilyResult`, a declared name contract, aligned zeros where there is nothing to read), needs no dependency the package lacks, and is pure numpy. It is registered nowhere, and the family package's docstring now names that category and says why a registry entry would be a promise a run over unlabelled prompts cannot keep. It arrives with the test its admission required. One comment changed: the level-2 quadrature convention is stated as a convention rather than as a temporary stand-in for an unmerged branch. |
| `anamnesis/scripts/analyze_signature_richness.py` (untracked) | *record, for now* | **Not admitted, with grounds.** Its two effective-rank metrics and its depth-slot common basis are real instrument capability, but its primary metric is intrinsic dimension, which `gauntlet/geometry.py` already provides — and the script reimplements the two TwoNN estimators in-file *because* `dadapy` and `skdim` were absent from the environment it was written in, not because the capability was missing. Admitting it as written would put two intrinsic-dimension implementations in one repository, which is the duplication the constitution exists to prevent. It is also a script with a hardcoded seven-bank table addressing a private outputs tree, it reaches into the gauntlet's `utils` for the z-score convention, and it has no test. The decision it waits on is whether the `geometry` extra resolves — if it does not, these dependency-free estimators are the better home for the repository's intrinsic dimension and the script's metrics should land as `analysis/richness.py` with its bank table externalized the way the run registry is. |
| `anamnesis/analysis/fast_lda.py` | *record* | Not ported, by ruling: it ports if and only if the richness readout is admitted and adopts it. |

## Orchestration

Six launcher scripts in the frozen record do one thing between them: partition a
pass across devices, spawn subprocesses, wait, and report. They differ in what they
partition — generation specs, manifest generation ids, a roster of cells — and in
almost nothing else, which is why there were six of them: each new arm copied the
nearest one and edited the middle. C3 keeps one implementation of the mechanics and
leaves the arm protocols in the record, where the `vmb_` prefix says they belong.

The mechanics are separable at all because of a property of the science, and the
package docstring says so rather than leaving it implied: a generation is seeded
from its own coordinates and a replay is teacher-forced from its own banked tokens,
so no output depends on which worker produced it or in what order. Scheduling is
therefore a cost decision, and a launcher is arithmetic.

**Multicell semantics won, per the §3 ruling.** The primary partition is
`plan_multicell` — each worker gets its slice of *every* cell in one job file, so a
roster costs one model load per worker instead of one per cell — and single-cell
fan-out is the same partition over one cell's work. The single-cell guard survives
into `orchestration/gpu.py` with live callers on both the generation and the replay
side, which is what keeps the fast path from being rediscovered as the slow one.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/parallel_replay.py` · `parallel_generate.py` · `vmb_stage0_generate.py` · `vmb_a2_generate.py` · `vmb_a5_gen_multicell.py` · `vmb_a5_replay_multicell.py` · `vmb_stage0_faithfulness.py` | `anamnesis/orchestration/launch.py` | **C3.** 404 lines against 1,387 of donors. What ports is the partition (`round_robin`, `LaunchPlan`, `plan_multicell`), the spawn-and-wait loop (`launch`, `LaunchResult`), the per-worker input files (`write_worker_inputs`), and run assembly (`assemble_run`). Three deliberate changes. The subprocess constructor is a parameter, so the whole fan-out is testable on a machine with no accelerator. Every worker's thread pools are pinned to one rather than inherited-if-set, which is the multicell donors' behaviour and the one that keeps a worker's arithmetic independent of how many siblings it has; the two behaviours differ only for a caller that deliberately raised a limit, and pinning is what the determinism floor was measured under. And `LaunchResult.raise_on_failure()` is available to every path rather than only the replay-multicell one — a launcher that returns zero while a worker died reports a complete pass over an incomplete corpus, which is the one failure a downstream contrast cannot see. |
| `parallel_generate.assemble` · `vmb_stage0_generate.assemble` | `anamnesis/orchestration/launch.py` · `assemble_run` | **The `assemble()` twins, routed through the schema's home.** Both donors carry a byte-identical inline copy of the replay-manifest schema, written from a dict literal. Assembly here builds `ReplayEntry` rows and writes through `manifest_from_entries` / `write_replay_manifest`, and `tests/test_launch.py` pins the output against the donors' literal bytes — key order, separators and all — because "one home for the schema" is a claim about the file, not only about the code. One behaviour changes, and improves: a record whose prompt/generated split does not partition a replayable sequence is **flagged** rather than written as a row. The donors wrote it regardless, which turned into a replay failure with nothing naming its cause; `FlaggedGeneration` is what the manifest already has for saying which of a run's generations it does not cover. |
| `anamnesis/scripts/_gpu.py` · `_single_cell_guard.py` | `anamnesis/orchestration/gpu.py` | The two underscore strays become one module about the launcher-to-worker boundary: `resolve_physical_gpus` (a `--gpus` value is a logical slot into the scheduler's assignment, never a physical index), `worker_environment` (the device plus the thread pins, replacing four inline environment dictionaries across the donors), and `enforce_single_cell_guard` with its constants. The guard's logic and its refusal message are unchanged but for the launcher names they point at. Its acceptance criterion is met by live callers: `run_replay.py` and `run_gen_tokens.py` both arm it on their single-cell fan-out path, so neither half of the capability is orphaned. |
| `anamnesis/scripts/_persistent_workers.py` · `persistent_replay_worker.py` | `anamnesis/orchestration/workers.py` | The file queue and the one job type that runs on it, in one module: `PersistentWorker` and `WorkerFleet` byte-for-byte in behaviour (the spawn environment now comes from `worker_environment`, which builds the same dictionary), plus the replay engine lifted out of the worker script — `ReplayJob` as a typed job payload, `replay_handler` returning the handler and its cleanup together so a worker cannot arm a write it forgets to detach, `dispatch` for the round-robin submit-and-collect, and `signature_mismatches` for the byte comparison. 390 lines against 409 of donors. The parity smoke's *orchestration* moves to `run_persistent_replay.py --parity`; what lives here is the comparison itself, which is therefore testable without a device. |
| `anamnesis/scripts/_a5_common.py` · `load_vector` | `anamnesis/extraction/interventions.py` · `load_vector` | The vector-bank reader follows the injection path it serves. |
| `anamnesis/scripts/_a5_common.py` · `teacher_forced_agreement`, `median_residual_norm` | *deferred* | **Not in this PR, and not into `orchestration/`.** The manifest's §2 line groups all four underscore helpers under orchestration, but these two are measurements, not scheduling: top-1 agreement against a forced continuation is the on-policy gate, and the median residual norm at a site is the unit an absolute dose is priced in. STRUCTURE §1 homes those in `steering/gates.py` and `steering/vectors.py`, and filing them under `orchestration/` would put them where no reader would look *and* leave them with zero callers, which G4 counts as an orphan. They arrive with the steering layer. |
| `vmb_stage0_faithfulness.py` | *partly here, rest deferred* | Its fan-out half is C3's, as the §2 rework directs: a worker per device, given exactly the generation ids scheduled onto that device, which `LaunchPlan` plus an explicit per-worker share expresses. Its protocol half — twenty continuations, ten stratified replays each, the synthetic manifest and the `replay_index.json` that maps a signature back to its device and component — is floors metrology and lands in `battery/stage0.py` per ⚑ SEAM 2, with the analysis layer. Nothing in it needs analysis internals today, so the deferral is about its home, not its dependencies. |
| `vmb_stage0_generate.load_stage0_protocol`, `build_specs`, `VMB_CANONICAL_DATE` · `vmb_a2_generate.conditions`, `build_specs` · `parallel_generate.load_source`, `flatten_topics`, `build_specs` | *record, except the data* | The launchers' spec builders are arm protocol, not launcher mechanics: a stratum table, a fixed condition order, a rule for extending a corpus from its source run's own instrument. The `vmb_` namespace does not exist in this repository, and the protocols' seed layouts are what make their banked corpora what they are. The stratum table itself *did* port, as data, inside `prompts/prompt_sets.json`; the reader arrives with the battery that needs it. A pass here supplies its specs as a file, which is what makes the launcher indifferent to which protocol produced them. |

### Capability lifted out of the entry points

The shims rule is that a script holds no logic another reader needs. Enforcing it
moved five things out of script bodies and into the package, and each landed where a
stranger would look for it rather than where it happened to be.

| old path | new home | notes |
|---|---|---|
| `run_replay_extraction._load_calibration` · `run_gpu_replay.load_calibration` | `anamnesis/extraction/calibration.py` | **The convergence PR 4 recorded as owed.** One reader for the two artifacts every feature-computing path depends on. Both banked shapes of the basis are accepted; absence is returned as `None` rather than raised, because a pass that does not touch the residual basis must still run while one that needs corrected features refuses for itself — and a missing positional means is warned about at the point of reading, since silently uncorrected features under the name of corrected ones is the failure this exists to prevent. The **per-layer** basis a corrected refit produces is deliberately *not* read here: it is consumed only by the offline recompute, and `feature_pipeline._load_pca_model` reads it there — the module the feature receipts pin, and therefore the module that keeps its own reader. Two readers remain for two shapes; the second is named rather than merged. |
| `run_replay_extraction._replay_cell` · its configs-and-load preamble | `anamnesis/extraction/replay/cell.py` | The production replay loop, and `load_replay_model` for the capture surface it runs against (keys, values, queries and attention outputs at every layer; families still consume the preset's sampled layers, so the vector is unchanged and the extra layers are banked raw). Logic transcribed exactly, including the per-generation start position, the gating check against the generated span, the resume-by-signature skip, and the `extraction_version` and routing-version riders written onto every signature. It lives here because three callers must run *the same* loop — the one-shot command, the load-once roster walk, and the resident worker — which is the whole basis on which the faster two are allowed to exist. |
| `run_gen_tokens._generate_specs` | `anamnesis/extraction/token_generation.py` | The token-banking generation loop, with `DecodePolicy` making the sampling settings a typed object rather than a dictionary passed by hand. One behaviour is pinned by test rather than by comment: at a repetition penalty of exactly one the argument is withheld from the sampler, so the default path's logits processors are the ones every banked corpus was produced under. |
| `run_gen_tokens._setup_injection`, `_setup_perturbation` · `run_replay_extraction._resolve_injection`, `_setup_replay_injection` · the two inline perturbation constructions | `anamnesis/extraction/interventions.py` | Four near-duplicate functions across two scripts, doing the same three steps on both sides of the instrument: read the description, build the spec, arm the hook. `InjectionSpec` and `PerturbationSpec` are typed; `injection_fields` writes the five travelling key names down once; `InjectionSpec.from_run_metadata` is what makes a steered generation and its replay provably the same intervention rather than two similar ones. `check_injection_gating` keeps both callers' expectations — a replay expects one write per generated token, a generation one fewer, because its first token comes from the last prompt position before the write — as an argument rather than as two copies of a conditional. |
| `run_gpu_replay.file_sha` | `anamnesis/provenance.py` | `qualify_box.py` imported it from `run_gpu_replay`, which is the shims rule's first assertion failing on the tree as it stood. It moves to the package root because both layers read it: extraction stamps what it produced, and an analysis deciding whether two banks may be joined reads the stamps back. `digest_of_shas` absorbs the identical inline construction both scripts used for the calibration digest, so a lane identity is computed from one definition; the bytes it produces are unchanged, which is why no lane id moves. |
| — | `generation_runner.trim_per_mode`, `build_prompt_swap_specs` | New, beside `build_generation_specs`, which is the capability they extend. `run_extraction` built both inline. The swap specs carry one forced change: the record set `mode_idx` to `-1` as a sentinel for "not a core mode", and the ported `GenerationSpec` requires a non-negative index, so the field now holds the pair's position in the swap set. The mode *name* was always what the label meant — `swap_<pair>` — and `vmb_a2_generate` indexed its swap conditions positionally for the same reason. The seed is untouched: it still derives from the frozen namespace slot, so a swap corpus regenerated here gets the seeds the record's would. ⚑ **Flagged for the operator** as the one data-visible adaptation in this PR. |

### The entry points

Five commands were asked for, and a sixth is here because the M7 primitive needs a
front door: a `workers.py` reachable only from a test would be a capability with no
way to run it. Every one of them is argparse, a configuration derived from a preset
row, and a call into the package — no numerics, and nothing a sibling imports.

The renames are recorded because the record cites the old names: `run_replay` for
`run_replay_extraction`, `run_recompute` for `run_recompute_v3`, `run_calibration`
for `run_8b_calibration`. The `parallel_` prefix does not survive, because fanning
out is now a flag on the command that does the work rather than a separate script
that spawns it.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/run_extraction.py` | `anamnesis/scripts/run_extraction.py` | 176 lines against 376. The preset-field copying is gone: `ExperimentConfig.from_preset()` derives the model, decode, extraction and calibration configs together, and the path constants the old script computed at import are the configuration package's. `--n-samples` trimming and the swap block come from `generation_runner`. The layer-count assertion against the loaded checkpoint becomes a refusal with a reason, since an `assert` is not a guard under optimisation. |
| `anamnesis/scripts/run_gen_tokens.py` + `parallel_generate.py` + `vmb_a5_gen_multicell.py` | `anamnesis/scripts/run_gen_tokens.py` | One command, four modes: a worker over a spec file, a fan-out over devices, a roster under one model load, and assembly. Merging the launcher into the worker is what removes the argv-mirroring problem — a launcher in a separate file has to restate every flag its worker accepts, which is how the donors' flag sets drifted apart. It re-invokes itself instead. The thread-pool pin that the donor set before importing torch is kept for a hand-run worker and is also in the environment a launcher hands out, which is earlier and therefore better. |
| `anamnesis/scripts/run_replay_extraction.py` + `parallel_replay.py` + `vmb_a5_replay_multicell.py` | `anamnesis/scripts/run_replay.py` | **Renamed**, and the same merge: one cell, a roster under one model load, either of those fanned out over devices. The single-cell guard fires on the fan-out path, pointing at `--cells-json`. The `--no-tier3` flag becomes `--no-pca`, which names the thing rather than a retired tier bin; the artifact it switches off is the same one. |
| `anamnesis/scripts/persistent_replay_worker.py` | `anamnesis/scripts/run_persistent_replay.py` | **Renamed** to sit with the other `run_` commands. Worker, driver and the parity gate, each a few lines of dispatch over `orchestration.workers`. The gate is renamed from a smoke to what it is — `--parity` prints `PERSISTENT_REPLAY_PARITY: PASS` or exits non-zero with the differing files named — and its second leg is `run_replay.py` rather than `parallel_replay`. |
| `anamnesis/scripts/run_recompute_v3.py` | `anamnesis/scripts/run_recompute.py` | **Renamed**, dropping a version from a command name. The family set is the one the banked vectors carry, stated with the reason it is *not* the replay surface's — that surface grew families the banked vectors do not contain, so a recompute inheriting it would widen its vector and stop being a recompute. `tests/test_entry_points.py` pins both configs field-for-field against the record's own construction, which is the G1 guard for this command: `feature_pipeline.py` is untouched and remains the single implementation. |
| `anamnesis/scripts/run_8b_calibration.py` + `run_corrected_pca.py` | `anamnesis/scripts/run_calibration.py` | **The §2 merge: one preset-parameterized command, with the corrected fit absorbed as its fix rather than as a sibling.** The default is the per-layer basis over positionally corrected states, because fitting on raw states and applying to corrected ones is a fit-and-apply mismatch; `--pooled` is the earlier shape, kept for reproducing the banks computed under it. Both artifacts come out of one pass now, where the record needed two scripts, and existing positional means are **reused rather than refitted** — the correction a bank was computed under must not move underneath it, which is the constraint `run_corrected_pca` stated in prose and enforced by not touching the file. `--refit-means` is the explicit override. The two donors' sampling differed in one detail and both are kept as they are: a pooled fit takes its three sample points as they fall, including when a short generation makes two of them the same step, and the per-layer fit takes the distinct ones. The legacy no-`--model` default is gone, since every preset row now carries its own calibration directory. |
| `anamnesis/scripts/run_gpu_replay.py` | `anamnesis/scripts/run_gpu_replay.py` | Amended, not re-ported: its local `load_calibration` and `file_sha` are now the package's, which is the convergence its own port-map row said was owed. The inline calibration digest becomes `digest_of_shas` over the same dictionary, so the bytes and therefore every lane id are unchanged. |
| `anamnesis/scripts/qualify_box.py` | `anamnesis/scripts/qualify_box.py` | Amended: it reaches the two helpers in the package instead of in a sibling script, which is the shims rule's first assertion satisfied on the tree as a whole. |
| `anamnesis/scripts/run_replay_multickpt.py` | *deferred* | Named as a PORT item in §2 and not in this PR's scope. Its `_build_configs`, which `persistent_replay_worker` imported across script boundaries, is `replay/cell.load_replay_model` here, so the coupling the shims rule forbids is already gone; what remains for it is the checkpoint-series iteration itself. |

### Tests

| old path | new home | notes |
|---|---|---|
| `tests/test_lane_guard.py` | `tests/test_lane_guard.py` | **The four deferred cases land**, now driving `signature_io.load_run4`: a lane preserved and a conflicting one refused, an untagged addon refused against a tagged lane, the legacy all-untagged read still available, and a tagged directory that cannot silently skip a file with no metadata. |
| — | `tests/test_signature_io.py` | New. The join rules rather than the arithmetic: tier discovery, the core-only filter and its exclusion of swap modes, addon merging and the two ways it must refuse, the text half, and a read of the banked `8b_fat_01` signatures that skips with a named reason where that bank is absent. |
| — | `tests/test_gauntlet_schemas.py` | New. That the split did not lose a name: every section module's models are re-exported, `__all__` and the modules agree both ways, the composite's field types are the sections' own models, `extra="forbid"` is live everywhere, and the two reshaping models round-trip their on-disk spelling. |
| — | `tests/test_gauntlet_run.py` | New. A real pass over a synthetic corpus with six of eleven sections running: dispatch, text loaded only when a section needs it, checkpoint, resume with rehydration, an error stub not counting as completed, and the composite validating. |
| — | `tests/test_gauntlet_classification.py` | New. Section 2's parts at small parameters, because its key-tier sweep is minutes of CPU per tier: the grouped-CV default, the permutation p-value's `1/(N+1)` floor, BH-FDR's monotonicity, and the readouts that name their own conditions. |
| — | `tests/test_battery.py` | New, and the battery's first tests in either repository. The stamp gate raising rather than warning, the law floored at permutation resolution, paired deltas drawn within a prompt class only, the cell masks, the channel split's two extremes, the manifest refusing a duplicate cell, and the Wave-1 stubs saying they are stubs. |
| — | `tests/test_audit_lib.py` | New. Each control against the property it exists for: what residualization removes and that the split version fits on train only, that folds hold out whole topics, that feature order is pinned and a missing feature fills with zero, that a surface vector's width does not depend on generation length, and that the Gram reduction preserves the inner products it claims to. |
| — | `tests/test_text_stats.py` | New. That a looping generation scores as degenerate, that the lexicons are word-bounded, that the prompt group is the unit, that the placebo floor is reproducible, and that a ratio to a near-zero denominator is suppressed in favour of the band readout. |
| — | `tests/test_binding_probe.py` | New, and the condition of the family's admission. Chiefly alignment, which is what a within-pair contrast rests on: names a function of the arguments alone, aligned zeros for a short generation and for an uncaptured layer, every refusal of a bad span table, and the increment-permutation null leaving level 1 identical while destroying level 2. |
| `tests/test_single_cell_guard.py` | `tests/test_single_cell_guard.py` | All twelve cases, unchanged in substance, with the launcher names and the multicell pointer following their move. Every case runs on a CPU against an isolated guard directory and an explicit job context. |
| `tests/test_persistent_workers.py` | `tests/test_persistent_workers.py` | Its eight cases plus one for `dispatch`, which is new here: the driver's total is the workers' own reported counts rather than a re-derivation, so a short pass cannot read as a full one. |
| `tests/test_multicell_bitwise.py` | *record* | Not ported. Both its cases shell out to `vmb_a5_multicell_smoke.py` and `vmb_a5_replay_multicell_smoke.py`, arm scripts that stay in the record, and both need a device, a checkpoint and a banked cell. The capability it gates — that a load-once path is byte-identical to the reload-per-cell one — is `run_persistent_replay.py --parity` over `workers.signature_mismatches`, whose comparison half is tested here without a device. |
| — | `tests/test_scripts_are_shims.py` | **New, and ⚑ SEAM 4 adopted.** Four assertions over the parsed sources: no script imports a script; no package module imports a script; a function a script defines is read by that script and its own test and nowhere else; and two scripts do not define the same name apart from the command-line roles. The third is the thinness proxy and it is a judgement — a diagnostic command's own verdict rendering is legitimately its own, while a function a second reader needs is capability — stated in the module's header alongside what the test cannot catch, which is paraphrase under two different names. It found the `qualify_box` → `run_gpu_replay` edge on the tree as it stood, and it found two of this PR's own helper collisions. |
| — | `tests/test_launch.py` | New. The partition, the environment each worker is handed, waiting for every worker after one fails, the multicell grouping, and assembly — all with a recording stand-in for the subprocess constructor, so none of it needs a device. It holds the consolidation's proof: the assembled manifest's bytes against the donors' literal form. |
| — | `tests/test_calibration_io.py` | New. What the one reader does with both banked shapes, with absence, and with the per-layer shape it refuses rather than silently reduces. |
| — | `tests/test_interventions.py` | New. The refusals, which are the substance: a partial spec, a bank missing a key, a write whose gating did not fire or fired the wrong number of times, and an unsteered cell asked to report a dose. |
| — | `tests/test_entry_points.py` | New. Each command's refusals before a model loads, the decode policy coming from the preset row rather than from a literal, and the recompute configs pinned field-for-field against the record's construction. |

## Steering — construction, screens, gates and readouts

Interventions are half the instrument, so the write side is in the repository
whole (manifest ruling 1): a contributor gets the read and the write loop in one
place. The thirteen donor scripts were thirteen experiments, and what they hold
between them is one vocabulary — a construction, a screen that says whether it
can be injected, a gate that says whether a cell is valid, and a readout that
says what it did. Each is now one module with a caller on golden path 6.

Everything here is **per-model**: a vector is built in one checkpoint's residual
basis, whitened by its covariance, dosed in its median residual norm and injected
at a site chosen by its own layer sweep, and none of those four transports.
Carrying a vector across models is a transport problem with its own instrument,
which is metabasis's to own, so this package offers no path that pretends
otherwise.

### Construction (C1)

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_a5_build_vectors.py` (**primary**) · `vmb_a5_whiten_steer_build.py` · `annex_band_pass.py` · `annex_perp_vectors.py` · `c5_sweep_and_build.py` | `anamnesis/steering/vectors.py` (**C1**) | 932 donor lines to 921. The primary donor's conventions win where donors disagree, and the disagreements are named rather than merged. **Mean difference:** `paired_mean_difference` is the primary's law (mean of per-pair, topic-matched differences, which is what the banked vectors are) and `mean_difference` is the unpaired difference of condition means; they coincide only on a complete balanced pairing, so both are callable and neither is a default the other hides behind. **Covariance:** `pooled_within_class_covariance` is the whitened-build donor's estimator and `target_covariance` the install donor's Σ-from-the-injection-target; the caller names one, because the choice is the science. **Median residual norm:** the primary's pooled-over-positions median wins over the whitened donor's median-of-per-generation-medians, which weights a short generation like a long one. **Degenerate-case conventions:** `heldout_cohens_d` takes `sd_floor` and `min_direction_norm` explicitly, because the sweep donor divides by `max(sd, 1e-8)` where the separation donor returns zero, and the two differ only where the statistic degenerates. Also here: `Spectrum` (descending on construction, from arrays, a covariance, observation rows or a banked npz), `band_pass` and its anatomy, `orthogonalize` with the hard cosine check, `random_unit_vectors` and `random_band_vector`, `dose_alpha`, `half_split_sweep`, the two banked builders, the install-lever builder, and bank IO under the frozen `a5_vectors.npz` / `a5_vectors_stamps.json` names. The five donors stay frozen in `anamnesis-pl`, where the banks they built were built. |
| `scripts/c5_ledoit_wolf_gpu.py` | *record, for this PR* | The GPU Ledoit–Wolf estimator the install donor calls. `ledoit_wolf_covariance` uses the scikit-learn estimator it was validated against, which is the reference of record and is what a CPU-only contributor can run; the GPU version is a stranded generic primitive on the manifest's own list and lands with whichever PR gives it a consumer. |
| kotodama's `steer_site_sweep.py` · `steer_install_build.py` | *record, another repository* | Convention donors only, and they are kotodama's. Their laws arrived here through `c5_sweep_and_build`, which is the anamnesis-side port of record: per-prompt averaging first, half-splits over prompts, Σ from the injection target, and the band-matched null. |

### Screens

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_a5_covariance_screen.py` · `vmb_a5_band_mass_readout.py` · `vmb_a5_layer_separation.py` · `vmb_a5_routing_separation.py` | `anamnesis/steering/screens.py` | 803 donor lines to 482. `screen_vector` and `screen_bank` are the Mahalanobis-and-eigenmass screen, promoted per the manifest's own reading of the donor's docstring: it is a statistic for any candidate vector at any site, not an arm's tool. `band_mass` is the band-health readout, conventions verbatim. `deformation_curve` is the graded-Goodhart leg, and it takes captured arrays rather than a model, so the dose/α² law is testable without a checkpoint. The two separation companions collapse into one pair of metrics (`two_fold_heldout_d`, `centroid_ratio`) over one row builder (`axis_separation_rows`), which is what makes the residual-stream and expert-routing curves comparable rather than nearly comparable — the donors carried two copies of the same arithmetic with different degenerate-case thresholds. `capture_site_outputs` uses the package's `attach_residual_write` instead of the donor's private pre-hook: the same math at the same positions, through one implementation. |

### Gates

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_a5_onpolicy_gate.py` · `vmb_a5_upstream_zero.py` · `annex_null.py` · `annex_shape_audit.py` | `anamnesis/steering/gates.py` | 697 donor lines to 523, and the manifest's **PORT-WITH-REWORK** carried out. The two annex gates graduate properly: `assert_against_own_null` and `audit_axis` / `audit_axes` take arrays and return verdicts, with no corpus loader anywhere in the module. The donors imported `annex_corpus` and `annex_spectrum.pca`, which self-declare as not buildable upon, only to reach a self-test and a principal-components fit — so the loaders stay in the record and the checks now run on any corpus, with the caller supplying the projections. `on_policy_gate` is the ≥0.85 matched-token bar with its `alpha = 0` baseline, over one write handle per site whose alpha is mutated per cell. `upstream_zero_check` is the exactly-zero standing check, keyed on the *deepest* layer a feature name reads so a cross-layer feature cannot be called upstream on the strength of its shallower index. |
| `anamnesis/scripts/_a5_common.py` | `anamnesis/steering/gates.py` · `anamnesis/steering/vectors.py` | One of its three helpers is a gate primitive and it lands as `gates.teacher_forced_agreement`; the dose currency it also held is `vectors.median_row_norm` and `vectors.capture_median_residual_norms`, and the bank reader is `vectors.load_vector`. The file is on the manifest's list of four underscore helpers that become `orchestration/`, so this is a private copy pending convergence with that module when it lands — nothing else imports it, and the function is eleven lines of arithmetic over one forward pass. |

### Readouts

Filed under `steering/` per the STRUCTURE seam-3 ruling: these read
*interventions*, where `analysis/` reads runs.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_a5_lever_readout.py` · `vmb_matched_support_efficiency.py` · `vmb_a6_directional_readout.py` · `vmb_partc_contrast.py` · `vmb_identity_histogram.py` | `anamnesis/steering/readouts.py` | 691 donor lines to 600. `lever_readout` is the model-agnostic lever; `matched_support_efficiency` is the same decomposition under the never-pool-nulls-across-supports constraint, and it is **C4's first package consumer** — it imports `analysis.text_stats.text_stats` for the coherence statistics that ride beside every cell, where the donor reached into a superseded analyzer for them. `directional_series`, `seed_floor` and `sign_flip_p` are the checkpoint-series readout; `contrast_fields` returns the frame alongside the fields so the contrast frame is stamped rather than assumed. `expert_usage_histogram` is the identity sidecar, warning included. `parse_cell_name` accepts both banked cell-directory spellings — the signed-dose one and the open-vector-name one the record-side scripts read through `vmb_arm_a5_analyze.parse_cell_dir` — because steered-cell naming is steering's own vocabulary, which is why it lands here rather than travelling with the superseded analyzer that C4 left in the record. |

### The entry point

| old path | new home | notes |
|---|---|---|
| — | `anamnesis/scripts/steer_vectors.py` | New, and not optional: without it C1 lands with no caller at all (staleness F3). Six subcommands, one per leg of golden path 6 up to the judge — `sweep`, `build`, `screen`, `gate`, `null`, `lever` — and it is a shim throughout: argparse, dispatch and JSON writing over package calls, with a test asserting its public surface holds nothing another module would import. The judge leg is `anamnesis/judging/`, which lands separately. |

### Tests

| old path | new home | notes |
|---|---|---|
| — | `tests/test_steering_vectors.py` | New; the donors carried no tests. Each convention against what it exists for: that an ascending eigendecomposition comes back descending, that the two mean-difference laws agree on a balanced pairing and disagree otherwise, that `Σ⁻¹Δ` leaves Δ under anisotropy, that a band member is confined and an amplified residue is visible as a low band mass, that orthogonalization refuses a parallel vector, that a band null stays in its band, that the pooled dose currency is not swayed by one outlying position, that the held-out d is smaller than the in-sample value on noise and that the two degenerate conventions differ only at zero variance, and that a sweep finds a planted layer. |
| — | `tests/test_steering_screens.py` | New. The screen's own claim, that the same unit magnitude costs more in the tail than at the top; that eigenmass entries are fractions; that a flat dose/α² ratio distinguishes a linear response from a super-linear one; and that a routing substrate's sparse layer list and a residual substrate's dense one both produce rows naming their layer and depth. |
| — | `tests/test_steering_gates.py` | New. That a cross-layer feature keys on its deepest layer and a delta feature is upstream one block shallower; that the upstream bar is exact zero, failing at `1e-12`; that the mean-difference null is the top-k trace share and the isotropic null is `k/d`; that the gate raises on a gradient and on a whitened direction rather than returning a number; that **the same vector reads above an isotropic null and inside its own**, which is the failure the gate exists for; and that the shape audit catches a six-row axis that both covariate legs call clean. |
| — | `tests/test_steering_readouts.py` | New. Both cell-name spellings and the refusal to default a non-cell directory; the target/off-target split and its dose invariance; a lever ratio separating a real vector from its controls over synthetic banks; that a band target with no band null reports no ratio rather than borrowing the tail's; that the coherence statistics distinguish a looping generation from a healthy one; and that the matched-control frame cancels exactly the drift the vs-base frame keeps. |
| — | `tests/test_steer_vectors_cli.py` | New. The shim rule as a test, the reachability of all four steering modules from the entry point, and `sweep` run end to end on synthetic banks so the CLI is a caller rather than a declaration. |

## Judging — the behavioural channel

Eight scripts judged banked text by blind two-alternative forced choice, and they
agreed on the paradigm while disagreeing on nearly every convention around it:
three pair constructions, two reply formats, two coherence windows, two ways of
holding the key apart, and two different tables of mode prose. One harness ports,
and where the donors disagree the resolution is recorded here rather than
inherited silently. The eight stay frozen in the record, which is where their
banked tables were produced and where a citation resolves.

**Which blinding pattern won, and why.** The A5 family (`vmb_a5_judge_formality`,
`vmb_a5_judge_socratic`, `vmb_d4_judge_analogical`) built its key dict inside the
same loop that rendered prompts, writing `key.json` beside `results.json`
afterwards: the guarantee rests on the loop's discipline. `c5_blind_judge` wrote
the key to a **separate file from the packet** and stated that law in the receipt,
which makes a slip recoverable rather than invisible. That is the pattern that
won, and it ports one level stronger: the key is separated at the *type*, not
only at the file. `BlindPair` — the only object a prompt is rendered from — has no
field that names a side, so the labels are not in scope where the text is
assembled, and `BlindPacket.write` refuses one path for packet and key.
`tests/test_judging_harness.py` pins it by swapping every key and asserting that
no rendered prompt changes.

**The conventions resolved, one row each.**

| conflict | donors | resolution |
|---|---|---|
| pair construction | A5/d4 enumerate target texts and sample a same-topic distractor; `c5_blind_judge` draws one random pair per prompt with a `--min-chars` filter; `run_2afc_mode_hardening` cycles a distractor *mode* per target index and picks deterministically | The A5/d4 construction is the manifest's primary framing and the shape the steering and census tables were banked under, so it is the core. `c5`'s length filter ports as `min_chars`, a control against a length tell rather than a second construction. The mode-hardening cycle ports as *named distractor sources* cycled across the candidate list, which makes target-versus-any-other and pole-versus-pole one function with one or several sources, and records the source in each key so `by_distractor` survives. |
| position draw | five donors use `rng.random() < 0.5`; `run_2afc_mode_hardening` uses `rng.choice(("A", "B"))` | `rng.random() < 0.5`, the majority and `c5`'s. The consolidated constructor's draw sequence is therefore its own: a *banked* `-pl` table is reproduced by the record's own script, and `verify_against_key` is what refuses a reconstruction that does not match. |
| reconstruct-then-verify vs re-judge the banked items | `vmb_judge_family2` reproduces a construction from the original seed and hard-verifies against the banked key before any call; `vmb_judge_family2_annex` re-judges the banked `pairs.md` items directly, with no rng at all | The annex pattern is the durable one and becomes the harness's answer: a pass **banks a judge-ready packet**, so a second family re-reads the items rather than rebuilding them, and `compare_families` refuses two passes whose packet digests differ. Reconstruction survives as `verify_against_key` for the tables that banked only a key, and `parse_pairs_md` / `write_pairs_md` keep the record's sheet format readable and writable. |
| coherence window | `vmb_a5_judge_socratic` rates the head of the text; `vmb_d4_judge_analogical` rates the tail, on the finding that degeneration reads late | Both available, `window="tail"` the default, and the window is recorded in the result — two cells compared across different windows are not comparable. |
| reply format | `run_2afc_{mode,temperature}_hardening` parse a JSON object with `choice` and `confidence`; the A5 family scans the first text block for a bare letter | Both kept, declared per prompt set, because a parse convention is part of a banked table's identity. The letter scan stays a scan rather than an equality test, which is what keeps a reasoning-leading reply's good judgement. |
| fallback trigger | mode/temperature retry the same model three times on 429/500/529 then fall to the fallback; the A5 family falls to the next model on *any* failure including an unparseable reply | Both: the backend waits out a transient status, and the harness's ladder escalates on an unparseable reply exactly as on an error. A judge that cannot be read has not judged. |
| mode prose | `run_2afc_mode_hardening` and `vmb_a5_judge_socratic` carry different five-mode description tables | Two prompt sets, not one with a flag (`MODE` and `SOCRATIC`). One asks which text followed a mode instruction, the other which text is *more* that mode; their rates are not comparable and the table says so. |

**`c5_blind_judge`'s three controls are core, not options.** The CEIL contrast
becomes `interpret_with_ceiling`, which returns `uninterpretable` rather than
`null` where the ruler has not shown it can separate anything — the asymmetry
stated as code, since a weak judge can hide an effect but cannot manufacture a
positive. The multi-model reader ladder becomes `run_reader_ladder`, where each
grade is a single-model ladder so no fallback can promote a weak reader's answer
into a strong reader's row, and `weakest_pass` is the grade a pass passes at. The
anti-circularity rule becomes `assert_not_circular`, checked against every prompt
set's declared `criterion_source` and refused before a call is made. `c5`'s
non-run rule ports beside them: zero scored pairs is `is_non_run`, an absence of
evidence rather than a null.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/run_2afc_mode_hardening.py` | capability in `anamnesis/judging/harness.py` (**C2**); prompt in `anamnesis/judging/prompts.py` · `MODE` | Donor frozen at `anamnesis-pl`. Contributed: the Wilson interval, the one-sided binomial against chance, the `≥.70` qualification rule, the cycled-distractor protocol, the JSON reply format, the transient-status retry, and the `calls > 0` warning that becomes `is_non_run`. |
| `anamnesis/scripts/run_2afc_temperature_hardening.py` | same (**C2**); prompt · `TEMPERATURE` | Donor frozen. The temperature axis has no pure-mode corpus, so its contrast is hot against same-topic cold; that is a caller's choice of banks here, not a second code path. |
| `anamnesis/scripts/vmb_a5_judge_formality.py` | same (**C2**); prompt · `FORMALITY` | Donor frozen. Contributed the phase discipline that makes concurrency safe: every RNG decision pre-drawn sequentially in the main thread, dispatch concurrent, aggregation last. `_dispatch` keeps it, and a test asserts one worker and sixteen produce the same rows. |
| `anamnesis/scripts/vmb_a5_judge_socratic.py` | same (**C2**); prompts · `SOCRATIC`, `COHERENCE_PROMPT` | Donor frozen. Contributed the coherence gate, the per-topic cap derived from `--n-pairs`, and the generalized mode-shift prompt. Its `SOCRATIC_PROMPT` back-compat alias does not port: it existed for one importer, which is `vmb_judge_family2`, and that importer's capability ports too. |
| `anamnesis/scripts/vmb_d4_judge_analogical.py` | same (**C2**); prompt · `ANALOGICAL` | Donor frozen. Contributed the tail-window coherence read and the frame-fair rule — a steered text pairs against a same-topic rider *from the same run*, same genre, same cap, same sampler — which is a property of the banks a caller hands `draw_pairs` and is stated in `Corpus`'s docstring rather than enforced in code. |
| `anamnesis/scripts/vmb_judge_family2.py` | same (**C2**) · `OpenRouterBackend`, `verify_against_key`, `compare_families` | Donor frozen. Contributed the second judge family, the hard-verify-before-any-call rule, and the pre-named reading that divergence is an instrument finding scoped per contrast and a flipped verdict is stop-and-surface — carried in `AgreementResult.reading`, so it is banked with the number rather than remembered. Its three per-table construction reproductions (`d4`, `d3`, `pg3`) do not port: they reproduce three specific banked tables, which is record work, and the harness's answer to the general problem is to bank the packet. |
| `anamnesis/scripts/vmb_judge_family2_annex.py` | same (**C2**) · `parse_pairs_md`, `packet_from_pairs_md`, `annex_prompt`; prompt · `ANNEX_TEMPLATE` | Donor frozen. Contributed the sheet format and the pattern that won for second passes. Its pair-heading regex is relaxed from `(\d+)` to a non-space token so a packet banked by this harness round-trips; a numbered sheet parses and sorts exactly as before. Its family-1 majority reader (`load_family1`, over `choices.json` and `j*.json`) does not port: those are the shapes of specific banked judge directories. |
| `anamnesis/scripts/c5_blind_judge.py` | same (**C2**) · `interpret_with_ceiling`, `run_reader_ladder`, `assert_not_circular`, `texts_by_prompt`, `AnthropicBackend` | Donor frozen. The strongest donor, and its three controls are core features. Also contributed: the structured-output path with the `effort` knob gated on the models that reject it (a lesson that cost a whole rater arm), the `stop_reason == "refusal"` check before touching content, the confidence clamp on read because a schema's numeric bounds are not enforced, the `min_chars` length-tell filter, and the pair floor. **Its persona prompt does not port** (ruling 4): the Computer #5 model-card prose belongs to that PoC, and it stays with it in the record. |
| `anamnesis/scripts/run_judge_scoring.py` | `anamnesis/judging/likert.py` | The non-2AFC paradigm, ported whole: the five-dimension rubric, the parse with its truncated-`reasoning` repair, purity as intended-minus-mean-of-others, the summary with its confusion matrix, resume by generation id, and the purity-against-centroid-distance correlation. Three adaptations. The judge call goes through the `JudgeBackend` contract, so the same stub that tests the 2AFC harness tests this. The retry stays on **one** model rather than falling to another, which is the opposite of the 2AFC ladder's rule and right for the opposite reason: a Likert number is a reading on one judge's internal scale. And the module-level `JUDGE_MODEL` rebinding trick — the donor reassigned its own module attribute to honour `--model` — is gone; the model is an argument. |
| `anamnesis/scripts/blind_reader.py` | *record* | Not ported. Desk tooling, and it solves a different problem than this layer does: it asks an *isolated* reader a question with none of the project's ambient context, because a reader launched from inside the repository inherits the repository's own notes and is no longer blind. That is a property of an agent harness, not of a pipeline, so the script stays with the desk. The lesson it carries is worth stating where a judge is run, though, and it is: a blind test has to be blind in the environment and not only in the prompt, which is why `run_reader_ladder` addresses named provider models over a plain API rather than sessions. |
| — | `anamnesis/judging/prompts.py` | New home for text that was scattered across eight scripts. Five versioned, provenance-stamped 2AFC sets — `MODE`, `SOCRATIC`, `ANALOGICAL`, `TEMPERATURE`, `FORMALITY` (ruling 4: all five) — plus the coherence rubric, the annex carrier and the Likert rubric. Every string is byte-equal to its donor's, including the two system prompts that are built by f-string in the donor and rebuilt here from a template; the placeholders are substituted with `replace` rather than `format` because two of the prompts contain a literal JSON example, and doubling those braces would make the stored text differ from the donor's. Each set carries `donor`, `version`, `criterion_source` and `axis`; `criterion_source` is what the anti-circularity rule reads. |
| — | `anamnesis/judging/__init__.py` | New. Documents the three modules and the one property that is load-bearing: judging is the only layer that talks to a provider, so `anthropic` and `requests` ship in the `[judge]` extra, their imports are deferred to first use, and naming the judging package pulls neither. API keys are read from the environment and from nowhere else — no backend takes one as an argument, which a test asserts by signature. |
| — | `anamnesis/scripts/judge_2afc.py` | New, and the golden path the eight donors each half-implemented. Draws a contrast from a target bank and named distractor banks, banks the packet and the key to two files, runs the ladder or the reader ladder, optionally runs a ceiling contrast and the coherence gate, and writes one receipt. `--seed` and `--scoring-instrument` are required: a contrast nobody can redraw, or whose circularity nobody can check, is not a measurement this entry point offers to take. `--dry-run` draws and banks without spending anything, which is also how a packet is prepared for a judge that is not an API. A non-run exits 2 rather than 0. |

### Tests

The frozen record has **no test covering the judging layer** — the one test that
mentions a judge (`tests/test_tool_fixes_2026_07_18.py`) tests
`vmb_arm_a3_analyze.merge_judge`, a record-side analyzer whose subject is join
alignment. So these are the judging layer's first tests in either repository, and
none of them reaches a provider: the backends are stubs, and neither client
library is installed in the environment the suite runs in.

| old path | new home | notes |
|---|---|---|
| — | `tests/test_judging_prompts.py` | New. The prompt table pinned as data: a digest per set and per shared rubric, and the two system prompts rebuilt from the donors' own f-strings and compared byte for byte. Also that `MODE` and `SOCRATIC` are different instruments, that every set carries its provenance, and that a set refuses a variant it does not hold. |
| — | `tests/test_judging_harness.py` | New. Each control against the failure it exists for: the blinding asserted by swapping every key and finding no rendered prompt changed, the banked packet asserted to hold no answer, one worker and sixteen asserted to produce the same rows, the ceiling's four verdicts including the two that refuse to call a null, the non-run flag, the ladder's escalation on an unparseable reply, the coherence gate's window, circularity refused four ways, and no backend accepting a key as an argument. |
| — | `tests/test_judging_likert.py` | New. Purity as a difference rather than a rating, including negative; the truncated-`reasoning` repair keeping the ratings it already had; an out-of-scale rating read as missing rather than zero; resume skipping what is scored; and the cross-channel correlation refusing a handful of points or a set missing a mode. |
| — | `tests/test_judge_2afc_cli.py` | New. The shim's own obligations: a dry run banking a real contrast while spending nothing, the same seed banking the same contrast, a circular study refused before anything is drawn, and the shim rule asserted — the script defines `main` and `parser` and nothing another script would import. |
| `tests/test_tool_fixes_2026_07_18.py` | *record* | Not ported with this layer. Its subject is `vmb_arm_a3_analyze.merge_judge`'s alignment to surviving generation ids, which is a record-side analyzer's join rather than a judging capability. |

## Analysis — the standing readings

The gauntlet answers one corpus's eleven questions. These are the readings *beside*
it: the confound test that says whether the signal is execution or prompt, the cut
that says which part of a family carries it, the transfer that asks whether two
corpora's mode vocabularies name the same thing, the cross-run report that reads
banked results rather than recomputing them, and the two gates a similarity
statistic and a hand-built projection have to pass before they are quoted.

Each of these arrived as one script holding both its capability and its command
line. What ports is the capability, into a module a stranger can find by what it
does; the command is a shim over it, and the split is what makes the numbers
testable without a corpus.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/run_binary_prompt_swap.py` | `anamnesis/analysis/prompt_swap.py` + `anamnesis/scripts/run_binary_prompt_swap.py` | The confound test, logic-identical: per swap pair, a binary forest trained on the two pure modes, predicting the swap generations, counted by which axis they land on and pooled at the 1.5:1 bar. The results are typed (`PromptSwapResult` and its parts) where the donor built nested dictionaries, so the document's shape is declared rather than assembled. It is the one module in the analysis layer that reads signature files directly instead of through the loader, and the reason is stated in it: swap generations are exactly what `load_run4`'s core filter exists to leave out. Named beside `modes/prompt_swap.py`, which holds the prompts this reads the results of. |
| `anamnesis/scripts/run_subfamily_decomp.py` | `anamnesis/analysis/subfamily.py` + `anamnesis/scripts/run_subfamily_decomp.py` | The four name classifiers, the nested operator groups and the coarse substrate cut, logic-identical; the accuracy is the same forest under the same stratified folds. Two adaptations. The donor's `get_feature_names` — a second reader of the signature npz and of its slice table — is gone: names come from `Run4Data.tier_feature_names`, which the loader already fills from that same table and which covers addon directories too, and a **length disagreement between the names and the matrix is now refused** rather than mapped through. That refusal is the one behavioural change, and it replaces the donor's fallback of handing a tier the *whole* vector's name list, which silently mis-assigned every column. `attention_flow`'s classifier also folds its two branches into one substring scan over the same nine signals in the same order — the donor's regex path and its fallback tested the same substrings against different haystacks, and the merged form agrees with both. |
| `anamnesis/scripts/run_cross_run_transfer.py` | `anamnesis/analysis/cross_run.py` + `anamnesis/scripts/run_cross_run_transfer.py` | Both transfer directions, the wildcard count, the pre-registered maps and the LDA direction test, logic-identical, with the 3B comparison travelling inside the result rather than printed after it. The MLP the donor inlined is now `analysis/contrastive_mlp.py` (below); the seed layout, the median-of-similarity and mean-of-assignment pooling, and the scoring that excludes the wildcard from both numerator and denominator are unchanged. The outputs root comes from the configuration package instead of three `parent` hops from `__file__`. |
| `anamnesis/scripts/analyze_complementarity.py` | `anamnesis/analysis/complementarity.py` + `anamnesis/scripts/analyze_complementarity.py` | All seven readings, logic-identical in their arithmetic: cross-run consistency at the five-point bar, resolution by pair difficulty, the hard-pair profile correlation with its zero-variance drop, importance by family and sub-family, the hardest confusion per tier, the ordering check, and value-add against the baseline composites. Three changes. The run table is an argument rather than a module constant, so a report can be taken over any analysis directory. The `--include-5way` flag is gone: the donor's two paths were the same path — a subset pass was read whenever it existed — so an inert switch is not kept. And the hardest-confusion table now reads its labels from `rf_5way.labels`, which is where a banked confusion matrix carries them; see the defect note below. |
| `anamnesis/scripts/train_contrastive_projection.py` · `run_cross_run_transfer.ProjectionNet`/`mine_triplets`/`train_full_data_mlp`/`embed_with_model` | `anamnesis/analysis/contrastive_mlp.py` + `anamnesis/scripts/train_contrastive_projection.py` | **Two donors, one network, two laws named separately.** The banked fit (grouped holdout by generation, kNN validation, early stopping, numpy weights out) and the analysis fit (full data, fixed epochs, several seeds, no holdout) trained the same architecture in both donors and differed in every rule around it, so both laws are callable and neither is the other's default — the same treatment C1 gave two mean-difference laws. Mining is likewise two functions: class-first for the banked fit, anchor-uniform for the analysis one. The training corpus is here as well (`load_hidden_state_samples`): the sampled layers now come from the preset's `contrastive_layers` rather than a hand-copied table, the group is the generation, and the positional correction is applied at the absolute position of each sampled step. It lives in `analysis/` because fitting is learned-probe machinery and because the family that *applies* these weights is pure numpy — a torch import in its closure would fail the anchor's purity guard. |
| `anamnesis/scripts/vmb_routing_cka_gate.py` | `anamnesis/analysis/leak_gate.py` + `anamnesis/scripts/leak_gate.py` | **Generalized, as the manifest's "instrument-grade, arm-independent" classification asks.** The law is the donor's exactly: length-residualize, LDA under GroupKFold-by-topic against the same folds' label-permutation null, the naive accuracy reported only as the other end of the leak gap, a topic-decode readout beside it, and the three-way verdict. What changed is what it is pointed at. The donor hard-coded a directory pattern (`vmb_a2_<model>_pure_<mode>`) and a feature-count assertion for one arm's six routing-CKA features; here the cells are named by the caller and the feature sets are selected by prefix or substring, with `--expect` available for a caller that wants the count pinned. The arm protocol that built those cells stays in the record, which is the same rule the launchers' spec builders were ported under. The length regression is `audit_lib.residualize_all` rather than a second implementation. |
| `anamnesis/scripts/vmb_s51_encoder_on_raw.py` | `anamnesis/analysis/encoder_ladder.py` + `anamnesis/scripts/encoder_on_raw.py` | The three-rung ladder — hand features, raw linear, raw encoder — on one GroupKFold-by-topic split with the fold preprocessing computed once for both architectures, logic-identical. **This is C5's first package consumer**, which closes the note that the audit library landed serving only its own test: the surface sampling, the Gram reduction and the readout pair are all read from `audit_lib` here. Two adaptations: the arms are typed (`Arm`) instead of six parallel lists, and the reading is a named function over the two pairs rather than a chained conditional, so the catch margin and the floor bar are constants a reader can see. The donor's module-level `RAW_SURFACES` global, rebound from `main`, is an argument. |
| `v3_audit/_common.py` · `make_encoder`, `train_eval` | `anamnesis/analysis/audit_lib.py` (**C5, scope completed**) | The readout pair the ladder is measured with: a linear classifier fitted by a convex solver — which is what makes it a floor rather than one optimizer's stopping point — and one nonlinear network as the check on whether the floor is the ceiling. Logic-identical, including the LBFGS-versus-AdamW pairing the encoder diagnostics pinned and the two epoch budgets for raw-wide and Gram-reduced inputs. C5's module is larger than its first extraction because its scope is now whole; the donor total is unchanged and G2's arithmetic still passes with room. |

### Metrology — Stage 0 and the census

⚑ SEAM 2, resolved as the desk leant: **the floors are battery metrology, so the
protocol that collects them lives beside them.** Nothing in either module spawns a
process or loads a model — a plan is an object, a law is arithmetic over banked
signatures, and the fan-out belongs to the launcher.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_stage0_faithfulness.py` · `vmb_stage0_law.py` | `anamnesis/analysis/battery/stage0.py` + `anamnesis/scripts/stage0_floors.py` | **The two halves of Stage 0, in one module and one command with two stages.** The protocol half is logic-identical and now typed: `floor_gid` is the gid layout as arithmetic, `select_continuations` takes seed 0 of every topic and errors on an absent id rather than returning a smaller set, and `plan_stratified_replays` returns the four-pinned-six-spread layout as `ReplayInstance` rows plus the synthetic manifest — whose entries are byte-for-byte the continuation's own manifest rows, ten times over. `replay_index.json` is written from those rows, so the mapping from a signature to its device and component is one definition rather than a dict literal beside a loop. The law half is `law_table_md` and `compute_stage0_law`, which computes the stochastic pass first *because* the faithfulness deltas are standardized on its scale — the donor's two-script arrangement let a caller skip it. The fan-out is the launcher's: `stage0_floors.py replays` builds a `LaunchPlan` over the plan's own per-device shares and invokes `run_replay.py`, which is the §2 rework note carried out (the donor carried a private copy of the C3 fan-out). Two refusals are new and both are about attribution: a device cannot be both pinned and spread, and the two faithfulness arguments come as a pair. |
| `anamnesis/scripts/vmb_subperceptual_census.py` | `anamnesis/analysis/battery/census.py` + `anamnesis/scripts/census.py` | The census, logic-identical: the declared bars, `gap = internals − max(content, likelihood)`, the content rung as a maximum over the declared detector set including the judge, the hardening annotation that voids a blind-judge gap where a forced-choice table shows the judge discriminating, the pending-and-appendix rows, and the class object per model. Rows are `CensusRow` rather than bare dictionaries, `extra="allow"` so a row keeps the per-axis fields it carries, and the hardening text is one function instead of a nested conditional inside a dictionary literal. The judge-defense gate still runs before anything is written — `census_document` calls it, so a census that has not been checked cannot be produced. The arm directory lists are module constants a caller can extend rather than tuples inside two functions. |

## Onboarding, trajectories, search and covariance

Four capabilities the manifest classified as instrument-grade and found stranded in
arm lanes. Each lands where its own subject lives rather than where it was written:
onboarding validates a capture path, so it is extraction's; a trajectory bank feeds
a feature family, so it is extraction's; a black-box search over a fitness function
is domain-free numerics, so it is the package root's; a shrinkage covariance at
width is what a whitened vector is built from, so it is steering's.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/vmb_onboard_validate.py` | `anamnesis/extraction/onboarding.py` + `anamnesis/scripts/onboard_model.py` | The four-step first-contact smoke, logic-identical in what it checks: the layer count, the hook target modules on the probed layers (with the latent-attention and routed-layer variants), `generate` returning attention weights at the preset's query-head count, the hidden-state depth, and one finite feature vector. The routed fifth step is unchanged too, including the routing family's arity and the taxonomy's zero-unclassified check. **The eleven `assert`s become named refusals** (`OnboardingError`): an assertion disappears under optimisation, and this is the one pass whose entire output is its refusals. The `ModelConfig` is built by `from_preset` instead of nine hand-copied fields, and the report is a typed object whose `lines()` render what the donor printed inline — which is what lets the checks be driven from a test with no checkpoint. `router_fields_from_hooks` in `generation_runner` loses its underscore in the same change, because it now has a second reader. |
| `anamnesis/scripts/pathsig_features.py` | `anamnesis/extraction/path_banks.py` + `anamnesis/scripts/pathsig_features.py` | The marshaller between a banked trajectory bank and the path-signature family, logic-identical: the ragged packing with its offset checks, the identity basis over already-projected coordinates (routed through the family's own basis object so the augmentation, integration and null paths stay the family's), the design-matrix stack that **drops a short path and counts it** rather than imputing, and the increment-permutation null at three seeds or more. Three changes: the result is typed (`PathDesignMatrix`) and carries the dropped count beside the kept indices, `from_list` is `from_paths` and checks that it was given one id per path, and the short-path catch narrows from a bare `except Exception` to the family's own `ShortPathError` — a malformed bank now raises where it used to be silently counted as short. The class is not called `SignatureMatrix`, because `audit_lib` has one of those and it is a different object. |
| `anamnesis/scripts/annex_sepcma.py` | `anamnesis/optimize.py` + `anamnesis/scripts/sepcma.py` | Separable CMA-ES, arithmetic-identical — the same weights, the same four learning rates, the same `(d+2)/3` speedup on the diagonal model, the same variance floor. It is at the package root because it knows nothing about signatures: a caller with a fitness function and a dimension is its whole audience, which is the same reason `provenance.py` sits there. The donor's self-test becomes the module's two landscapes (`sphere_fitness`, `alignment_fitness`) plus `probe_budget`, which is the command: on the idealized version of a real direction search, does the optimizer climb at `10·d` evaluations? A typed `BudgetProbe` carries the verdict and the script's exit status is that verdict, so a budget decision can be scripted. `search` is new and is the loop a caller would otherwise write; `state`/`restore` keep the donor's checkpointing, generator included, and `tell` now refuses a population that did not come from its own `ask`. |
| `anamnesis/scripts/c5_ledoit_wolf_gpu.py` | `anamnesis/steering/covariance.py` + `anamnesis/scripts/ledoit_wolf_gpu.py` | The fast shrinkage covariance, arithmetic-identical: both algebraic rearrangements, the float32 matmul with float64 for the shrinkage, the solve and the eigendecomposition, TF32 switched off around the matmul, and the reference's `min(beta, delta)` clamp and zero-numerator guard. It is a sibling of `steering/vectors.py` rather than part of it: the estimator of record there is scikit-learn's, this computes the same number faster, and the two carry **different names** (`shrinkage_covariance` against `ledoit_wolf_covariance`) so a reader always knows which they hold. Three changes: the default device is the CPU, which is what makes the agreement check runnable in CI; the check is a function returning a typed `AgreementCheck` with the three tolerances as named constants, rather than prints and a `SystemExit`; and the cache-emptying call is guarded by the device, so the whole module runs where no accelerator exists. The command is the check — one box, one verdict, one exit status. |
| `anamnesis/scripts/vmb_a5_qual_extract.py` | `anamnesis/steering/readouts.py` · `cell_ladder`, `matched_generations`, `qualitative_markdown` + `anamnesis/scripts/qual_extract.py` | The qualitative browser, logic-identical: the dose ladder and its two controls at a matched dose, matched by generation id, rendered as markdown with the two caveats a reader needs — that seeds differ per cell so this is a style comparison rather than a token-level control, and that the strongest dose collected sits below the mode-induction peak. It joins `readouts.py` rather than becoming its own module because steered-cell naming is that module's vocabulary: the ladder's directory names are the banked spellings and they parse through the same `parse_cell_name` grammar, which a test asserts. The ladder is a function of the site and the vector rather than a literal list, so a caller can read any vector's ladder. |
| `anamnesis/scripts/run_replay_multickpt.py` · `build_partc_replay_cells.py` | `anamnesis/extraction/replay/checkpoint_series.py` + `anamnesis/scripts/run_replay_multickpt.py` + `anamnesis/scripts/build_checkpoint_series.py` | **The deferred PR-6 item, and its builder, sharing one schema.** The adapter-swap loop is logic-identical: load the base once, wrap it, and per checkpoint load-select-restore-merge-replay-unmerge-delete, with the pristine snapshot taken once and restored before each merge. `require_pristine_restore` keeps the donor's hard refusal for a multi-checkpoint swap, and it is now checked before a model is touched. The series document — `{"checkpoints": [{label, adapter_path, run_dir}, …]}` — is `CheckpointSeries`, so the builder that enumerates a training directory and the replay that walks it are the same object rather than two readings of one JSON shape; `series_from_adapter_dir` keeps the donor's step ordering with `final` last and its refusal of a full-weight checkpoint, naming the command that can replay one. The donor's `_build_configs` is `replay/cell.load_replay_model`, which is where the coupling the shims rule forbids already went, and the fan-out is `orchestration/launch.py` instead of a private `Popen` loop. The builder's name drops the arm it was written for: it builds a checkpoint series, and the cohort layout it writes into is an argument. |

### The entry points

Eighteen commands, each argparse and a call into the package. The renames are
recorded because the record cites the old names: `run_gauntlet` for
`run_unified_analysis`, `judge_likert` for `run_judge_scoring`, `stage0_floors` for
the two `vmb_stage0_*` scripts, `onboard_model` for `vmb_onboard_validate`,
`census` for `vmb_subperceptual_census`, `leak_gate` for `vmb_routing_cka_gate`,
`encoder_on_raw` for `vmb_s51_encoder_on_raw`, `qual_extract` for
`vmb_a5_qual_extract`, `sepcma` for `annex_sepcma`, `ledoit_wolf_gpu` for
`c5_ledoit_wolf_gpu`, and `build_checkpoint_series` for
`build_partc_replay_cells`. The `vmb_`, `annex_` and `c5_` prefixes do not exist
here.

⚑ **`run_unified_analysis` gets no alias.** The STRUCTURE draft left the question
open; the ruling recorded here is no. An alias is a second name for one command
that has to be kept working, documented and eventually removed, and the reader it
serves — someone holding a citation of the old name — is served better by this
table, which says where the command went and what else moved with it.

| old path | new home | notes |
|---|---|---|
| `anamnesis/scripts/run_unified_analysis.py` | `anamnesis/scripts/run_gauntlet.py` | 98 lines against 111, and thin throughout: the run registry resolves the signature and addon directories, the mode filter's output directory comes from `outputs_root()`, and `run_full_analysis` does the rest. An unknown run name names every run the registry holds. |
| `anamnesis/scripts/run_binary_prompt_swap.py` | `anamnesis/scripts/run_binary_prompt_swap.py` | 83 lines against 339. A run whose bank has no swap generations is named and skipped rather than reported as an ambiguous result, and a pass over no testable run exits non-zero. |
| `anamnesis/scripts/run_subfamily_decomp.py` | `anamnesis/scripts/run_subfamily_decomp.py` | 90 lines against 451. |
| `anamnesis/scripts/run_cross_run_transfer.py` | `anamnesis/scripts/run_cross_run_transfer.py` | 87 lines against 710. |
| `anamnesis/scripts/analyze_complementarity.py` | `anamnesis/scripts/analyze_complementarity.py` | 68 lines against 758; the analysis directory is an argument and an empty one names the command that fills it. |
| `anamnesis/scripts/train_contrastive_projection.py` | `anamnesis/scripts/train_contrastive_projection.py` | 111 lines against 501. The artifact is verified before the command exits — loaded back through the feature family that will apply it and used to project one row, because a file its consumer cannot read is not a calibration artifact. |
| `anamnesis/scripts/run_judge_scoring.py` | `anamnesis/scripts/judge_likert.py` | New name, for symmetry with `judge_2afc.py`: the two paradigms are two commands over one backend contract. 145 lines against 526, because the paradigm itself landed with the judging layer. The receipt is written as it fills, so a killed pass resumes from what it scored; a pass that scored nothing exits 2, as a non-run rather than a null. |
| `anamnesis/scripts/vmb_stage0_faithfulness.py` · `vmb_stage0_law.py` | `anamnesis/scripts/stage0_floors.py` | One command, two stages in the order they run: `replays` plans and fans out, `law` computes the floors and the table. A dry run banks the plan and stops, which is what makes the plan an artifact rather than a side effect of a launch. |
| `anamnesis/scripts/vmb_onboard_validate.py` | `anamnesis/scripts/onboard_model.py` | Argparse, one call, and the report's own lines. A refused claim exits 1 with the reason. |
| `anamnesis/scripts/vmb_subperceptual_census.py` | `anamnesis/scripts/census.py` | Argparse, one call, and the class object printed per model. |
| `anamnesis/scripts/vmb_routing_cka_gate.py` | `anamnesis/scripts/leak_gate.py` | Cells and feature sets are named on the command line (`LABEL=RUN_DIR`, `NAME=prefix:…`), `--expect` pins a feature count, and the exit status is the first feature set's verdict. |
| `anamnesis/scripts/vmb_s51_encoder_on_raw.py` | `anamnesis/scripts/encoder_on_raw.py` | The two arms, the source run's metadata, and a device that falls back to the CPU with a warning rather than failing where no accelerator exists. |
| `anamnesis/scripts/pathsig_features.py` | `anamnesis/scripts/pathsig_features.py` | 100 lines against 170: the bank, the layer, the rank, the level, the optional null seeds, and one npz out carrying the matrix, its names, the kept indices and the generation ids behind them. |
| `anamnesis/scripts/annex_sepcma.py` | `anamnesis/scripts/sepcma.py` | The budget probe as a command: one or more dimensions, the budget multiple, and an exit status that is the verdict. `--sphere` runs the convex sanity check instead, which separates an arithmetic fault from a budget one. |
| `anamnesis/scripts/c5_ledoit_wolf_gpu.py` | `anamnesis/scripts/ledoit_wolf_gpu.py` | The agreement check against scikit-learn, with the device as an argument. Run it once per box before building vectors through the fast path. |
| `anamnesis/scripts/vmb_a5_qual_extract.py` | `anamnesis/scripts/qual_extract.py` | The ladder, the site, the prompts and the character budget; the vector and its doses are arguments rather than a literal, so it reads any vector's ladder. |
| `anamnesis/scripts/run_replay_multickpt.py` | `anamnesis/scripts/run_replay_multickpt.py` | 183 lines against 217. Worker, launcher and dry run, with the pristine-restore refusal checked in all three before anything loads. |
| `anamnesis/scripts/build_partc_replay_cells.py` | `anamnesis/scripts/build_checkpoint_series.py` | 67 lines against 69; the arm and the cohort root are arguments and the refusal of a full-weight checkpoint comes from the package. |

### Tests

| old path | new home | notes |
|---|---|---|
| — | `tests/test_battery_stage0.py` | New. The protocol, which is where a floor gets attributed to the wrong thing: the gid layout by value, the four-pinned-six-spread plan with its device and component per replay, the two refusals, the synthetic manifest repeating one continuation's own tokens, the law table's conservative PLAN column and its exactly-zero row, and a law pass over a synthetic corpus where the faithfulness floor comes out below the stochastic one. |
| — | `tests/test_census.py` | New. The bars at their own boundaries, the maximum that makes membership conservative, the judge asymmetry in all three of its states (voided, surviving, pending), the class object's empty entry as a reading, and the command's refusal of a root with no records. |
| — | `tests/test_prompt_swap.py` | New. The corpus is planted twice — swap generations carrying the execution mode's value, then the prompt mode's — so the verdict has to flip; plus the label grammar, the complete-coverage filter on tiers, and the 1.5:1 bar at its boundary. |
| — | `tests/test_subfamily.py` | New. Every classifier against its family's naming convention, the unknown bucket as the honest answer, the names-versus-columns refusal, the empty subset that returns a reason, the nested operator groups' growing widths, and the whole family scored beside its parts. |
| — | `tests/test_cross_run.py` | New. Scoring against a pre-registered map with the wildcard excluded from both numerator and denominator, the nearest-centroid assignment, the cosine similarity's scale-freeness, the layer filter's two spellings, the refusals on a missing key and a width mismatch, the LDA leg on a shared and an unshared corpus, and one end-to-end pass at two seeds. |
| — | `tests/test_contrastive_mlp.py` | New. The banked fit's four arrays plus its standardization, the grouped split and its degenerate fallback, the two mining laws' different properties, the single-class refusal, the unit-sphere embedding, and the corpus loader's three rules: swaps excluded, group is the generation, correction applied at the absolute position. The raw captures are written through the real saver, so the layer-offset assertion is about the reconstruction the instrument performs. |
| — | `tests/test_complementarity.py` | New. Each reading over banked files this test writes, including the two refusals that keep a report honest — an unreadable run is skipped, an unvalidatable section leaves the file readable — and the hardest-confusion reading that the frozen record's own lookup could not produce. |
| — | `tests/test_leak_gate.py` | New. A planted condition-carrying feature clears its null and a noise set reads as inert; the verdict law is then tested directly on the three shapes of evidence it is defined over, because which corpus produces which shape is a property of the classifier and what must not drift is the reading. Also that the grouped null sits above chance for a narrow feature set, which is why chance is not the bar. |
| — | `tests/test_encoder_ladder.py` | New. The three readings named apart, matching by generation across two arms with gaps on both sides, the refusal when the arms share nothing, and the two ways a raw capture yields no row rather than a row of zeros. |
| — | `tests/test_optimize.py` | New. The sphere as the convex check, the planted direction at `10·d`, the probe's failing verdict, the two landscapes' own properties, the ask-and-tell contract's refusals, and a restored search continuing the run it was rather than a statistically similar one. |
| — | `tests/test_steering_covariance.py` | New. The port's exactness against scikit-learn on a CPU — shrinkage, Sigma, and the whitened direction's cosine, which is the one a vector is built from — plus the singular-Sigma degradation and the eigendecomposition's reconstruction. |
| — | `tests/test_path_banks.py` | New. The packing's round trip and its two refusals, the short path dropped and counted, the rank bound, and the increment-permutation null leaving level 1 identical while moving level 2 — which is the whole reason it is the control for a level-2 result. |
| — | `tests/test_checkpoint_series.py` | New. The series schema's round trip, enumeration in step order with `final` last, the full-weight refusal naming the command that can read it, the pristine-restore requirement, and the snapshot restoring a mutated wrapped weight exactly. |
| — | `tests/test_onboarding.py` | New. Every refusal driven against a stand-in whose shape is wrong in one way — the shape a bad preset row produces — including the one that earns the whole pass: `generate` returning no attention weights. |
| — | `tests/test_stage0_floors.py` | New. The dry run banking its plan before any device is touched, the faithfulness arguments as a pair, and the law stage's artifacts. |
| — | `tests/test_sepcma.py` | New. The command's exit status, which is its interface: zero when the search climbed on the idealized landscape, one when it did not. |
| `tests/test_audit_lib.py` | `tests/test_audit_lib.py` | **Extended** with the readout pair: the two architectures and the refusal of a third, the convex solver reaching a separable problem's optimum, determinism under a seed, and a training accuracy that makes an unconverged fit visible. |
| `tests/test_steering_readouts.py` | `tests/test_steering_readouts.py` | **Extended** with the qualitative readout: the ladder's banked cell spellings parsing through the cell grammar, a cell with no metadata reading as a gap, and the document naming its two caveats. |

### Defects found in the frozen record during this port

Four, all recorded rather than silently fixed, and two of them fixed here with the
reason stated because no banked number moves:

1. **`analyze_complementarity`'s hardest-confusion table never printed.** The donor
   read `class_labels` off the per-tier classification result; the field is `labels`
   and it lives on `rf_5way`, so the lookup always returned empty and the table was
   skipped in every run. A comment in the donor recorded the mismatch and kept the
   broken lookup. Fixed here, because the section's only output was a printed table
   and no banked number changes — unlike `gauntlet/classification.length_only`,
   which is ported faithfully broken for exactly that reason.
2. **`vmb_subperceptual_census`'s markdown table split its own first column.** Row
   names spell a contrast the way the arm did — `A1:temperature(t03|t09)` — and an
   unescaped pipe inside a markdown cell shifts every column after it, so every
   banked census markdown has a malformed header row. Fixed here by escaping the
   delimiter; the JSON census was never affected.
3. **`run_subfamily_decomp`'s feature-name fallback mis-assigned columns.** Where a
   bank's metadata carried no slice table, the donor handed a tier the *whole*
   vector's name list and then indexed it as if it were the tier's, so a
   decomposition could report sub-families over the wrong features. Refused here
   rather than fixed, because there is no correct mapping to fall back to.
4. **The sub-family classifier's `attention_flow` vocabulary has drifted from the
   family's.** The classifier reads `sysprompt_mass` and `head_diversity_sysprompt`;
   the family emits `prompt_mass` and `head_diversity_prompt`. Ported as-is, since
   the cut is by name and the record's tables were produced under these names — the
   consequence is a large unknown bucket on a v3 bank, which is what the bucket is
   for. Flagged for the nomenclature sweep, where the family names and the cut can
   be brought together in one change.

## Manifest §2 — the completeness table

Every item the port manifest's §2 names, with where its capability lives now. The
list is the manifest's own, in its own order and grouping; forty-two items port and
six are ruled to the record, and there is no row here that reads "see elsewhere".

**Core pipeline entry points**

| §2 item | where it lives now |
|---|---|
| `run_extraction` | `scripts/run_extraction.py` — PR 6 |
| `run_gen_tokens` | `scripts/run_gen_tokens.py` + `extraction/token_generation.py` — PR 6 |
| `run_replay_extraction` | `scripts/run_replay.py` + `extraction/replay/cell.py` — PR 6 |
| `run_replay_multickpt` | `extraction/replay/checkpoint_series.py` + `scripts/run_replay_multickpt.py` — **this PR** |
| `run_recompute_v3` | `scripts/run_recompute.py` — PR 6 |
| `run_8b_calibration` (absorbing `run_corrected_pca`) | `scripts/run_calibration.py` — PR 6 |
| `run_unified_analysis` | `scripts/run_gauntlet.py` over `analysis/gauntlet/` — **this PR** (gauntlet package, PR 5) |
| `run_binary_prompt_swap` | `analysis/prompt_swap.py` + `scripts/run_binary_prompt_swap.py` — **this PR** |
| `run_subfamily_decomp` | `analysis/subfamily.py` + `scripts/run_subfamily_decomp.py` — **this PR** |
| `run_cross_run_transfer` | `analysis/cross_run.py` + `analysis/contrastive_mlp.py` + `scripts/run_cross_run_transfer.py` — **this PR** |
| `run_judge_scoring` | `judging/likert.py` — PR 8; `scripts/judge_likert.py` — **this PR** |
| `analyze_complementarity` | `analysis/complementarity.py` + `scripts/analyze_complementarity.py` — **this PR** |
| `train_contrastive_projection` | `analysis/contrastive_mlp.py` + `scripts/train_contrastive_projection.py` — **this PR** |
| `persistent_replay_worker` | `orchestration/workers.py` + `scripts/run_persistent_replay.py` — PR 6 |
| `_gpu` | `orchestration/gpu.py` — PR 6 |
| `_a5_common` | `extraction/interventions.py` · `load_vector` — PR 6; `steering/gates.py` · `teacher_forced_agreement` and `steering/vectors.py` · `median_row_norm` — PR 7 |
| `_persistent_workers` | `orchestration/workers.py` — PR 6 |
| `_single_cell_guard` | `orchestration/gpu.py` · `enforce_single_cell_guard` — PR 6 |

**Metrology and gates**

| §2 item | where it lives now |
|---|---|
| `vmb_stage0_faithfulness` | protocol in `analysis/battery/stage0.py`, fan-out through `orchestration/launch.py` (the §2 rework note), command `scripts/stage0_floors.py replays` — **this PR**; its C3 donor row landed PR 6 |
| `vmb_stage0_law` | `analysis/battery/stage0.py` · `law_table_md`, `compute_stage0_law` + `scripts/stage0_floors.py law` — **this PR** |
| `vmb_onboard_validate` | `extraction/onboarding.py` + `scripts/onboard_model.py` — **this PR** |
| `vmb_a5_onpolicy_gate` | `steering/gates.py` · `on_policy_gate` — PR 7 |
| `vmb_subperceptual_census` | `analysis/battery/census.py` + `scripts/census.py` — **this PR** |
| `vmb_routing_cka_gate` | `analysis/leak_gate.py` + `scripts/leak_gate.py` — **this PR**, generalized past the one arm's directory pattern |
| `annex_null` (PORT-WITH-REWORK) | `steering/gates.py` · `assert_against_own_null` — PR 7; the corpus loaders stayed in the record, which was the rework |
| `annex_shape_audit` (PORT-WITH-REWORK) | `steering/gates.py` · `audit_axis` / `audit_axes` — PR 7 |
| `vmb_a5_upstream_zero` | `steering/gates.py` · `upstream_zero_check` — PR 7 |

**Explicitly classified post-validation — LEAVE-IN-RECORD, by ruling**

| §2 item | ruling |
|---|---|
| `vmb_s51_resolver` | Not ported. One-shot, unstamped, tied to its own legs; V2 inspected it and ruled it record. Its dependency on the audit library is served here by `analysis/audit_lib.py` for anything that succeeds it. |
| `pathsig_project_residual` | Not ported, same ruling. The projection it performs is what produces a trajectory bank; reading one is `extraction/path_banks.py`. |
| `pathsig_incremental` | Not ported, same ruling. |
| `pathsig_read_e1` | Not ported, same ruling. The reads it performed are `scripts/pathsig_features.py` plus a caller's own analysis. |
| `pathsig_constant_injection` | Not ported, same ruling. |
| `pathsig_s51_regen` | Not ported, same ruling. |

**Steering and analysis readouts**

| §2 item | where it lives now |
|---|---|
| `vmb_a5_lever_readout` | `steering/readouts.py` · `lever_readout` — PR 7 |
| `vmb_matched_support_efficiency` | `steering/readouts.py` · `matched_support_efficiency` — PR 7 |
| `vmb_a5_band_mass_readout` | `steering/screens.py` · `band_mass` — PR 7 |
| `vmb_a5_layer_separation` | `steering/screens.py` · `axis_separation_rows`, `two_fold_heldout_d` — PR 7 |
| `vmb_a5_routing_separation` | `steering/screens.py` · the same pair over the routing substrate — PR 7 |
| `vmb_a6_directional_readout` | `steering/readouts.py` · `directional_series`, `seed_floor`, `sign_flip_p` — PR 7 |
| `vmb_partc_contrast` | `steering/readouts.py` · `contrast_fields` — PR 7 |
| `vmb_identity_histogram` | `steering/readouts.py` · `expert_usage_histogram` — PR 7 |
| `vmb_s51_encoder_on_raw` (needs audit_lib) | `analysis/encoder_ladder.py` + `scripts/encoder_on_raw.py` — **this PR**; C5's first package consumer, with `make_encoder`/`train_eval` extracted to complete its scope |
| `build_partc_replay_cells` | `extraction/replay/checkpoint_series.py` · `series_from_adapter_dir` + `scripts/build_checkpoint_series.py` — **this PR** |

**Generic primitives found stranded in arm lanes**

| §2 item | where it lives now |
|---|---|
| `annex_sepcma` | `optimize.py` + `scripts/sepcma.py` — **this PR** |
| `c5_ledoit_wolf_gpu` | `steering/covariance.py` + `scripts/ledoit_wolf_gpu.py` — **this PR** |
| `pathsig_features` | `extraction/path_banks.py` + `scripts/pathsig_features.py` — **this PR** |
| `vmb_a5_covariance_screen` | `steering/screens.py` · `screen_vector`, `screen_bank` — PR 7 |
| `vmb_a5_qual_extract` | `steering/readouts.py` · `cell_ladder`, `matched_generations`, `qualitative_markdown` + `scripts/qual_extract.py` — **this PR** |

**The §2 rework note.** `vmb_stage0_faithfulness` ports as a *consumer* of
`orchestration/launch.py` rather than with its own copy of the fan-out, which is
what `scripts/stage0_floors.py replays` does: it builds a `LaunchPlan` over the
stratified plan's own per-device shares and invokes `run_replay.py` once per
device. C3's donor table already counts the script's 152 lines.

With that, manifest §2 is exhausted: forty-two items live in this repository and
six are in the frozen record by ruling, with no item unaccounted for.
