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
