# Architecture

A map for a reader who has just cloned this and has not seen the code. It says what the
pieces are, which constraints are load-bearing, and which file to open next. It is not a
manual: the modules carry their own documentation, and this page exists to tell you which
of them to read first.

## What a signature is

**A signature is a lossy compression of the causal history of a forward pass over a span
of text** — what the computation did, not what the text said. `anamnesis/__init__.py`
states it in one line, and the whole instrument follows from the choice of *span of text*
over *generation*.

A decoder forward is a deterministic function of its input tokens. So if you know the
token sequence a generation realized, you can recover the states that produced it by
teacher-forcing that sequence through one instrumented forward — you do not need to have
been present while it was sampled. Two consequences, and they are why the definition is
worth being fussy about:

- A signature can be collected over text the model never chose. Any span is a legitimate
  subject, including one written by a person or by a different model.
- The engine that produced the tokens is a free choice. Sampling and measuring are
  separate passes over the same ids.

Had the object been "a generation", both would be false, and every feature would be
entangled with the sampler that drew it. `anamnesis/extraction/replay/__init__.py` is the
statement of this in the code.

## Where to start reading

Five files, in this order. Each is a package docstring written to be read as prose, and
between them they cover the instrument.

| open this | for |
|---|---|
| `anamnesis/__init__.py` | the definition, and what each top-level module is for |
| `anamnesis/extraction/__init__.py` | the write side: what a forward pass is read into |
| `anamnesis/modes/__init__.py` | the processing-mode prompts a run's labels refer to, the registry they are rows of, and the confound test that travels with them |
| `anamnesis/analysis/gauntlet/__init__.py` | the read side: the eleven standing analyses and the section registry that dispatches them |
| `anamnesis/scripts/__init__.py` | the command inventory, grouped by the order a run moves through it |

Everything below is orientation around those five.

## Extraction: three layers, kept apart

`anamnesis/extraction/` separates three concerns, and collapsing them is the one
refactor this package will not accept.

- **`extraction/model_loader.py`** — the checkpoint on a device, with hooks on it. Loads
  the model and tokenizer with the eager attention kernel, registers forward hooks on the
  projection modules a capture reads — `k_proj` for keys, and `q_proj`, `v_proj` and
  `gate_proj` where those substrates are wanted — and manages hook lifetime. It knows
  nothing model-specific: every architectural fact comes from a `ModelConfig` preset.
- **`extraction/state_extractor.py`** — the numeric anchor. Raw arrays in, a named
  feature vector out, **with no torch import, no model awareness and nothing beyond numpy,
  scipy and this package's own configuration on its import line.** That is a design
  constraint rather than an accident: it means the numbers can be reproduced, and the
  module tested, on a machine with no accelerator and no weights, and it gives every
  faster path something to be *defined as agreeing with*.
- **`extraction/generation_runner.py`** — orchestration. Prompt, seed, generate, convert,
  call the extractor, save, and write the replay manifest beside the metadata.

Two modules exist because of that separation. `extraction/state_extractor_reference.py`
is the golden master the anchor's optimisations are checked against —
`tests/test_extraction_equivalence.py` is where the two meet. `extraction/raw_saver.py`
is the on-disk form of a capture, and the reason the engineered families in
`extraction/feature_families/` can run with no model loaded at all: a capture is banked
once and re-read as often as the feature set changes.

Three capture facts are load-bearing, and each is stated where it is enforced:

- **Eager attention is required.** Flash attention and SDPA return no attention weights,
  so the eager kernel is a correctness requirement rather than a preference.
  `anamnesis/config/models.py` carries the constant.
- **Keys, values and queries are captured pre-RoPE, from the projection modules rather
  than from the cache.** A post-RoPE key has its position baked in, so geometric features
  computed over one would be reading position.
- **Query heads and key/value heads are separate facts about a model.** Every model's row
  in `anamnesis/config/models.json` carries both, and the validator in
  `anamnesis/config/models.py` refuses a row whose grouped-query group size is not exact,
  because under grouped-query attention the two counts differ and a per-head reading has to
  say which of them it is counting.

## The replay gateway

The pipeline splits sampling from measuring into two passes over the same token ids.

1. `anamnesis/scripts/run_gen_tokens.py` samples text and writes down the *realized*
   token ids — prompt and generation together. No hooks, no calibration, no feature
   arithmetic. `anamnesis/extraction/token_generation.py` is the loop.
2. `anamnesis/scripts/run_replay.py` teacher-forces exactly those ids back through an
   instrumented forward and produces the signatures.

Banking the ids rather than the decoded string is the point: a decoded string has to be
re-tokenised to be replayed, and the first generated token is not recoverable from the
decode at all. `anamnesis/extraction/replay/manifest.py` holds a reconstruction path,
with validation, for a bank that carries text only — a pass through phase one never needs
it.

What the split buys:

- **The capture surface is free to change.** Adding a feature family or hooking another
  layer costs a replay, not a re-generation, and the text under both is provably the same
  text because it is the same ids.
- **The generating engine is a free choice.** Nothing downstream of the ids knows what
  sampled them.
- **Replay over a fixed token stream is bitwise deterministic on one machine.** Same ids,
  same weights, same arithmetic, same last digits — which is what licenses the exactly-zero
  upstream-delta checks in `anamnesis/steering/gates.py`.
- **Work is partitionable.** Every generation is seeded from its own coordinates, so which
  worker produced it, and in what order, cannot reach the output. That is what makes
  `anamnesis/orchestration/` a cost decision rather than a scientific one.

## The fast lane, and `lane_id`

`anamnesis/extraction/fast/` emits the anchor's named vector without materialising raw
tensors on the host: attention weights are reduced inside each layer's forward hook, so
the all-layer attention tensor never exists at once. It is the same vector by a cheaper
route, and its right to exist rests entirely on being measurably equal to the anchor's.

Because a feature vector is only comparable to another vector computed the same way, the
way is pinned into an identifier. `GpuFeatureLane` in `anamnesis/extraction/fast/features.py`
builds an identity dictionary and hashes it:

```
lane_id = "torch-eager-reduce-v1-" + sha256(json(identity, sort_keys=True))[:20]
```

The identity holds a SHA-256 of the bytes of each of the five lane source files —
`features.py`, `ops.py`, `attention.py`, `families.py`, `batch_layout.py` — alongside the
extraction and family configuration, the calibration digest, the device type, which replay
path the lane runs (`cached` or `full`), the torch, transformers, numpy and CUDA-runtime
versions, the CUBLAS workspace setting, a digest of the feature-name schema, and the
deterministic-algorithms, TF32 and preferred-BLAS flags.

The consequence a contributor needs to know in advance: **a change to the bytes of any of
those five files changes every `lane_id` the lane stamps, so every already-banked corpus
carries an identity the changed code cannot reproduce, and re-banking is the only way to
join the two.** Signatures banked under different lanes are not two measurements of the
same quantity, and `anamnesis/analysis/lane_guard.py` refuses the join on the read side —
every loader that assembles a matrix from more than one file passes its metadata through it
first. Untagged historical banks stay readable and cannot be mixed with tagged ones.

Whether a given machine's lane agrees with the anchor is a property of the machine, not of
the code: floating-point reduction order differs across BLAS builds and thread counts.
`anamnesis/extraction/equivalence/` holds the evidence checks and
`anamnesis/scripts/qualify_box.py` is how an operator qualifies their own box.

## What runs on which model

Two paths compute a signature, and they do not accept the same models.

| | Hook path — the numeric anchor | Fast lane |
|---|---|---|
| Commands | `run_extraction` (generate and capture), `run_replay` (teacher-force banked ids) | `run_gpu_replay`, `qualify_box`, and in process `prepare_fast_lane` + `harvest_loaded` |
| Architectures | Dense decoders whose layers sit at `model.model.layers` with k/q/v/o/gate projections (the Llama, Qwen-2 and OLMo-2 families); Gemma-3, whose text decoder nests inside a multimodal wrapper; DeepSeek-V2, whose latent attention and routed experts get their own capture surface | Dense Llama only. Any other `model_type` is refused by `check_loaded_model` once loaded |
| Placement | Whatever device map the preset's configuration gives | One device holding every parameter |
| Attention kernel | Eager, which returns weights; a fused kernel is refused at the preset | Eager |
| How a box checks it | `onboard_model` | `qualify_box`, against the anchor on the same box |

Of the shipped presets, `3b` and `8b` run on both paths; `olmo2-7b`, `qwen-7b`,
`gemma3-27b` and `dsv2-lite` run on the hook path. A row added through `ANAMNESIS_MODELS`
runs on whichever path its architecture meets.

Bringing a model up, in order:

1. **A registry row**, then `onboard_model`: the layer plan, the hook targets and the
   attention kernel, each refused by name if wrong.
2. **A calibration**, `run_calibration`. `--reach-from` names the replay manifest of the
   runs the calibration will correct and requires the last position they read, so a
   calibration never stops short of its corpus; one that would is refused before it is
   written. The directory it writes holds the positional means, a per-layer basis, the
   sequences the fit was taken over and a build receipt.
3. **What the basis determines**, `calibration_stability`: how many directions two fits
   over disjoint halves of those sequences share, per layer. Components past that count
   are properties of the sample, not of the model. It prints the counts as a
   `pca_components_by_layer` entry; set in the row, a calibration fits each layer to its
   own count and the extraction projects each layer onto what its basis holds.
4. **The fast lane**, for a dense Llama: `qualify_box` on the machine that will run it.
5. **Signatures**, by either path. Both read the per-layer basis step 2 writes, and both
   refuse, before computing anything, a sequence that reaches past the positions the
   calibration fills: `run_extraction` from its longest prompt and its token budget, the
   replays and the fast lane from each banked sequence. Extraction prompts carry a mode's
   system prompt and calibration prompts do not, so a calibration at the same budget ends
   short of an extraction; `--reach-from` or `--required-through` on step 2 is what
   closes that gap.

## Fail closed

This is the habit most worth taking from the repository, and the quickstart in
`README.md` is a live demonstration of it.

`anamnesis/shortfall.py` gives every command one way to state what it was asked for and
what it produced. A `Shortfall` carries `requested`, `produced` (the requested ids whose
artifact is *on disk* when the pass ends), `excluded` (ids the pass was never going to
attempt, and why) and `failures` (ids that were attempted and raised, with the reason).
`refuse_unless_complete` then exits non-zero when any pass came up short, after writing a
receipt beside the output it describes.

- `EXIT_SHORT` (3) — short, and nobody asked for a short pass.
- `EXIT_SHORT_SANCTIONED` (4) — short, and the invocation passed `--allow-partial`. Still
  non-zero, because partial work is permitted and hiding it is not.

A complete pass deletes its own receipt, so a directory holding one is short as of its
last pass rather than as of some earlier one. A model validator refuses an accounting that
does not close: you cannot report a shortfall against a set you never requested, and an
exclusion is not a request.

The rule this enforces: **nothing is silently zero and nothing absent is scored.** A mean
over four of five generations is a number, and nothing about the number says five were
asked for. A section that cannot measure intrinsic dimension because an optional
dependency is absent says so and the pass refuses, rather than the pass reporting itself
complete. `section_shortfall` in `anamnesis/analysis/gauntlet/__init__.py` is the same
accounting applied to analysis sections.

## Analysis: the reading side

`anamnesis/analysis/__init__.py` routes this whole package. The major pieces:

**`analysis/gauntlet/`** — eleven standing analyses over one corpus, run as a single pass
because a claim about signatures is usually a claim about several of them agreeing: an
accuracy means one thing beside a clean semantic-orthogonality result and another beside a
length-only baseline that reaches the same number. The sections are data integrity and
descriptive statistics; classification; the readout over the stored blocks and feature
importance; intrinsic dimension; cross-condition generalization; topology and
hyperbolicity; clustering; contrastive projection; semantic independence; the prediction
scorecard; and manifold geometry. They are declared in the `SECTIONS` registry and
dispatched by one loop, which imports each section module only when its section runs — so a
pass that skips the contrastive and semantic sections never imports torch or
sentence-transformers. Results are checkpointed after each section, and `--resume` reads
the checkpoint back, validates it against the section schemas, and skips what is already
there.

**`analysis/battery/`** — the metrology that comes before an experimental arm rather than
after it: how large a difference has to be to be resolvable, how many samples it takes to
see one, and which rows the internals catch that a cheap reader misses. Typed throughout,
and a number leaves it stamped rather than bare — `ResultStamp` and `StampedValue` in
`anamnesis/analysis/battery/stats.py` are what a reading is carried in.

**`analysis/subfamily.py`** — which *part* of a family carries its signal. The taxonomy in
`anamnesis/feature_map.py` cuts across families by substrate; this cuts a single family
into the signals its own feature names spell out, and reads each part's accuracy against
the whole family's.

**`analysis/complementarity.py`** — the readings that exist only *between* banked results.
It reads the gauntlet's own JSON rather than signatures, so cross-run questions cost a
re-read instead of a re-run: whether a block is stable across corpora, which blocks fail on
*different* hard pairs and are therefore worth combining, and which are redundant.

Also here, each answering one question: `analysis/prompt_swap.py` (is the signal the
instruction or the execution — the confound the programme rests on), `analysis/leak_gate.py`
(does a feature set carry the signal or the topic it was measured on),
`analysis/audit_lib.py` (the length control, the leak-free folds, the readout pair),
`analysis/encoder_ladder.py` and `analysis/contrastive_mlp.py` (whether a hand-built
projection or the raw state is the limitation, which is the only way to read a null off
features), `analysis/cross_run.py` (do two corpora's mode vocabularies name the same
thing), and `analysis/text_stats.py` (the text channel beside the signature, which catches a
degenerate generation reading as a change).

## The other two halves

**Steering — `anamnesis/steering/`.** The write side: whether a direction in the residual
stream can *make* a pass run a given way. Four modules stand between a candidate direction
and a citable result: `vectors.py` (construction, matched nulls, the dose currency),
`screens.py` (whether a direction can be injected at all), `gates.py` (whether a cell is
valid — on-policy agreement, the exactly-zero upstream delta a deterministic replay owes, a
direction's own matched null) and `readouts.py` (movement along the axis against movement
off it). Everything in it is **per-model**: a vector is built in one model's residual basis,
whitened by that model's covariance, dosed in that model's residual norm, at a site chosen
by that model's layer sweep, and none of the four transports.
`anamnesis/extraction/interventions.py` is where a banked intervention spec becomes an
armed hook, on both the generation and the replay side, so the two cannot drift onto
different specs.

**Judging — `anamnesis/judging/`.** The behavioural channel, held to the same standard as
any other instrument: a blind two-alternative forced choice with the key in a type no
prompt renders from, a ceiling control that decides whether a null may be called, prompts
pinned by hash because what a judge was asked is part of what its number means, and Wilson
intervals. It is the one part that talks to a model provider, so the client libraries are
an optional extra, their imports are deferred to first use, and the tests mock the
transport — which is also what guarantees no test can reach an API. Keys are read from the
environment and from nowhere else.

## The supporting spine

- **`anamnesis/config/`** — per-model presets, one pass's settings, and the run registry,
  in three modules behind one import surface. Nothing here imports torch, so a run is
  describable on a machine with no GPU. The presets and the runs are data — `models.json`
  and `runs.json` beside the modules that read them — and `ANAMNESIS_MODELS` names further
  model files, so a checkpoint this package never shipped is addressable by name without a
  code change. A model's layer count is written in its row and nowhere else: the depth bands
  in `anamnesis/feature_map.py` and the battery's per-model metadata read it from there, so
  three copies cannot drift apart.
- **`anamnesis/feature_map.py`** — the executable `source × method × depth` taxonomy: what
  a feature name *means*, read by extraction and analysis alike. The contiguous blocks a
  vector is stored in are addresses into the vector and nothing more; three of the four span
  several substrates, so a block's accuracy localizes nothing. Address a stored artifact by
  its block, describe a feature by its cell.
- **`anamnesis/provenance.py`** — the digests a bank is stamped with and read back through,
  which is how two banks are known to be joinable.
- **`anamnesis/synthetic_bank.py`** — a bank of the right shape drawn from a seed, so
  everything on the reading side can be exercised before there is a model to run. It states
  what the construction deliberately does and does not put in the numbers; read it before
  quoting anything the demo prints.
- **`anamnesis/orchestration/`** — which device a worker gets, how work is partitioned, and
  workers that stay loaded. Nothing here computes a feature.
- **`anamnesis/optimize.py`** — domain-free black-box search, plus the probe that says
  whether a budget is worth spending.

## Gates

Four gates guard a change, and each is a command you can run yourself. They live in
`tools/`:

| gate | checker |
|---|---|
| G1 data compatibility — banked artifacts in, identical features out | `tools/g1_hash_manifest.py` |
| G2 test retention — a change that shrinks the suite shrinks the code by at least as many lines | `tools/check_test_retention.py` |
| G3 documentation — the two documentation rules, over comments, docstrings and the message a `raise` or a log says out loud | `tools/check_timelessness.py` and `tools/check_referents.py` |
| G4 import closure — every module reachable from a command or a test; no orphans | `tools/check_import_closure.py` |

`CONTRIBUTING.md` states the command for each and — read this before writing any prose
here — the documentation rule G3 enforces. `.github/workflows/gates.yml` is what runs on a
pull request: the suite, G2, both halves of G3 and G4, on every supported interpreter, plus
a wheel built and installed outside the checkout so an import that only worked from the
source tree fails. G1 reads banked signatures, which a runner has no copy of, so it runs
where the data is and its receipt is attached instead. `tools/surface_report.py` reports the
size of the codebase in code tokens and documentation words; it is a trend instrument, never
a gate.

## A first hour

1. Run the quickstart in `README.md`. It needs no model, no accelerator and no network,
   and its refusal is the doctrine above in one command.
2. Read `anamnesis/__init__.py`, then `anamnesis/extraction/__init__.py`.
3. Read `anamnesis/extraction/state_extractor.py`'s module docstring and one
   `extract_*` function. That is what a feature *is*, in the plainest form the repository
   has.
4. Read `anamnesis/analysis/gauntlet/__init__.py` and skim the results of the pass you
   just ran.
5. Read `anamnesis/scripts/__init__.py` to find the command for whatever you came here to
   do.
