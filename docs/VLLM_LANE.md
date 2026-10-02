# The vLLM lane

The vLLM lane computes the same named feature vector as the fast lane
(`anamnesis/extraction/fast/`), from a vLLM engine instead of a hooked eager forward. It
exists for models whose replay is too large or too slow to run through Hugging Face
transformers on the hardware at hand, and it is a separate lane: its rows are never
combined with fast-lane rows inside one contrast.

A lane is self-consistency plus provenance (`docs/ARCHITECTURE.md`, "What a lane is"): the
same tokens give the same signature every time, under one declared model, engine, arithmetic
and host. Every host that runs this lane repeats itself byte for byte; whether it also
reproduces the qualified lane, or is a lane of its own, is what the install check decides.

## What it covers, and what it refuses

| | |
|---|---|
| Models | `3b` (Llama-3.2-3B-Instruct, float16), `8b` (Llama-3.1-8B-Instruct, bfloat16), `70b` (Llama-3.1-70B-Instruct, bfloat16) |
| Placement | One GPU holding every parameter, plus 2 GiB of KV cache. The 70B model in bfloat16 needs a card with roughly 160 GB |
| Spans | A prompt and at least two generated tokens, 1024 tokens in all at most |
| Execution | Whole-prompt prefill, one request at a time or eight in a batch |
| Engine | `vllm==0.16.0`, `torch==2.9.1`, `triton==3.5.1`: the `vllm` extra |

Everything outside that table is refused by name before an engine is built:
`anamnesis/extraction/vllm/envelope.py` holds the declaration and the guard. A fine-tune of
one of these models can join as an extension lane of its own (see
[Extension lanes](#extension-lanes-fine-tunes-of-a-qualified-model)).

## Setting up

Install from a clone into a virtual environment of its own, because the extra pins torch
exactly: `uv pip install -e ".[vllm]"`. Triton compiles a small launcher when it first runs,
so the Python interpreter's development headers must be present; the interpreters `uv`
installs carry them. An interpreter without its C headers runs the lane only from a Triton
cache prebuilt where they exist, named by `TRITON_CACHE_DIR`.

`--model-path` is a local directory holding the checkpoint's `config.json` and its
`*.safetensors` shards at the top level: Meta's Instruct release of the model, as published
on Hugging Face. The check hashes those files and refuses, before capturing anything, a
checkpoint whose digest differs from the one the fixtures were produced from; the refusal
prints both digests.

The lane reads the first GPU the process can see, and a receipt names that card. On a host
with several, `CUDA_VISIBLE_DEVICES` chooses one, and a different card is checked again.

Global memory capacity is not the only hardware limit. The pinned engine's float16
batch-invariant matrix multiply requests 104 KiB of shared memory per thread block with its
three-stage pipeline, and some devices offer less.
`anamnesis/extraction/vllm/matmul_launch.py` caps float16 staging at two on devices below
104 KiB, or one below 56 KiB, chosen from the device's capacity alone and independent of
requests and batching. It retains the engine's kernel, tiles, reduction order and dtype;
other dtypes and devices fitting the stock launch keep their launch settings. The capture
record carries `matmul_policy`, and the host fingerprint hashes it along with the lane's
source, so a host with a smaller launch is checked like any other and lands in whichever
tier its captures earn.

The lane reduces every row with the calibration its fixtures were reduced with, not one
fitted on the host, so banks from any host running the lane share one calibration. It is
downloaded once into the output root; a copy elsewhere is named with `--calib-dir` and
verified against the sizes and digests in `anamnesis/extraction/vllm/hub.py`.

## Where the fixtures and tolerance come from

Each model's fixtures and tolerance come from one comparison of the lane with the numeric
anchor, the hook path (`anamnesis/extraction/state_extractor.py`), on the same populations,
per engine release. The **fixture vectors** are what the lane produced on 44 rows of that
comparison, chosen for where a host is likeliest to move: the largest attention shifts,
coverage shifts and gate-sparsity displacements, the largest spectral deviations, and 16
rows evenly spaced over the rest. The **tolerance** is the spread between the lane and the
anchor that comparison measured while the lane's effects were retained. Its row ceilings are
ratios of a row's **distance** (its standardized L2 difference over one component of the
vector: the **covered substrate**, which is every coordinate outside the attention
families, or the **attention** families) to the row's **path floor** (how far the anchor's
own two execution paths, its one-forward replay and its token-by-token path, disagree on that
row). A path floor is measured under the lane's arithmetic, the cuBLAS workspace pin
(`CUBLAS_WORKSPACE_CONFIG=:16:8`) and deterministic algorithms, which is the arithmetic every
reference the lane is scored against was computed in. Without the pin, cuBLAS picks its GEMM
algorithms by matrix shape, so a floor measured unpinned mixes the path disagreement with
algorithm-selection noise that grows with the model; tolerance contract 2 is the one measured
under the pin. A few rows whose path floor is too fragile to divide by are left out of the
ceilings and named in the file. The records the tolerance was
read from belong to that comparison and are not published; the file names each by its role
and pins it by sha256.

## Before first use on a host: the install check

```
python -m anamnesis.scripts.qualify_vllm --model 8b \
    --model-path /path/to/Llama-3.1-8B-Instruct --work-dir ./qualify-8b
```

It captures 44 fixture rows shipped with the package, twice one at a time and once in
batches of eight, reduces them, and compares them with the qualified lane's vectors. The
calibration those vectors were reduced with is fetched on first use and verified against its
pinned digests; `--calib-dir` names a local copy instead.

The work directory holds the raw captures of the three passes until each is reduced: about
44 GB per pass for the 70B model, far less for the smaller ones.

Different hardware gives different numbers, and that is expected rather than wrong. The check
ends in one of three tiers:

- **identical** — every vector byte-identical to the fixtures. The host runs the qualified
  lane itself, and its outputs carry that lane's id.
- **own-lane** — the vectors differ, and the host is a lane of its own: every fixture row
  byte-identical across its two single passes and its batched pass, every feature finite,
  and per component (covered substrate, attention) the median of the rows' ratios at or below
  the maximum ratio the qualification recorded. The host's lane id is derived from its
  fingerprint, and it is fully usable.
- **refused** — a different checkpoint, a host whose repeated or batched captures disagree,
  a non-finite feature, or a component whose median ratio exceeds that maximum: not a lane,
  or plainly broken. The reasons are printed.

The median bound is coarse on purpose. It catches a host computing something else (the 3B
model run in bfloat16 against its float16 fixtures sits at a substrate median of 3.55 against
a recorded maximum of 2.87) and leaves arithmetic differences to the lane identity. Every
other ceiling is **reported, not gated**: each row's distance against its recorded ceiling,
each family's largest |δ|/σ_cal against its recorded maximum, the discrete crossings, and the
evenly spaced rows' median and tail against the recorded 90th and 99th percentiles. They say
how far this lane sits from the qualified one.

The receipt is cached under the output root, keyed by the host's fingerprint (GPU, driver,
CUDA runtime, torch, vLLM and anamnesis versions, checkpoint, fixtures, tolerance, engine
settings and the lane's source), and reused only while every field is equal. Change any of them
and the check runs again; `--refresh` runs it regardless. A receipt of another contract is
decided again, never reinterpreted.

Exit status 0 means identical or own-lane, 1 refused, 2 that the check could not run (a
missing engine, checkpoint or calibration, with the reason).

## Banking signatures

```
python -m anamnesis.scripts.run_vllm_replay --model 8b \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --manifest runs/<run>/replay_manifest.json --output banks/<run>-vllm
```

The manifest is the `replay_manifest.json` a banked run carries, written by `run_extraction`
or `run_gen_tokens` into the run's directory under the output root; `metadata.json` beside it
is read when present. Spans must fit the lane's 1024-token context, and a span too short to
carry the fast lane's full schema (fewer than eight predicted positions) is refused.

It refuses to start without a cached receipt for this host that did not refuse it. Rows are
captured and reduced in chunks (`--chunk-rows`), because a captured row's substrate is large,
up to about a gigabyte for the 70B model, and `--work-dir` holds one chunk's captures at a
time. The output is the banked format the fast lane writes: a vector and a metadata sidecar
per generation, beside a `deployment.json`.

Every row carries `lane_id`, the id its host's receipt assigned, and an `extraction_lane`
record naming the tier, the receipt digest, the calibration and the feature schema.
`anamnesis/analysis/lane_guard.py` refuses to combine rows of different lanes inside one
contrast, so a bank from an `identical` host joins other `identical` banks of the same model,
and an `own-lane` host's bank joins only banks from that host. Every reader that takes a
signature directory reads it, `run_gauntlet` among them.

## Resident sessions

A caller that captures a few rows at a time for as long as it runs (a server harvesting one
draw at a time, say) holds the lane open instead of starting it per call:

```python
from anamnesis.extraction.vllm.session import LaneSession

with LaneSession.open("8b", model_path, work_dir=work, cache_dir=receipts) as lane:
    result = lane.capture(rows)   # rows: generation_id, input_ids, prompt_length
    result.vectors[gid], result.receipts[gid], lane.extraction_lane(row, result.receipts[gid])
```

It is the same lane, with the same id: the same two processes, started the same way, with the
same engine settings and readout, behind the same install-check receipt, which it refuses to
start without. `anamnesis/extraction/vllm/session.py` holds it. What differs is how long
the processes live and three things that follow from that:

- **Rows cross in memory.** Each row's substrate goes from the engine to the readout through a
  shared-memory segment (`anamnesis/extraction/vllm/handoff.py`), not a file, and the readout
  reduces it while the engine captures the next group. The engine still computes each row's
  content receipt, the sha256 of every tensor's native bytes, over the bytes it handed off,
  on a background thread; the readout checks that the segment is the one the engine published
  for that row (row id, receipt id, tensor names, shapes and dtypes) instead of re-hashing it.
  No row's vector is returned before its receipt is written beside the call's schedule
  records in the session's work directory.
- **The non-interference check runs on a cadence.** Every group is run unhooked and hooked
  and compared until the session's first eight groups have passed and each condition the
  session has used has a passing group; after that, every sixteenth group by its index in the
  session, and the first group under a condition the session has not used before. Each group's record
  and each row's receipt say `checked` or `not_checked`, and an unchecked group never claims
  `hook_noninterference`. A failed check stops the session and names every row it returned
  since the last passing check as unverified, to be captured again or dropped. Banks, install
  checks and transfer audits check every group.
- **One engine, two conditions.** The engine is built for batches of eight. A capture of eight
  rows or more runs as `full-b8-order0` (a short final batch filled with other rows of the
  capture, never read back); a capture of fewer runs one request at a time, as
  `full-b1-order0`, with no filler rows. The install check certifies the two conditions
  byte-identical on the host, and each group's recorded schedule proves the concurrency it
  ran at. Every receipt, group record and `extraction_lane` record names the condition its
  row ran under (`condition_id`) beside the condition the engine was built for
  (`engine_condition: full-b8-order0`).

A session that ends without closing leaves its shared-memory directory
(`anamnesis-lane-<pid>-<token>`) behind. Opening a session removes every such directory whose
process is gone, and lists them in `session.json` as `removed_stale_handoff_dirs`.

A session holds one GPU: the engine and the readout share the first visible device. A
tensor-parallel lane is refused.

## Extension lanes (fine-tunes of a qualified model)

A fine-tune of `3b`, `8b` or `70b` is the same network with different weights. It can use its
base's lane as an **extension lane** of its own, without editing this package: the owner of the
fine-tune declares it, and this package learns no fine-tune's paths.

### What carries over

Everything that depends only on the architecture carries over from the base unchanged: the
instrumented kernel, the determinism of the dispatch, the feature schema, the capture routes, the
dtype, the logprob handling and every engine setting in `anamnesis/extraction/vllm/envelope.py`.
An extension may not set any of them.

An extension is admitted on its identity: a registry preset that `extends` the base's, its own
calibration pinned by digest, its checkpoint digest, and its own fixtures and tolerance carrying
that identity. Each host that runs it then passes the install check on those fixtures, with the
same three tiers as a base.

### The lane-agreement audit

`transfer_vllm` measures how far the fine-tune's vLLM lane sits from its fast lane on a sample of
its own rows, and reads that against the regime its base's qualification measured between the
same two lanes. Run it when a claim needs the two lanes to agree: before a finding about the
fine-tune is said to hold beyond one lane, when a serving change should have left the
computation alone, or when conclusions from the two lanes are joined at the effect level. Its
verdict is information, recorded with the extension when the entry names it; it is not a
condition of using the lane. The scoring (`check_transfer` in
`anamnesis/extraction/vllm/transfer.py`) takes matched vectors from any two lanes of one schema.

The audit scores the sample with the **base's** tolerance, under the **base's rule set**
(`transfer_rules.json` beside the base's tolerance), and reads it inside the base's regime when:

- per component (covered substrate, attention), at most two rows are over the base's maximum
  ratio of distance to path floor;
- 16 ordinary rows, evenly spaced over the sample's generation ids and chosen before anything is
  measured, keep their median ratio within the base's 90th percentile, with at most one of them
  over its 99th;
- per feature family, over the same 16 ordinary rows, the median of each row's largest
  |δ|/σ_cal in the family stays within the base's 90th percentile of that statistic;
- every row repeats byte for byte, one at a time and in batches of eight.

A single row's extremity is **reported, never gated**. Every row over a component's or a
family's base maximum is listed with its multiple, beside a bifurcation diagnostic the audit
runs itself: it compares the vLLM lane's residual stream with the numeric anchor's, token by
token, over the blocks from 35% of the depth, and reports the largest divergence and, at that
token, the largest one-sided channel. The largest single-row deviations found so far are tokens
where one engine forms a massive activation and the other does not.

The path floor is the fine-tune's own: how far the anchor's one-forward replay and its
token-by-token path disagree on the row under the lane's arithmetic, over a σ_cal fitted on the
sample. A row whose floor exceeds the base's limit (`BASE_MAX_FLOOR` in
`anamnesis/extraction/vllm/transfer.py`) is named and left out of the scoring, because on a
fragile reference the deviation measures the reference.

The audit counts rows and reads medians where the install check reads only a median, because
its deviation is a full vLLM-versus-fast-lane deviation, distributed like the base's own: some of
44 in-regime rows land over a maximum read from a couple of hundred rows by chance alone.

**Measured characteristics of each base's rule set.** Simulated in-regime samples, 300 of 56 rows
drawn from the base's qualification rows and scored against ceilings read from the rows left out,
read outside the regime 9.3% (3B), 9.0% (8B) and 5.0% (70B) of the time. The family gate is strong
for the attention families and weaker for some substrate families; the rule set and every receipt
name the families where a 1.5× drift is caught less than 80% of the time, with the drift it does
catch:

| base | weakly guarded families (the drift caught 80% of the time) |
|---|---|
| 3B | attention, keys, qk, values (2×); attn-spectral, output (2.5×) |
| 8B | attention, attn-spectral, keys, qk, values (2×); gate, output (2.5×) |
| 70B | keys, qk, values (2×); gate, attn-spectral (2.5×); residual (3×); output (4×) |

An audit against a base without a rule set is refused, naming the base.

### Running the audit

It needs, for the fine-tune:

1. a registry preset that `extends` the base's preset with the same architecture, dtype and
   layer plan, in a file named by `ANAMNESIS_MODELS` (see `CONTRIBUTING.md`);
2. its own calibration, fitted with `run_calibration` at that preset, and so at the base's
   dtype;
3. a replay manifest of 44 to 60 rows it generated in its own prompt regime, within the lane's
   1024-token context.

```
python -m anamnesis.scripts.transfer_vllm --key my-finetune --extends 70b \
    --preset my-finetune --model-path /path/to/checkpoint --calib-dir /path/to/calibration \
    --manifest runs/<run>/replay_manifest.json --out transfer-my-finetune
```

It captures the sample through the vLLM lane first, then loads the checkpoint through the fast
lane and the anchor on the same GPU. `--out` receives `transfer_receipt.json` and, whatever the
verdict, the fine-tune's `fixtures/` (the sample's vLLM vectors, with a tolerance read from the
sample) and `lane-entry.json`: the fixtures are the fine-tune's own lane output, not a certificate
of agreement. Only a sample the audit cannot score (a lane that disagrees with itself, a schema or
sample outside its scope) yields no fixtures. Exit status 0 means pass, 1 refuse (the reasons are printed and recorded in
the receipt), 2 that the audit could not run.

### Declaring the lane

`ANAMNESIS_VLLM_LANES` names lane files, separated the way `PATH` is. The `lane-entry.json` the
audit writes is one, and can be named as it is:

```json
{"lanes": {"my-finetune": {
  "extends": "70b",
  "preset": "my-finetune",
  "checkpoint_sha256": "<the checkpoint digest>",
  "calibration_dir": "/path/to/calibration",
  "calibration": {"positional_means.npz": {"size": 0, "sha256": "<digest>"},
                  "pca_model.pkl": {"size": 0, "sha256": "<digest>"}},
  "fixtures_dir": "fixtures",
  "transfer_receipt": "transfer_receipt.json",
  "transfer_receipt_sha256": "<the receipt's digest>"}}}
```

`transfer_receipt` and its digest are optional, and named together or not at all. Relative paths
resolve against the lane file's directory. With the file named, both commands take the key:

```
export ANAMNESIS_VLLM_LANES=transfer-my-finetune/lane-entry.json
python -m anamnesis.scripts.qualify_vllm --model my-finetune \
    --model-path /path/to/checkpoint --work-dir ./qualify-my-finetune
python -m anamnesis.scripts.run_vllm_replay --model my-finetune \
    --model-path /path/to/checkpoint --manifest runs/<run>/replay_manifest.json \
    --output banks/<run>-vllm
```

The calibration is read from the declared directory, or `--calib-dir`, and verified against
the declared sizes and digests before anything reads it; nothing is downloaded.

Every use passes the extension's guard (`anamnesis/extraction/vllm/extensions.py`): the lane is
refused by name unless its fixtures and tolerance load and carry its key, lane id, checkpoint
and calibration, and its preset still extends the base's with its architecture and layer plan
unchanged. When the entry names a receipt, the receipt must exist, match its declared digest,
and be about this key, base, lane id, checkpoint, calibration, fixtures and tolerance; its
verdict is recorded, not gated. A file whose entry reuses a key, repeats a checkpoint on one
base, extends anything but a shipped lane, or sets an inherited setting is refused whole.

An extension's lane id is the digest of its base's lane identity, the base's key and its
checkpoint digest. It differs from the base's and from every other extension's, so
`anamnesis/analysis/lane_guard.py` keeps an extension's rows out of any contrast with the base's
rows, exactly as it keeps two hosts' rows apart.

## How it works

Two processes per pass, never one. The engine process builds vLLM with an instrumented
attention backend that registers under the `TRITON_ATTN` name
(`anamnesis/extraction/vllm/backend.py`): an instrumented copy of the engine's Triton kernel
accumulates each query row's attention statistics inside the loop that computes the
attention output, and a bounded second pass reads the normalized probabilities of the rows
the attention families need. The capture (`capture.py`, `runner.py`) verifies every
scheduler step against the requested rows, hooks the model runner for the rest of the
substrate, runs each batch once without hooks and once with them, and refuses unless both
produced the same tokens and logprobs. Each row's substrate goes to disk beside a receipt of
its tensor contents.

The readout process (`readout.py`, `adapter.py`) has never imported the engine: vLLM's
batch-invariant mode replaces torch's matrix products for the whole process. It checks each
receipt against its file, then reduces the capture with the fast lane's reducers to the full
named vector. `anamnesis/extraction/vllm/runtime.py` starts both processes with the
environment each requires, so nothing needs to be exported by hand.

A model too large for one GPU can declare a tensor-parallel size in the envelope. Its engine
then runs one spawned worker per GPU, with the engine's custom all-reduce off, and the capture
runs inside every worker (`tensor_parallel.py`, `tp_worker.py`). Each rank's slice of the
attention heads, the QKV projection and the MLP gate is gathered in rank order to the
single-GPU layouts before anything reduces across heads, and a batch refuses unless every
rank captured the same bytes. The workers add each layer's partial products in their own
order, so the vectors differ from a single-GPU run of the same weights. A tensor-parallel lane
is therefore a lane of its own, with its own id, and a host checks it against that lane's own
fixtures; no shipped model declares one.
