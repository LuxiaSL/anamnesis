# The vLLM lane

The vLLM lane computes the same named feature vector as the fast lane
(`anamnesis/extraction/fast/`), from a vLLM engine instead of a hooked eager forward. It
exists for models whose replay is too large or too slow to run through Hugging Face
transformers on the hardware at hand, and it is a separate lane: its rows are never
combined with fast-lane rows inside one contrast.

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
installs carry them.

`--model-path` is a local directory holding the checkpoint's `config.json` and its
`*.safetensors` shards at the top level: Meta's Instruct release of the model, as published
on Hugging Face. The check hashes those files and refuses, before capturing anything, a
checkpoint whose digest differs from the one the fixtures were produced from; the refusal
prints both digests.

The lane reads the first GPU the process can see, and a receipt names that card. On a host
with several, `CUDA_VISIBLE_DEVICES` chooses one, and a different card is checked again.

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
rows evenly spaced over the rest. The **tolerance** is the largest deviation from the anchor
that comparison measured while the lane's effects were retained. Its row ceilings are
ratios of a row's **distance** (its standardized L2 difference over one component of the
vector: the **covered substrate**, which is every coordinate outside the attention
families, or the **attention** families) to the row's **path floor** (how far the anchor's
own two execution paths disagree on that row). A few rows whose path floor is too fragile to
divide by are left out of the ceilings and named in the file. The records the tolerance was
read from belong to that comparison and are not published; the file names each by its role
and pins it by sha256.

## Before first use on a host: the install check

```
python -m anamnesis.scripts.qualify_vllm --model 8b \
    --model-path /path/to/Llama-3.1-8B-Instruct --work-dir ./qualify-8b
```

It captures 44 fixture rows shipped with the package, twice one at a time and once in
batches of eight, reduces them, and compares them with the vectors the lane produced when
it was measured against the numeric anchor. The calibration those vectors were reduced with
is fetched on first use and verified against its pinned digests; `--calib-dir` names a local
copy instead.

The work directory holds the raw captures of the three passes until each is reduced: about
44 GB per pass for the 70B model, far less for the smaller ones. On one B200 the 70B check takes
about 35 minutes, most of it the three capture passes and a first read of the checkpoint to
digest it; a replay then runs at about 20 seconds a row, engine start included.

The check ends in one of three tiers:

- **identical** — every vector byte-identical to the fixtures. The host runs the lane
  itself, and its outputs carry the lane's recorded id.
- **conformant** — the vectors differ, but stay inside the perturbation the lane's
  measurement against the anchor showed harmless: each row's distance within its recorded
  ceiling, each continuous coordinate within its family's recorded maximum, and the
  unselected rows' median and tail within the recorded 90th and 99th percentiles. The host
  is a lane of its own, with an id derived from its fingerprint, and fully usable. A GPU
  model other than the one the fixtures were produced on is expected to land here, because
  floating-point reductions differ across hardware.
- **refused** — a different checkpoint, a host whose repeated or batched captures disagree,
  or a deviation outside the tolerance. The reasons are printed.

The receipt is cached under the output root, keyed by the host's fingerprint (GPU, driver,
CUDA runtime, torch, vLLM and anamnesis versions, checkpoint, fixtures, tolerance, engine
settings and the lane's source), and reused only while every field is equal. Change any of them and the check runs
again; `--refresh` runs it regardless.

Exit status 0 means identical or conformant, 1 refused, 2 that the check could not run (a
missing engine, checkpoint or calibration, with the reason). A refusal lists what differed:
a host whose repeated or batched captures disagree is not deterministic in this
configuration, and a deviation outside the tolerance means this hardware and software do
not compute what the lane computes closely enough to share its results.

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
and a `conformant` host's bank joins only banks from that host. Every reader that takes a
signature directory reads it, `run_gauntlet` among them.

## Extension lanes (fine-tunes of a qualified model)

A fine-tune of `3b`, `8b` or `70b` is the same network with different weights. It can use its
base's lane as an **extension lane** of its own, without a qualification campaign and without
editing this package, once a transfer check has shown that the lane measures it inside the
regime its base was qualified in. The owner of the fine-tune runs the check and keeps the
result; this package learns no fine-tune's paths.

### What carries over, and what the check measures

Everything that depends only on the architecture and its shapes carries over from the base
unchanged: the instrumented kernel and its second pass, the determinism of the dispatch, the
feature schema, the capture routes, the dtype, the logprob handling and every engine setting in
`anamnesis/extraction/vllm/envelope.py`. An extension may not set any of them.

What the weights can change is how far the vLLM lane drifts from the fast lane on a row:
threshold proximity, large activations and gate sparsity are statistics of the activations.
The base's qualification measured its effects surviving the lane as a function of the size of
that drift, and its tolerance records the drift it measured. So the check measures the
fine-tune's drift on a sample of its own rows and scores it with the **base's** tolerance, the
way the install check scores a host:

- each row's covered-substrate and attention distance over its path floor, under the base's
  ceilings, and each continuous coordinate under its family's recorded maximum;
- the ordinary rows' median ratio under the base's 90th percentile, and at most one ordinary row
  over its 99th. The 16 ordinary rows are evenly spaced over the sample's generation ids, chosen
  before anything is measured: the median and tail gates compare the sample with population
  quantiles of the base, which only means something over rows drawn without regard to how far
  they deviate. The other rows are labelled by the deviation-ranked selection rules for the
  fixture set, and those labels feed no gate;
- the path floor is the fine-tune's own: how far the numeric anchor's one-forward replay and
  its token-by-token path disagree on the row, over a σ_cal fitted on the sample. A row whose
  floor exceeds the base's limit (`BASE_MAX_FLOOR` in
  `anamnesis/extraction/vllm/transfer.py`) is named and left out of the scoring, because on a
  fragile reference the deviation measures the reference rather than the lane;
- every row is also captured again one at a time and in batches of eight, and all three
  vectors must be byte-identical.

The tail rule is deliberately looser than the install check's, which bounds every ordinary
row by the 99th percentile. A host reproducing a qualified lane deviates from its fixtures by
far less than the qualification's own deviations, so a single row over the 99th percentile is
already a sign the host is somewhere else. A fine-tune's deviation is a full vLLM-versus-fast-lane
deviation on new weights, expected to be distributed like the base's own: each ordinary row then
lands over the base's 99th percentile about one time in a hundred even when the fine-tune is
exactly in regime, and bounding all 16 would refuse such a fine-tune about 15% of the time.
Allowing one keeps that false refusal near 1% while still refusing a heavy tail. The receipt
records how many ordinary rows exceeded the 90th and the 99th percentile. The fine-tune's own
fixtures keep the same id-chosen ordinary rows, and its hosts' install checks apply the install
check's own rules to them.

A fine-tune inside the base's regime inherits the base's evidence that effects survive the
lane, for the same reason a conformant host does. One outside it needs a qualification of its
own. Which contrasts the fine-tune resolves is a question about the fine-tune, answered by its
owner's own experiments; the check answers only whether the lane measures it faithfully.

A pass also writes the fine-tune's own fixtures and tolerance, built from the sample by the same
rules as the shipped ones, so each host that runs the fine-tune checks its install exactly as
it would for a base.

### Running the check

The check needs, for the fine-tune:

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
lane and the anchor on the same GPU. `--out` receives `transfer_receipt.json`, and on a pass
`fixtures/` and `lane-entry.json`. Exit status 0 means pass, 1 refuse (the reasons are
printed and recorded in the receipt), 2 that the check could not run.

### Declaring the lane

`ANAMNESIS_VLLM_LANES` names lane files, separated the way `PATH` is. The `lane-entry.json`
a pass writes is one, and can be named as it is:

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

Relative paths resolve against the lane file's directory. With the file named, both commands
take the key:

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

Every use passes the extension's guard (`anamnesis/extraction/vllm/extensions.py`), which
refuses the lane by name unless its receipt exists, matches its declared digest and passed;
unless the receipt, the entry and the fixtures agree on the base lane, the checkpoint, the
calibration, the fixture set and the tolerance; unless the base's shipped tolerance is still the
one the receipt was scored against; and unless the preset still descends from the base's with
its architecture and layer plan unchanged. A file that gives an entry a shipped key or another
entry's key, declares one checkpoint on one base twice, extends anything but a shipped lane, or
sets an inherited setting is refused whole.

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
