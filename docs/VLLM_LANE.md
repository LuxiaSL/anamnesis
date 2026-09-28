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
`anamnesis/extraction/vllm/envelope.py` holds the declaration and the guard.

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
44 GB per pass for the 70B model, far less for the smaller ones.

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
CUDA runtime, torch, vLLM and anamnesis versions, checkpoint, fixtures, tolerance and engine
settings), and reused only while every field is equal. Change any of them and the check runs
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
