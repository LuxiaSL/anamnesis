# anamnesis

**A signature is a lossy compression of the causal history of a forward pass over a span of text.**

Anamnesis is an instrument for collecting, analyzing, and intervening on these signatures:
extracting internal-state features from transformer forward passes (attention flow, gate
dynamics, key geometry, residual trajectories), classifying *how* a span was processed
orthogonally to *what* it says, replaying banked generations bitwise-deterministically, and
building gated steering vectors from the same substrate the readouts measure.

The instrument is the whole of it: extraction, the replay gateway, the eleven standing
analyses, the steering side and the judging channel, with a test suite and four gates over
them. [`CONTRIBUTING.md`](CONTRIBUTING.md) states how code and claims arrive, the
documentation rule both are held to, and the gates a change passes.

## The question it was built for

The shipped protocol asks a model to write about a topic in one of five *processing
modes* — linear, analogical, socratic, contrastive, dialectical — each a system prompt such
as *"Develop your exploration through a sequence of questions and provisional answers"*,
and all five under one format constraint: flowing paragraphs, no lists, no headers. Four of
the five produce prose a reader struggles to tell apart. A run asks whether the signatures
of those forward passes separate the modes anyway, and whether what separates them is the
computation rather than the words, the length or the topic — which is what the length
baseline, the semantic section and the prompt-swap test exist to rule out. The prompts are
rows in [`anamnesis/modes/mode_sets.json`](anamnesis/modes/mode_sets.json); a researcher's
own modes are a file of the same shape named by `ANAMNESIS_MODE_SETS`.

## Start here

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the map: what a signature is, the three
extraction layers and why they stay apart, the replay gateway, the arithmetic-lane identity
every signature carries, the fail-closed doctrine, and which file to open next. Ten minutes.

The four package docstrings it routes you to are worth reading in their own right:

- [`anamnesis/__init__.py`](anamnesis/__init__.py) — the definition, and what each
  top-level module is for.
- [`anamnesis/extraction/__init__.py`](anamnesis/extraction/__init__.py) — the write side:
  what a forward pass is read into, and the code that reads it.
- [`anamnesis/modes/__init__.py`](anamnesis/modes/__init__.py) — the processing-mode
  prompts a run's labels refer to, and the confound test that travels with them.
- [`anamnesis/analysis/gauntlet/__init__.py`](anamnesis/analysis/gauntlet/__init__.py) —
  the read side: the eleven analyses and the registry that dispatches them.

[`anamnesis/scripts/__init__.py`](anamnesis/scripts/__init__.py) is the command inventory,
grouped by the order a run moves through the pipeline.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"

# A corpus, drawn from a seed. No model, no accelerator, no network.
python -m anamnesis.scripts.make_synthetic_bank

# Read it: the eleven standing analyses over that corpus. Budget ~15 minutes.
python -m anamnesis.scripts.run_gauntlet --run synthetic_demo
```

The gauntlet over the synthetic corpus takes about fifteen minutes of CPU, roughly
two-thirds of it inside the classification section. It prints each section as it starts and
its elapsed time as it finishes, so the pass is legible while it runs — but a command that
sits for ten minutes on one line is working, not hung.

All eleven sections run. The command then **exits 3, not 0**, and that is the instrument
working. Two sections measure against something the `dev` install does not bring: the
intrinsic-dimension section needs the `geometry` extra's estimators, and the semantic section
measures content by sentence embeddings, which the `semantic` extra provides. Without them each
section states what it lacks, three scorecard rows read `INSUFFICIENT_DATA` naming the reading
they miss, and the pass refuses rather than reporting itself complete. Install
`".[dev,geometry,semantic]"` for a full pass — the embedding model downloads on first use — or
`--skip` those sections to make the narrowing explicit.
Nothing is silently zero and nothing absent is scored — which is the single habit worth
taking from this repository.

The synthetic numbers are about nothing: they come from a generator, not a forward pass.
[`anamnesis/synthetic_bank.py`](anamnesis/synthetic_bank.py) states what the construction
deliberately does and does not put in them — read it before quoting anything the demo
prints. One consequence catches readers out: the demo still ranks feature blocks against
each other, and that ranking follows the block widths and the seed. It is a property of
the fixture, and the pass says so where it prints it.

## A first real run, on a CPU

The quickstart's numbers come from a generator. A real signature needs a real model, and
[`examples/models/qwen2.5-0.5b-instruct.json`](examples/models/qwen2.5-0.5b-instruct.json)
adds one a laptop CPU can run: a small, ungated checkpoint, as a registry row the package
does not ship.

```bash
uv pip install -e ".[dev,semantic]"
export ANAMNESIS_MODELS=examples/models/qwen2.5-0.5b-instruct.json
export ANAMNESIS_OUTPUTS=$PWD/outputs

# Does the capture path work on this model at all?
python -m anamnesis.scripts.onboard_model --model qwen2.5-0.5b

# Its calibration, far enough to cover what the extraction below reads.
python -m anamnesis.scripts.run_calibration --model qwen2.5-0.5b --num-prompts 50 \
    --required-through 600 --suppress-eos

# Fifty generations, five modes, and the families over their banked tensors.
python -m anamnesis.scripts.run_extraction --model qwen2.5-0.5b --run-name first \
    --n-samples 10 --save-raw
python -m anamnesis.scripts.run_recompute --model qwen2.5-0.5b --run-dir outputs/runs/first \
    --calib-dir outputs/calibration/qwen2.5-0.5b --raw-subdir raw_tensors \
    --metadata-subdir signatures

python -m anamnesis.scripts.run_gauntlet --run first --sig-dir outputs/runs/first/signatures
```

Each step refuses rather than guessing. This model's answers mostly end before a late
position, so too few calibration prompts would reach position 600 on their own and the
calibration would refuse to write; `--suppress-eos` keeps each one generating to its budget,
at the cost of the latest positions' means coming partly from text past the end of an
answer. If the extraction's prompts reach further than position 600, it stops before
sampling and names the position to calibrate through. On a laptop CPU the whole walk takes
tens of minutes and most of it is generation.

## Installing

Artifacts — runs, calibration, analysis results — are written under the XDG data
directory, never inside the installation. `ANAMNESIS_OUTPUTS` points that elsewhere, and is
how a corpus on another disk is read.

`torch` is a core dependency, and a default install fetches its CUDA build and the vendor
libraries beside it — several gigabytes. On a machine with no accelerator,
`uv pip install --torch-backend=cpu -e ".[dev]"` fetches the CPU build instead; the
quickstart and everything on the reading side run on it unchanged.

Four extras, each carrying what one part needs and nothing else: `geometry`
(intrinsic-dimension estimators and persistent homology), `semantic` (sentence
embeddings), `judge` (provider clients), `adapters` (checkpoint-series replay). The
instrument imports and runs without all four, and a part that needs one reports its
absence rather than failing the import.

## Lineage

- [`anamnesis-pl`](https://github.com/LuxiaSL/anamnesis-pl) — the frozen record: the
  instrument exactly as used for the battery era. Every historical citation of a script path
  resolves there, forever.
- The systema — the project's knowledge base of graded claims and operational notes — is not
  public yet. Until it is, the claims this README states are the ones the repository stands
  behind, and `CONTRIBUTING.md` says how a new one is raised.

## License

[MIT](LICENSE)
