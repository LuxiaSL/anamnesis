# anamnesis

**A signature is a lossy compression of the causal history of a forward pass over a span of text.**

Anamnesis is an instrument for collecting, analyzing, and intervening on these signatures:
extracting internal-state features from transformer forward passes (attention flow, gate
dynamics, key geometry, residual trajectories), classifying *how* a span was processed
orthogonally to *what* it says, replaying banked generations bitwise-deterministically, and
building gated steering vectors from the same substrate the readouts measure.

This repository is being refounded from the research codebase; the structure and code are
landing in reviewed increments. [`CONTRIBUTING.md`](CONTRIBUTING.md) states how code and claims
arrive, the documentation rule both are held to, and the gates a change passes.

## Quickstart

```bash
pip install -e ".[dev]"

# A corpus, drawn from a seed. No model, no accelerator, no network.
python -m anamnesis.scripts.make_synthetic_bank

# Read it: the eleven standing analyses over that corpus.
python -m anamnesis.scripts.run_gauntlet --run synthetic_demo
```

All eleven sections run. The command then **exits 3, not 0**, and that is the instrument
working: the sections needing the `geometry` extra cannot measure intrinsic dimension
without it, so they state that, three scorecard rows read `INSUFFICIENT_DATA` naming the
reading they lack, and the pass refuses rather than reporting itself complete. Install
that extra for a full pass, or `--skip` those sections to make the narrowing explicit.
Nothing is silently zero and nothing absent is scored — which is the single habit worth
taking from this repository.

The synthetic numbers are about nothing: they come from a generator, not a forward pass.
[`anamnesis/synthetic_bank.py`](anamnesis/synthetic_bank.py) states what the construction
deliberately does and does not put in them — read it before quoting anything the demo
prints.

## Installing

Artifacts — runs, calibration, analysis results — are written under the XDG data
directory, never inside the installation. `ANAMNESIS_OUTPUTS` points that elsewhere, and is
how a corpus on another disk is read.

Four extras, each carrying what one part needs and nothing else: `geometry`
(intrinsic-dimension estimators and persistent homology), `semantic` (sentence
embeddings), `judge` (provider clients), `adapters` (checkpoint-series replay). The
instrument imports and runs without all four, and a part that needs one reports its
absence rather than failing the import.

## Lineage

- [`anamnesis-pl`](https://github.com/LuxiaSL/anamnesis-pl) — the frozen record: the
  instrument exactly as used for the battery era. Every historical citation of a script path
  resolves there, forever.
- Operational documentation and ratified claims will live in the systema wiki (link to come).

## License

[MIT](LICENSE)
