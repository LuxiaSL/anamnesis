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

## Lineage

- [`anamnesis-pl`](https://github.com/LuxiaSL/anamnesis-pl) — the frozen record: the
  instrument exactly as used for the battery era. Every historical citation of a script path
  resolves there, forever.
- [`anamnesis-phase0`](https://github.com/LuxiaSL/anamnesis-phase0) — the archived origin
  (Phase 0, early 2026), superseded; see its banner before reading anything into it.
- [`PORT-MAP.md`](PORT-MAP.md) — where a name from the frozen record lives here, and which
  names stayed there.
- Operational documentation and ratified claims will live in the systema wiki (link to come).

## License

[MIT](LICENSE)
