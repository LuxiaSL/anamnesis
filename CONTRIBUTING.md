# Contributing

Two kinds of contribution arrive here, and they are handled differently.

**Code** is a pull request against this repository. CI runs the gates described below; a change
merges when they pass.

**Claims** are never merged as claims. A contribution that asserts a finding — that a signature
separates something, that an intervention has an effect, that a number holds — enters as a
candidate entry in the systema, is evaluated and replicated where warranted, is graded, and only
then becomes visible. Code that *enables* a claim is welcome in a pull request; the claim itself
travels the other road.

## The documentation rule

Documentation and comments here obey two rules at once. Both are mechanically checked, and both
apply to comments and docstrings — never to string data, which is content the program handles
rather than prose a reader is meant to believe.

### 1. State what is true now

A comment describes the code as it stands. It does not narrate how the code got there.

No `previously`, `used to <verb>`, `changed in`, `we now`, `no longer`. No bare `TODO`, `FIXME`
or `HACK` — a known gap is either a refusal the code makes explicitly, a test that pins the
current behaviour, or an issue, not a word left in a file for somebody to find. A date belongs
in a comment only when it dates *evidence*: a measurement, a ruling, a pre-registration, a
finding that a reader may want to weigh. A date on an edit is history and goes.

```python
# Good — the constraint, and why it is load-bearing.
# Flash attention and SDPA return no attention weights, so the eager kernel is a
# correctness requirement rather than a preference.

# Good — a date on evidence.
# Sub-perceptual at this dose (census 2026-07-12, n=80 per cell).

# Bad — narrates an edit.
# We now pin the thread pools; this used to inherit the parent's setting.
```

### 2. Every referent must be reachable from this repository

If a comment or docstring names something, a reader holding only this repository must be able to
go look at it. That means a module path that exists here, a file in this tree, a symbol this
package defines, or a public URL.

It does **not** mean a path into a private tree, a planning document, a design note, an internal
ticket, or a conversation. It does not mean "see the earlier discussion", "as decided elsewhere",
or a pointer whose substance lives somewhere the reader cannot follow. A buried referent is the
same failure as a stale date: it makes the reader depend on context they do not have, and it ages
into a dead end.

When the substance is short, state it inline. When it is long and public, link it. When it is
long and private, restate the part this code depends on — one sentence of standing fact beats a
citation nobody can open.

```python
# Good — names a module in this repository.
# The read-side gate on lane identity is anamnesis.analysis.lane_guard.

# Good — the substance, stated, with nothing to chase.
# Signatures from different machines must not be joined inside one contrast:
# floating-point reductions are not associative, so the same code on another box
# produces different last digits.

# Bad — a referent this repository does not contain.
# See research/notes/v3-delta-memo for why the localization claim was revised.

# Bad — defers the meaning to a conversation.
# Kept for the reason discussed when this was ruled on.
```

The rule has one consequence worth stating plainly: **a claim in a docstring must match the code
under it.** "This replaces X" is false while X is still running, and a reader who checks will
trust the next sentence less. Describe what is, and let what is be the argument.

### Keep the prose style already here

Present tense. Say why, not only what — the constraint a reader would otherwise violate, the
failure a refusal exists to prevent. Document a refusal where the refusal happens. Module
docstrings carry the shape of the module; a function's docstring carries what a caller needs:
arguments, units, the shape of what comes back, and how it fails.

## The gates

A pull request merges when these pass. Each is a command you can run.

| gate | what it checks | command |
|---|---|---|
| tests | the suite | `pytest tests/` |
| G3 documentation | the two rules above | `python -m tools.check_timelessness --root anamnesis tools tests` and `python -m tools.check_referents --root anamnesis tools tests` |
| G4 import closure | every module reachable from a command or a test; no orphans | `python -m tools.check_import_closure --package anamnesis --roots anamnesis/scripts tests` |
| G1 data compatibility | banked artifacts in, identical features out | `python -m tools.g1_hash_manifest` — see its own help; needs banked data |
| G2 consolidation | a consolidated module smaller than the donors it replaced; test lines not shrinking | `python -m tools.g2_loc_report --repo .` |

`tools/surface_report.py` reports the size of the codebase in code tokens and documentation
words. It is a trend instrument, not a gate: it says which direction the repository is moving.

G1 and G2 need artifacts and a tree state a runner does not have, so they are run per pull
request and their output attached rather than executed in CI. A gate may be waived only in
writing, with the reason recorded beside the receipt — the failure is recorded, never the gate
rewritten.

## Adding to the instrument

- **A feature family** implements the contract in `anamnesis/extraction/feature_families/`
  (`__init__.py` states it), registers its names with the taxonomy in `anamnesis/feature_map.py`,
  and arrives with a test. `path_signature.py` is the worked example of a large family;
  `attn_res.py` is the worked example of a family that activates only when an architecture
  supplies its substrate.
- **A model** is a preset row in `anamnesis/config/models.py` and a validation pass:
  `python -m anamnesis.scripts.onboard_model`. Every per-model fact lives in the row, because a
  layer index means a different fraction of the network in each model.
- **A machine** qualifies itself: `python -m anamnesis.scripts.qualify_box`. Different hardware
  gives different numbers, which is expected rather than wrong; results from different machines
  must not be mixed inside one contrast.
