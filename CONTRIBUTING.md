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
apply to three surfaces: comments, docstrings, and the message a `raise` or a logging call says
out loud. A stranger meets that last one at the moment something breaks, which is the worst
moment to hand them a pointer they cannot follow. The rules never apply to string data — a
fixture, a dict key, a filename, a row a run writes into a report — which is content the program
handles rather than prose a reader is meant to believe. Inside an f-string, the substitutions are
code and are read as such.

### 1. State what is true now

A comment describes the code as it stands. It does not narrate how the code got there.

No `previously`, `used to <verb>`, `changed in`, `we now`, `no longer`. No bare `TODO`, `FIXME`
or `HACK` — a known gap is either a refusal the code makes explicitly, a test that pins the
current behaviour, or an issue, not a word left in a file for somebody to find.

**No dates.** Not on an edit, and not on evidence either. Git records when a line changed, and a
dated measurement or decision belongs in the record that holds it, where a reader can weigh the
whole thing instead of a fragment of it. What the code needs is the standing fact the code
depends on, stated in the present tense; the date of the run that established it is provenance,
and provenance is not what a comment is for.

```python
# Good — the constraint, and why it is load-bearing.
# Flash attention and SDPA return no attention weights, so the eager kernel is a
# correctness requirement rather than a preference.

# Good — the standing fact, with no date and nothing to chase.
# The effect is below the threshold a blind judge resolves at this dose, so the
# readout is reported as a bound rather than as a difference.

# Bad — narrates an edit.
# We now pin the thread pools; this used to inherit the parent's setting.

# Bad — dates the evidence instead of stating it.
# Sub-perceptual at this dose (census 2026-07-12, n=80 per cell).
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

It also does not mean a machine, a host, an account, a scheduler, or a sibling repository. A
reader has no login on the box that taught us something, and a comment crediting that box states
provenance where the constraint belongs — the thread budget, the memory ceiling, the shared-core
arithmetic. Those hold wherever the code runs, which is what made them worth a comment in the
first place. Unlike the rules above, this one is a reviewer's catch rather than a checker's, by
design: a public gate would have to enumerate the names, and publishing the list defeats the
point of keeping them out.

It also does not mean a **provenance citation** — a `§` section of a document that is not here, a
named arm or milestone code, a pre-registration, an addendum or codicil, a bare item code like
`14e`, a commit hash. The code is the receipt for what the code does, and git is the receipt for
how it came to do it; a citation of the plan it was written under resolves nowhere for the reader
in front of it. Numbered sections of the analysis are a different thing and stay: section 9 is a
module in this tree, and a reader can open it.

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

# Bad — credits the machine instead of stating the budget it implies.
# One BLAS thread per pool worker (the shared-box convention on our cluster).

# Bad — cites the plan instead of stating the constraint.
# One analysis template per arm x model (prereg §6b).
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
| G2 test retention | a change that shrinks the suite shrinks the code by at least as many lines | `python -m tools.check_test_retention --base main` |

G2 counts physical lines of Python under `tests/` (the suite) and everywhere else (the code), at
your branch and at the point it left `main`. Adding tests, adding code, or deleting a module
together with its tests all pass: the suite runs at about half the size of the code, so code
removed with its own tests takes out roughly two code lines for every test line. What fails is a
suite that shrinks while the code it covered stays. If it fails, either restore the tests or
delete the code they covered in the same change. Merging duplicate tests without touching the code
fails too, because the line counts cannot tell it apart from dropped coverage; that is a change to
waive in writing, with the reason beside the receipt.

Each G3 rule is pinned by a corpus rather than by reading: `tests/test_gate_fixtures.py` holds the
strings each checker must catch and the legitimate prose it must stay quiet on, one case per
idiom, on each of the three surfaces. A rule that stops seeing something fails a test there
instead of quietly reporting PASS. Closing a blind spot starts by adding the string that slipped
through.

Both G3 pattern files — `tools/timelessness_allowlist.txt` and `tools/referents_allowlist.txt` —
carry a reason above every entry, and an entry is a claim that its reason is true *now*. When a
reason expires, the entry goes, whatever it used to protect.

`tools/surface_report.py` reports the size of the codebase in code tokens and documentation
words. It is a trend instrument, not a gate: it says which direction the repository is moving.

G1 is the one gate CI cannot run: it reads banked signatures, hundreds of gigabytes of them, and
asks whether the same inputs still produce the same features — and a different machine's
floating-point reductions differ anyway, so it runs on the box that holds the data and its receipt
is attached to the pull request. The other three, G2 included, run in CI. A gate may be waived only
in writing, with the reason recorded beside the receipt — the failure is recorded, never the gate
rewritten.

## Adding to the instrument

- **A feature family** implements the contract in `anamnesis/extraction/feature_families/`
  (`__init__.py` states it), registers its names with the taxonomy in `anamnesis/feature_map.py`,
  and arrives with a test. `path_signature.py` is the worked example of a large family;
  `attn_res.py` is the worked example of a family that activates only when an architecture
  supplies its substrate.
- **A model** is a row of data and a validation pass. The row goes in a JSON file of the shape
  `anamnesis/config/models.json` has, and `ANAMNESIS_MODELS` names it — several files, separated
  the way `PATH` is — so your own checkpoint is onboarded without editing this package. Then
  `python -m anamnesis.scripts.onboard_model --model <your key>`, which loads it through the
  instrument's own loader and refuses with a reason if the layer plan, the hook targets or the
  attention kernel are wrong. Every per-model fact lives in the row, because a layer index means a
  different fraction of the network in each model — and the row is the only place the layer count is
  written, so the depth bands, the battery metadata and the loader cannot disagree about it. A row
  may not redefine a shipped one: banked signatures mean the shipped row, so a variant is a new key,
  and a collision is refused by name rather than merged.
- **A mode set** is the same shape of change. Rows go in a file like
  `anamnesis/modes/mode_sets.json`, named by `ANAMNESIS_MODE_SETS`, and
  `run_extraction.py --modes <your set>` offers it as soon as it is readable. Declare each mode's
  `index` explicitly: an index reaches the generation seed, so it is part of what a banked
  coordinate reproduces and is never left to a dict's order. A set that `extends` another inherits
  the parent's modes, indices and format constraint unchanged, which is how the five-mode subset
  sits inside the eight. A vocabulary a corpus carries that this package has no prompts for is a
  `label_vocabularies` row instead, and a prediction relating two vocabularies is a
  `mode_mappings` row.
- **A machine** qualifies itself: `python -m anamnesis.scripts.qualify_box`. Different hardware
  gives different numbers, which is expected rather than wrong; results from different machines
  must not be mixed inside one contrast.
