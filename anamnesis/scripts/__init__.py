"""Scripts: the command-line entry points, thin over package capability.

A script here parses arguments, resolves paths, calls into the package and
prints or writes a receipt. It holds no numerics and no logic another script
would want to import — capability that two entry points need belongs in the
package, not in a sibling script. ``tests/test_scripts_are_shims.py`` enforces
that: no script imports a script, no package module imports a script, and a
function a script defines is used by that script and its own test and nowhere
else. This is the single structural defence against the directory growing back
into a pile of near-duplicates, which is what it grew into once.

**The pipeline, in the order a run moves through it.**

* `run_calibration.py` — measure a model's positional means and residual basis.
  Everything corrected downstream rests on these two files.
* `run_extraction.py` — generate, featurise and save in one process. The
  shortest path to a signature, and the right one for a run that fits on one
  device.
* `run_gen_tokens.py` — generate and bank the realized token ids, nothing else.
  Phase one of the replay gateway, on one device or fanned out over many.
* `run_replay.py` — teacher-force banked token sequences back into signatures.
  Phase two, and the reason a change to the capture surface costs a replay
  rather than a corpus.
* `run_persistent_replay.py` — the same replay through workers that stay
  loaded, plus the byte-identity gate that licenses them.
* `run_recompute.py` — recompute signatures from banked raw tensors on a CPU, so
  a feature set is an experimental variable rather than a commitment.

**The fast lane, and qualifying a machine for it.**

* `run_gpu_replay.py` — replay through the fast lane on a single device.
* `qualify_box.py` — measure whether this machine's fast lane agrees with the
  numeric anchor, and name the lane identity its outputs will carry.
* `judge_2afc.py` — draw a blind two-alternative forced choice over banked text,
  bank the key apart from the packet, and report the rate with its interval.
* `judge_likert.py` — the other judging paradigm: rate one text on every mode at
  once, which is what yields purity and the cross-channel correlation.

**A corpus without a device.**

* `make_synthetic_bank.py` — write a bank of the right shape from a seed, so everything
  below can be run before there is a model to run it on. The numbers are drawn and mean
  nothing about any model; `anamnesis.synthetic_bank` states what the construction does
  and does not put in them.

**Reading a bank.**

* `run_gauntlet.py` — the eleven standing analyses over one corpus.
* `run_binary_prompt_swap.py` — whether the signal is the instruction or the
  execution.
* `run_subfamily_decomp.py` — which part of a family carries its signal.
* `run_cross_run_transfer.py` — whether two corpora's mode vocabularies name the
  same thing.
* `analyze_complementarity.py` — the readings that exist only between banked
  results.
* `encoder_on_raw.py` — hand features against the raw state they summarize, on one
  split.
* `leak_gate.py` — whether a feature set carries the signal or the topic.
* `pathsig_features.py` — banked trajectories to a path-signature design matrix
  and its null.

**Metrology, and onboarding a model.**

* `onboard_model.py` — whether a new preset row is true of its checkpoint. Run it
  before spending anything else.
* `stage0_floors.py` — the faithfulness replays, then the floors and the n-min law
  they imply.
* `census.py` — which rows the internals see that the cheap readers miss.
* `ledoit_wolf_gpu.py` — whether this box's fast covariance path agrees with the
  reference it is a port of.
* `sepcma.py` — whether a direction search can climb at all on the budget it would
  be given.

**Steering, and the series a replay runs against.**

* `steer_vectors.py` — sweep, build, screen, gate, null and lever.
* `qual_extract.py` — the same dose ladder read by eye, matched by prompt.
* `build_checkpoint_series.py` — a training directory's adapter checkpoints as a
  replay series.
* `run_replay_multickpt.py` — that series replayed on one model load per worker.
* `train_contrastive_projection.py` — fit the learned projection a model's
  signatures are computed through.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
