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
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
