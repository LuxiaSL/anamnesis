"""Scripts: the command-line entry points, thin over package capability.

A script here parses arguments, resolves paths, calls into the package and
prints or writes a receipt. It holds no numerics and no logic another script
would want to import — capability that two entry points need belongs in the
package, not in a sibling script.

* `run_gpu_replay.py` — replay a banked run's realized token sequences through
  the fast lane and write signatures.
* `qualify_box.py` — measure whether this machine's fast lane agrees with the
  numeric anchor, and name the lane identity its outputs will carry.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
