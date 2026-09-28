"""The child processes of a vLLM lane pass: ``capture`` and ``reduce``.

:func:`anamnesis.extraction.vllm.runtime.run_step` starts each as
``python -m anamnesis.extraction.vllm.steps <step> <spec.json>`` in the
environment that step requires. They are separate interpreters because the
engine's batch-invariant mode replaces torch's matrix products for the whole
process, and the readout refuses to run anywhere that has happened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from anamnesis.extraction.vllm.runtime import capture_rows, reduce_rows

STEPS = {"capture": capture_rows, "reduce": reduce_rows}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step from its spec file; the exit code is the result."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in STEPS:
        print(f"usage: python -m {__name__} {{{','.join(STEPS)}}} SPEC.json", file=sys.stderr)
        return 2
    STEPS[args[0]](json.loads(Path(args[1]).read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
