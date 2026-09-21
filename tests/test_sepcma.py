"""The budget probe as a command: what it prints, and what its exit status means.

The command exists so a budget question can be answered before any real evaluation is
spent, which makes its **exit status** the interface: zero when the search climbed on the
idealized landscape, one when it did not. A caller scripting a decision around it reads
that and nothing else, so it is what is pinned here, for both the planted-direction probe
and the convex sanity check.
"""

from __future__ import annotations

from anamnesis.scripts.sepcma import main


def test_an_adequate_budget_exits_zero() -> None:
    assert main(["--dim", "64"]) == 0


def test_an_inadequate_budget_exits_one() -> None:
    assert main(["--dim", "64", "--budget-multiple", "1"]) == 1


def test_several_dimensions_are_probed_in_one_call_and_any_failure_shows() -> None:
    assert main(["--dim", "32", "--dim", "64"]) == 0


def test_the_convex_check_is_its_own_verdict() -> None:
    assert main(["--sphere", "--dim", "32"]) == 0
