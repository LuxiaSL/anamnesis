"""The write side of the instrument: building an intervention and gating it.

Reading a signature says how a forward pass ran. Steering asks the converse
question — whether a direction in the residual stream can *make* a pass run that
way — and it is the half of the instrument where a number is easiest to get and
hardest to earn. Four modules are the four things that stand between a candidate
direction and a citable steering result:

:mod:`~anamnesis.steering.vectors`
    Construction. Mean difference, whitened mean difference, band projection,
    orthogonalization, matched nulls and the dose currency, with the sweep law
    that picks a site.

:mod:`~anamnesis.steering.screens`
    Whether a direction can be injected at all: where it sits in the residual
    covariance, how much of it lives inside a chosen eigenband, and which layer
    separates the axis you mean to steer.

:mod:`~anamnesis.steering.gates`
    Whether a cell is valid: matched-token on-policy agreement, the
    exactly-zero upstream delta a deterministic replay owes, a direction's own
    matched null, and the shape of the score distribution a claim rests on.

:mod:`~anamnesis.steering.readouts`
    What an intervention did: movement along the axis against movement off it,
    read against support-matched nulls, and the checkpoint-series analogue for
    installs.

**Everything here is per-model.** A vector is built in one model's residual
basis, whitened by that model's covariance, dosed in that model's residual norm
and injected at a site chosen by that model's own layer sweep. None of those
four quantities transports: carrying a vector across models is a transport
problem with its own instrument, and this package does not pretend to solve it
by reusing an array of the right width.

:mod:`~anamnesis.steering.covariance` sits beside the four as an implementation
detail worth naming: the shrinkage covariance a whitened vector is built from,
computed at residual width on whichever device is at hand. It is a port of the
estimator :mod:`~anamnesis.steering.vectors` calls rather than a second estimator,
and it carries a different function name so a reader always knows which of the two
they hold — with a command that measures the two against each other on one box.
"""
