"""Enrichment passes.

The capture (phase 1) writes the trace skeleton; a pass fills fields in it.
Every pass has the same shape — take a trace and its residual tensor, mutate
the trace in place, and hand back a record of what it did:

    class Pass(Protocol):
        name: str
        def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord: ...

Because the residuals live on disk next to the trace, a pass never needs the
model. Re-running the SAE encoder on a saved trace costs no generation and no
6-second gemma load, which is what makes iterating on this code bearable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schema import PassRecord, Trace


class Pass(Protocol):
    name: str

    def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord: ...


def apply(pass_: Pass, trace: Trace, residuals: np.ndarray) -> PassRecord:
    """Run `pass_` and record it on the trace, replacing any earlier run of it.

    Re-running a pass overwrites the fields it owns, so its record should be
    replaced too — otherwise the trace claims two conflicting provenances for
    the same features.
    """
    record = pass_.run(trace, residuals)
    trace.passes = [p for p in trace.passes if p.name != record.name] + [record]
    return record
