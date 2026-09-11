"""Enrichment passes.

The capture (phase 1) writes the trace skeleton; a pass fills fields in it.
Every pass has the same shape — take a trace and its residual tensor, mutate
the trace in place, and hand back a record of what it did:

    class Pass(Protocol):
        name: str
        def run(self, trace: Trace, residuals: np.ndarray) -> PassRecord: ...

Because the residuals live on disk next to the trace, most passes never need
the model. Re-running the SAE encoder on a saved trace costs no generation and
no 6-second gemma load, which is what makes iterating on this code bearable.

The logit lens is the exception: `W_U` is 2304 x 256_000, far too big to sit in
a sidecar beside every trace, so it takes the model handle instead. It is
optional on the pass (`LogitLensPass.model`) and resolved from `model_cache` if
absent, so the signature above still holds.
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

    Each run replaces that pass's complete result, even when a layer subset
    is requested. SAE reruns also invalidate dependent labels and positions.
    A failure leaves the previous document and its provenance intact.
    """
    # Work on a private document. A failing pass must not leave half a new
    # measurement alongside the old provenance. Residual tensors are not copied.
    candidate = trace.model_copy(deep=True)
    invalidated = {pass_.name}
    if pass_.name == "sae":
        invalidated.update({"labels", "layout"})
        candidate.labels = {}
        candidate.layout = {}
        for step in candidate.steps:
            for state in step.layers:
                state.features = []
                state.l0 = None
    elif pass_.name == "labels":
        candidate.labels = {}
    elif pass_.name == "layout":
        candidate.layout = {}
    elif pass_.name in {"lens", "attribution"}:
        for step in candidate.steps:
            for state in step.layers:
                if pass_.name == "lens":
                    state.logit_lens = None
                else:
                    state.edges = []
    record = pass_.run(candidate, residuals)
    candidate.passes = [p for p in candidate.passes if p.name not in invalidated] + [record]
    for name in type(trace).model_fields:
        setattr(trace, name, getattr(candidate, name))
    return record
