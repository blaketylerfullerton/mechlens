"""The trace schema — the JSON contract the whole project writes into.

Shape:

    Trace
      └── steps: [TokenStep]          one per token in the final sequence
            ├── token:  TokenInfo     what this position holds
            ├── logits: LogitSummary  what the model predicts *after* it
            └── layers: [LayerState]  one per transformer layer (26 for gemma-2-2b)
                  ├── resid_norm      filled now (phase 1)
                  ├── logit_lens      filled by the logit-lens pass (phase 2)
                  ├── features        filled by the SAE encoder    (phase 3)
                  └── edges           filled by the attribution pass (phase 4)

Everything after phase 1 *adds to* this structure rather than reshaping it, so
the later passes are pure enrichment: load a trace, fill a field, save it back.

Raw residuals do not live in the JSON. A [n_tokens, n_layers, d_model] float32
tensor is ~12MB for a 50-token trace, which makes the JSON unreadable and slow
to parse in the browser. It is written next to the JSON as a .npy sidecar and
described by `Trace.residuals` (see store.py).

This module deliberately imports nothing but pydantic — a consumer that only
wants to read traces should not need torch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# Bump on any breaking change to the models below. Readers should check it.
SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# tokens & logits
# --------------------------------------------------------------------------


class TokenInfo(BaseModel):
    """A single token at a single position in the sequence."""

    position: int  # absolute index in the sequence, BOS at 0
    token_id: int
    text: str  # decoded piece, leading whitespace preserved
    source: Literal["prompt", "generated"]


class TopToken(BaseModel):
    """One entry of a next-token distribution."""

    token_id: int
    text: str
    logit: float
    prob: float


class LogitSummary(BaseModel):
    """The model's next-token distribution at a position, summarised.

    The full 256k-entry vocab distribution is not worth keeping; top_k plus
    entropy is enough to see where the model is confident and where it is not.
    """

    top_k: list[TopToken]
    entropy: float  # nats, over the full vocab
    chosen: TopToken | None = None  # token actually appended; None at the last position


# --------------------------------------------------------------------------
# per-layer state — mostly placeholders that later phases fill in
# --------------------------------------------------------------------------


class Feature(BaseModel):
    """An active SAE feature at this layer/position (phase 3)."""

    index: int
    activation: float
    label: str | None = None  # e.g. from Neuronpedia


class LogitLens(BaseModel):
    """This layer's residual stream decoded through the unembed (phase 2)."""

    top_k: list[TopToken]
    entropy: float


class NodeRef(BaseModel):
    """Points at one site in the trace: a layer/position, optionally a feature.

    `feature is None` means the residual stream itself at that site.
    """

    layer: int
    position: int
    feature: int | None = None


class Edge(BaseModel):
    """An attribution edge *into* the LayerState that holds it (phase 4).

    The target is implied by the containing LayerState, so only the source and
    the strength are stored — that keeps the trace roughly linear in edge count
    instead of duplicating the target on every edge.
    """

    source: NodeRef
    weight: float
    kind: Literal["attn", "mlp", "resid", "sae"] = "resid"


class LayerState(BaseModel):
    """What one layer holds at one token position."""

    layer: int
    resid_norm: float  # L2 norm of hook_resid_post — cheap health signal
    logit_lens: LogitLens | None = None
    features: list[Feature] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


# --------------------------------------------------------------------------
# the trace itself
# --------------------------------------------------------------------------


class TokenStep(BaseModel):
    """One token position, with the full stack of layer states beneath it."""

    step: int  # index into Trace.steps; equals token.position today
    token: TokenInfo
    logits: LogitSummary
    layers: list[LayerState]


class ResidualRef(BaseModel):
    """Where the raw residual tensor for this trace lives, and what it is."""

    path: str  # relative to the trace JSON's own directory
    format: Literal["npy"] = "npy"
    hook: str  # TransformerLens hook these came from, e.g. "hook_resid_post"
    shape: tuple[int, int, int]  # [n_tokens, n_layers, d_model]
    dtype: str  # always float32: bf16 has no numpy dtype and fp16 can
    # overflow on Gemma 2's large late-layer activations


class Trace(BaseModel):
    """A single prompt → generation run, instrumented at every layer."""

    schema_version: str = SCHEMA_VERSION
    trace_id: str
    created_at: datetime = Field(default_factory=_utcnow)

    model: str
    device: str
    dtype: str
    n_layers: int
    d_model: int

    prompt: str
    completion: str  # generated text only, prompt excluded
    n_prompt_tokens: int
    n_generated_tokens: int
    stop_reason: Literal["max_tokens", "eos"] = "max_tokens"
    elapsed_s: float = 0.0

    residuals: ResidualRef | None = None
    steps: list[TokenStep] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.prompt + self.completion
