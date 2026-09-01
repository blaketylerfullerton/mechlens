"""The trace schema — the JSON contract the whole project writes into.

Shape:

    Trace
      ├── passes: [PassRecord]        which enrichment passes have been run
      ├── labels: {"L/F": FeatureLabel}  Neuronpedia text  (phase 3)
      └── steps: [TokenStep]          one per token in the final sequence
            ├── token:  TokenInfo     what this position holds
            ├── logits: LogitSummary  what the model predicts *after* it
            └── layers: [LayerState]  one per transformer layer (26 for gemma-2-2b)
                  ├── resid_norm      filled by the capture      (phase 1)
                  ├── features, l0    filled by the SAE encoder  (phase 2)
                  ├── logit_lens      filled by the logit lens   (unclaimed)
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
#   1.1  LayerState.l0, Trace.passes
#   1.2  Trace.labels (side table); Feature.label removed
SCHEMA_VERSION = "1.2"


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
    """An active SAE feature at this layer/position (phase 2).

    Deliberately just an index and a number. The human-readable label lives in
    `Trace.labels`, keyed by "layer/index" — a feature recurs across positions
    (~2x on the traces measured so far), so a label stored here would be the
    same string written out thousands of times.
    """

    index: int
    activation: float


class LogitLens(BaseModel):
    """This layer's residual stream decoded through the unembed."""

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

    # `features` is truncated to the top-k activations; `l0` is how many were
    # actually non-zero, which is the number that says whether the SAE is
    # behaving. Without it a top-16 list looks identical whether 20 features
    # fired or 16000 did.
    features: list[Feature] = Field(default_factory=list)
    l0: int | None = None

    edges: list[Edge] = Field(default_factory=list)


class FeatureLabel(BaseModel):
    """What Neuronpedia says a feature means (phase 3).

    Held in `Trace.labels` under "layer/index" rather than on every Feature.

    No URL field: the link is a pure function of (model_id, source_set, index),
    and the labels pass records that mapping once in its PassRecord.params. Six
    thousand copies of the same f-string is not data.

    `explainer` is not decoration. Neuronpedia's export does not use one
    explainer throughout — for gemma-2-2b/16k, layers 16, 18, 20, 22 and 24
    carry gemini-2.5-flash-lite explanations while the other 21 layers carry
    gpt-4o-mini, and the two write in visibly different styles. Anything that
    compares labels across layers has to be able to see that split.
    """

    text: str
    explainer: str | None = None  # e.g. "gpt-4o-mini"
    explanation_type: str | None = None  # e.g. "oai_token-act-pair"
    score: float | None = None  # usually absent; Neuronpedia scores few labels


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


class PassRecord(BaseModel):
    """One enrichment pass that has been applied to a trace.

    Without this a trace on disk cannot answer "where did these features come
    from?" — the same top-16 list looks the same whether it came from the 16k
    or the 262k SAEs. `stats` is where a pass leaves its own sanity numbers.
    """

    name: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    stats: dict[str, float | list[float]] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


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

    # TransformerLens normalization_type at capture time. A "...Pre" value
    # (LNPre/RMSPre) means the weights were LayerNorm-folded, which moves
    # resid_post off the distribution Gemma Scope's SAEs were fitted on.
    # Optional so traces written before this field still load.
    normalization: str | None = None

    prompt: str
    completion: str  # generated text only, prompt excluded
    n_prompt_tokens: int
    n_generated_tokens: int
    stop_reason: Literal["max_tokens", "eos"] = "max_tokens"
    elapsed_s: float = 0.0

    residuals: ResidualRef | None = None
    passes: list[PassRecord] = Field(default_factory=list)

    # "layer/index" -> label, for every feature appearing anywhere in `steps`.
    # Flat and string-keyed so it survives a JSON round trip unchanged.
    labels: dict[str, FeatureLabel] = Field(default_factory=dict)

    steps: list[TokenStep] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.prompt + self.completion

    def pass_record(self, name: str) -> PassRecord | None:
        return next((p for p in self.passes if p.name == name), None)

    def label(self, layer: int, feature: int) -> FeatureLabel | None:
        return self.labels.get(label_key(layer, feature))


def label_key(layer: int, feature: int) -> str:
    """The one place the `Trace.labels` key format is written down."""
    return f"{layer}/{feature}"
