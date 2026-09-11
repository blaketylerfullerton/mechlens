# Attribution Specification

## Purpose

Explains, for every token position and layer in a trace, which upstream
computation — the carried-over residual, an attention head reading an earlier
position, or the MLP at that position — produced the residual stream the rest
of the trace already describes.

## Requirements

### Requirement: Residual stream decomposed into attributable edges
For every `LayerState` at layer 1 through the last layer, `edges` SHALL
decompose that layer's residual stream into carry-over, attention, and MLP
contributions. Edge weights SHALL be identified as L2 contribution magnitudes,
not conserved shares or causal effects. Full-vector reconstruction SHALL be
measured separately from the sum of serialized edge weights.

#### Scenario: Decomposition covers every token and layer
- **WHEN** the attribution pass runs on a trace with residuals present for
  every layer
- **THEN** every `LayerState` from layer 1 through the last layer has at
  least one edge of kind `resid`, at least one of kind `mlp`, and one or more
  of kind `attn`

#### Scenario: Decomposition is checkable
- **WHEN** the pass reconstructs the residual from full residual, attention and MLP vectors
- **THEN** it records the relative reconstruction gap in `PassRecord.stats`
- **AND** edge weights are not claimed to sum to `resid_norm`, because vector norms do not preserve cancellation

### Requirement: Layer 0 has no residual carry-over source
Layer 0 is the first transformer block; there is no earlier `LayerState` for
a `kind="resid"` edge to point at, so the pass SHALL NOT emit one for layer
0.

#### Scenario: First layer's edges omit the carry-over kind
- **WHEN** the attribution pass processes layer 0 for any token position
- **THEN** its edges include only `kind="attn"` and `kind="mlp"` entries, and
  no `kind="resid"` edge
- **AND** the sum-to-`resid_norm` check in the previous requirement does not
  apply to layer 0

### Requirement: Attention edges attribute per source position
Attention edges SHALL attribute across the source positions attention drew
from, not collapse to a single scalar, so a reader can see which earlier
token(s) a layer's attention output came from.

#### Scenario: Multiple source positions contribute
- **WHEN** a layer's attention output for a given position draws weight from
  more than one earlier position
- **THEN** that position's edges include a separate `kind="attn"` entry per
  contributing source position, each with a `NodeRef` identifying that source
  position

### Requirement: Edge list stays bounded per layer
Edges SHALL be truncated to the top contributors by weight, mirroring how
`Feature` lists are truncated to top-k activations, so a long trace does not
produce an unbounded number of edges.

#### Scenario: Long trace does not blow up trace size
- **WHEN** a trace has many tokens, so attention could in principle draw
  from every earlier position
- **THEN** only the top-k highest-weight `attn` edges are kept per
  layer/position, and the truncation depth used is recorded in
  `PassRecord.params`

### Requirement: Pass runs against traces already on disk
The attribution pass SHALL run on any trace and residual sidecar already
produced by capture, without requiring the trace to be re-captured or the
sidecar format to change.

#### Scenario: Existing trace without re-capture
- **WHEN** attribution is run on a trace produced before this pass existed
- **THEN** the pass completes and fills edges without error and without
  requiring the trace to be regenerated

### Requirement: CLI exposes attribution
Users SHALL be able to request attribution the same way they request the SAE,
labels, and lens passes.

#### Scenario: Requesting attribution at trace or enrich time
- **WHEN** a user passes `--attribution` to `cli trace` or `cli enrich`
- **THEN** the resulting trace has `edges` populated and an `attribution`
  entry in `Trace.passes`

#### Scenario: Inspecting attribution
- **WHEN** a user runs `cli show <trace> --attribution`
- **THEN** a per-layer table of attribution edges is printed for the
  requested token position

### Requirement: Feature-level attribution is out of scope for this pass
This pass SHALL NOT produce `kind="sae"` edges. Attributing a feature's own
activation to upstream contributions is deferred to a later phase.

#### Scenario: No sae edges emitted
- **WHEN** the attribution pass runs on any trace
- **THEN** no edge with `kind="sae"` appears anywhere in the resulting trace

### Requirement: Steered traces are rejected until intervention replay is supported
The pass SHALL reject steered traces before loading a model or emitting edges.
Replaying the original intervention and accounting for its additive contribution
are prerequisites to lifting this restriction.
