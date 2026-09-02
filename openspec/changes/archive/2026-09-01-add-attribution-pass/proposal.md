## Why

`LayerState.edges` has been defined in the trace schema since phase 1 and has
sat empty through phases 2–4: nothing in a trace currently explains *why* a
layer's residual stream holds what it holds. SAE features say what's active,
the logit lens says what the model would answer at that depth, but neither
says which upstream computation — the carried-over residual, an attention head
reading an earlier position, or the MLP at this position — produced it. Phase
5 fills that gap.

## What Changes

- New `AttributionPass` (`app/passes/attribution.py`), following the existing
  `Pass` protocol: takes a trace and its residuals, fills `LayerState.edges`,
  returns a `PassRecord`.
- The decomposition is exact and additive, not approximate: for every
  `(layer, position)`, `resid_post = resid_pre + attn_out + mlp_out`. Each term
  becomes one or more `Edge`s:
  - `kind="resid"` — one edge, carried over from the same position at `layer -
    1`.
  - `kind="attn"` — one edge per source position, weighted by that position's
    share of `attn_out` (via the attention pattern).
  - `kind="mlp"` — one edge, from the same position's post-attention residual
    at this layer.
  - No gradients, no patching, no corrupted-run baseline: the residual stream
    is literally a sum of these terms, so the decomposition is checkable the
    same way the lens pass's `final_layer_agreement` is — the edges into a
    `LayerState` should sum back to its `resid_norm`.
- Runs at enrich time, the same way the lens pass does: takes the model
  handle, runs one additional forward pass with `pattern` / `attn_out` /
  `mlp_out` hooks added. `capture.py` and the `.npy` sidecar format are
  unchanged; the pass works on all traces already on disk without
  re-capturing them.
- New CLI surface: `--attribution` on `cli trace` and `cli enrich`; a per-layer
  attribution table in `cli show --attribution`.
- Edge weight is the norm of each term's contribution to the residual — a
  structural measure, not tied to any particular downstream token or feature.
- **Not in this phase**: `kind="sae"` edges (attributing a feature's
  activation to upstream contributions). That requires pushing the
  decomposition through the SAE encoder's linear-before-JumpReLU projection
  and introduces the SAE's own reconstruction error — a different
  correctness story than the exact resid/attn/mlp decomposition here. Left
  for a later phase.

## Capabilities

### New Capabilities
- `attribution`: exact, additive decomposition of each layer's residual
  stream into resid / attention / MLP contributions, recorded as
  `LayerState.edges`.

### Modified Capabilities
(none — `Edge` and `NodeRef` already exist in the schema at 1.2; this phase
populates them without changing their shape)

## Impact

- `app/passes/attribution.py` — new pass.
- `app/cli.py` — `--attribution` flag on `trace` / `enrich` / `show`.
- `tests/test_attribution_pass.py` — new, including a decomposition-sums-to-
  `resid_norm` check in the spirit of `final_layer_agreement`.
- No change to `app/schema.py`, `app/capture.py`, or the sidecar format; no
  `SCHEMA_VERSION` bump anticipated.
