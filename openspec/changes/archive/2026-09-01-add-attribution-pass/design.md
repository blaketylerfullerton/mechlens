## Context

See proposal.md for motivation. The relevant current state:

- Only `hook_resid_post` is captured at trace time (`capture.py`); attention
  patterns, per-head values, and the block-level `attn_out`/`mlp_out` are not
  saved anywhere. Per the confirmed direction, this pass fetches them at
  enrich time via a second forward pass — the same shape as
  `LogitLensPass.model` — rather than extending capture or the sidecar.
- Verified against the installed TransformerLens source
  (`components/transformer_block.py`, `components/abstract_attention.py`):
  `hook_attn_out` and `hook_mlp_out` are captured **after** Gemma-2's
  sandwich norm (`ln1_post` / `ln2_post`) is applied — the code's own comment
  says as much ("precede the hook ... to capture the additive
  contribution"). So `resid_mid = resid_pre + attn_out` and `resid_post =
  resid_mid + mlp_out` hold **by construction**, not as something to
  approximate. This is the same class of guarantee as `final_layer_agreement`
  in phase 4: a correctness gate that should read ~exact, not a best-effort
  estimate.
- `resid_pre(layer)` for `layer >= 1` is exactly `resid_post(layer - 1)` — no
  operation happens between blocks in TransformerLens. It is already on disk
  in the `.npy` sidecar; the pass does not need a hook for it.
- Layer 0's `resid_pre` is the embedding output, which is not captured
  anywhere and is out of scope per the spec (layer 0 has no `resid`-kind
  edge).

## Goals / Non-Goals

**Goals:**
- Exact (modulo bf16 rounding) decomposition of `resid_post` into carry-over,
  attention, and MLP contributions.
- Per-source-position decomposition of the attention contribution, so an edge
  points at the specific earlier token attention drew from, not just "layer
  L's attention" as a whole.
- An honest, reported gap between what's decomposed and what's stored (top-k
  truncation), not a silently-lossy one.

**Non-Goals:**
- `kind="sae"` edges (deferred to a later phase — see proposal.md).
- Per-head attribution. Edges are per source *position*; a future phase could
  split further by head, but nothing in the schema or spec requires it here.
- Attributing bias terms (`b_O`, MLP's own output bias) to a source position.
  `Edge.source` is a required `NodeRef` — there is no "no source" sentinel in
  the schema for a position-independent constant, so bias contributions are
  not represented as edges at all (see Decision 3).

## Decisions

**1. Fetch attention/MLP tensors at enrich time, not capture time.**
Adds `hook_attn_out`, `hook_mlp_out`, `hook_pattern`, `hook_v` to a
`names_filter` on a fresh `run_with_cache`, mirroring how the lens pass takes
`model` and pays one forward pass. `capture.py` and the sidecar format are
untouched; existing traces work without re-tracing.
*Alternative considered*: extend phase-1 capture to save these tensors so
attribution needs no model at enrich time. Rejected (per direction already
confirmed) — it would grow the sidecar for every trace whether or not
attribution is ever run on it, and require re-tracing the 5 traces already on
disk.

**2. `resid`-kind edge reads the previous layer's residual from the sidecar.**
No new hook; `resid_pre(L) == resid_post(L-1)`, already loaded from the
`.npy` file the same way every other pass reads it.

**3. Per-source attention decomposition, and where the bias goes.**
For a destination position `d` at layer `L`:

```
raw_contrib(s) = sum_h  pattern[h, d, s] * (v[h, s, :] @ W_O[h])   # pre-post-norm
raw_attn_out(d) = sum_s raw_contrib(s) + b_O
attn_out(d)     = ln1_post(raw_attn_out(d))                        # what's on the residual
```

`ln1_post` is RMSNorm: `x * gain / rms(x)`, where `rms(x)` and `gain` are
computed from the *whole* vector `x = raw_attn_out(d)`, not per source. That
scale factor is therefore the same multiplier for every term in the sum it
was computed from, so it distributes exactly:

```
attn_out(d) = sum_s (raw_contrib(s) * scale(d))  +  (b_O * scale(d))
```

Each `raw_contrib(s) * scale(d)` becomes one `kind="attn"` edge, source =
`NodeRef(layer=L, position=s)`, weight = its L2 norm. `b_O * scale(d)` has no
source position — it is not stored as an edge (Non-Goal above); it is folded
into the "unattributed" stat described in Decision 6.
*Alternative considered*: attribute `b_O` to the destination position itself
(a self-loop edge). Rejected — a self-loop under `kind="attn"` would read as
"this position attended to itself" when it's actually a fixed per-layer
constant unrelated to attention at all; better to leave it out and report it
honestly than mislabel it.

**4. `mlp`-kind edge is a single, undecomposed term.**
Unlike attention, the MLP is already position-wise — there is no second
position to decompose across. One `kind="mlp"` edge per `(layer, position)`,
source = the same position at the same layer (`NodeRef(layer=L,
position=p)`), weight = `norm(mlp_out)`. `b_out` (the MLP's own output bias)
is likewise not attributed to a source and folds into the same unattributed
stat.

**5. Edge weight is contribution norm.**
Confirmed direction: structural, not answer-focused. `weight = ||contribution
vector||`. A DLA-style (unembed-direction-weighted) variant is explicitly
deferred, not designed here.

**6. Two separate correctness stats, not one.**
Conflating "is the decomposition correct" with "did truncation lose
information" would hide a real regression behind an expected, benign gap —
the same mistake the project already guards against elsewhere (`echo_by_layer`
vs `entropy_by_layer` in phase 4). So the pass records both:
  - `reconstruction_max_rel_gap` — `||resid_post - (resid_pre + attn_out +
    mlp_out)|| / ||resid_post||`, computed from the **untruncated** sums.
    Should read ~0 (bf16 precision) by construction (Decision context above);
    anything else means a hook or a sign is wrong, not that k was too small.
  - `attn_topk_coverage` — mean fraction of total per-position attention
    contribution norm retained after top-k truncation. Expected to be < 1.0
    on prompts where attention spreads across many positions; reported like
    the SAE pass already reports `mean L0` / `explained_variance` — an honest
    number, not a target to force to 1.0.

**7. Top-k truncation for attn edges.**
Default `k=8` (attention is typically concentrated on a handful of source
positions, versus the SAE pass's `k=16` for a wider, sparser feature space).
Configurable the same way `SAEPass.top_k` and `LogitLensPass.top_k` are;
recorded in `PassRecord.params`.

**8. Layer 0 has no `resid` edge.**
Per spec — there is no prior `LayerState` to point at, and layer 0's true
`resid_pre` (the embedding) is out of scope to fetch. Layer 0's edges are
`attn` and `mlp` only.

## Risks / Trade-offs

- **GQA hook granularity unverified for gemma-2-2b.** `hook_v` may be at
  raw KV-head granularity or already expanded to query-head granularity
  depending on TransformerLens's grouped-query-attention handling. → Mitigate
  by asserting the expected shape against `model.cfg` at pass-init time and
  failing loudly rather than silently mis-attributing; confirm during
  implementation against the actual gemma-2-2b config, not assumed here.
- **A second forward pass per attribution run.** Same cost class as the lens
  pass (~model load + one pass), on top of whatever passes already ran. →
  Acceptable per the confirmed direction; if this becomes the dominant cost
  in practice, capture-time hooks become worth revisiting as a later change.
- **Bias terms are invisible in the trace.** A reader summing edges will
  always come up slightly short of `resid_norm`, and nothing in the schema
  says why unless they read `PassRecord.stats`. → `reconstruction_max_rel_gap`
  and a recorded mean bias-contribution norm make the shortfall legible
  rather than silent; if this proves confusing in practice, a schema change to
  give `Edge` an optional-source "bias" representation is a natural follow-up,
  out of scope here.
- **Top-k truncation is lossy on long, diffuse-attention traces.**
  `attn_topk_coverage` makes this visible per the design above, but a caller
  who ignores `PassRecord.stats` can still be misled by a truncated edge list
  that looks complete. Same shape of risk the SAE pass already accepts with
  its own top-k features.

## Migration Plan

None needed. This is a pure enrichment pass: it only adds to
`LayerState.edges`, which already exists in the schema at 1.2. No
`SCHEMA_VERSION` bump, no change to `capture.py` or the sidecar format, and
existing traces are unaffected until `--attribution` is explicitly requested
on them.
