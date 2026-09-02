## 1. Pass scaffold

- [x] 1.1 Create `app/passes/attribution.py` with `AttributionPass` following
  the `Pass` protocol (`name`, `run(trace, residuals) -> PassRecord`) and an
  injectable `model: HookedTransformer | None = None` field like
  `SAEPass`/`LogitLensPass`. Verify `from app.passes.attribution import
  AttributionPass` imports cleanly and satisfies the `Pass` protocol.
- [x] 1.2 Wire a `names_filter` on `run_with_cache` for `hook_attn_out`,
  `hook_mlp_out`, `hook_pattern`, `hook_v`, one layer at a time (as `lens.py`
  decodes one layer at a time) to bound memory. Verify the cache returned for
  a small trace contains exactly one entry per hook per layer.

## 2. Residual decomposition (resid + mlp edges, top-level check)

- [x] 2.1 Implement the `kind="resid"` edge from the previous layer's
  `resid_post` already in the `.npy` sidecar (no new hook), omitted entirely
  for layer 0. Verify layer 0's `LayerState.edges` contain no `kind="resid"`
  entry, and every layer 1..N-1 contains exactly one.
- [x] 2.2 Implement the `kind="mlp"` edge as a single edge sourced from the
  same `(layer, position)`, weight = `norm(hook_mlp_out)`. Verify every
  `LayerState` at every layer has exactly one `kind="mlp"` edge.
- [x] 2.3 Compute `reconstruction_max_rel_gap = max over (layer, position) of
  ||resid_post - (resid_pre + attn_out + mlp_out)|| / ||resid_post||`, using
  the untruncated attention sum, and record it in `PassRecord.stats`. Verify
  it reads near-zero (e.g. < 1e-3) on a real capture from `tiny-stories-1M`
  on CPU.

## 3. Attention per-source decomposition

- [x] 3.1 Implement `raw_contrib(s) = sum_h pattern[h, d, s] * (v[h, s, :] @
  W_O[h])` for every destination position `d`. Verify against a hand-computed
  2-token, 1-head case in a unit test.
- [x] 3.2 Implement the RMSNorm scale-distribution step: compute `scale(d)`
  from the whole `raw_attn_out(d) = sum_s raw_contrib(s) + b_O` via
  `ln1_post`'s rms + gain, then apply that same `scale(d)` to every
  `raw_contrib(s)`. Verify `sum_s (raw_contrib(s) * scale(d)) + b_O *
  scale(d)` matches the model's own `hook_attn_out(d)` within bf16 tolerance,
  on a real model where `ln1_post` is `nn.Identity` (`tiny-stories-1M`).
- [x] 3.3 Add a test against a small synthetic config with
  `use_normalization_before_and_after=True` (Gemma-2-style sandwich norm),
  since `tiny-stories-1M`'s post-norm is `Identity` and does not exercise this
  path — mirrors how the lens pass tested the softcap directly because
  `tiny-stories-1M`'s cap is 0. Verify the per-source decomposition still
  reconstructs `hook_attn_out` exactly under a real `RMSNorm` post-norm.
- [x] 3.4 Before relying on `hook_v`'s per-head granularity for gemma-2-2b,
  assert its shape against `model.cfg.n_heads` at pass-init time and fail
  loudly (not silently mis-attribute) if grouped-query-attention has already
  collapsed or not yet expanded the head dimension. Verify the assertion
  fires on a deliberately mismatched shape in a test.

## 4. Truncation and stats

- [x] 4.1 Truncate `attn`-kind edges to the top-k by weight, default `k=8`,
  configurable via `AttributionPass.top_k` (same pattern as
  `SAEPass.top_k`/`LogitLensPass.top_k`); record the k used in
  `PassRecord.params`. Verify a trace with more than k source positions keeps
  exactly k attn edges per `(layer, position)`.
- [x] 4.2 Compute `attn_topk_coverage` — mean fraction of each position's
  total attention-contribution norm retained after truncation — and record
  it in `PassRecord.stats`. Verify it equals exactly 1.0 when `n_tokens <=
  k` in a test, and is reported (not silently dropped) when it is less than
  1.0.

## 5. CLI integration

- [x] 5.1 Add `--attribution` to `cli trace` and `cli enrich`, wired the same
  way `--lens` is (resolve the model, apply `AttributionPass` via
  `app.passes.apply`). Verify `python -m app.cli enrich <trace>
  --attribution` fills `edges` and adds an `"attribution"` entry to
  `Trace.passes`.
- [x] 5.2 Add a per-layer attribution table to `cli show --attribution`,
  printing the top edges by weight for a requested token position. Verify
  manually against one of the traces in `backend/traces/`.

## 6. Tests and docs

- [x] 6.1 Write `tests/test_attribution_pass.py` covering the bookkeeping
  (layer-0 has no resid edge, top-k truncation, `PassRecord.params`/`stats`
  shape) against a stub, plus the real-model identity checks from groups 2
  and 3. Verify `pytest backend/tests/test_attribution_pass.py -q` passes.
- [x] 6.2 Run the full suite and confirm no regressions. Verify `pytest
  backend/tests -q` passes with the new tests included.
- [x] 6.3 Update the README's phase table (phase 5: attribution → done) and
  layout table with the new module, following the existing style. Verify a
  new measured-stats row (e.g. `reconstruction_max_rel_gap`,
  `attn_topk_coverage`) is added once run against the traces already in
  `backend/traces/`.
