## Phase 2 — SAE encoding (the "real sparse vector")

- Use **SAELens** to load Gemma Scope SAEs: release `gemma-scope-2b-pt-res-canonical`, one SAE per
  layer, 16k width to start (65k/262k later if you want). `SAE.from_pretrained(release, sae_id)`
  returns the SAE directly in SAELens 6.x — older tutorials show a `(sae, cfg, sparsity)` tuple.
- ~~Practical constraint: 26 SAEs won't comfortably sit in GPU memory alongside the model. Keep them
  on CPU and encode there.~~ **Not true on this box.** The GB10 has one 128GB unified pool with
  ~84GB free, so "moving to CPU" frees nothing. All 26 SAEs at 16k are ~8GB fp32 next to gemma's
  5GB — keep them all resident on GPU and skip the load/unload dance. Encoding a 31-token trace is
  26 matmuls of `[31, 2304] @ [2304, 16384]`: milliseconds. Revisit only at 262k width.
  The real cost is a one-time ~7.9GB download from HuggingFace.
- For each (token, layer), store only the **top-k active features** (k=10–20) with their activation
  values. That's what goes in the trace; the full sparse vector is thousands of mostly-zero entries
  you don't need. Also store **`l0`** — the true count of active features — because top-k truncation
  otherwise hides how dense the vector really was.

### Run it as an enrichment pass, not a re-generation

Phase 1 put the residuals on disk, so this phase must never re-run gemma:

    python -m app.cli enrich traces/golden-gate.json --sae

loads the `.npy`, encodes, and writes `features` back into the same JSON. Iterating on SAE code then
costs zero generation and zero model load. Build that path first, then the encoder.

Passes record themselves in `Trace.passes` (name, params, stats), so a trace on disk says which SAE
release produced its features.

### Gotchas that cost time

- **Never set `MECHLENS_PROCESS_WEIGHTS`.** Gemma Scope is trained on raw activations; LayerNorm
  folding changes `resid_post` and the features become silent garbage. The pass asserts this.
- **Assert the hook site.** `-res-` SAEs are trained on `hook_resid_post` only; `ResidualRef.hook`
  records what was captured, so check it rather than assume it.
- **JumpReLU, not ReLU**: `acts = pre * (pre > threshold)`. Plain ReLU yields thousands of tiny
  activations and the sanity check below fails confusingly.
- **Position 0 (BOS) looks insane** — huge, meaningless activations. Known Gemma Scope artifact, not
  a bug. Excluded from the summary stats.
- Residuals are fp32 on disk; load the SAE weights fp32 too.

**Done when:** each layer-state in your trace contains `[(feature_idx, activation), ...]` and the
numbers look sane — checked mechanically, not by eye:

- **L0 ≈ 30–150** active features per (token, layer) for the canonical SAEs. 16384 or 0 means a
  threshold bug.
- **Explained variance ≳ 0.7** for `decode(encode(x))` against the original residual.
- A handful of strong activations with a long weak tail, i.e. the top activation is well clear of
  the k-th.

Done — `python -m app.cli enrich traces/<id>.json --sae` over all 26 layers:
**mean L0 78.3, mean explained variance 0.880** (per-layer 0.83–0.96), 0.3s to encode a 31-token
trace once the SAEs are loaded (7s from disk, 202s the first time while ~7.9GB downloads).
Pass in `backend/app/passes/sae.py`, SAE loading in `sae_cache.py`.

BOS behaved exactly as warned: l0=6291 with a 1598-magnitude top feature, against ~78 everywhere
else. It is excluded from the stats and recorded per-token anyway.
