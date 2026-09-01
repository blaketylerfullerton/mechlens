## Phase 1 — Residual stream capture (the trace skeleton)

This is the data structure everything else hangs off.

- Generate token-by-token (a loop, not `model.generate`), and for each generated token, cache the residual stream at every layer (26 layers for Gemma 2 2B).
- Define your **trace schema** now, in Pydantic. Something like: `Trace → [TokenStep] → per-step: token, logits summary, [LayerState] → per-layer: features, logit_lens, edges`. This JSON contract is the most important design decision in the project — the frontend, the SAE encoder, and the attribution pass all write into it.
- Save traces to disk as JSON (or msgpack later). Tensor of shape `[n_tokens, n_layers, d_model]` is your intermediate.

**Done when:** running a prompt produces a saved trace file with raw residuals per token per layer.

Done — `cd backend/app && python run_trace.py -n 20` writes `backend/traces/<id>.json` plus
`<id>.residuals.npy` (`[31, 26, 2304]` float32, `hook_resid_post`). Schema: `backend/app/schema.py`.
Capture loop: `backend/app/capture.py`. Tests: `pytest backend/tests -q`.
