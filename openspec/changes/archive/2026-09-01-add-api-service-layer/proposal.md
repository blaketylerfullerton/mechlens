## Why

Every phase so far runs from the CLI against files on disk — `trace` captures,
`enrich` fills in a saved trace, `show`/`view` reads one back. That has been
the right call while the schema and passes were still moving (phases 1–5,
schema 1.2), but it means the model is reloaded per invocation and there is no
way to drive a trace from anything but a terminal. `viewer.py` already flags
itself as a placeholder for this: "no /trace or /steer... those belong in the
FastAPI layer." This phase adds that layer: one process holding the model
once, HTTP endpoints for tracing and steering, and a feature-metadata proxy —
without changing anything about how phases 1–5 work as a library.

## What Changes

- New `app/service/` package: a FastAPI app that loads the model once at
  startup (`model_cache.get_model()`) and exposes:
  - `POST /trace {prompt, max_tokens}` — enqueues a capture job (bare
    capture, no enrichment passes — matching the CLI's own default `trace`
    behavior), returns `{job_id}` immediately.
  - `GET /trace/{id}` — job status, or the full `Trace` JSON once done.
  - `POST /steer {prompt, max_tokens, layer, feature_idx, coefficient}` —
    enqueues a trace job where generation adds `coefficient *
    sae.W_dec[feature_idx]` to the layer's residual stream at every
    generated position. Same job/result shape as `/trace`.
  - `GET /feature/{layer}/{idx}` — looks up the Neuronpedia label and stats
    for one SAE feature via the existing `LabelStore`, without needing a
    trace.
- New in-process job queue: a single background worker thread consuming a
  `queue.Queue`, one job at a time. Job records (`pending` / `running` /
  `done` / `error`, plus result or error message) live in an in-memory dict
  keyed by job id — no persistence, no retries; a process restart drops
  in-flight jobs, which is acceptable for a v1 whose traces take seconds to
  tens-of-seconds and whose only client is the local viewer/dev use.
- New GPU serialization: a single `threading.Lock` (or semaphore of size 1)
  held for the duration of any model forward pass, so `/trace` and `/steer`
  jobs never overlap on the GPU even though HTTP requests can arrive
  concurrently. One model instance, one job running at a time — enforced by
  the worker thread design itself (a single consumer), the lock exists to
  protect any future second worker or direct model access from the request
  handlers.
- New steering hook in generation: `capture.py`'s `generate_trace` gains an
  optional intervention hook parameter (added via TransformerLens's
  `model.add_hook` on `blocks.{layer}.hook_resid_post`, active only when a
  steering job is running). The traced residuals reflect the *steered*
  activations, matching what SAE/labels/lens/attribution enrichment would
  see if run on that trace afterward — steering is not undone before
  passes run.
- `Trace` gains an optional `steering` field recording the layer, feature
  index, and coefficient a steered trace was generated with (`None` for an
  ordinary trace). Additive and optional, so every trace on disk today still
  loads; `SCHEMA_VERSION` bumps to 1.3 per the convention in `schema.py`.
- CLI, `capture.py`'s existing signature for unsteered traces, and
  `store.py` are otherwise unchanged; this phase only adds a new consumer
  (the service) and one new optional capability inside generation.

## Capabilities

### New Capabilities
- `api-service`: the FastAPI app, its job queue, GPU serialization, and the
  `/trace`, `/steer`, `/feature/{layer}/{idx}` endpoints.
- `steering`: the generation-time intervention that adds a scaled SAE
  feature decoder direction to a layer's residual stream during generation,
  and the trace produced by it.

### Modified Capabilities
(none — `capture.py`'s existing unsteered `generate_trace` behavior, the
trace schema, and all phase 1–5 passes are unchanged; steering is additive
and off by default)

## Impact

- `backend/app/service/` — new: FastAPI app, job queue, request/response
  models, GPU lock.
- `backend/app/capture.py` — `generate_trace` gains an optional steering
  hook parameter; no change to its default (unsteered) behavior or return
  type.
- `backend/app/schema.py` — new optional `Trace.steering` field;
  `SCHEMA_VERSION` 1.2 → 1.3.
- `backend/requirements.txt` — adds `fastapi` and an ASGI server
  (`uvicorn`).
- `backend/tests/` — new tests for the job queue, the steering hook's effect
  on residuals, and the three endpoints (via FastAPI's `TestClient`, no real
  model load needed for queue/endpoint-shape tests).
- No change to `app/store.py` or any existing CLI command.
