## Context

`app/model_cache.get_model()` already gives a process-wide singleton
(`lru_cache(maxsize=1)`); the comment on it says "FastAPI: call `get_model()`
once on startup; every request reuses it" — this phase is that call site.
`generate_trace` (capture.py) drives generation itself via
`model.run_with_cache`, one explicit forward pass per token, so there is
already a single, well-defined place to add a hook. `get_sae(layer)`
(sae_cache.py) returns a loaded SAE with `W_dec[feature_idx]` giving that
feature's decoder direction in residual space — the SAE for a layer is
already loaded on demand, no new loading path needed. `LabelStore.get(layer,
feature)` is the existing feature → label lookup. See proposal.md - Why for
why this phase exists now.

## Goals / Non-Goals

**Goals:**
- One process, one model, serving `/trace`, `/steer`, `/feature/{layer}/{idx}`.
- Jobs are async from the client's perspective (submit, poll) but strictly
  sequential on the GPU.
- Steering reuses the existing SAE/label infrastructure; no new "vector
  store" or intervention config format.

**Non-Goals:**
- No persistence of job state across a process restart.
- No auth, rate limiting, or multi-tenant isolation — v1 is single-user/dev,
  matching the CLI's current trust model.
- No multi-intervention steering (list of interventions per request) or raw
  vector injection — deferred per the proposal's scope decision.
- No horizontal scaling / multi-worker deployment; `model_cache`'s
  `maxsize=1` and the GPU lock are both explicitly single-process
  assumptions.

## Decisions

**Layout**: new `app/service/` package —
- `app/service/app.py` — FastAPI app + route handlers.
- `app/service/jobs.py` — job queue, worker thread, job records.
- `app/service/models.py` — Pydantic request/response models (distinct from
  `schema.py`'s `Trace`, which stays the on-disk/wire trace format).
- `app/service/steering.py` — builds the TransformerLens hook function from
  a `(layer, feature_idx, coefficient)` triple.

**Job queue: a thread, not asyncio, not Celery/RQ.** `generate_trace` is
synchronous, CPU/GPU-bound torch code — running it inside an `async def`
route would block the event loop for the whole trace. A single background
`threading.Thread` pulling from a `queue.Queue` keeps FastAPI's async
handlers non-blocking (they just enqueue and return) while guaranteeing
one job runs at a time, which is exactly the "one model instance, one
request at a time" requirement — no separate lock needed to enforce
ordering, since there is exactly one consumer. A process-wide dict
(`job_id -> JobRecord`) holds status/result; `GET /trace/{id}` reads it
directly. Alternative considered: `asyncio` + `run_in_executor` with a
`asyncio.Semaphore(1)` — rejected as more moving parts for the same
guarantee, and harder to reason about when a second worker is added later.
A real queue (Celery/RQ/Redis) is overkill for one in-process worker and a
dev-scale tool; revisit if the service ever needs to survive a restart or
run multiple workers.

**GPU serialization**: the worker thread itself is the serialization
point — jobs are pulled off `queue.Queue` one at a time, so no explicit
semaphore is needed for the happy path. A `threading.Lock` is still taken
around the forward-pass section as a defensive guard (per the proposal's
Impact note) in case a future change adds a second reader of the model
(e.g. `/feature` triggering a live SAE forward pass instead of a pure
label lookup) — cheap insurance, not load-bearing for v1's single-worker
design.

**Steering hook**: implemented as a TransformerLens hook function
`(resid_post, hook) -> resid_post + coefficient * W_dec[feature_idx]`,
registered with `model.add_hook(f"blocks.{layer}.hook_resid_post", fn)` for
the duration of one `generate_trace` call and removed afterward (`finally:
model.reset_hooks()`). This sits at the same hook point `capture.py`
already reads from (`RESID_HOOK = "hook_resid_post"`) — the hook mutates the
tensor before `capture.py`'s own read of the cache, which is what makes
"captured residuals reflect the intervention" (steering spec) true without
`capture.py` needing to know the hook exists. `generate_trace` takes an
optional `intervention: Callable | None` parameter; `None` is today's
behavior exactly, satisfying "an unsteered trace is unaffected."
`coefficient=0` naturally reduces to a no-op add, satisfying that scenario
without a special case.

**Where steering is validated**: layer range and feature-index range are
checked in the `/steer` route handler (via `model.cfg.n_layers` and the
loaded SAE's `W_dec.shape[0]`) before a job is enqueued, not inside the
worker — so an invalid request never occupies a queue slot, matching "no
job enqueued" in the spec's rejection scenarios.

**Trace schema addition**: `Trace.steering: SteeringInfo | None = None`
where `SteeringInfo` is a small model (`layer`, `feature_idx`,
`coefficient`) living in `schema.py` beside `Trace`, not in
`app/service/models.py` — it is part of the trace's on-disk contract
(read by anything loading a trace later, service or CLI), not a
request/response shape. `SCHEMA_VERSION` bumps to 1.3 for the same reason
1.1 and 1.2 bumped it: a new field on a versioned document.

**Feature endpoint reuses `LabelStore` directly** — no new caching layer;
`LabelStore` is already SQLite-backed and fast (see README: 6234 features in
1.5s, no network). The route opens a `LabelStore` per request (cheap local
SQLite connection) rather than holding one open process-wide, mirroring
`get_label`'s doc comment that a server "should build its own `LabelStore`."

## Risks / Trade-offs

- **In-memory job state is lost on restart** → acceptable for v1 (Non-Goal);
  a client mid-poll gets a 404 on `GET /trace/{id}` after a restart, same as
  an id that never existed.
- **Single worker thread means a slow/stuck job blocks everything** →
  matches "don't overthink concurrency for v1"; a future phase can add a
  timeout or job cancellation if this becomes a problem in practice.
- **Steering hook mutates shared model state (`add_hook`) while other
  requests could theoretically read it** → mitigated by the worker thread
  being the only caller of `generate_trace`, so hook registration and
  removal are never interleaved with another job's forward pass.
- **`SCHEMA_VERSION` bump touches every future trace, and every existing
  reader that switches on the version string** → the field is optional and
  additive (same pattern as 1.1's `l0` and 1.2's `labels`), so old traces
  still load; no migration of traces already on disk is needed.

## Migration Plan

No data migration: existing traces lack `steering` and load with it
defaulting to `None`. Deploying the service is additive — `python -m
app.service` (or equivalent) starts a new process; nothing about the
existing CLI workflow changes or needs to run alongside it. Rollback is
simply not running the service process; no schema rollback needed since
the field is optional.
