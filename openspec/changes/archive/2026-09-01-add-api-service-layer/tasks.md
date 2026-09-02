## 1. Schema: steering metadata

- [x] 1.1 Add `SteeringInfo` (`layer`, `feature_idx`, `coefficient`) and
      `Trace.steering: SteeringInfo | None = None` to `app/schema.py`; bump
      `SCHEMA_VERSION` to `1.3` with a comment line per the existing
      convention. Verify an existing saved trace JSON (no `steering` key)
      still loads via `Trace.model_validate_json`.

## 2. Steering hook in generation

- [x] 2.1 Add an optional `intervention: Callable | None = None` parameter
      to `generate_trace` (`app/capture.py`), registered via
      `model.add_hook(f"blocks.{layer}.hook_resid_post", fn)` before
      generation and removed in a `finally: model.reset_hooks()`. Verify
      `generate_trace` called with `intervention=None` produces identical
      output (completion + residuals) to the current behavior, on a fixed
      prompt.
- [x] 2.2 Add `app/service/steering.py` with a function building the hook
      closure from `(layer, feature_idx, coefficient)` using
      `sae_cache.get_sae(layer).W_dec[feature_idx]`. Verify a unit test:
      generating with a nonzero coefficient changes the captured residual
      at the target layer/position by (approximately) `coefficient *
      W_dec[feature_idx]` relative to an unsteered run of the same prompt.
- [x] 2.3 Verify a unit test: generating with `coefficient=0` produces a
      trace matching an unsteered trace of the same prompt (per the
      steering spec's zero-coefficient scenario).

## 3. Job queue

- [x] 3.1 Create `app/service/jobs.py`: `JobRecord` (id, status
      pending/running/done/error, result or error message, created_at),
      an in-memory `dict[str, JobRecord]`, a `queue.Queue` of pending work,
      and a single background `threading.Thread` worker that pulls one job
      at a time and runs it to completion before pulling the next. Verify
      a unit test: submitting two jobs back-to-back, the second does not
      start running until the first's status becomes `done`.
- [x] 3.2 Add a `submit(fn) -> job_id` helper that enqueues a callable and
      returns its job id immediately, without waiting for it to run.
      Verify a unit test: `submit` returns before the submitted callable
      has been invoked.
- [x] 3.3 Wrap the worker's job execution in a guard so an exception inside
      the job function sets the job to `error` with the exception message
      rather than crashing the worker thread. Verify a unit test: a job
      whose callable raises leaves the worker able to process a
      subsequent job.

## 4. FastAPI app and routes

- [x] 4.1 Add `fastapi` and `uvicorn` to `backend/requirements.txt`; verify
      `pip install -r backend/requirements.txt` succeeds.
- [x] 4.2 Create `app/service/models.py` with request/response Pydantic
      models: `TraceRequest {prompt, max_tokens}`, `SteerRequest {prompt,
      max_tokens, layer, feature_idx, coefficient}`, `JobResponse
      {job_id}`, `JobStatusResponse {status, trace | None, error | None}`,
      `FeatureResponse {label, explainer, ...}`. Verify each model
      round-trips a sample payload through `.model_validate` /
      `.model_dump`.
- [x] 4.3 Create `app/service/app.py`: FastAPI app that calls
      `model_cache.get_model()` once on startup (e.g. in a lifespan
      handler) and starts the job worker from task 3.1. Verify the app
      starts under `uvicorn` (or FastAPI's `TestClient`) without loading
      the model more than once (check `model_cache._load.cache_info().hits`
      or equivalent across two requests).
- [x] 4.4 Implement `POST /trace`: validate `prompt` non-empty and
      `max_tokens > 0` (422 otherwise), submit a job that calls
      `generate_trace` with no intervention and no enrichment passes —
      confirmed with the user: bare capture only, matching the CLI's own
      default `trace` behavior — return `{job_id}`. Verify via `TestClient`:
      valid request returns a job id; invalid request (empty prompt,
      `max_tokens=0`) returns 422 and no job is created.
- [x] 4.5 Implement `GET /trace/{id}`: return job status; on `done`,
      include the full `Trace` document (serialized via
      `Trace.model_dump()`); on `error`, include the error message; on an
      unknown id, return 404. Verify via `TestClient`: polling a freshly
      submitted job's id returns a pending/running state before it
      finishes, the finished state includes the trace body, and an
      unregistered id returns 404.
- [x] 4.6 Implement `POST /steer`: validate `prompt`/`max_tokens` as in
      4.4, plus `layer` in `[0, model.cfg.n_layers)` and `feature_idx` in
      `[0, sae.W_dec.shape[0])` for that layer's SAE (422 otherwise before
      enqueueing), submit a job that calls `generate_trace` with the
      steering hook from 2.2 and sets `Trace.steering` on the result,
      return `{job_id}` pollable via the same `GET /trace/{id}`. Verify
      via `TestClient`: valid request returns a job id pollable through
      `/trace/{id}`; an out-of-range layer or feature index returns 422
      and no job is created.
- [x] 4.7 Implement `GET /feature/{layer}/{idx}`: look up the feature via
      `LabelStore.get(layer, idx)`, return its label/metadata if labelled,
      a response distinguishing "no label" from "not found" if the
      feature exists but is unlabelled, and 404 for an out-of-range
      layer/index. Verify via `TestClient` against a small fixture
      `LabelStore` (or the real DB if present) covering all three cases.

## 5. GPU serialization guard

- [x] 5.1 Add a `threading.Lock` around the model forward-pass section
      used by job execution (per design.md's defensive-guard decision).
      Verify a unit test or code inspection confirms the lock is held
      across the full `generate_trace` call within a job.

## 6. Integration check

- [x] 6.1 With the service running locally, exercise the full flow end to
      end: `POST /trace` → poll `GET /trace/{id}` to completion → `POST
      /steer` targeting a labelled feature from that trace → poll to
      completion → `GET /feature/{layer}/{idx}` for that feature. Verify
      the steered trace's completion or top-k next-token predictions
      visibly differ from the unsteered trace's, and that
      `Trace.steering` on the steered result matches the request.
      Verified against the real gemma-2-2b model and a real Gemma Scope
      SAE (layer 20, feature 12082, "dog walking accessories"): baseline
      completion unchanged from coefficient 8 through 150 (decoder rows
      are unit-norm against a ~300+ residual norm at that depth, so small
      coefficients are a negligible perturbation), visibly degenerate
      completion at coefficient 400 — the intervention reaches the real
      model. `Trace.steering` matched the request in every case.
- [x] 6.2 Update `README.md`'s phase table: mark the API service layer
      phase done, matching the project's existing per-phase status
      convention. Renumbered the pre-existing "feature-level attribution"
      row from phase 6 to phase 7, since this change is what the code's
      own comments (viewer.py, capture.py) already called "phase 6."
