## Why

The brain was the premise, and right now it is not on screen at all.

`frontend/src/components/Brain.tsx` is a complete 329-line three.js
rendering — glass shell, coherent-noise gyri, cerebellum, brainstem, traced
sulcus boundaries, bloom — and **nothing imports it**. Commit `ad31387`
("updating to use backend") replaced the full-screen `<Brain />` in
`App.tsx` with `<TraceViewer />`, and the brain has been orphaned code ever
since. The product a user sees is a residual-norm spreadsheet.

Remounting it alone would not fix the premise either. `Brain()` takes no
props: it is a decorative rotating mesh that never sees `trace`, `status`,
layers, tokens, or activations. And it has nothing to show even if it did —
`POST /trace` runs only `generate_trace`, no enrichment passes, so the
frontend receives `resid_norm` and nothing else; the `logit_lens` data that
actually captures an answer crystallising by depth never reaches the
browser. `JobRecord` exposes only `pending → running → done`, so there is no
signal at all describing *where* a run is while it is running.

The result is that the one thing the project measures best — a fact
resolving layer by layer, `top1_agreement_by_layer` climbing to a
`crossover_layer` — is invisible in the interface built to show it.

## What Changes

- **Mount the brain, beside the grid.** `App.tsx` renders a split layout:
  the 3D brain on one side, the existing `TraceViewer` grid and panels on
  the other, both visible at once, sharing one selection. Clicking a grid
  cell moves the brain's highlighted layer band; the brain stays the stage
  the run plays out on.
- **Give `Brain` props and a data-driven surface.** `Brain` accepts the
  trace, the current selection, and the live job progress, and renders
  depth-ordered layer bands whose colour comes from the same
  `answer`/`echo`/`other` classification and the same crossover-layer
  marker the archived brain-view design defined — so brain and grid agree on
  what a colour means.
- **Run the logit-lens pass inside the trace job.** `POST /trace` gains an
  opt-in `passes` list; when it includes `"lens"`, the service runs
  `LogitLensPass` over the residuals `generate_trace` already returns (and
  currently throws away) before resolving the job, so the returned `Trace`
  carries `logit_lens` per (token, layer) plus the lens `PassRecord.stats`.
- **Report forward-pass progress on the job.** `JobRecord` gains a
  `progress` field — a phase (`generating` / `lens`) plus the token and
  layer counters that phase is at — updated from the worker thread as the
  run proceeds and returned by `GET /trace/{id}`. The frontend polls faster
  while a job is running and drives the brain's layer sweep from it, so the
  brain lights up depth by depth as the model actually computes.
- **Pulse honestly when there is nothing to report.** Between submit and
  the first progress reading — and during model warm-up, when `/trace`
  answers 503 — the brain shows an undifferentiated "working" shimmer that
  makes no claim about which layer is executing.
- **Do not imply layer/anatomy correspondence.** The layer bands are drawn
  as a distinct depth-ordered overlay with a legend, and the anatomical
  sulcus lines are de-emphasised while data is displayed, so no one reads
  "layer 14" as "the parietal lobe". This carries forward the archived
  brain-view design's explicit rejection of mapping layer ranges onto
  named brain regions.

## Capabilities

### New Capabilities
- `frontend-brain-view`: the React frontend's 3D brain, mounted beside the
  trace grid, rendering a selected token's layer-by-layer logit-lens
  progression and animating live from a running job's forward-pass
  progress.

### Modified Capabilities
- `api-service`: `POST /trace` can run enrichment passes (starting with
  `lens`) as part of the trace job, and `GET /trace/{id}` reports in-flight
  progress for a running job rather than only its status.

## Impact

- `frontend/src/App.tsx`: split layout, shared selection state lifted out of
  `TraceViewer`, `<Brain />` mounted again.
- `frontend/src/components/Brain.tsx`: props, a data-driven layer-band
  overlay, progress-driven and idle animation states, disposal on unmount
  unchanged.
- `frontend/src/components/TraceViewer.tsx`: selection becomes a controlled
  prop instead of local `useState`, so the brain and the grid cannot
  disagree; the "layer readouts arrive when the API runs the logit-lens
  enrichment pass" placeholder stops being the normal case.
- `frontend/src/hooks/useTrace.ts`: request the `lens` pass, surface
  `progress`, poll faster while running.
- `frontend/src/lib/api-client.ts`, `api-types.ts`: `passes` on the trace
  request, `progress` on the job status response.
- `backend/app/service/app.py`: `POST /trace` keeps `CaptureResult.residuals`
  and runs the requested passes; the job callable reports progress.
- `backend/app/service/jobs.py`: `JobRecord.progress`, and a way for a job
  callable to publish into it.
- `backend/app/service/models.py`: `TraceRequest.passes`,
  `JobStatusResponse.progress`.
- `backend/app/capture.py`: an optional progress callback on
  `generate_trace`, called per generated token and per layer as the forward
  pass sweeps.
- `backend/app/passes/lens.py`: an optional progress callback per decoded
  layer, and an optional `hook` the caller vouches for so the pass can run on
  an in-memory capture that has no on-disk residual sidecar (see design.md).
- No trace-schema change: `logit_lens` and the lens `PassRecord` already
  exist at schema 1.3. Progress is job metadata, not trace data, and is
  never persisted.

## Non-Goals

- No post-run replay or scrubber. The brain animates from live progress and
  then holds the finished trace's state for whatever is selected; a
  timeline scrub over a completed trace is out of scope.
- No SAE-feature or attribution rendering on the brain. Lens data only for
  this change; `features`/`edges` stay in the grid's panels.
- No persistence of progress across a service restart. `JOBS` remains an
  in-memory dict, per the api-service design's stated non-goals.
- Not a claim that a brain region computes anything. See the disclosure
  requirement above.
