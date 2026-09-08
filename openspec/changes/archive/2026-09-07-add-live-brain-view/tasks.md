## 1. Backend: job progress plumbing

- [x] 1.1 Add a progress snapshot type and a `progress` field to `JobRecord`
      in `backend/app/service/jobs.py`, plus a way for a job callable to
      publish an update (e.g. `submit` passing a reporter into `fn`, or a
      `jobs.report(job_id, ...)` helper); assignment replaces the whole
      snapshot rather than mutating one in place, per design.md
- [x] 1.2 Verify in `backend/tests/test_jobs.py` that a job which publishes
      progress is observable through `jobs.get(...)` while still running,
      that a `pending` job's progress is absent, and that a finished job's
      terminal status does not depend on a progress reading
- [x] 1.3 Verify progress reported for a phase is non-decreasing across
      successive reads of the same running job
- [x] 1.4 Verify a second submitted job stays `pending` (no running phase)
      until the first finishes, so progress reporting has not disturbed the
      single-worker serialization

## 2. Backend: progress callbacks in capture and the lens pass

- [x] 2.1 Add an optional progress callback to `generate_trace` in
      `backend/app/capture.py`, invoked once per completed generation step
      with the number of tokens generated so far and the budget; verify
      `backend/tests/test_capture.py` still passes and that a test callback
      receives one call per step with monotonically increasing counts
- [x] 2.2 Verify `generate_trace` behaves identically when no callback is
      passed (default `None`), including with an `intervention` registered,
      so `/steer` and the CLI are unaffected
- [x] 2.3 Add an optional per-layer progress callback to `LogitLensPass` in
      `backend/app/passes/lens.py`, invoked as each layer is decoded; verify
      `backend/tests/test_lens_pass.py` still passes and that a test
      callback sees one call per decoded layer in layer order
- [x] 2.4 Verify the lens pass's `final_layer_agreement` and
      `crossover_layer` stats are unchanged by the callback addition, by
      re-running the existing lens pass tests

## 3. Backend: `/trace` runs requested passes and reports progress

- [x] 3.1 Add `passes: list[str]` (default empty) to `TraceRequest` and a
      `progress` field to `JobStatusResponse` in
      `backend/app/service/models.py`, and reject an unrecognised pass name
      with a 422; verify in `backend/tests/test_service_app.py` that an
      unknown pass returns a client error and enqueues no job
- [x] 3.2 Rewire `POST /trace` in `backend/app/service/app.py` to keep
      `CaptureResult.residuals` and, when `"lens"` is requested, run
      `LogitLensPass(model=m, verbose=False)` over them inside
      `_forward_lock` before resolving the job; verify the returned trace has
      `logit_lens` populated per (token, layer) and a `lens` entry in
      `Trace.passes`
- [x] 3.3 Verify `POST /trace` with no `passes` returns a capture-only trace
      exactly as it does today (no `logit_lens`, no extra pass record), so
      existing clients see no change
- [x] 3.4 Wire the capture and lens callbacks from tasks 2.1/2.3 into the
      job's progress reporter, tagging each with its phase, and verify
      `GET /trace/{id}` reports the generating phase with a token count and
      the lens phase with a layer count while a job runs
- [x] 3.5 Verify a failing enrichment pass fails the job — `GET /trace/{id}`
      reports error with a message and no trace document — rather than
      returning a capture-only trace
- [x] 3.6 Verify `POST /steer` is unchanged: still enqueues, still pollable
      via `GET /trace/{id}`, and `backend/tests/test_steering.py` and
      `test_service_gpu_lock.py` still pass
- [x] 3.7 Run the full backend suite and confirm no regression against the
      recorded baseline (118 passed, 1 skipped)

## 4. Frontend: types, client, and the trace hook

- [x] 4.1 Add `passes` to the trace request type and a progress type to
      `frontend/src/lib/api-types.ts`, mirroring the backend models, and
      thread `passes` through `postTrace` in
      `frontend/src/lib/api-client.ts`
- [ ] 4.2 Have `useTrace` request `["lens"]`, expose `progress` in
      `UseTraceResult`, and poll at the faster interval while a job is
      pending or running; verify the existing generation guard still
      discards responses from a superseded run and that the 503 warm-up
      retry path still works (stop the backend mid-poll, restart it)
- [x] 4.3 Verify `progress` is cleared when a new run starts, so a new
      trace never briefly shows the previous run's counters
- [ ] 4.4 Distinguish "model loading" (the 503 retry path) from "trace
      running" in the hook's exposed state, so the UI can say which; verify
      by submitting a prompt against a freshly started backend

## 5. Frontend: shared lens classification

- [x] 5.1 Add `frontend/src/lib/lens.ts` exporting a per-(position, layer)
      `answer` / `echo` / `other` classification with a confidence, and a
      class-hue ramp, following the archived brain-view design's semantics;
      verify against a trace whose per-layer readouts are known from the
      grid's existing detail panel
- [x] 5.2 Add a band-binning helper partitioning `0..n_layers-1` into
      contiguous ranges, and verify every layer lands in exactly one
      non-empty band for several layer/band-count combinations, including
      counts that do not divide evenly
- [x] 5.3 Add a band-blend helper returning a band's majority class (ties
      toward the higher layer) and the mean confidence of the layers
      matching it, and verify it ignores layers with no lens data rather
      than counting them as any class
- [x] 5.4 Verify the helpers do not throw on a trace with no `logit_lens`
      at all, returning a neutral result the caller can render as "no lens
      data"

## 6. Frontend: `Brain` takes props and renders bands

- [ ] 6.1 Give `Brain` props for the trace, the selection, and the job
      progress, and split its `useEffect` so scene/geometry/composer setup
      runs once while data-driven updates run on prop change; verify no
      renderer, composer, geometry, or material is recreated when only the
      selection changes
- [ ] 6.2 Add a per-vertex `color` `BufferAttribute` to the brain geometry
      with a band index derived from each vertex's anterior-posterior
      position, and verify updating it repaints bands without rebuilding
      geometry (check frame time does not spike on selection change)
- [ ] 6.3 Colour bands from the shared helpers in task 5 for the selected
      token position, and verify a band whose layers all agree with the
      final answer uses the same hue the grid uses for the `answer` class
- [ ] 6.4 Mark the band containing `crossover_layer` distinctly when it is
      non-negative and in range, and draw no marker when it is negative or
      absent; verify against a trace where the crossover is known and one
      where it is `-1`
- [ ] 6.5 Add per-band hover detail disclosing the band's layer range and
      its per-layer classification breakdown, and verify a mixed band's
      breakdown matches the grid's readouts for the same layers
- [ ] 6.6 Drop the sulcus boundary lines to a subordinate opacity while
      band data is displayed, and add a legend stating that a band is a
      range of transformer layers; verify no named anatomical region is
      labelled or presented as the site of a layer range in any state
- [ ] 6.7 Verify the brain renders neutrally, with a stated reason and no
      fabricated classification colour, for a trace with no lens data

## 7. Frontend: live and idle animation states

- [ ] 7.1 Drive the active band from the lens phase's reported layer,
      easing between readings without advancing past the last reported
      value; verify by watching a real trace that the sweep never reaches
      the final band before the job completes
- [ ] 7.2 Drive the token indicator from the generating phase's reported
      token count, and verify no specific layer is presented as executing
      during that phase
- [ ] 7.3 Add the undifferentiated activity shimmer for `pending`, for
      `running` before any progress arrives, and for the model-loading
      retry path; verify against a freshly started backend that the
      interface says the model is loading rather than that a trace is
      running
- [ ] 7.4 Verify the brain leaves its activity state when a job errors, and
      that the error is surfaced rather than the animation continuing
- [ ] 7.5 Verify the animation loop is cancelled and all resources disposed
      on unmount (mount/unmount repeatedly and confirm no growth in WebGL
      contexts or memory)

## 8. Frontend: split layout and shared selection

- [ ] 8.1 Lift `selection` out of `TraceViewer` into `App.tsx`, including
      the existing trace-id fallback, and change `TraceViewer` to take
      `selection` + `onSelectionChange` props
- [ ] 8.2 Mount `<Brain />` in `App.tsx` in a split layout beside
      `TraceViewer`, with `ChatPanel` docked as it is today; verify both
      surfaces are visible on first load with no trace, and with a trace
      loaded, without any view-mode control
- [ ] 8.3 Verify selection is shared both ways: clicking a grid cell moves
      the brain's highlighted band, and selecting a token updates the
      brain's rendered position and the grid's selected column
- [ ] 8.4 Verify a newly arrived trace resets both surfaces to the same
      default selection rather than each holding its own
- [ ] 8.5 Verify the layout holds up at a narrow viewport (the split
      stacking rather than clipping either surface) and that the grid still
      scrolls independently

## 9. End-to-end verification

- [ ] 9.1 With the backend and frontend running, submit each starter prompt
      and confirm: the brain shimmers while the model loads, advances by
      token during generation, sweeps depth during the lens phase, then
      holds the selected position's band colouring — with no console errors
- [ ] 9.2 Confirm the brain's band colouring and crossover marker agree
      with the grid's per-layer readouts for the same position, by spot
      checking two positions on one trace
- [ ] 9.3 Confirm a trace requested without the lens pass (via `curl`)
      still returns a capture-only trace, and that the frontend shows its
      neutral no-lens state if handed one
- [ ] 9.4 Confirm the CLI is unaffected: run a `trace` and an
      `enrich --lens` from `backend/` and compare output against the
      README's documented behaviour
