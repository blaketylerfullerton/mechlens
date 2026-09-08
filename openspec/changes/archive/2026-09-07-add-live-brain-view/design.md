## Context

The React frontend is Vite + Tailwind. `App.tsx` calls `useTrace()` and
renders `TraceViewer` (which owns its own `selection` in local `useState`)
plus `ChatPanel`. `Brain.tsx` builds an `IcosahedronGeometry(1.4, 20)`,
displaces it with layered value noise into gyri, carves a longitudinal
fissure, adds a cerebellum bulge, a brainstem, traced sulcus boundary
lines, four lights and an `UnrealBloomPass`, and animates a slow rotation.
It is self-contained: one `useEffect` with an empty dependency array, no
props, and nothing imports it.

On the backend, `generate_trace` returns `CaptureResult(trace, residuals)`
where `residuals` is `[n_tokens, n_layers, d_model]` float32. The service's
`/trace` job does `generate_trace(...).trace` — it drops the residual array
on the floor. `LogitLensPass.run(trace, residuals)` is exactly the consumer
of that array, needs the model for `ln_final` + `W_U`, and measures at 2.3s
for 26 layers x 31 tokens. `jobs.JobRecord` holds `status`, `result`,
`error`, `created_at`; one worker thread runs one job at a time, and `JOBS`
is an in-memory dict with no persistence by design.

The archived `2026-09-01-add-brain-view` change designed a brain view for
the *static* `backend/viewer/index.html`, and was never implemented there
(`grep -i brain backend/viewer/index.html` returns nothing). Its
classification, blend, and crossover-marker decisions are still the right
ones and are reused here; its target surface is not.

See proposal.md for why the brain is invisible today and why remounting
alone would not be enough.

## Goals / Non-Goals

**Goals:**
- The brain is on screen next to the grid, driven by trace data and by real
  job progress, sharing one selection with the grid.
- One implementation of the lens classification and colour ramp, consumed by
  both the brain and the grid, so the two cannot disagree about what a
  colour means.
- Progress that describes work the model actually did, at a granularity the
  transport can honestly deliver.
- No trace-schema change and no new dependency.

**Non-Goals:**
- Not a replay/scrub timeline over a finished trace.
- Not SAE features or attribution edges on the brain.
- Not a streaming transport (SSE/WebSocket) — see the transport decision.
- Not anatomical accuracy, and explicitly not a layer-to-region mapping.

## Decisions

**Selection moves up to `App`; `TraceViewer` becomes controlled.**
`TraceViewer` currently owns `selection` in `useState` and derives a
fallback when the trace id changes. Two surfaces reading one selection means
exactly one owner, so `App` holds `{layer, position, traceId}` and passes
`selection` + `onSelectionChange` to both `TraceViewer` and `Brain`. The
existing trace-id fallback logic moves up with it, unchanged in behaviour —
this is why the spec has a scenario about a new trace resetting both
surfaces to the same default rather than each keeping its own.

**A shared `frontend/src/lib/lens.ts`, not a second colour system.**
The archived design's `classify()` / `ramp()` / `CLASS_HUE` live in the
static HTML viewer; the React app has neither. Rather than write brain-only
colour logic (and inevitably drift from whatever the grid grows), add one
module exporting `classifyLayer(step, layer)` → `answer | echo | other`
with a confidence, and a class-hue ramp. The grid's existing `heatColor()`
stays as-is for the residual-magnitude heatmap — that encodes a different
quantity — but any lens-classified surface, brain or grid, goes through
`lens.ts`. This is what makes the spec's "a colour means the same thing in
both views" requirement enforceable rather than aspirational.

**Bands are per-vertex colour on the existing geometry, not new meshes.**
`buildBrainGeometry()` is the expensive part (~8.8k faces of noise
evaluation) and must not re-run per selection. So the band overlay is a
`color` `BufferAttribute` on that same geometry: each vertex's band index
comes from its position along the chosen depth axis, and updating the render
is a write into that attribute plus `needsUpdate = true`. No geometry
rebuild, no per-band mesh, and the spec's "update without rebuilding the
renderer or base geometry" requirement falls out of the representation
instead of needing discipline. Band boundaries get a thin emissive line so
bands are countable at a glance.

**Depth axis: anterior → posterior, layer 0 at the front.**
The grid header already reads "input → output" left to right; front-to-back
gives the brain the same directional reading. The honest problem is that any
axis across a brain shape partly coincides with named anatomy — and the
archived design rejected lobe mapping precisely because it implies a
functional correspondence that does not exist in a transformer. The
mitigation is not a cleverer axis (there isn't one on a shell geometry;
radial depth doesn't vary) but disclosure: the sulcus boundary lines drop
to a low opacity whenever data is displayed so they read as silhouette, the
band boundaries are the visually dominant division, and a legend states
that a band is a range of transformer layers. That is why the spec makes
non-implication a requirement with its own scenarios rather than leaving it
to taste.

**Band count: 6-8 contiguous bins, not one band per layer.**
Same reasoning as the archived design: 26 bands on a 1.4-radius shell are
wafer-thin, unreadable, and unclickable. Bins are contiguous layer ranges
covering every layer, with the remainder distributed so no bin is empty.
A band's class is the majority class among its layers (ties toward the
higher layer, the more settled reading) and its intensity is the mean
confidence of the layers matching that class; the per-layer breakdown is
always available on hover, so the blend is never the only source of truth.

**Progress granularity: per generated token while generating, per layer
while running the lens pass.**
This is forced by what is observable. `generate_trace`'s inner
`for layer in range(n_layers)` loop is *copying out of an already-complete
cache* — `model.run_with_cache` has run the whole forward by then, so that
loop is not a layer sweep in time. Real per-layer timing during generation
would need a hook per layer's `RESID_HOOK`, and those fire roughly every
2-3ms on a 26-layer forward: far below any poll interval, so reporting them
would be data nobody can observe. Per generated token (~0.3-0.7s each) is
observable. The lens pass, by contrast, genuinely iterates layer by layer
(`_decode_layer` per layer, 26 layers in 2.3s ≈ 90ms/layer) — so the lens
phase is where the honest depth sweep lives, and it is also the phase whose
output the bands display. During the generating phase the brain advances its
token, not a fabricated layer position; this is what the spec's "does not
run ahead of the service" scenario is protecting.

**Transport: a `progress` field on the existing job-status response, polled
faster — not SSE.**
`useTrace` already polls `GET /trace/{id}` every 500ms with generation
guarding against stale responses. Adding `progress` to `JobStatusResponse`
reuses that loop, its cancellation semantics, and its 503 warm-up retry for
free. SSE would mean a new endpoint, `EventSource` on the client, a second
CORS surface, and its own reconnect story — real cost for a local-only v1
whose jobs last seconds. The poll interval drops to ~150ms while a job is
pending or running (a localhost GET returning a few integers), which
resolves the lens phase's 90ms/layer to within about two layers. Reverting
to SSE later changes only `useTrace` and one route.

**Progress is published as a whole immutable snapshot, no lock.**
The worker thread writes and the event loop reads. Rather than mutate a
dict in place, the job callable assigns a fresh progress object to
`JobRecord.progress` on each update; a reader therefore sees either the old
snapshot or the new one, never a half-updated one, without a lock on the
hot path. Monotonicity within a phase (a spec requirement) comes from the
producer only ever assigning forward, not from the reader sorting it out.

**`/trace` takes a `passes` list; the frontend always asks for `lens`.**
The brain needs lens data, but hard-wiring the pass into the route would
slow every other client's traces by ~2.3s with no way to opt out. So the
request carries `passes: ["lens"]`, validated against a known set (unknown
pass → 422, no job), and the frontend sends it on every trace. The job
keeps `CaptureResult.residuals` alive in its closure to feed
`LogitLensPass.run` — about 7MB for a 31-token trace, released when the job
callable returns — and runs the pass inside `_forward_lock`, because the
lens pass also touches the model and the GPU.

**The lens pass takes a vouched `hook` so it can run on an in-memory capture.**
`LogitLensPass.run` guards itself with `_check_compatible`, which requires a
`ResidualRef` on the trace and refuses anything not captured at
`hook_resid_post` — the lens's correctness test (the last layer's lens *is*
the model's output) only holds for `resid_post`. The CLI satisfies that guard
because `cmd_trace` calls `save_trace()` first, writing the `.npy` sidecar and
attaching the ref, and only then enriches. The service never touches disk, so
`trace.residuals` is `None` and `ResidualRef.path` — documented as "relative to
the trace JSON's own directory" — has no truthful value to hold.

So `LogitLensPass` gains an optional `hook` field: when the trace carries no
ref, that field is what the guard checks instead. This is not a weakened
guard, it is the same question asked of whichever source actually knows the
answer. For a disk caller the array is *loaded through* the ref, so the ref is
the provenance; for an in-memory caller the array arrives as an argument, so
"do you have it" is already settled by the call existing and only provenance
needs carrying — and `CaptureResult.hook` already carries it. The field
defaults to `None`, leaving the guard byte-for-byte today's behaviour for
every existing caller, and the one caller that sets it passes
`CaptureResult.hook` rather than a hand-typed string. The ref wins whenever
it is present, since it describes bytes that actually landed on disk.

`backend/app/passes/attribution.py` has the identical guard and is left alone
here; phase 7 (feature-level attribution) can follow the same pattern rather
than rediscovering the problem.

**A failed pass fails the job.**
`_run_one` already turns an exception into `status="error"`. Returning a
capture-only trace after a requested pass raised would hand the brain a
trace that looks lens-less and is indistinguishable from one that never
asked — so a pass failure propagates, and the spec says so.

## Risks / Trade-offs

- **The lens pass makes every frontend trace ~2.3s slower.** Accepted: it is
  the data the view exists to show, and the progress reporting means that
  time is now visible work rather than dead air. The `passes` list keeps the
  CLI and any other client unaffected.
- **Poll-driven layer sweeps will look slightly steppy.** ~150ms polling
  against ~90ms/layer means the brain advances roughly two bands at a time
  in the lens phase. Mitigated by easing between reported values *without*
  overshooting the last reported layer, per the spec.
- **Two heavy surfaces side by side.** The bloom composer plus a scrolling
  26-row grid in one viewport costs more than either alone; the split
  halves the brain's canvas, and the existing `min(devicePixelRatio, 2)` cap
  stays. If it does not hold up, bloom strength/resolution is the first
  thing to cut, not the band overlay.
- **Front-to-back bands will partly coincide with lobes no matter what.** A
  reader may still infer "frontal lobe = early layers". Mitigated by
  disclosure, not by geometry, as above — and the mitigation is specified,
  so it cannot be quietly dropped during implementation.
- **Lifting selection into `App` re-renders both surfaces on every click.**
  The grid is ~26 x N buttons. If that stutters, memoising rows is the fix;
  splitting selection back into two sources of truth is not.

## Open Questions

- Exact band count (6, 7, or 8) and the easing curve between progress
  readings are implementation-time visual tuning; any choice satisfying the
  spec's coverage and non-overshoot requirements is fine.
- Whether the generating phase should also show a slow non-committal sweep
  or hold steady per token is a visual call, constrained only by the
  requirement that it not present a specific layer as executing.
