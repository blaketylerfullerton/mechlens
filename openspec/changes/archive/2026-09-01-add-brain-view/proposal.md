## Why

The viewer's existing grid HUD reads as a spreadsheet: rows are layers,
columns are token positions, cells are logit-lens guesses. That is the right
shape for auditing a single cell, but it does not read as "watching the model
think" — the layer-by-layer crystallization of an answer, which is exactly
what the logit lens data already captures (`top1_agreement_by_layer`,
`echo_by_layer`, `crossover_layer`). A visualization that maps depth-through-
the-model onto depth-through-a-brain gives that same data a form a
non-specialist can read at a glance: a token's answer "lighting up" region by
region as it moves from echoing its input to holding the model's final
answer.

## What Changes

- Add a new view mode, **brain**, to the existing viewer
  (`backend/viewer/index.html`), alongside the current grid HUD — selected
  the same way `answer` / `probability` / `entropy` are today.
- The brain view renders one token position at a time (the position already
  selected via `PIN`/`HOVER` in the existing grid), as a brain-shaped SVG
  divided into regions mapped to layer ranges (front-to-back or outer-to-
  inner, low layers to high layers).
- Each region's fill encodes that layer range's logit-lens state for the
  selected position, reusing the existing `answer` / `echo` / `other`
  classification and OKLab ramp already built for the grid — so brain and
  grid agree on what a color means.
- A layer-indexed "settle point" annotation on the brain shows where
  `crossover_layer` falls for the selected position, echoing the marker
  already drawn on the existing curve chart.
- No new backend pass, no new trace fields, no new HTTP endpoint: the brain
  view is a second renderer over the same `Trace.steps[*].layers[*].logit_lens`
  and `PassRecord.stats` data the grid already reads.
- Traces without a `lens` pass show the existing "no logit lens on this
  trace" message in place of the brain view, same as the grid does today.

## Capabilities

### New Capabilities
- `viewer-brain-view`: a brain-shaped, per-position rendering of a trace's
  logit-lens layer progression in the static HTML viewer, as an additional
  view mode alongside the existing grid HUD.

### Modified Capabilities
(none — no existing spec describes the viewer today, so this is additive
only)

## Impact

- `backend/viewer/index.html`: add a `brain` entry to the `#modes` segmented
  control, a new render function producing the brain SVG, and wiring so it
  shares `PIN`/`HOVER`/theme state with the existing grid.
- No changes to `backend/app/*` (schema, passes, service) — this consumes
  trace JSON already produced by the existing `lens` pass.
- No new dependencies: the SVG is hand-drawn markup plus the viewer's
  existing OKLab ramp code, consistent with the viewer's no-build-step,
  no-CDN constraint.
