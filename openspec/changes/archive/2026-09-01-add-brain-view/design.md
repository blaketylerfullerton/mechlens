## Context

`backend/viewer/index.html` is a single static file, no build step, no CDN
(see file header comment) — it reads a `Trace` JSON document and renders it
client-side. The existing grid HUD (`cardHud`/`buildGrid`/`buildCell`)
already computes, per position and layer, a three-way classification
(`answer` / `echo` / `other`, via `classify()`) and colors cells with an
OKLab-interpolated ramp (`ramp()`, `readPalette()`) driven by theme-aware CSS
custom properties. Selection state (`PIN`, `HOVER`) and re-render
(`renderAll`, `renderDetail`, `highlight`) are global, page-level functions
already shared across the meta card, curve chart, grid, and detail panel.
See proposal.md for why a brain-shaped view earns its place next to that
grid.

## Goals / Non-Goals

**Goals:**
- Add a `brain` view mode that renders the currently selected position's
  layer-by-layer logit-lens classification as a brain-shaped SVG, reusing
  the existing classification, ramp, and selection state rather than
  duplicating them.
- Keep the file dependency-free: no new library, no build step, consistent
  with the rest of the viewer.

**Non-Goals:**
- Not a new backend pass or trace schema field — purely a client-side
  renderer over `logit_lens` data the `lens` pass already produces.
- Not an anatomically accurate brain illustration — a stylized, clearly
  segmented diagram (e.g. concentric rings or lobe-like blobs) is enough to
  carry "depth through the model" as "depth through the brain."
- Not an animated playback across positions/layers (the user has already
  scoped this to a static, per-position render, matching how the grid and
  detail panel behave today).
- Not a replacement for the grid HUD — an additional mode, selectable
  alongside it.

## Decisions

**Region shape: concentric rings, not anatomical lobes.**
Mapping `N` layers onto lobes (frontal/temporal/parietal/occipital) implies a
functional correspondence between layer ranges and specific brain regions
that does not exist in a transformer — that would misrepresent what the
trace is showing. Concentric rings (or radial wedges) let "layer order"
map onto "distance from center" or "angle" with no implied semantics beyond
depth, which is the one thing this view is actually claiming. An outer
brain-like outline (a single clipped path) around the rings gives the
"looks like a brain" read the user asked for, without over-claiming
anatomical meaning region by region.

**Region count: fixed small number of bins, not one ring per layer.**
Gemma-2-2b has 26 layers; one ring per layer produces 26 wafer-thin
concentric rings that read as a bullseye, not a brain, and are unclickable
at typical viewport sizes. Bin layers into a fixed number of regions (e.g.
6-8), each spanning a contiguous layer range, matching how the proposal's
"combining layers' classifications" scenario is written. The exact bin
count is an implementation-time visual-tuning choice, not a spec-level
one — the spec only requires every layer be represented in some region, in
order.

**Per-region color: dominant class by layer count, confidence-weighted
opacity/lightness.**
A region's classification is the majority class among its layers (ties
broken toward the higher layer, since that is the more settled reading);
its color intensity (the `t` fed to the existing `ramp()`) is the mean
`top1_agreement`-equivalent confidence of layers matching that class. This
reuses `ramp()`/`CLASS_HUE` verbatim rather than inventing a second color
system. The spec's hover/detail text requirement means a reader can always
see the underlying per-layer breakdown on demand, so the blend is never the
only source of truth.

**Rendering: inline SVG built in JS, same pattern as the existing curve
chart (`cardCurves`).**
`cardCurves` already builds an SVG string with computed `path()`/`x()`/`y()`
helpers and inlines it via `innerHTML`. The brain view follows the same
pattern: a `cardBrain()` function computing ring/wedge `<path>` d-strings
from `N` layers binned into regions, so no new rendering technique enters
the file.

**Integration point: a new value in the existing `#modes` segmented
control, gated on `passOf("lens")` exactly like `cardCurves`/`cardHud`
already are in `renderAll()`.**
The existing `MODE` variable currently only affects grid cell coloring
(`answer`/`prob`/`entropy`); this adds a mode that swaps which card renders
(grid vs. brain) rather than how a grid cell is colored. `renderAll()`
branches on this the same way it already branches on `passOf("lens")` for
`noLens()`.

**Selection stays global (`PIN`/`HOVER`), no new state.**
The brain view reads whichever position is active exactly as `renderDetail`
does today, so `highlight()`/arrow-key navigation/click-to-pin all keep
working without new event listeners beyond re-rendering the brain card
alongside the existing detail card.

## Risks / Trade-offs

- **Blended-region color can obscure a genuinely mixed layer range** (e.g. a
  region that's half strongly-answer, half strongly-echo reads as a
  muddy mid-tone with the dominant-class rule) → mitigated by the spec's
  requirement that hover/detail text always disclose the per-layer
  breakdown, and by choosing a small enough bin count that mixed regions
  are the exception rather than the rule.
- **A stylized brain shape risks reading as decoration rather than
  information** → mitigated by keeping the outline load-bearing only as a
  frame (it does not carry data itself) and keeping every data-bearing
  element (region fill, crossover marker) inside it, matching the rest of
  the viewer's rule that hue always means the same one thing.
- **Adding a second big rendering path to a single-file viewer risks
  bloating `index.html` further** → mitigated by reusing existing
  `ramp()`/`classify()`/`readPalette()` rather than parallel copies; the new
  code is additive (new functions), not a fork of existing ones.

## Open Questions

- Exact bin count and ring-vs-wedge geometry are left to implementation-time
  visual tuning; either satisfies the spec as written.
