## 1. Mode toggle and gating

- [ ] 1.1 Add a `brain` button to the `#modes` segmented control in
      `backend/viewer/index.html` and a page-level `VIEW` state
      (`grid` | `brain`) separate from the existing cell-color `MODE`, and
      verify clicking it toggles `aria-pressed` the same way the existing
      buttons do
- [ ] 1.2 In `renderAll()`, branch on `VIEW` to render `cardHud()` or the new
      `cardBrain()` when `passOf("lens")` is present, and keep `noLens()` as
      the fallback for both views when it is not; verify by loading a trace
      without a `lens` pass and confirming the same "no logit lens" message
      appears in both view modes

## 2. Layer binning and classification blend

- [ ] 2.1 Write a `binLayers(nLayers, binCount)` helper that partitions
      `0..nLayers-1` into contiguous ranges and verify with a unit-style
      manual check (console/log) that every layer index appears in exactly
      one bin across a few `nLayers`/`binCount` combinations, including
      cases where `nLayers` does not divide evenly
- [ ] 2.2 Write a `regionState(pos, layerRange)` helper that, given the
      existing `classify(pos, layer)` per layer in the range, returns the
      majority class (ties toward the higher layer) and the mean confidence
      of layers matching that class, reusing `classify()`/`lensOf()`
      verbatim; verify against a hand-picked trace position where the
      per-layer breakdown is known from the existing grid/detail view
- [ ] 2.3 Verify `regionState` never throws when a layer in the range was
      not decoded (sparse `lens.top_k`/missing layer), matching how
      `buildCell` already guards `!lens`

## 3. Brain SVG rendering

- [ ] 3.1 Implement `cardBrain()` producing an SVG with a brain-like outline
      path and `binCount` concentric ring (or wedge) regions in layer order,
      filled via the existing `ramp(hue, t)` using `CLASS_HUE` and each
      region's blended state from task 2.2; verify visually in a browser
      that region order matches layer order (outer/first = lowest layers or
      a clearly documented convention, applied consistently)
- [ ] 3.2 Mark the region containing `crossover_layer` (when non-negative
      and in range) with a distinct stroke/marker, and render no marker when
      `crossover_layer` is absent or negative; verify against a trace where
      `crossover_layer` is known from the existing tile, and against one
      where it is `-1`
- [ ] 3.3 Add hover/detail text (title attribute or the existing `#detail`
      side panel) disclosing each region's constituent layer range and
      per-layer classification breakdown, so a blended color is never the
      only information available; verify by hovering a mixed-class region
      and confirming the breakdown matches the grid's per-cell tooltips for
      the same layers

## 4. Selection and theme integration

- [ ] 4.1 Wire `cardBrain()` to read the same `PIN`/`HOVER` state the grid
      uses (no new selection state) and re-render on the existing
      click/mouseover/keydown handlers; verify arrow-key navigation and
      clicking a grid cell both update the brain view when it is the active
      `VIEW`
- [ ] 4.2 Verify the brain view's colors update on theme toggle by reusing
      `readPalette()`/`rampCache` exactly as the grid does — no separate
      palette lookup — checked by toggling light/dark and confirming region
      fills change without a reload

## 5. Manual verification

- [ ] 5.1 Load an existing trace from `backend/traces/` with a `lens` pass
      in the viewer, switch to the brain view, and confirm the rendering
      matches the design's ring/wedge + classification-blend approach with
      no console errors
- [ ] 5.2 Confirm `backend/viewer/index.html` still opens and renders
      correctly via `file://` (drag-and-drop trace loading), since the file
      remains dependency-free and build-step-free per design.md's Goals
