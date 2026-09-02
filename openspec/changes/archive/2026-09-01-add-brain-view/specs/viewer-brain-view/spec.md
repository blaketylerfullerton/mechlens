## Purpose

Gives the static HTML trace viewer a second, brain-shaped rendering of a
token position's logit-lens progression through the model's layers, so a
reader can see an answer crystallize depth by depth without reading the
existing layer/position grid cell by cell.

## ADDED Requirements

### Requirement: Brain view is a selectable mode
The viewer SHALL offer a "brain" view mode, selectable the same way the
existing grid's `answer` / `probability` / `entropy` cell-color modes are
selected, without navigating away from the loaded trace.

#### Scenario: Selecting the brain view
- **WHEN** a trace with a `lens` pass is loaded and the user selects the
  brain view mode
- **THEN** the brain rendering replaces or appears alongside the grid HUD
  without reloading or re-fetching the trace

### Requirement: Brain view renders one token position's layer progression
For the currently selected token position, the brain view SHALL render a
brain-shaped diagram divided into regions, where each region corresponds to
a contiguous range of layers ordered from the model's earliest layer to its
last, so depth through the model maps to depth through the diagram.

#### Scenario: Regions cover every layer
- **WHEN** the brain view renders for a position on a trace with `N`
  decoded layers
- **THEN** every decoded layer 0 through N-1 is represented within some
  region of the diagram, in layer order

### Requirement: Region color reuses the existing lens classification
Each region's fill SHALL be derived from the same per-position,
per-layer classification the grid HUD already computes — `answer`, `echo`,
or `other` — and the same color ramp, so a color means the same thing in
both views.

#### Scenario: A region spanning only answer-class layers
- **WHEN** every layer inside a region agrees with the model's final answer
  at the selected position
- **THEN** the region is filled using the same hue the grid uses for the
  `answer` classification

#### Scenario: A region spanning mixed-class layers
- **WHEN** a region's layers do not all share one classification
- **THEN** the region's fill is derived by combining those layers'
  classifications and confidences (for example, an area-weighted or
  dominant-class blend), and never silently picks one layer to represent
  the whole region without disclosing that in the hover/detail text

### Requirement: Crossover layer is annotated on the brain
When the selected position's trace has a `crossover_layer` value from the
lens pass stats, the brain view SHALL mark the region containing that layer
distinctly from unmarked regions.

#### Scenario: Crossover falls inside a rendered region
- **WHEN** `crossover_layer` is non-negative and less than the number of
  decoded layers
- **THEN** the region containing that layer is visually marked as the
  point where half the positions hold the final answer

#### Scenario: No crossover recorded
- **WHEN** `crossover_layer` is negative (the answer never settles) or
  absent
- **THEN** the brain view renders without a crossover marker, and does not
  fabricate one

### Requirement: Brain view shares selection state with the grid
The brain view SHALL render the same token position currently pinned or
hovered in the existing grid HUD, and changing that selection (click,
hover, or arrow-key navigation) SHALL update the brain view without a
separate selection control.

#### Scenario: Arrow-key navigation updates the brain view
- **WHEN** a position is pinned and the user moves the pin with the arrow
  keys
- **THEN** the brain view re-renders for the newly pinned position

### Requirement: Brain view respects the active theme
The brain view's colors SHALL be derived from the same theme-aware CSS
custom properties and OKLab ramp the grid uses, so toggling light/dark mode
updates the brain view exactly as it updates the grid.

#### Scenario: Toggling theme
- **WHEN** the user toggles light/dark mode while the brain view is shown
- **THEN** the brain view's region fills and any text update to the new
  theme's palette without a page reload

### Requirement: Brain view degrades when the lens pass is absent
When a loaded trace has no `lens` pass, the brain view SHALL show the same
"no logit lens on this trace" guidance the grid HUD shows today, rather
than rendering an empty or broken diagram.

#### Scenario: Trace without a lens pass
- **WHEN** a trace lacking a `lens` entry in `Trace.passes` is loaded and
  the brain view mode is selected
- **THEN** the viewer shows guidance to run the lens enrichment pass,
  and no brain diagram is drawn
