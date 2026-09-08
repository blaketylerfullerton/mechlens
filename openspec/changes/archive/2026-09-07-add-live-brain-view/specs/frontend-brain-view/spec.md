## Purpose

Puts the 3D brain on screen beside the trace grid and makes it reflect the
model run: a selected token's answer crystallising layer by layer once the
trace lands, and a depth sweep driven by real forward-pass progress while
the run is still executing.

## ADDED Requirements

### Requirement: The brain is always on screen alongside the grid
The frontend SHALL render the brain and the trace grid simultaneously in a
split layout, with neither behind a mode toggle, from the moment the app
loads.

#### Scenario: First load, no trace yet
- **WHEN** the app loads and no trace has been requested
- **THEN** the brain is visible and animating, and the grid side shows its
  empty-state guidance

#### Scenario: Trace loaded
- **WHEN** a trace has finished and is displayed
- **THEN** the brain and the grid are both visible without the user
  selecting a view mode

### Requirement: Brain and grid share one selection
The brain and the grid SHALL read the same selected (layer, token position)
state, and changing that selection from either surface SHALL update both.

#### Scenario: Selecting a grid cell
- **WHEN** the user clicks a cell in the residual grid
- **THEN** the brain highlights the band containing that cell's layer, for
  that token position

#### Scenario: Selecting a token in the strip
- **WHEN** the user selects a different token position
- **THEN** the brain re-renders that position's layer progression and the
  grid's selected column follows, with no second selection control

#### Scenario: A new trace replaces the old one
- **WHEN** a new trace arrives while a selection from the previous trace is
  held
- **THEN** both surfaces fall back to the same default selection for the
  new trace rather than showing selections from different traces

### Requirement: Layer bands cover every layer in depth order
For the selected token position, the brain SHALL render depth-ordered bands
where every decoded layer `0` through `n_layers - 1` is represented within
exactly one band, in layer order.

#### Scenario: Every layer is represented
- **WHEN** the brain renders for a trace with `N` decoded layers
- **THEN** each layer index `0..N-1` falls inside exactly one band, and the
  band order follows layer order under a single stated convention

#### Scenario: Layer count differs from the band count
- **WHEN** the number of layers does not divide evenly by the number of
  bands
- **THEN** every layer is still assigned to exactly one band and no band is
  empty

### Requirement: Band colour reuses the lens classification and crossover marker
Each band's colour SHALL be derived from the same per-position, per-layer
`answer` / `echo` / `other` classification the grid uses, and the band
containing the trace's `crossover_layer` SHALL be marked distinctly, so a
colour means the same thing on the brain as in the grid.

#### Scenario: A band whose layers all hold the final answer
- **WHEN** every layer in a band agrees with the model's final answer at the
  selected position
- **THEN** the band is filled with the same hue the grid uses for the
  `answer` classification

#### Scenario: A band spanning mixed classifications
- **WHEN** a band's layers do not all share one classification
- **THEN** the band's fill is derived by combining those layers'
  classifications and confidences, and the per-layer breakdown for that
  band is available on demand rather than the blended colour being the only
  information shown

#### Scenario: Crossover layer is recorded
- **WHEN** the lens pass stats report a `crossover_layer` that is
  non-negative and less than the layer count
- **THEN** the band containing that layer is marked distinctly from
  unmarked bands

#### Scenario: No crossover recorded
- **WHEN** `crossover_layer` is negative, or the trace has no lens pass
  stats
- **THEN** the brain renders with no crossover marker and does not
  fabricate one

### Requirement: The brain animates from live forward-pass progress
While a trace job is running and the service is reporting progress, the
brain SHALL animate the position that progress describes — advancing its
active band as the reported layer advances and its active token as the
reported token advances — so the lighting corresponds to work the model has
actually done.

#### Scenario: Progress advances through layers
- **WHEN** the job reports progress in a phase that carries a layer counter,
  and successive readings report increasing layer indices
- **THEN** the brain's active band advances to the band containing each
  reported layer, in order

#### Scenario: Progress advances through generated tokens
- **WHEN** successive progress readings report an increasing generated-token
  count
- **THEN** the brain reflects that advance, and the count it shows never
  exceeds the number of tokens the service has reported

#### Scenario: Progress does not run ahead of the service
- **WHEN** no new progress reading has arrived since the last one
- **THEN** the brain does not advance its active band or token past the last
  reported values

### Requirement: Undifferentiated activity is shown as undifferentiated
When a job is submitted or running but no progress reading is available —
including while the service is still loading the model and answering 503 —
the brain SHALL show a general activity state that makes no claim about
which layer or token is executing.

#### Scenario: Between submit and the first progress reading
- **WHEN** a trace has been submitted and the job status is `pending`, or is
  `running` with no progress reported yet
- **THEN** the brain shows a general activity animation and no specific
  layer or token is presented as currently executing

#### Scenario: Model still warming up
- **WHEN** the request is being retried because the service is still loading
  the model
- **THEN** the brain shows the same general activity state, and the
  interface says the model is loading rather than that a trace is running

#### Scenario: Job fails
- **WHEN** the job reports an error
- **THEN** the brain leaves its activity state and the error is surfaced,
  rather than the brain continuing to animate as though work were ongoing

### Requirement: The brain does not claim layers map to anatomical regions
The brain view SHALL present its layer bands as a depth encoding distinct
from the rendering's anatomical features, and SHALL NOT label or colour a
named anatomical structure as though it corresponded to a layer range.

#### Scenario: Data displayed
- **WHEN** layer bands are displayed for a selected position
- **THEN** a legend or caption states that a band is a range of transformer
  layers, and the anatomical boundary lines are visually subordinate to the
  data-bearing bands

#### Scenario: No anatomical labelling
- **WHEN** the brain view is rendered in any state
- **THEN** no named brain region is presented as the site of a layer range,
  a feature, or a computation

### Requirement: The brain degrades when lens data is absent
When the displayed trace has no logit-lens data, the brain SHALL render
without fabricated classification colour and the interface SHALL say what is
missing, rather than showing an empty or misleading diagram.

#### Scenario: Trace captured without the lens pass
- **WHEN** a trace without `logit_lens` data is displayed
- **THEN** the brain renders in a neutral state, the interface states that
  the lens pass did not run for this trace, and no band is coloured as
  though a classification were known

#### Scenario: A single layer was not decoded
- **WHEN** some layers in a band have lens data and others do not
- **THEN** the band is derived only from the layers that have data, and does
  not treat a missing layer as any classification

### Requirement: The brain releases its GPU and animation resources
The brain SHALL stop its animation loop and dispose of its rendering
resources when it is unmounted, and SHALL NOT leak a renderer, a composer,
or geometry per trace or per selection change.

#### Scenario: Unmount
- **WHEN** the brain component unmounts
- **THEN** its animation frame is cancelled and its renderer, composer,
  geometries, and materials are disposed

#### Scenario: New trace or new selection
- **WHEN** a new trace arrives, or the selection changes
- **THEN** the brain updates without rebuilding its renderer or its base
  geometry from scratch
