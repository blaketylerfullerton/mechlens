## ADDED Requirements

### Requirement: The brain renders features as nodes placed by the atlas
The brain SHALL render each SAE feature as a node whose position comes from
the feature atlas, and SHALL NOT compute or adjust a node's position from the
trace being displayed.

#### Scenario: Nodes are placed from the atlas
- **WHEN** the brain renders nodes for a trace
- **THEN** each node sits at the position the atlas gives for its (layer,
  feature index), and two traces displaying the same feature place it
  identically

#### Scenario: A feature with no atlas position
- **WHEN** a trace reports an active feature the atlas has no position for
- **THEN** that feature is listed as unplaced with its activation, and no
  position is invented for it

#### Scenario: Atlas version mismatch
- **WHEN** the atlas the trace was drawn against differs from the atlas the
  interface holds
- **THEN** the interface says so and does not present the positions as
  corresponding to that trace

### Requirement: Node brightness comes from measured activation
A node's visual prominence SHALL be derived from the activation value recorded
in the trace for that feature at the displayed scope, and SHALL NOT be derived
from its position, its cluster, or its proximity to other lit nodes.

#### Scenario: An active feature is lit in proportion to its activation
- **WHEN** two features are active at the displayed scope with different
  activation values
- **THEN** the one with the higher recorded activation is the more prominent,
  independent of where either sits

#### Scenario: A dim feature next to a bright cluster stays dim
- **WHEN** a feature with a low activation is positioned near a group of
  strongly active features
- **THEN** it is rendered at its own low prominence and is not brightened by
  its neighbours

#### Scenario: A feature that did not fire is not lit
- **WHEN** a feature is present in the atlas but absent from the trace at the
  displayed scope
- **THEN** it is rendered as inactive and is distinguishable from a feature
  that fired weakly

### Requirement: The displayed scope is selectable and stated
The brain SHALL let the viewer choose whether the lit nodes are those active
at the selected (layer, token) cell, at the selected token across all layers,
or across the whole trace, and SHALL state which scope is displayed and how
many features it covers.

#### Scenario: Cell scope
- **WHEN** the cell scope is selected
- **THEN** only features the trace records as active at that layer and that
  token position are lit

#### Scenario: Token scope
- **WHEN** the token scope is selected
- **THEN** features active at that token position in any layer are lit, and
  the scope and feature count are stated

#### Scenario: Trace scope
- **WHEN** the trace scope is selected
- **THEN** features active anywhere in the trace are lit, the aggregation used
  to combine a feature's activations across positions is stated, and the scope
  and distinct-feature count are stated

### Requirement: The selection drives the brain as a transport
Changing the shared (layer, token) selection SHALL advance what the brain
lights, and the interface SHALL offer a control that steps that selection
through layers so the viewer can watch the lit features change with depth.

#### Scenario: Stepping through layers
- **WHEN** the viewer advances the transport at a fixed token position
- **THEN** the selected layer advances and the lit nodes change to those
  active at each successive layer

#### Scenario: Transport reaches the last layer
- **WHEN** the transport reaches the model's last layer
- **THEN** it stops there and does not wrap to layer 0 or continue past the
  layer count

#### Scenario: Transport with no feature data
- **WHEN** the transport is used on a trace with no SAE feature data
- **THEN** the control is unavailable or inert and the interface says why,
  rather than stepping through empty states

### Requirement: Areas are rendered from atlas clusters and named only when the atlas names them
The brain SHALL render the atlas's clusters as areas, SHALL label an area only
with the name the atlas recorded for it, and SHALL render an unnamed cluster as
unnamed.

#### Scenario: A named area
- **WHEN** the atlas recorded a name for a cluster
- **THEN** the area carries that name, and the interface can disclose that the
  name came from a member feature's own label

#### Scenario: An unnamed area
- **WHEN** the atlas recorded a cluster as unnamed
- **THEN** the area is rendered without a name and is not labelled by
  summarising its members in the interface

#### Scenario: Area prominence is aggregated from member activations
- **WHEN** an area contains features active at the displayed scope
- **THEN** its prominence is derived from those members' recorded activations,
  and the interface states how member activations were combined and how many
  of the area's features were active

### Requirement: Detail is disclosed progressively
The brain SHALL present areas, individual nodes, and per-feature detail at
increasing levels of detail, and per-feature detail SHALL name the feature and
its activation rather than only its colour.

#### Scenario: Overview
- **WHEN** the brain is viewed without a node or area focused
- **THEN** areas and the lit nodes are visible, and no single feature's
  identity is required to read the view

#### Scenario: Inspecting one node
- **WHEN** the viewer focuses a single node
- **THEN** the interface shows its layer, feature index, activation, its label
  if one exists, and a link to that feature's Neuronpedia page

#### Scenario: A node whose feature has no label
- **WHEN** the focused feature has no explanation
- **THEN** the interface shows the absence of a label rather than an empty
  string or a placeholder that reads as a label

### Requirement: Lit nodes are filterable by label text
The brain SHALL let the viewer restrict the lit nodes to those whose feature
labels match a text query, and SHALL state that unlabelled features cannot
match.

#### Scenario: A query with matches
- **WHEN** the viewer enters a query matching some active features' labels
- **THEN** only those features remain lit, and the number matched out of the
  number active is stated

#### Scenario: A query with no matches
- **WHEN** the query matches no active feature's label
- **THEN** the interface says no active feature's label matched, rather than
  showing an unfiltered or empty view without explanation

#### Scenario: Unlabelled features under a filter
- **WHEN** a filter is active and some active features have no label
- **THEN** those features are excluded and the interface states that
  unlabelled features are not searchable

### Requirement: The brain states what its nodes leave out
Where the trace records only a truncated set of a layer's active features, the
brain SHALL state how many features it is showing against how many the trace
records as having fired.

#### Scenario: Truncated feature list
- **WHEN** the trace records a top-k feature list for a layer alongside the
  true count of features that fired
- **THEN** the interface states both, so the lit nodes are not read as the
  complete set of what fired

#### Scenario: The first token position
- **WHEN** the displayed scope would include the sequence's first position
- **THEN** that position's features are excluded from the lit nodes and the
  interface states that they are a known artifact of the SAE at that position

### Requirement: The brain does not present its layout as a metric space
The brain SHALL NOT present distance between nodes or between areas as
meaningful, SHALL NOT render axes, gridlines, or a distance scale, and SHALL
state that the layout preserves local neighbourhoods while distorting larger
distances.

#### Scenario: No distance affordances
- **WHEN** the brain is rendered in any state
- **THEN** no axis, coordinate readout, distance scale, or measurement tool is
  offered, and no copy compares how far apart two nodes or areas are

#### Scenario: The layout's limits are stated
- **WHEN** nodes or areas are displayed
- **THEN** the interface states that positions come from a dimensionality
  reduction that preserves nearby relationships and distorts distant ones, and
  makes the atlas's recorded neighbourhood-preservation measurement available

### Requirement: The brain does not present its rendering as anatomy
The brain view SHALL NOT present any named anatomical structure as the site of
a feature, an area, a layer, or a computation, SHALL keep the anatomical shell
visually subordinate to the nodes and areas, and SHALL NOT imply left/right or
bilateral meaning in the layout.

#### Scenario: Data displayed
- **WHEN** nodes and areas are displayed
- **THEN** the shell and its surface detail are visually subordinate to them,
  and a legend states that a node is an SAE feature and an area a cluster of
  similar directions in the model's residual stream

#### Scenario: No anatomical labelling
- **WHEN** the brain view is rendered in any state
- **THEN** no named brain region is presented as the site of a feature, an
  area, a layer range, or a computation

#### Scenario: No implied symmetry
- **WHEN** the layout places nodes across the rendering's midline
- **THEN** the interface does not present either side as meaning anything, and
  no mirrored or symmetrised placement is introduced to make the rendering look
  bilateral

### Requirement: The brain has structure with no trace loaded
Before any trace has been requested, the brain SHALL render the atlas's
structure in an inactive state, and SHALL NOT present any feature as active.

#### Scenario: First load
- **WHEN** the app loads and no trace has been requested
- **THEN** the brain renders its nodes and areas in an inactive state and the
  interface invites a prompt

#### Scenario: No atlas available
- **WHEN** no atlas is available to the interface
- **THEN** the brain renders the shell alone, states that the feature atlas is
  unavailable, and does not place nodes at arbitrary positions

### Requirement: The brain degrades when feature data is absent
When the displayed trace has no SAE feature data, the brain SHALL render its
nodes in an inactive state and the interface SHALL state that the feature pass
did not run for that trace, rather than showing an empty or misleading view.

#### Scenario: Trace captured without the feature pass
- **WHEN** a trace with no recorded features is displayed
- **THEN** every node renders inactive, the interface states that the SAE pass
  did not run for this trace, and no node is lit as though an activation were
  known

#### Scenario: Some layers have features and others do not
- **WHEN** features were recorded for only a subset of layers
- **THEN** the lit nodes are drawn only from the layers that have data, the
  layers without data are named, and a missing layer is not treated as a layer
  in which nothing fired

#### Scenario: Features present but labels absent
- **WHEN** a trace records features but no labels were resolved for them
- **THEN** nodes are still lit by activation, per-feature detail reports the
  missing labels, and label filtering states that it has nothing to match on

## MODIFIED Requirements

### Requirement: Brain and grid share one selection
The brain and the grid SHALL read the same selected (layer, token position)
state, and changing that selection from either surface SHALL update both.

#### Scenario: Selecting a grid cell
- **WHEN** the user clicks a cell in the residual grid
- **THEN** the brain lights the features the trace records as active at that
  cell's layer and token position, at whatever scope includes that cell

#### Scenario: Selecting a token in the strip
- **WHEN** the user selects a different token position
- **THEN** the brain re-lights for that position and the grid's selected column
  follows, with no second selection control

#### Scenario: A new trace replaces the old one
- **WHEN** a new trace arrives while a selection from the previous trace is
  held
- **THEN** both surfaces fall back to the same default selection for the
  new trace rather than showing selections from different traces

### Requirement: The brain animates from live forward-pass progress
While a trace job is running and the service is reporting progress, the
brain SHALL animate the position that progress describes — lighting the
features of each layer the reported progress has reached, and advancing its
active token as the reported token advances — so the lighting corresponds to
work the model has actually done.

#### Scenario: Progress advances through layers
- **WHEN** the job reports progress in a phase that carries a layer counter,
  and successive readings report increasing layer indices
- **THEN** the brain lights the features of each reported layer in order, and
  does not light a layer the service has not reported

#### Scenario: Progress advances through generated tokens
- **WHEN** successive progress readings report an increasing generated-token
  count
- **THEN** the brain reflects that advance, and the count it shows never
  exceeds the number of tokens the service has reported

#### Scenario: Progress does not run ahead of the service
- **WHEN** no new progress reading has arrived since the last one
- **THEN** the brain does not advance past the last reported layer or token

#### Scenario: A layer's features are not yet available
- **WHEN** progress reports a layer whose features have not yet reached the
  client
- **THEN** the brain indicates that layer is being computed without lighting
  nodes that no activation has been received for

### Requirement: The brain releases its GPU and animation resources
The brain SHALL stop its animation loop and dispose of its rendering
resources when it is unmounted, and SHALL NOT leak a renderer, a composer,
geometry, or atlas buffers per trace or per selection change.

#### Scenario: Unmount
- **WHEN** the brain component unmounts
- **THEN** its animation frame is cancelled and its renderer, composer,
  geometries, materials, and atlas buffers are disposed

#### Scenario: New trace or new selection
- **WHEN** a new trace arrives, or the selection or displayed scope changes
- **THEN** the brain updates without rebuilding its renderer, its shell
  geometry, or its atlas position buffers from scratch

## REMOVED Requirements

### Requirement: Layer bands cover every layer in depth order
**Reason**: Depth is no longer a layer axis. Spending a spatial dimension on
layer made every cluster of features appear once per layer as a thin slice, so
nothing in the rendering read as a region. All three spatial dimensions now
carry the feature layout, and layer is expressed through the selection and the
transport instead.

**Migration**: Layer-by-layer reading moves to the trace grid, which already
renders layer x token and remains the surface for depth-ordered comparison.
Watching depth on the brain is now done with the transport control described in
"The selection drives the brain as a transport".

### Requirement: Band colour reuses the lens classification and crossover marker
**Reason**: The brain no longer reads logit-lens data. Colouring bands by the
`answer` / `echo` / `other` classification made the brain a second rendering of
the signal the trace grid already shows in the same colours from the same
module. The brain now shows SAE feature activity, which nothing else displays.

**Migration**: The lens classification and the `crossover_layer` marker remain
in the trace grid, which is unchanged and remains the single place that
classification is rendered. No brain surface consumes lens data.

### Requirement: The brain degrades when lens data is absent
**Reason**: The brain no longer depends on lens data, so a lens-shaped
degradation requirement no longer describes it. Its replacement covers the data
the brain does depend on.

**Migration**: See "The brain degrades when feature data is absent", which
states the equivalent behaviour for missing SAE features, partially covered
layers, and missing labels.

### Requirement: The brain does not claim layers map to anatomical regions
**Reason**: The prohibition was written for layer bands and is too narrow for a
rendering that now shows discrete nodes grouped into visible areas — a cluster
invites being read as a brain region far more strongly than a depth band did.

**Migration**: See "The brain does not present its rendering as anatomy", which
carries the same prohibition and extends it to nodes, areas, and implied
bilateral meaning, and "The brain does not present its layout as a metric
space", which covers the reading errors the layout itself invites.
