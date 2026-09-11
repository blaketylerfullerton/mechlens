# Feature-Atlas Specification

## Purpose

Gives every SAE feature a fixed position and an optional named cluster, so a
trace's active features can be shown as sites on a map instead of rows in a
list. The layout is projected from one of two recorded representations — the
features' explanation embeddings or the model's own decoder directions — is
reproducible, and carries the measurements that say how much of the original
geometry survived the projection.

## Requirements

### Requirement: The atlas assigns every feature one position
The atlas SHALL assign a position to every (layer, feature index) pair in the
model's SAE set, and each such pair SHALL have exactly one position.

#### Scenario: Complete coverage
- **WHEN** an atlas is built for a model with `L` layers and an SAE width of
  `W` features per layer
- **THEN** the atlas holds a position for every pair `(layer, index)` with
  `0 <= layer < L` and `0 <= index < W`, and no pair has more than one

#### Scenario: A feature with no label still has a position
- **WHEN** a feature has no Neuronpedia explanation
- **THEN** it still receives a position, and its absence of a label does not
  exclude it from the atlas

### Requirement: An atlas records which representation its positions came from
An atlas SHALL be built from exactly one named source representation — the
features' SAE decoder directions, or their explanation-text embeddings — and
SHALL record which. A consumer SHALL be able to determine an atlas's source
without inspecting its positions.

Two source representations answer two different questions, and neither
subsumes the other. Decoder directions are the model's own geometry: near
means the two features write similarly into the residual stream. Explanation
embeddings are descriptions of features: near means an explainer described them
similarly. Both are legitimate maps and they are not interchangeable, so the
source is recorded rather than assumed.

#### Scenario: Source recorded
- **WHEN** an atlas is built
- **THEN** its record names the source representation the positions were
  projected from

#### Scenario: Two atlases coexist
- **WHEN** atlases are built from both source representations
- **THEN** both are retained under distinct versions, each with its own source
  recorded, and neither overwrites the other

#### Scenario: Changing one source does not move the other's positions
- **WHEN** an atlas is built from decoder directions
- **THEN** its positions are unchanged by any difference in the explanation
  labels or their embeddings

### Requirement: An atlas built from explanation text does not present itself as model geometry
An atlas whose positions come from explanation embeddings SHALL record that its
layout describes explanations of features rather than the model's own
computation, and SHALL record the distribution of explainers across the
features it placed.

This is the standing hazard for a text-derived layout: Neuronpedia's export
does not use one explainer, so a layout over label text could separate features
by writing style rather than by meaning. Recording the distribution is what
makes that detectable; measuring it is required below.

#### Scenario: Text-derived layout is labelled as such
- **WHEN** an atlas built from explanation embeddings is read
- **THEN** its record states that its positions derive from explanation text,
  so a consumer can say so rather than presenting the layout as the model's
  internal geometry

#### Scenario: Explainer influence on the layout is measured
- **WHEN** an atlas is built from explanation embeddings and its features are
  clustered
- **THEN** the record carries a measure of how far the cluster assignment is
  predicted by which explainer wrote the members' labels, so a layout that
  separates on writing style is identifiable as such

#### Scenario: Explainer influence is reported, not acted on
- **WHEN** that measure is high
- **THEN** it is recorded as measured and the atlas is still produced, rather
  than the build silently substituting another source

### Requirement: The atlas is deterministic and versioned
Rebuilding the atlas from the same inputs and the same recorded seed SHALL
produce the same positions and the same clusters. The atlas SHALL carry a
version identifier and a content hash of its positions.

#### Scenario: Rebuild reproduces the layout
- **WHEN** the atlas is rebuilt from the same SAE weights with the recorded
  seed and parameters
- **THEN** the resulting positions and cluster assignments are identical and
  the content hash matches

#### Scenario: Changed inputs produce a new version
- **WHEN** the atlas is rebuilt with different parameters, a different seed, or
  a different SAE release or width
- **THEN** it is recorded under a different version identifier and a different
  content hash, rather than replacing the previous layout under its identity

#### Scenario: A consumer can tell which atlas it holds
- **WHEN** a consumer reads positions from the atlas
- **THEN** the version identifier and content hash are available alongside
  them, so a stale or mismatched layout is detectable rather than silent

### Requirement: The atlas records how much structure the projection preserved
The atlas build SHALL measure and record the fraction of each feature's
nearest neighbours in the original decoder space that remain among its nearest
neighbours in the projected space, aggregated across features, and that
measurement SHALL be readable by any consumer of the atlas.

#### Scenario: Preservation is recorded
- **WHEN** an atlas is built
- **THEN** its record carries the neighbourhood-preservation measurement, the
  neighbourhood size it was computed at, and the number of features sampled

#### Scenario: Poor preservation is reported, not hidden
- **WHEN** the measured preservation is low
- **THEN** the atlas is still produced and the measured value is recorded as
  measured, rather than being suppressed, rounded up, or omitted

### Requirement: Clusters are named only when their members' labels agree
The atlas SHALL group features into clusters, and SHALL assign a cluster a
name only when its members' measured label coherence exceeds that cluster's
own measured random baseline by at least a recorded margin. A cluster that
does not clear its baseline by that margin SHALL be recorded as unnamed.

The gate SHALL be relative to the baseline rather than an absolute coherence
value. An absolute threshold is not a measure of agreement: embeddings from
one explainer share a similarity floor well above zero, so a fixed number
admits clusters whose members agree no more than a random group of the same
size does.

#### Scenario: A coherent cluster is named
- **WHEN** a cluster's measured label coherence exceeds its baseline by at
  least the recorded margin
- **THEN** the cluster carries a name drawn from its members' own labels, and
  the record states which member the name came from, the coherence measured,
  the baseline it was measured against, and the margin in force

#### Scenario: A cluster that only matches chance stays unnamed
- **WHEN** a cluster's measured coherence is above zero but does not exceed its
  baseline by the recorded margin
- **THEN** the cluster is recorded as unnamed with both its coherence and its
  baseline, and no name is synthesised for it

#### Scenario: Coherence is measured against a baseline
- **WHEN** cluster coherence is computed
- **THEN** it is recorded alongside the coherence of a randomly assembled group
  of the same size, so the reported figure can be read as better or worse than
  chance

#### Scenario: No baseline could be measured
- **WHEN** a cluster's coherence was measured but no baseline could be — the
  pool of features with embeddings is smaller than the cluster itself
- **THEN** the cluster is recorded as unnamed with its coherence and no
  baseline, rather than being named against an absolute value

#### Scenario: Explanation embeddings unavailable
- **WHEN** the label store holds no explanation embeddings
- **THEN** clusters are produced and left unnamed, and the atlas record states
  that naming was not attempted, rather than names being derived by another
  means

### Requirement: The atlas records cross-source cluster agreement
The atlas build SHALL compare its decoder-direction clustering against a
clustering of the same features derived from their explanation embeddings, and
SHALL record the agreement between the two.

#### Scenario: Agreement is recorded
- **WHEN** an atlas is built and explanation embeddings are available
- **THEN** its record carries an agreement measure between the
  decoder-direction clusters and the explanation-embedding clusters

#### Scenario: Agreement is not a gate
- **WHEN** the recorded agreement is low
- **THEN** the atlas is still produced with its decoder-direction positions and
  clusters, and the low agreement is recorded rather than causing the build to
  substitute the other clustering

### Requirement: The atlas does not pool explanation text across explainers without saying so
Where the atlas uses explanation text or its embeddings, it SHALL record the
distribution of explainers across the features involved, so a cluster that is
an artifact of differing explainer styles is identifiable.

#### Scenario: Explainer distribution recorded per named cluster
- **WHEN** a cluster is named or its coherence is measured
- **THEN** the record includes which explainers produced its members' labels
  and in what proportion

#### Scenario: A single-explainer cluster is visible as such
- **WHEN** a cluster's members' labels come predominantly from one explainer
  while the model as a whole uses several
- **THEN** that concentration is recorded with the cluster, so it can be read
  as a possible explainer artifact rather than a property of the model

### Requirement: The atlas is available without a trace
Atlas positions, cluster assignments, cluster names and the atlas record SHALL
be retrievable independently of any trace, and SHALL NOT require a model load.

#### Scenario: Query positions with no trace
- **WHEN** a consumer asks for the positions of a set of (layer, feature)
  pairs and no trace exists
- **THEN** the positions are returned and the model is not loaded

#### Scenario: Atlas missing
- **WHEN** no atlas has been built
- **THEN** a request for positions reports the atlas's absence rather than
  returning fabricated or zeroed positions
