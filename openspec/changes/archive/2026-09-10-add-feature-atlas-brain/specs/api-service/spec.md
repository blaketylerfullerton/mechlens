## ADDED Requirements

### Requirement: A returned trace carries the atlas positions for the features it reports
When a trace job has run the feature pass and an atlas is available, the trace
returned for that job SHALL carry a position for each (layer, feature index) it
reports as active, together with the identity of the atlas those positions came
from.

#### Scenario: Trace with features and an atlas
- **WHEN** a client polls a completed job whose trace records active features
  and an atlas is available
- **THEN** the trace carries a position for each reported (layer, feature)
  pair, and records the atlas version and content hash those positions came
  from

#### Scenario: No atlas available
- **WHEN** a trace records active features and no atlas is available
- **THEN** the trace is returned with its features and no positions, and
  records the absence of an atlas rather than carrying zeroed or fabricated
  positions

#### Scenario: A reported feature is absent from the atlas
- **WHEN** the trace reports a feature the atlas has no position for
- **THEN** that feature is returned without a position, and the response
  distinguishes it from a feature whose position is present

#### Scenario: Trace without the feature pass
- **WHEN** a trace job ran without the feature pass
- **THEN** the returned trace carries no positions and no atlas identity,
  because it reports no features to position

### Requirement: Atlas positions and areas are queryable independent of any trace
The service SHALL expose the atlas — feature positions, cluster assignments,
cluster names where the atlas recorded them, and the atlas's own record — for
retrieval without a trace and without loading the model.

#### Scenario: Query the atlas with no trace
- **WHEN** a client requests the atlas
- **THEN** the response carries the positions, cluster assignments, recorded
  cluster names, and the atlas record including its version, content hash and
  neighbourhood-preservation measurement, and the model is not loaded

#### Scenario: Unnamed clusters
- **WHEN** the atlas recorded a cluster as unnamed
- **THEN** it is returned as unnamed, and no name is synthesised at request
  time

#### Scenario: Atlas missing
- **WHEN** no atlas has been built
- **THEN** the request reports the atlas's absence rather than returning an
  empty layout that reads as an atlas with no features

## MODIFIED Requirements

### Requirement: A trace request may ask for enrichment passes
`POST /trace` SHALL accept an optional list of enrichment passes to run as
part of the trace job, and the trace returned for that job SHALL include the
fields those passes produce. The accepted passes SHALL include the logit-lens
pass, the SAE feature pass, and the feature-label pass.

#### Scenario: Requesting the lens pass
- **WHEN** a client sends `POST /trace` asking for the `lens` pass
- **THEN** the trace returned when the job completes carries per-layer
  logit-lens readouts for its token positions, and a pass record for `lens`
  among its passes

#### Scenario: Requesting the feature pass
- **WHEN** a client sends `POST /trace` asking for the SAE feature pass
- **THEN** the trace returned when the job completes carries per-layer,
  per-position active features with their activations and the true count of
  features that fired, and a pass record for that pass among its passes

#### Scenario: Requesting the label pass
- **WHEN** a client sends `POST /trace` asking for the feature-label pass
- **THEN** the trace returned when the job completes carries labels for the
  features it reports, and a pass record for that pass among its passes

#### Scenario: Requesting labels without features
- **WHEN** a client asks for the label pass without asking for the feature
  pass
- **THEN** the request is rejected with a client error naming the missing
  dependency, and no job is enqueued

#### Scenario: Requesting no passes
- **WHEN** a client sends `POST /trace` without asking for any pass
- **THEN** the job runs capture only and the returned trace carries no
  enrichment fields, exactly as it does today

#### Scenario: Requesting an unknown pass
- **WHEN** a client sends `POST /trace` naming a pass the service does not
  support
- **THEN** the request is rejected with a client error and no job is
  enqueued

#### Scenario: An enrichment pass fails
- **WHEN** capture succeeds but a requested enrichment pass raises
- **THEN** the job reports failure with an error message, rather than
  returning a trace that silently lacks the requested fields

#### Scenario: The label store is unavailable
- **WHEN** the label pass is requested and no label store is available
- **THEN** the job reports failure naming the missing store, rather than
  returning a trace whose features silently carry no labels

### Requirement: A running job reports where it is
`GET /trace/{id}` SHALL report in-flight progress for a job that is
running — which phase of the job is executing and how far through that
phase's units of work it is — in addition to the job's status. Every phase
SHALL report in the units that phase genuinely advances through.

#### Scenario: Progress during generation
- **WHEN** a client polls a job that is generating tokens
- **THEN** the response identifies the generation phase and reports how many
  tokens have been generated out of how many were requested

#### Scenario: Progress during an enrichment pass
- **WHEN** a client polls a job that is running the lens pass
- **THEN** the response identifies that phase and reports how many layers
  have been decoded out of the model's layer count

#### Scenario: Progress during the feature pass
- **WHEN** a client polls a job that is running the SAE feature pass
- **THEN** the response identifies that phase and reports how many layers have
  been encoded out of the model's layer count

#### Scenario: Phases are reported in the order the job reaches them
- **WHEN** a client polls a job that runs several passes and compares
  successive readings
- **THEN** the reported phase never moves back to a phase the job has already
  left

#### Scenario: Progress before work starts
- **WHEN** a client polls a job that is still `pending`
- **THEN** the response carries no progress reading, distinguishing "queued"
  from "running and at token 0"

#### Scenario: Progress after completion
- **WHEN** a client polls a job that has finished, successfully or with an
  error
- **THEN** the response reports the terminal status and the client does not
  depend on a progress reading being present

#### Scenario: Progress never goes backwards within a phase
- **WHEN** a client polls the same running job repeatedly
- **THEN** the counters reported for a given phase are non-decreasing across
  those responses
