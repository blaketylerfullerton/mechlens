# Api-Service Specification

## Purpose

Exposes trace generation, steering, and feature-metadata lookup over HTTP so
a client other than the CLI (a browser, the viewer) can drive the model
without reloading it per request.

## Requirements

### Requirement: Trace requests are accepted asynchronously
`POST /trace` SHALL accept a prompt, a maximum token count, and an optional
set of enrichment passes, enqueue a trace job, and respond immediately with
a job identifier rather than blocking until generation and any requested
passes complete.

#### Scenario: Trace request returns before generation finishes
- **WHEN** a client sends `POST /trace` with a valid `prompt` and
  `max_tokens`
- **THEN** the response arrives with a job id before the trace has finished
  generating, and the response does not include the full trace body

#### Scenario: Trace request with passes returns before the passes run
- **WHEN** a client sends `POST /trace` asking for enrichment passes
- **THEN** the response arrives with a job id before those passes have run,
  and the passes execute as part of the same job

#### Scenario: Missing or invalid parameters are rejected
- **WHEN** a client sends `POST /trace` without a `prompt`, or with a
  non-positive `max_tokens`
- **THEN** the request is rejected with a client error and no job is
  enqueued

### Requirement: Job status and result are retrievable by id
`GET /trace/{id}` SHALL report the current state of the job identified by
`id`, SHALL report in-flight progress while that job is running, and SHALL
return the complete trace document once the job has finished successfully.

#### Scenario: Poll before completion
- **WHEN** a client requests `GET /trace/{id}` for a job that is still
  pending or running
- **THEN** the response indicates that state and does not include a trace
  document

#### Scenario: Poll after completion
- **WHEN** a client requests `GET /trace/{id}` for a job that has finished
  successfully
- **THEN** the response includes the complete trace document for that job,
  including the fields produced by any enrichment passes the request asked
  for

#### Scenario: Poll after failure
- **WHEN** a client requests `GET /trace/{id}` for a job that failed during
  generation or during a requested enrichment pass
- **THEN** the response indicates failure and includes an error message,
  not a trace document

#### Scenario: Unknown job id
- **WHEN** a client requests `GET /trace/{id}` for an id that was never
  issued
- **THEN** the response is a not-found error

### Requirement: Steer requests follow the same job lifecycle as trace requests
`POST /steer` SHALL accept the same prompt/length parameters as `POST
/trace` plus a feature steering specification, and SHALL enqueue and report
the resulting job through the same id, polling, and result shape as
`/trace`.

#### Scenario: Steer request returns a pollable job
- **WHEN** a client sends `POST /steer` with a valid prompt, length, and
  steering specification
- **THEN** the response arrives with a job id, and that id is pollable via
  `GET /trace/{id}` exactly as a `/trace` job would be

#### Scenario: Invalid steering target is rejected
- **WHEN** a client sends `POST /steer` naming a layer outside the model's
  range, or a feature index not present in that layer's SAE
- **THEN** the request is rejected with a client error and no job is
  enqueued

### Requirement: At most one generation job runs at a time
The service SHALL serialize trace and steer job execution so that no two
jobs run generation concurrently, regardless of how many requests arrive
at once.

#### Scenario: Concurrent submissions are queued, not run in parallel
- **WHEN** two or more `POST /trace` or `POST /steer` requests arrive
  before the first has finished
- **THEN** each is accepted and assigned its own job id, and their
  generation work executes one at a time rather than overlapping

### Requirement: Feature metadata is queryable independent of any trace
`GET /feature/{layer}/{idx}` SHALL return the label and associated metadata
for the SAE feature at the given layer and index, without requiring a
trace to exist.

#### Scenario: Known feature
- **WHEN** a client requests `GET /feature/{layer}/{idx}` for a feature
  that has been looked up before or has a label available
- **THEN** the response includes that feature's label text and metadata

#### Scenario: Feature with no explanation
- **WHEN** a client requests `GET /feature/{layer}/{idx}` for a feature
  that Neuronpedia has no explanation for
- **THEN** the response reflects the absence of a label rather than an
  error, distinguishing "no label" from "not found"

#### Scenario: Out-of-range feature or layer
- **WHEN** a client requests `GET /feature/{layer}/{idx}` for a layer or
  index outside the model's or SAE's range
- **THEN** the response is a not-found or client error

### Requirement: A trace request may ask for enrichment passes
`POST /trace` SHALL accept an optional list of enrichment passes to run as
part of the trace job, and the trace returned for that job SHALL include the
fields those passes produce.

#### Scenario: Requesting the lens pass
- **WHEN** a client sends `POST /trace` asking for the `lens` pass
- **THEN** the trace returned when the job completes carries per-layer
  logit-lens readouts for its token positions, and a pass record for `lens`
  among its passes

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

### Requirement: A running job reports where it is
`GET /trace/{id}` SHALL report in-flight progress for a job that is
running — which phase of the job is executing and how far through that
phase's units of work it is — in addition to the job's status.

#### Scenario: Progress during generation
- **WHEN** a client polls a job that is generating tokens
- **THEN** the response identifies the generation phase and reports how many
  tokens have been generated out of how many were requested

#### Scenario: Progress during an enrichment pass
- **WHEN** a client polls a job that is running the lens pass
- **THEN** the response identifies that phase and reports how many layers
  have been decoded out of the model's layer count

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

### Requirement: Progress reporting does not change job serialization or persistence
Reporting progress SHALL NOT allow two generation jobs to run concurrently,
and progress SHALL NOT be persisted beyond the lifetime of the service
process.

#### Scenario: Concurrent submissions still queue
- **WHEN** several trace or steer requests arrive at once and each is polled
  for progress
- **THEN** at most one job reports a running phase at any time, and the
  others report `pending`

#### Scenario: Service restart
- **WHEN** the service process restarts
- **THEN** previously issued job ids report not-found, and no progress from
  before the restart is served
