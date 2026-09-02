# Api-Service Specification

## Purpose

Exposes trace generation, steering, and feature-metadata lookup over HTTP so
a client other than the CLI (a browser, the viewer) can drive the model
without reloading it per request.

## Requirements

### Requirement: Trace requests are accepted asynchronously
`POST /trace` SHALL accept a prompt and a maximum token count, enqueue a
trace job, and respond immediately with a job identifier rather than
blocking until generation completes.

#### Scenario: Trace request returns before generation finishes
- **WHEN** a client sends `POST /trace` with a valid `prompt` and
  `max_tokens`
- **THEN** the response arrives with a job id before the trace has finished
  generating, and the response does not include the full trace body

#### Scenario: Missing or invalid parameters are rejected
- **WHEN** a client sends `POST /trace` without a `prompt`, or with a
  non-positive `max_tokens`
- **THEN** the request is rejected with a client error and no job is
  enqueued

### Requirement: Job status and result are retrievable by id
`GET /trace/{id}` SHALL report the current state of the job identified by
`id`, and SHALL return the complete trace document once the job has
finished successfully.

#### Scenario: Poll before completion
- **WHEN** a client requests `GET /trace/{id}` for a job that is still
  pending or running
- **THEN** the response indicates that state and does not include a trace
  document

#### Scenario: Poll after completion
- **WHEN** a client requests `GET /trace/{id}` for a job that has finished
  successfully
- **THEN** the response includes the complete trace document for that job

#### Scenario: Poll after failure
- **WHEN** a client requests `GET /trace/{id}` for a job that failed during
  generation
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
