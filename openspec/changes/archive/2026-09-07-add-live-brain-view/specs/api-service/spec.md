## ADDED Requirements

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

## MODIFIED Requirements

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
