# Self-hosted auto-interp

## Decision

Auto-interp is a local, asynchronous labeling workflow for a completed custom
SAE. It produces a *candidate* explanation from checkpoint-scoped activation
examples, then tests that candidate on held-out positives and matched controls.
A generated label is never silently treated as ground truth.

MechLens owns the job queue, evidence selection, evaluation, provenance, and
storage. A configurable local inference provider only generates structured
candidate explanations. This separation means the serving stack or model can
change without changing the evidence or evaluation contract.

The initial transport is an OpenAI-compatible local endpoint. Ollama is the
default quick-start target; vLLM and llama.cpp work by changing configuration.
Direct `transformers` loading is a later provider, not a separate job format.

## User path

1. Train an SAE and collect examples for one or more feature IDs.
2. Configure or select a local inference profile.
3. Create a bounded job in **Auto-interp jobs**.
4. MechLens snapshots the examples and creates candidate labels.
5. MechLens evaluates each candidate on examples not provided to the model.
6. The viewer shows the label, evidence, score, and provenance. Failed or weak
   labels remain **unverified**.

Jobs are scoped to one checkpoint artifact ID. Re-training or choosing a
different checkpoint always makes a different job; IDs never migrate between
dictionaries.

## Stable provider contract

Providers implement one operation:

```python
generate_candidate(request: CandidateRequest) -> CandidateResponse
```

`CandidateRequest` contains only:

- a versioned prompt template and schema version;
- the requested feature ID and checkpoint artifact ID;
- a bounded, ordered set of positive activation examples;
- no held-out examples or controls.

`CandidateResponse` must parse against this schema:

```json
{
  "label": "short, specific feature description",
  "summary": "one or two sentence hypothesis",
  "evidence": ["example indices that support the hypothesis"],
  "uncertainty": "limits or ambiguity in the observed examples"
}
```

Invalid JSON, schema failures, timeouts, and provider errors fail the feature
record without deleting already completed records in the same job. Raw model
responses are stored only when the user enables debug retention; the parsed
response and request fingerprint are stored always.

No provider receives a feature index alone: examples are the evidence. No
provider receives evaluation data, so evaluation cannot be leaked into prompt
construction.

## Provider profiles

Profiles are declarative. The job saves a fully resolved profile snapshot, not
only a mutable profile name.

```toml
[auto_interp.profile.local]
provider = "openai_compatible"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:14b"
api_key_env = "MECHLENS_AUTO_INTERP_API_KEY" # optional for Ollama
timeout_s = 120
max_parallel_requests = 1
response_mode = "json_schema"

[auto_interp.profile.spark]
provider = "openai_compatible"
base_url = "http://127.0.0.1:8000/v1"
model = "your-32b-instruct-model"
timeout_s = 180
max_parallel_requests = 1
response_mode = "json_schema"
```

The configuration lives outside the repository, at
`$MECHLENS_CONFIG_DIR/config.toml` (default: the backend data directory). The
UI may select a profile but never writes secrets. Environment values are read
at dispatch time and never persisted to SQLite, logs, reports, or the browser.

Supported provider values will be:

| Provider | Use case | Configuration that changes |
| --- | --- | --- |
| `openai_compatible` | Ollama, vLLM, llama.cpp | URL, model name, optional auth |
| `transformers_local` | single-process experimental mode | local model path/ID, dtype, device |

The initial release implements only `openai_compatible`. Adding
`transformers_local` must satisfy the exact same request/response contract.

## Resource policy for this host

This GB10 host has 128 GB of coherent unified memory; the current backend has
roughly 89 GiB available. MechLens also retains Gemma 2B and may retain an SAE,
so auto-interp must leave headroom for the OS, model runtime, KV cache, and the
interactive workload.

Default profiles and limits:

| Profile | Intended explainer | Quantization | Policy |
| --- | --- | --- | --- |
| `local` | 8B–14B instruct | 4-bit | baseline and laptop-friendly |
| `spark` | 14B–32B instruct | 4-bit | default on this host |
| `large` | up to 70B instruct | 4-bit | explicit advanced opt-in; one request only |

The app must not estimate safety solely from parameter count. Before dispatch,
the provider health check reports its model ID; MechLens records available host
memory, selected concurrency, requested context budget, and the active Gemma
model. The UI warns when available memory is below 30 GiB. It does not launch
parallel jobs by default.

Direct `transformers_local` cannot be selected while a model load would violate
the configured memory reserve. The initial reserve is 30 GiB and is configurable
only in the server configuration file.

## Job and result records

The durable job record contains:

- job ID, checkpoint artifact ID, SAE layer/feature count, and feature IDs;
- resolved provider profile minus secrets;
- prompt/schema versions and source-example report fingerprint;
- state (`queued`, `running`, `evaluating`, `completed`, `failed`, `cancelled`,
  or `interrupted`), counters, timestamps, and per-feature errors.

Each label record contains the parsed candidate response, status, evaluation
metrics, source example IDs, held-out/control selection fingerprint, provider
model ID, and prompt version. Labels with an unavailable or failed evaluation
are visibly `unverified`, never shown as ordinary feature labels.

Results publish atomically after each feature. Restart recovery converts an
in-flight job to `interrupted`; a resume action reuses completed feature records
only if its checkpoint, profile snapshot, prompt version, and evidence
fingerprints match exactly.

## Evaluation v1

For each feature, split examples deterministically into prompt positives and
held-out positives. Sample matched controls from the same corpus/source and
activation scan, preferring low-activation positions. The evaluator asks a
separate, deterministic local scoring routine whether the candidate label
