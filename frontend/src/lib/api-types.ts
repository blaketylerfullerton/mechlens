// Mirrors backend/app/schema.py and backend/app/service/models.py.
// Keep field names and literal unions in lockstep with those two files —
// this is the wire contract, not an independent frontend model.

export interface TokenInfo {
  position: number
  token_id: number
  text: string
  source: 'prompt' | 'generated'
}

export interface TopToken {
  token_id: number
  text: string
  logit: number
  prob: number
}

export interface LogitSummary {
  top_k: TopToken[]
  entropy: number
  chosen: TopToken | null
}

export interface Feature {
  index: number
  activation: number
}

export interface LogitLens {
  top_k: TopToken[]
  entropy: number
}

export interface NodeRef {
  layer: number
  position: number
  feature: number | null
}

export interface Edge {
  source: NodeRef
  weight: number
  kind: 'attn' | 'mlp' | 'resid' | 'sae'
}

export interface LayerState {
  layer: number
  resid_norm: number
  logit_lens: LogitLens | null
  features: Feature[]
  l0: number | null
  edges: Edge[]
}

export interface FeatureLabel {
  text: string
  explainer: string | null
  explanation_type: string | null
  score: number | null
}

export interface TokenStep {
  step: number
  token: TokenInfo
  logits: LogitSummary
  layers: LayerState[]
}

export interface ResidualRef {
  path: string
  format: 'npy'
  hook: string
  shape: [number, number, number]
  dtype: string
}

export interface SteeringInfo {
  layer: number
  feature_idx: number
  coefficient: number
}

export interface PassRecord {
  name: string
  params: Record<string, string | number | boolean>
  stats: Record<string, number | number[]>
  elapsed_s: number
  created_at: string
}

export interface Trace {
  schema_version: string
  trace_id: string
  created_at: string
  model: string
  device: string
  dtype: string
  n_layers: number
  d_model: number
  normalization: string | null
  prompt: string
  completion: string
  n_prompt_tokens: number
  n_generated_tokens: number
  stop_reason: 'max_tokens' | 'eos'
  elapsed_s: number
  residuals: ResidualRef | null
  passes: PassRecord[]
  steering: SteeringInfo | null
  labels: Record<string, FeatureLabel>
  steps: TokenStep[]
}

// -- HTTP request/response shapes (service/models.py) --

// The enrichment passes POST /trace will run as part of the trace job. Mirrors
// `TracePass` in service/models.py: an unrecognised name is a 422 there, so
// this union is not cosmetic.
export type TracePass = 'lens' | 'sae' | 'labels'

export interface TraceRequest {
  prompt: string
  max_tokens: number
  // `labels` without `sae` is a 422: the label pass labels the features the
  // SAE pass records, so it cannot run on its own.
  passes?: TracePass[]
  // Which layers the SAE pass encodes; omit for every layer. The subset exists
  // because 26 resident 16k SAEs come to ~7.9GB — fine on a unified-memory
  // box, not fine on a 16GB discrete GPU. Ignored unless `passes` includes
  // `sae`, and a layer at or past `n_layers` is a 422.
  sae_layers?: number[] | null
}

export interface SteerRequest {
  prompt: string
  max_tokens: number
  layer: number
  feature_idx: number
  coefficient: number
}

// Mirrors HealthResponse in backend/app/service/models.py: "loading" while the
// model warms up (the server binds its port first), "error" if the load failed.
export interface HealthResponse {
  status: 'loading' | 'ready' | 'error'
  detail?: string | null
}

export interface JobResponse {
  job_id: string
}

export type JobStatus = 'pending' | 'running' | 'done' | 'error'

// Which phase of a trace job is executing, in the order a job reaches them.
// `generating` counts tokens; `sae` and `lens` count layers, because those are
// the units each phase can honestly report — the capture loop computes a whole
// forward pass at once, so there is no moment during it at which one layer is
// "executing", while both later passes genuinely walk depth one layer at a
// time. A reading never moves back to a phase the job has already left.
export type JobPhase = 'generating' | 'sae' | 'lens'

export interface JobProgress {
  phase: JobPhase
  done: number
  total: number
}

export interface JobStatusResponse {
  status: JobStatus
  trace: Trace | null
  error: string | null
  // null for a queued job, and for a running one that has not reported yet:
  // "pending" and "running, at token 0" are different answers.
  progress: JobProgress | null
}

export interface FeatureResponse {
  layer: number
  feature_idx: number
  label: string | null
  explainer: string | null
  explanation_type: string | null
  score: number | null
  url: string
}
