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

export interface TraceRequest {
  prompt: string
  max_tokens: number
}

export interface SteerRequest {
  prompt: string
  max_tokens: number
  layer: number
  feature_idx: number
  coefficient: number
}

export interface JobResponse {
  job_id: string
}

export type JobStatus = 'pending' | 'running' | 'done' | 'error'

export interface JobStatusResponse {
  status: JobStatus
  trace: Trace | null
  error: string | null
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
