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
  // Schema 1.4. Keyed "layer/index" exactly as `labels` is, and for the same
  // reason: a feature recurs across token positions, so three floats copied
  // onto every occurrence would be the same numbers thousands of times.
  //
  // A feature the atlas cannot place is simply *absent* here — that is what
  // lets a reader tell "unplaced" from "placed at the origin", and it is why
  // nothing in the view may substitute a default position. Empty overall means
  // either no atlas was available or the layout pass did not run; the `layout`
  // pass record is what distinguishes those two.
  layout: Record<string, NodePosition>
  steps: TokenStep[]
}

// Where one feature sits in the atlas (schema 1.4, `Trace.layout`).
export interface NodePosition {
  x: number
  y: number
  z: number
  // -1 means the feature belongs to no cluster: a real answer from a
  // density-based clustering, not a missing value.
  cluster: number
}

// -- HTTP request/response shapes (service/models.py) --

// The enrichment passes POST /trace will run as part of the trace job. Mirrors
// `TracePass` in service/models.py: an unrecognised name is a 422 there, so
// this union is not cosmetic.
export type TracePass = 'lens' | 'sae' | 'labels'

export interface TraceRequest {
  live?: boolean
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
  partial_trace?: Trace | null
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

// -- GET /atlas ------------------------------------------------------------
//
// The same wire shape `build_feature_atlas.py` writes for the static idle
// asset, deliberately: `parseAtlas` in lib/atlas.ts reads both, so the brain
// has one parser rather than two that can drift. The extra fields here over
// the asset's are the atlas record's — identity and provenance, which the
// endpoint can afford to carry and a 571KB static file need not.
//
// 404 rather than an empty atlas means no atlas has been built. An atlas that
// placed *nothing* is a 200 with empty node arrays — different facts, and a
// client that conflates them renders one as the other.

// Column-wise, not a list of node objects: it is the same four values repeated
// ~98,000 times, and per-object key names would dominate the payload. `xyz` is
// flat int16 triples over `extent` (position = value / 32767 * extent).
export interface AtlasNodesPayload {
  layer: number[]
  feature: number[]
  cluster: number[]
  xyz: number[]
}

export interface AtlasResponse {
  atlas_version: string
  source: string
  note: string
  // The layout's own measured fidelity. null means not measured — never 0
  // standing in for it.
  knn_preservation: number | null
  knn_k: number | null
  explainer_ami: number | null
  // Identity, so a trace drawn against a different atlas is detectable rather
  // than silently misplaced.
  positions_sha256: string
  release: string
  width: string
  seed: number
  // Which layers this atlas covers, e.g. "0,5,10,16,20,25". The view states
  // its own scope from this rather than implying it covers the whole model.
  layers: string
  extent: number
  quantisation: string
  n_sampled: number
  n_total: number
  nodes: AtlasNodesPayload
  areas: AtlasAreaPayload[]
}

// One cluster. `name` is null whenever the naming was not earned — a
// measurement outcome to show, never a field to fill in.
export interface AtlasAreaPayload {
  cluster: number
  name: string | null
  n_members: number
  centroid: [number, number, number]
  spread: number
  coherence: number | null
  baseline_coherence: number | null
  explainers: string
}
