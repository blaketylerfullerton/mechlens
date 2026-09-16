import { request } from './api-client'

/**
 * The Circuits API: attribution graphs over transcoder features.
 *
 * A transcoder feature is not a Gemma Scope residual-SAE feature. Different
 * dictionary, different feature space — nothing here may be joined to
 * `getFeature(layer, idx)` in `api-client` by layer/index, and no label from
 * there may be shown against a node from here.
 */

/** Node kinds, in the order they sit in the backend's adjacency matrix. */
export type CircuitNodeKind = 'feature' | 'error' | 'embed' | 'logit'

export interface CircuitNode {
  id: string
  kind: CircuitNodeKind
  layer: number | null
  position: number | null
  /** Index into the transcoder's feature space. Never a residual-SAE index. */
  feature_idx: number | null
  activation: number | null
  influence: number
}

export interface CircuitEdge {
  source: string
  target: string
  /** Signed: a negative contribution pushes the target token down. */
  weight: number
}

export interface CircuitSource {
  trace_id: string
  model: string
  prompt: string | null
  prefix_token_ids: number[]
  prefix_texts: string[] | null
  target_token_id: number
  target_text: string | null
  target_position: number
}

/**
 * How much of the computation the served graph actually covers.
 *
 * Rendered next to the graph, never behind a disclosure: an omitted node or
 * edge is below a retention threshold, not a zero effect, and a reader who
 * cannot see the difference is being misled by the picture.
 */
export interface CircuitCoverage {
  total_nodes: number
  retained_nodes: number
  total_edges_nonzero: number
  edges_after_pruning: number
  sent_nodes?: number
  sent_edges: number
  n_active_features: number
  n_selected_features: number
  n_error_nodes: number
  n_layers: number
  n_positions: number
  node_threshold: number
  edge_threshold: number
  max_edges_per_node: number
  max_feature_nodes: number
  note: string
}

export type AnalysisStatus = 'queued' | 'running' | 'done' | 'error' | 'interrupted'

export interface AnalysisSummary {
  analysis_id: string
  created_at: number
  status: AnalysisStatus
  error: string | null
  source: CircuitSource
  provenance: Record<string, unknown>
  coverage: Partial<CircuitCoverage>
  target_probability: number | null
  elapsed_s: number | null
}

export interface CircuitGraphResponse {
  analysis_id: string
  source: CircuitSource
  target_probability: number | null
  nodes: CircuitNode[]
  edges: CircuitEdge[]
  coverage: CircuitCoverage
}

/** One high-activation context for a feature, with its per-token activations. */
export interface FeatureExample {
  quantile: string | null
  activation: number
  peak_index: number
  tokens: string[]
  activations: number[]
}

export interface FeatureEvidence {
  layer: number
  feature_idx: number
  transcoder_set: string
  transcoder_revision: string | null
  /**
   * Always null. The transcoder artifact carries activation examples but no
   * written label — these features have never been auto-interpreted. The UI
   * shows the id and an explicit unlabeled state rather than borrowing a name
   * from a different dictionary.
   */
  label: string | null
  label_available: boolean
  act_min: number | null
  act_max: number | null
  /** Unembed directions the feature writes toward. Noisy; not a description. */
  top_logits: string[] | null
  bottom_logits: string[] | null
  examples: FeatureExample[]
  n_examples_available: number
}

export function createAnalysis(source: CircuitSource): Promise<{ analysis_id: string; job_id: string }> {
  return request('/circuits/analyses', { method: 'POST', body: JSON.stringify({ source }) })
}

export function getAnalysis(id: string): Promise<AnalysisSummary> {
  return request(`/circuits/analyses/${id}`)
}

export function listAnalyses(limit = 50): Promise<{ analyses: AnalysisSummary[]; total: number }> {
  return request(`/circuits/analyses?limit=${limit}`)
}

export function deleteAnalysis(id: string): Promise<{ deleted: string }> {
  return request(`/circuits/analyses/${id}`, { method: 'DELETE' })
}

/**
 * `minInfluence` re-filters what the analysis already retained; it never
 * computes more. Widening past the retained set needs a new analysis, which is
 * why `coverage` travels with every response.
 */
export function getGraph(id: string, minInfluence = 0, maxEdgesPerNode?: number): Promise<CircuitGraphResponse> {
  const params = new URLSearchParams({ min_influence: String(minInfluence) })
  if (maxEdgesPerNode != null) params.set('max_edges_per_node', String(maxEdgesPerNode))
  return request(`/circuits/analyses/${id}/graph?${params}`)
}

export function getFeatureEvidence(layer: number, featureIdx: number, maxExamples = 6): Promise<FeatureEvidence> {
  return request(`/circuits/features/${layer}/${featureIdx}?max_examples=${maxExamples}`)
}

/** Build the request body for a generated token, enforcing the one invariant.
 *
 * For a token generated at position j the explanatory prefix is token_ids[:j].
 * The target is never part of its own prefix, and ids are carried through
 * rather than re-derived from text: decoding and re-encoding a gemma prefix
 * silently doubles the BOS.
 */
export function sourceForStep(
  trace: { trace_id: string; model: string; prompt: string; steps: { token: { token_id: number; text: string; source: string; position: number } }[] },
  step: number,
): CircuitSource | null {
  const target = trace.steps[step]
  if (!target || target.token.source !== 'generated') return null
  const prefix = trace.steps.slice(0, step)
  if (prefix.length === 0) return null
  return {
    trace_id: trace.trace_id,
    model: trace.model,
    prompt: trace.prompt,
    prefix_token_ids: prefix.map((s) => s.token.token_id),
    prefix_texts: prefix.map((s) => s.token.text),
    target_token_id: target.token.token_id,
    target_text: target.token.text,
    target_position: prefix.length,
  }
}
