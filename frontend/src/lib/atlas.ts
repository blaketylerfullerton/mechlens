// The feature atlas: where each SAE feature is drawn, and what the layout is
// and is not allowed to claim.
//
// A node is one SAE feature. Its position comes from a UMAP projection —
// computed once, offline, under a fixed seed
// (backend/scripts/build_feature_atlas.py) and served as a fixed table.
// Nothing here computes a position, and nothing here adjusts one for the trace
// being displayed: two traces showing the same feature must place it
// identically, or they were drawn against different atlases and the interface
// has to say so.
//
// What the layout means, exactly
// ------------------------------
// Near means similar — but similar *in what*, and that depends on the atlas's
// source, which is why `AtlasSource` exists and why the view states it. See
// that type: a label-derived atlas maps how features were described, a
// decoder-derived one maps how the model writes them. Reading one as the other
// is the mistake this module is arranged to prevent.
//
// Far means nothing, in either. UMAP preserves local neighbourhoods and
// distorts larger distances, so "these two nodes are close" is a real
// statement and "these two regions are far apart" is not one. That is why this
// module exposes no distance function and the view offers no axes, no
// coordinate readout and no scale: there is no honest way to answer a question
// about distance here.
//
// How much to trust it is a number, not a vibe: `knnPreservation` is carried
// on the atlas and rendered in the legend.
//
// Deliberately free of runtime imports -- types and pure functions only -- so
// the parser can be exercised without a WebGL context, the way lib/lens.ts is.

/** One feature's place in the atlas. */
export interface AtlasNode {
  layer: number
  feature: number
  x: number
  y: number
  z: number
  /** -1 means the feature belongs to no cluster: a real answer, not a gap. */
  cluster: number
}

/**
 * One cluster of the atlas, as an area on the brain.
 *
 * `name` is null whenever the naming was not earned, and that is the design: an
 * area is named only when its members' label agreement clears its own random
 * baseline by a recorded margin. On the shipping atlas 27 of 32 clear it; on
 * the decoder atlas none of 84 attempted groups did. A null name is a
 * measurement outcome to be shown, never a field to fill in.
 */
export interface AtlasArea {
  cluster: number
  name: string | null
  /**
   * `[layer, feature]` of the member whose own label became `name` — the
   * cluster's medoid.
   *
   * Carried so the view can disclose *whose* label is speaking. An area name
   * is one member's sentence standing for hundreds of features, which is the
   * same lossiness a blended band had, and it is disclosed the same way:
   * by naming the member it came from rather than letting the name read as a
   * summary of the cluster. Null whenever `name` is null.
   */
  name_source?: [number, number] | null
  n_members: number
  centroid: [number, number, number]
  spread: number
  coherence: number | null
  baseline_coherence: number | null
  explainers: string
  /** Present when a name was withheld, saying why. */
  name_withheld?: string
}

/**
 * Which representation an atlas's positions were projected from.
 *
 * Not cosmetic: the two answer different questions and the view has to say
 * which one it is showing, because "near" means something different in each.
 *
 * - `labels` — the features' Neuronpedia explanation embeddings. Near means an
 *   explainer *described* two features similarly. This is the default because
 *   it is the only source that yields areas at all, and its measured numbers
 *   are better across the board (0.29 preservation against 0.15, 27 of 32
 *   clusters earning a name against none of 84). The cost is that the map
 *   describes descriptions, not the model's computation.
 * - `decoder` — the features' SAE decoder directions. Near means the two
 *   features *write* similarly into the model's residual stream. The model's
 *   own geometry, and a continuum: no clusterable areas exist in it.
 */
export type AtlasSource = 'labels' | 'decoder'

export interface Atlas {
  version: string
  source: AtlasSource
  /** What this file is, in its own words — rendered in the legend, not paraphrased. */
  note: string
  nodes: AtlasNode[]
  areas: AtlasArea[]
  /** How many of the atlas's features this file carries, and how many exist. */
  sampled: number
  total: number
  /**
   * The layout's own measured fidelity: the fraction of a feature's `knnK`
   * nearest neighbours in the source space that are still among its nearest in
   * three dimensions. Null when the atlas did not record it.
   *
   * Surfaced so the view can publish how much to trust the picture instead of
   * asking the reader to take it on trust.
   */
  knnPreservation: number | null
  knnK: number | null
  /**
   * How far the clustering is predicted by which explainer wrote the labels.
   * ~0 means the areas are about meaning; near 1 would mean they are an
   * artifact of the labeller. Null when it could not be measured.
   */
  explainerAmi: number | null
}

/** The idle asset's on-disk shape, as `build_feature_atlas.py` writes it. */
interface AtlasPayload {
  atlas_version: string
  source?: string
  note: string
  knn_preservation?: number | null
  knn_k?: number | null
  explainer_ami?: number | null
  extent: number
  n_sampled: number
  n_total: number
  nodes: {
    layer: number[]
    feature: number[]
    cluster: number[]
    /** Flat int16 triples; position = value / 32767 * extent. */
    xyz: number[]
  }
  areas: AtlasArea[]
}

/** int16 quantisation divisor, matching `atlas.quantise_positions`. */
const INT16_SCALE = 32767

export class AtlasFormatError extends Error {}

/**
 * Parse the idle atlas asset.
 *
 * Throws rather than returning a partial atlas: a truncated node list would
 * silently misplace features, which is worse than not drawing them. The caller
 * is expected to treat a failure as "no atlas available" and say so — see the
 * requirement that the brain renders the shell alone rather than placing nodes
 * at arbitrary positions.
 */
export function parseAtlas(raw: unknown): Atlas {
  const payload = raw as AtlasPayload
  if (!payload || typeof payload !== 'object') throw new AtlasFormatError('not an object')
  if (typeof payload.atlas_version !== 'string' || payload.atlas_version === '') {
    throw new AtlasFormatError('no atlas_version')
  }

  const { nodes, extent } = payload
  if (!nodes || !Array.isArray(nodes.xyz)) throw new AtlasFormatError('no nodes.xyz')
  if (typeof extent !== 'number' || !(extent > 0)) {
    throw new AtlasFormatError(`extent must be positive, got ${extent}`)
  }

  const count = nodes.layer?.length ?? 0
  if (nodes.feature?.length !== count || nodes.cluster?.length !== count) {
    throw new AtlasFormatError('layer, feature and cluster lengths disagree')
  }
  if (nodes.xyz.length !== count * 3) {
    throw new AtlasFormatError(`xyz holds ${nodes.xyz.length} values for ${count} nodes`)
  }

  const parsed: AtlasNode[] = new Array(count)
  for (let i = 0; i < count; i++) {
    parsed[i] = {
      layer: nodes.layer[i],
      feature: nodes.feature[i],
      cluster: nodes.cluster[i],
      x: (nodes.xyz[i * 3] / INT16_SCALE) * extent,
      y: (nodes.xyz[i * 3 + 1] / INT16_SCALE) * extent,
      z: (nodes.xyz[i * 3 + 2] / INT16_SCALE) * extent,
    }
  }

  return {
    version: payload.atlas_version,
    // Defaults to the shipping source rather than throwing: an atlas built
    // before the field existed is a label atlas, and refusing to draw it over
    // a missing string would be worse than the small assumption.
    source: payload.source === 'decoder' ? 'decoder' : 'labels',
    note: typeof payload.note === 'string' ? payload.note : '',
    nodes: parsed,
    areas: Array.isArray(payload.areas) ? payload.areas : [],
    sampled: payload.n_sampled ?? count,
    total: payload.n_total ?? count,
    knnPreservation: numberOrNull(payload.knn_preservation),
    knnK: numberOrNull(payload.knn_k),
    explainerAmi: numberOrNull(payload.explainer_ami),
  }
}

/** A recorded measurement, or null. Never 0 standing in for "not measured". */
function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/**
 * What "near" means in this atlas, in one clause.
 *
 * The view must not leave this to be inferred: a label-derived layout maps
 * descriptions of features and a decoder-derived one maps the model's own
 * writing, and a reader who assumes the wrong one draws wrong conclusions.
 */
export function sourceClaim(source: AtlasSource): string {
  return source === 'decoder'
    ? 'nearby dots write similarly into the model\u2019s residual stream'
    : 'nearby dots were described similarly by the model that labelled them'
}

/** Load the idle atlas, or null if it is not there or not readable. */
export async function loadAtlas(url = '/atlas-idle.json'): Promise<Atlas | null> {
  try {
    const response = await fetch(url)
    if (!response.ok) return null
    return parseAtlas(await response.json())
  } catch {
    // Deliberately swallowed: every failure here means the same thing to the
    // caller — there is no atlas — and the brain's job is then to say that
    // rather than to place nodes it does not have positions for.
    return null
  }
}

/** The key `Trace.layout` and `Trace.labels` are both indexed by. */
export function nodeKey(layer: number, feature: number): string {
  return `${layer}/${feature}`
}

/**
 * Positions as a flat Float32Array, ready for a BufferAttribute.
 *
 * One draw call is the point of the whole node layer: 20,000 dim nodes plus
 * the few thousand a trace lights is nothing for the GPU as a single
 * `THREE.Points`, and everything if it were a mesh per feature.
 */
export function positionBuffer(nodes: AtlasNode[]): Float32Array {
  const out = new Float32Array(nodes.length * 3)
  for (let i = 0; i < nodes.length; i++) {
    out[i * 3] = nodes[i].x
    out[i * 3 + 1] = nodes[i].y
    out[i * 3 + 2] = nodes[i].z
  }
  return out
}

/** Which layers this atlas actually covers, ascending. */
export function atlasLayers(atlas: Atlas): number[] {
  return [...new Set(atlas.nodes.map((node) => node.layer))].sort((a, b) => a - b)
}

/** The atlas's areas by cluster id, for joining against a node's `cluster`. */
export function areasByCluster(atlas: Atlas): Map<number, AtlasArea> {
  return new Map(atlas.areas.map((area) => [area.cluster, area]))
}

/**
 * How many of this atlas's areas carry a name.
 *
 * Read by the legend so the view states the measurement outcome rather than
 * showing an empty area list with no explanation.
 */
export function namedAreaCount(atlas: Atlas): number {
  return atlas.areas.filter((area) => area.name !== null).length
}
