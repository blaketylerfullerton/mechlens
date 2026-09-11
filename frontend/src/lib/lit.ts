// Which features a trace lights on the atlas, and what the view has to say
// about the set it is showing.
//
// The atlas says where every feature *is*; a trace says which ones *fired*.
// This module is the join, and it is deliberately the only place that join
// happens, because almost every honesty requirement on the brain view is a
// statement about this set rather than about the picture:
//
//   - a lit node is one the trace recorded an activation for, at a stated
//     scope, and nothing is lit that the trace did not record
//   - a feature the atlas cannot place is carried as unplaced, with its
//     activation, rather than being dropped or given a position
//   - what is shown is a top-k slice of what fired, and both numbers travel
//     together so the slice never reads as the whole
//   - the first sequence position is excluded, always, with a reason
//   - a layer with no data is named as such, not shown as a layer in which
//     nothing fired
//
// Free of runtime imports — pure functions over plain data, the way lens.ts
// is — so it can be exercised under node without a WebGL context.

import type { Trace, NodePosition, PassRecord } from './api-types'
import type { Atlas, AtlasArea } from './atlas'
import { areasByCluster, nodeKey } from './atlas'

/**
 * How much of the trace is lit at once.
 *
 * - `cell`   — one (layer, token). The smallest honest unit: exactly the
 *              features the SAE pass recorded for that position and depth.
 * - `token`  — every layer at one token. What the model was representing at
 *              that point in the sequence.
 * - `trace`  — every layer at every token. Aggregated, and the aggregation is
 *              stated rather than left to be inferred: see `AGGREGATION`.
 */
export type LitScope = 'cell' | 'token' | 'trace'

/**
 * How a feature's activation is combined when one scope covers it more than
 * once. Reported in the interface verbatim.
 *
 * Max rather than sum or mean: a feature that fires hard at one position is a
 * real event, and summing would make a weak feature that recurs at twenty
 * positions outrank it for a reason about sequence length rather than about
 * the model. Mean has the mirror problem — it dilutes exactly the spikes worth
 * looking at.
 */
export const AGGREGATION = 'max' as const

/** The first sequence position, excluded everywhere. See `BOS_REASON`. */
export const BOS_POSITION = 0

/**
 * Why position 0 never lights.
 *
 * Gemma Scope's SAEs produce meaningless activations on the BOS token — a
 * known artifact of the residual stream at that position, not a finding about
 * the prompt. The backend keeps those features per-token but excludes them
 * from every summary statistic; the view does the same, and says so.
 */
export const BOS_REASON =
  'position 0 (BOS) is excluded: Gemma Scope SAEs produce meaningless activations there'

/** One feature the atlas could place, ready to draw. */
export interface LitNode {
  layer: number
  feature: number
  /** The activation at the displayed scope, combined per `AGGREGATION`. */
  activation: number
  x: number
  y: number
  z: number
  cluster: number
}

/**
 * One feature the trace reported that the atlas has no position for.
 *
 * Carried rather than dropped: it fired, and the reader is entitled to know
 * it fired even though there is nowhere honest to draw it. No position is
 * invented for it — that is the whole reason this is a separate list.
 */
export interface UnplacedFeature {
  layer: number
  feature: number
  activation: number
}

export interface LitSet {
  scope: LitScope
  /** Placed features, descending by activation. */
  nodes: LitNode[]
  unplaced: UnplacedFeature[]
  /** Largest activation in the set, for normalising prominence. 0 when empty. */
  maxActivation: number
  /** Distinct features shown — the top-k slice the trace recorded. */
  shown: number
  /**
   * How many features actually fired across the cells in scope, from the
   * trace's own `l0`. Always >= `shown`, usually far larger: the SAE pass
   * keeps the top 16 of a mean ~78. Null when no cell in scope reported one.
   */
  fired: number | null
  /** Layers the trace has feature data for, ascending. */
  layersWithData: number[]
  /**
   * Layers the trace has no feature data for, ascending — a partial SAE run,
   * not layers in which nothing fired. Named so the view can say which.
   */
  layersWithoutData: number[]
  /** True when position 0 was in range and deliberately left out. */
  bosExcluded: boolean
  /** The combination used, when the scope covers a feature more than once. */
  aggregation: typeof AGGREGATION | null
  /**
   * Set when the scope is empty and there is a reason worth stating — a
   * trace with no features, a cell on BOS. Null when the set is non-empty.
   */
  emptyReason: string | null
}

const EMPTY_BASE = {
  nodes: [] as LitNode[],
  unplaced: [] as UnplacedFeature[],
  maxActivation: 0,
  shown: 0,
  fired: null,
  layersWithData: [] as number[],
  layersWithoutData: [] as number[],
  bosExcluded: false,
  aggregation: null,
}

/**
 * The features a trace lights at `scope`, joined against its own layout.
 *
 * Positions come from `trace.layout` and never from the atlas asset: the asset
 * is a 20,000-node *sample* for the idle view, so a feature this trace
 * reported is usually not in it. Reading positions from the sample would light
 * whichever features happened to be sampled, which is a different set than the
 * one that fired.
 */
export function litSet(
  trace: Trace | null,
  scope: LitScope,
  selection: { layer: number; position: number } | null,
): LitSet {
  if (trace === null || trace.steps.length === 0) {
    return { ...EMPTY_BASE, scope, emptyReason: 'no trace' }
  }

  const steps = trace.steps
  const position = Math.min(selection?.position ?? steps.length - 1, steps.length - 1)

  // Which token positions the scope covers, before BOS is taken out.
  const requested =
    scope === 'trace' ? steps.map((_, i) => i) : [position]
  const covered = requested.filter((i) => i !== BOS_POSITION)
  const bosExcluded = requested.includes(BOS_POSITION)

  if (covered.length === 0) {
    // Only reachable at cell/token scope on position 0 itself.
    return { ...EMPTY_BASE, scope, bosExcluded: true, emptyReason: BOS_REASON }
  }

  // Best activation seen per (layer, feature), and the true fired count.
  const best = new Map<string, { layer: number; feature: number; activation: number }>()
  const withData = new Set<number>()
  const allLayers = new Set<number>()
  let fired: number | null = null
  let covers = 0

  for (const index of covered) {
    const step = steps[index]
    if (!step) continue
    for (const state of step.layers) {
      allLayers.add(state.layer)
      if (scope === 'cell' && selection !== null && state.layer !== selection.layer) continue

      if (state.features.length > 0) withData.add(state.layer)
      if (state.l0 !== null) fired = (fired ?? 0) + state.l0

      for (const feature of state.features) {
        const key = nodeKey(state.layer, feature.index)
        const seen = best.get(key)
        // `AGGREGATION` — max, and the only place it is applied.
        if (seen === undefined || feature.activation > seen.activation) {
          best.set(key, {
            layer: state.layer,
            feature: feature.index,
            activation: feature.activation,
          })
        }
      }
      covers += 1
    }
  }

  const nodes: LitNode[] = []
  const unplaced: UnplacedFeature[] = []
  let maxActivation = 0

  for (const [key, entry] of best) {
    if (entry.activation > maxActivation) maxActivation = entry.activation
    const at: NodePosition | undefined = trace.layout[key]
    if (at === undefined) {
      unplaced.push({ layer: entry.layer, feature: entry.feature, activation: entry.activation })
    } else {
      nodes.push({ ...entry, x: at.x, y: at.y, z: at.z, cluster: at.cluster })
    }
  }

  nodes.sort((a, b) => b.activation - a.activation)
  unplaced.sort((a, b) => b.activation - a.activation)

  const layersWithData = [...withData].sort((a, b) => a - b)
  const layersWithoutData = [...allLayers].filter((l) => !withData.has(l)).sort((a, b) => a - b)
  const shown = best.size

  return {
    scope,
    nodes,
    unplaced,
    maxActivation,
    shown,
    fired,
    layersWithData,
    layersWithoutData,
    bosExcluded,
    // Only meaningful where the scope could cover a feature twice.
    aggregation: covers > 1 ? AGGREGATION : null,
    emptyReason: shown === 0 ? emptyReason(trace, layersWithData) : null,
  }
}

function emptyReason(trace: Trace, layersWithData: number[]): string {
  if (layersWithData.length > 0) return 'no features recorded at this scope'
  return trace.passes.some((p) => p.name === 'sae')
    ? 'the SAE pass recorded no features for this trace'
    : 'this trace was run without the SAE pass, so it records no features'
}

/**
 * Prominence for one node, in 0..1.
 *
 * Normalised against the brightest node *in the same set*, so a scope with one
 * strong feature and a scope with twenty read comparably. Deliberately a plain
 * per-node ratio and nothing else: a dim node beside a bright cluster must
 * stay dim, so nothing here may borrow brightness from a neighbour.
 */
export function prominence(activation: number, maxActivation: number): number {
  if (!(maxActivation > 0)) return 0
  const t = activation / maxActivation
  return t < 0 ? 0 : t > 1 ? 1 : t
}

// --------------------------------------------------------------------------
// areas: a cluster of the atlas, as this trace lit it
// --------------------------------------------------------------------------

/**
 * How an area's member activations combine into one prominence.
 *
 * Max, the same rule and for the same reason as `AGGREGATION`: an area is as
 * bright as its brightest member fired, so one hard-firing feature is a real
 * event rather than something averaged away by the hundreds of members that
 * did not fire. Summing would rank a large area above a small one for a
 * reason about cluster size, which is a fact about the atlas and not about
 * the trace.
 *
 * It is deliberately *not* a function of how many members fired — that number
 * is reported beside it instead, because "one feature fired hard" and "forty
 * fired weakly" are different findings and must not collapse into one glow.
 */
export const AREA_AGGREGATION = 'max' as const

/** One atlas area, with what this trace lit inside it. */
export interface LitArea {
  area: AtlasArea
  /** How many of the area's members are lit at the displayed scope. */
  active: number
  /** The combined activation, per `AREA_AGGREGATION`. */
  activation: number
  /** 0..1 against the brightest area in the same set. */
  prominence: number
}

/**
 * The areas this lit set touches, brightest first.
 *
 * Only areas with at least one active member are returned: an area with
 * nothing lit in it is not lit, and returning it with a zero would invite a
 * renderer to draw it as though it had been measured and found dark. A node
 * whose cluster is -1 belongs to no area and contributes to none.
 */
export function litAreas(set: LitSet, atlas: Atlas | null | undefined): LitArea[] {
  if (!atlas) return []
  const byCluster = areasByCluster(atlas)

  const best = new Map<number, { active: number; activation: number }>()
  for (const node of set.nodes) {
    if (node.cluster < 0) continue
    const seen = best.get(node.cluster)
    if (seen === undefined) {
      best.set(node.cluster, { active: 1, activation: node.activation })
    } else {
      seen.active += 1
      // AREA_AGGREGATION.
      if (node.activation > seen.activation) seen.activation = node.activation
    }
  }

  const peak = Math.max(...[...best.values()].map((v) => v.activation), 0)
  const areas: LitArea[] = []
  for (const [cluster, tally] of best) {
    const area = byCluster.get(cluster)
    // A cluster the atlas has no record for cannot be drawn as an area: there
    // is no centroid to put it at and no membership count to state.
    if (area === undefined) continue
    areas.push({
      area,
      active: tally.active,
      activation: tally.activation,
      prominence: prominence(tally.activation, peak),
    })
  }

  areas.sort((a, b) => b.activation - a.activation)
  return areas
}

// --------------------------------------------------------------------------
// filtering the lit set by what its features are called
// --------------------------------------------------------------------------

/**
 * The lit nodes whose labels match `query`, and what had to be left out.
 *
 * Case-insensitive substring, not a ranked search: the query is a filter over
 * a set already on screen, and a relevance order would imply a judgement
 * about which match is better that nothing here can support.
 *
 * `unlabelled` is the count of active features with no explanation at all.
 * They can never match — there is no text to match against — and the view is
 * required to say so rather than let them read as features the query ruled
 * out.
 */
export interface LabelFilter {
  query: string
  nodes: LitNode[]
  /** Active features carrying a label, matched or not. */
  labelled: number
  /** Active features with no label, which no query can match. */
  unlabelled: number
}

export function filterByLabel(
  set: LitSet,
  trace: Trace | null,
  query: string,
): LabelFilter {
  const needle = query.trim().toLowerCase()
  let labelled = 0
  let unlabelled = 0
  const nodes: LitNode[] = []

  for (const node of set.nodes) {
    const label = trace?.labels[nodeKey(node.layer, node.feature)] ?? null
    const text = label?.text?.trim() ?? ''
    if (text === '') {
      unlabelled += 1
      continue
    }
    labelled += 1
    if (needle === '' || text.toLowerCase().includes(needle)) nodes.push(node)
  }

  return { query: needle, nodes, labelled, unlabelled }
}

// --------------------------------------------------------------------------
// what the trace says about the atlas it was drawn against
// --------------------------------------------------------------------------

/** The `layout` pass record, if the trace carries one. */
export function layoutRecord(trace: Trace | null): PassRecord | null {
  return trace?.passes.find((p) => p.name === 'layout') ?? null
}

export type AtlasAgreement =
  /** The trace was placed against the atlas being drawn. */
  | { kind: 'match'; version: string }
  /** Placed against a *different* atlas — positions are not comparable. */
  | { kind: 'mismatch'; traceVersion: string; atlasVersion: string }
  /** The trace carries no positions, and says why. */
  | { kind: 'no-atlas'; reason: string }
  /** No layout pass ran: the trace reports no features to position. */
  | { kind: 'no-layout' }
  /** No atlas loaded in the view yet, so there is nothing to compare against. */
  | { kind: 'no-view-atlas' }

/**
 * Whether the trace's positions correspond to the atlas on screen.
 *
 * Two traces showing one feature must place it identically, or they were drawn
 * against different atlases. When they disagree the view may not present the
 * positions as this atlas's — the areas, the names and the neighbourhoods all
 * belong to a different map, and silently overlaying them is exactly the kind
 * of plausible-looking error the rest of this project refuses.
 */
export function atlasAgreement(trace: Trace | null, atlas: Atlas | null | undefined): AtlasAgreement {
  const record = layoutRecord(trace)
  if (record === null) return { kind: 'no-layout' }

  if (record.params.atlas_available === false) {
    return {
      kind: 'no-atlas',
      reason: 'no atlas was available when this trace ran, so its features have no positions',
    }
  }

  const traceVersion = String(record.params.atlas_version ?? '')
  if (!atlas) return { kind: 'no-view-atlas' }
  if (traceVersion !== '' && traceVersion !== atlas.version) {
    return { kind: 'mismatch', traceVersion, atlasVersion: atlas.version }
  }
  return { kind: 'match', version: atlas.version }
}

/**
 * The layers an atlas covers but the trace has no feature data for.
 *
 * The shipping atlas is a six-layer pilot, so a trace run over all 26 layers
 * has features the atlas cannot place, and a trace pinned to the pilot six has
 * none. Either way the view states its own scope rather than implying the
 * picture covers the whole model.
 */
export function coverageNote(set: LitSet, atlas: Atlas | null | undefined): string | null {
  if (set.layersWithoutData.length === 0) return null
  const missing = set.layersWithoutData
  const list = missing.length > 6 ? `${missing.slice(0, 6).join(', ')}, …` : missing.join(', ')
  void atlas
  return `no feature data for layer${missing.length === 1 ? '' : 's'} ${list} — the SAE pass did not run there`
}
