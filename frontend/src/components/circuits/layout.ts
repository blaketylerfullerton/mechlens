import type { CircuitEdge, CircuitNode } from '@/lib/circuits-api'

/**
 * Where a node sits: token position across, layer up.
 *
 * The two axes are the model's own coordinates, so the picture is readable as
 * computation rather than as an abstract network — inputs along the bottom,
 * the explained token at the top, depth increasing upward.
 *
 * This is a *placement*, not a claim about execution order. Several features in
 * one cell fired at the same layer and position; spreading them sideways is a
 * way to see them, not a sequence. Nothing here implies one chain of thought.
 */

export const COLUMN = 104
export const ROW = 22
export const NODE = 11
/** Preferred sideways step for co-located nodes. Tightened when a cell is
 *  crowded so a busy layer cannot spill into its neighbours' columns. */
const FAN = 13

export interface PlacedNode extends CircuitNode {
  x: number
  y: number
}

export interface Bounds {
  width: number
  height: number
}

/**
 * Rows, top to bottom: the logit, then layers n-1 down to 0, then the embeds.
 * Embeds get their own row below layer 0 because an input token is not a layer;
 * drawing it inside one would imply it was computed there.
 */
export function rowFor(node: CircuitNode, nLayers: number): number {
  if (node.kind === 'logit') return 0
  if (node.kind === 'embed') return nLayers + 1
  return nLayers - (node.layer ?? 0)
}

export function layout(
  nodes: CircuitNode[],
  nLayers: number,
  nPositions: number,
): { placed: PlacedNode[]; bounds: Bounds } {
  // Group by cell first so co-located nodes can be fanned out deterministically
  // — the same graph must lay out identically every time it is opened.
  const cells = new Map<string, CircuitNode[]>()
  for (const node of nodes) {
    const row = rowFor(node, nLayers)
    const column = node.kind === 'logit' ? nPositions - 1 : (node.position ?? 0)
    const key = `${row}:${column}`
    const bucket = cells.get(key)
    if (bucket) bucket.push(node)
    else cells.set(key, [node])
  }

  const placed: PlacedNode[] = []
  for (const [key, bucket] of cells) {
    const [row, column] = key.split(':').map(Number)
    // Strongest first, and centred on the column, so the most influential node
    // in a cell lands closest to the column's true position.
    const ordered = [...bucket].sort((a, b) => b.influence - a.influence)
    // With hundreds of features retained, one cell can hold dozens of nodes. A
    // fixed step would run them out of the column and overlap the next one, so
    // the step tightens to whatever fits and the column stays the truth about
    // which token a node belongs to.
    const fan = ordered.length > 1
      ? Math.min(FAN, (COLUMN - NODE) / (ordered.length - 1))
      : 0
    const spread = (ordered.length - 1) * fan
    ordered.forEach((node, index) => {
      placed.push({
        ...node,
        x: column * COLUMN + COLUMN / 2 - NODE / 2 - spread / 2 + index * fan,
        y: row * ROW,
      })
    })
  }

  return {
    placed,
    bounds: { width: nPositions * COLUMN, height: (nLayers + 2) * ROW + NODE },
  }
}

/** Drop edges whose endpoints are not both visible, so nothing dangles. */
export function visibleEdges(edges: CircuitEdge[], visible: Set<string>): CircuitEdge[] {
  return edges.filter((edge) => visible.has(edge.source) && visible.has(edge.target))
}

/**
 * Line width for an edge, in px, from its magnitude against the strongest edge
 * on screen. Floored so a weak edge still draws rather than disappearing, and
 * capped so one dominant edge cannot swamp the picture.
 */
export function edgeWidth(weight: number, max: number): number {
  if (!(max > 0)) return 1
  return 0.6 + Math.min(Math.abs(weight) / max, 1) * 2.2
}

/**
 * Opacity for a node from its influence, relative to the strongest on screen.
 * A floor keeps a retained-but-weak node visible: it was kept for a reason and
 * fading it to nothing would hide a fact the pruning decided to keep.
 */
export function nodeOpacity(influence: number, max: number): number {
  if (!(max > 0)) return 1
  return 0.42 + Math.min(influence / max, 1) * 0.58
}
