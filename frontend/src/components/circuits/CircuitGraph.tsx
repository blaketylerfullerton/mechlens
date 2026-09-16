import { useMemo } from 'react'

import type { CircuitEdge, CircuitNode } from '@/lib/circuits-api'
import { COLUMN, NODE, edgeWidth, layout, nodeOpacity, visibleEdges } from './layout'

/**
 * The attribution graph: token position across, layer up.
 *
 * Drawn as one SVG rather than with a node library. The graph is a fixed grid
 * of up to a few hundred small marks with no dragging, no connecting and no
 * handles — everything a flow library exists to provide — and hand-drawing it
 * keeps the marks at 11px with hairline edges instead of fighting a component
 * kit's shadows, pills and radii back down to this system.
 */

const KIND_LABEL: Record<CircuitNode['kind'], string> = {
  feature: 'transcoder feature',
  error: 'unexplained computation',
  embed: 'input token',
  logit: 'explained token',
}

function describe(node: CircuitNode): string {
  const where = node.kind === 'embed'
    ? `position ${node.position}`
    : node.kind === 'logit'
      ? 'output'
      : `layer ${node.layer}, position ${node.position}`
  const activation = node.activation != null ? `, activation ${node.activation.toFixed(2)}` : ''
  return `${KIND_LABEL[node.kind]} ${node.id} at ${where}, influence ${node.influence.toFixed(4)}${activation}`
}

export function CircuitGraph({
  nodes, edges, nLayers, nPositions, tokens, selected, onSelect,
}: {
  nodes: CircuitNode[]
  edges: CircuitEdge[]
  nLayers: number
  nPositions: number
  tokens: string[]
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  const { placed, bounds } = useMemo(
    () => layout(nodes, nLayers, nPositions), [nodes, nLayers, nPositions],
  )
  const byId = useMemo(() => new Map(placed.map((n) => [n.id, n])), [placed])
  const drawn = useMemo(
    () => visibleEdges(edges, new Set(placed.map((n) => n.id))), [edges, placed],
  )

  const maxInfluence = useMemo(
    () => placed.reduce((m, n) => Math.max(m, n.influence), 0), [placed],
  )
  const maxWeight = useMemo(
    () => drawn.reduce((m, e) => Math.max(m, Math.abs(e.weight)), 0), [drawn],
  )

  // Selecting a node dims everything not directly wired to it. Its own inputs
  // and outputs stay at full strength, which is what "follow this backward"
  // means on a graph that is too dense to read all at once.
  const related = useMemo(() => {
    if (!selected) return null
    const set = new Set<string>([selected])
    for (const edge of drawn) {
      if (edge.source === selected) set.add(edge.target)
      if (edge.target === selected) set.add(edge.source)
    }
    return set
  }, [selected, drawn])

  const labelGutter = 18
  const width = bounds.width
  const height = bounds.height + labelGutter

  return (
    <svg
      role="group"
      aria-label={`Attribution graph, ${placed.length} nodes and ${drawn.length} edges. Token position runs left to right, layer runs bottom to top.`}
      viewBox={`0 0 ${width} ${height}`}
      className="h-full w-full"
      onClick={() => onSelect(null)}
    >
      {/* Column rules: one hairline per token position, so a node's column is
          readable without counting across. */}
      {tokens.map((token, index) => (
        <g key={`col-${index}`}>
          <line
            x1={index * COLUMN + COLUMN / 2} x2={index * COLUMN + COLUMN / 2}
            y1={0} y2={bounds.height}
            stroke="var(--color-border-subtle)" strokeWidth={1}
          />
          <text
            x={index * COLUMN + COLUMN / 2} y={height - 5}
            textAnchor="middle" className="fill-text-tertiary font-mono"
            style={{ fontSize: 10 }}
          >
            {token.length > 11 ? `${token.slice(0, 10)}…` : token}
          </text>
        </g>
      ))}

      {drawn.map((edge, index) => {
        const from = byId.get(edge.source)
        const to = byId.get(edge.target)
        if (!from || !to) return null
        const dim = related ? !(related.has(edge.source) && related.has(edge.target)) : false
        const negative = edge.weight < 0
        return (
          <line
            key={`${edge.source}->${edge.target}-${index}`}
            x1={from.x + NODE / 2} y1={from.y + NODE / 2}
            x2={to.x + NODE / 2} y2={to.y + NODE / 2}
            // Sign is carried by dash as well as colour: a reader who cannot
            // separate the two hues still sees which edges work against the
            // explained token.
            stroke={negative ? 'var(--color-err)' : 'var(--color-fn)'}
            strokeDasharray={negative ? '2 2' : undefined}
            strokeWidth={edgeWidth(edge.weight, maxWeight)}
            opacity={dim ? 0.05 : 0.3}
          />
        )
      })}

      {placed.map((node) => {
        const isSelected = node.id === selected
        const dim = related ? !related.has(node.id) : false
        const size = node.kind === 'logit' ? NODE + 4 : NODE
        // Tier, not category: the explained token and the features are the
        // point and carry the accent; error and input nodes are structure and
        // stay in the greys.
        const fill = node.kind === 'error'
          ? 'var(--color-rule)'
          : node.kind === 'embed'
            ? 'var(--color-text-tertiary)'
            : node.kind === 'logit'
              ? 'var(--color-text-primary)'
              : 'var(--color-fn)'
        return (
          <rect
            key={node.id}
            x={node.x} y={node.y} width={size} height={size} rx={2}
            fill={fill}
            fillOpacity={dim ? 0.12 : nodeOpacity(node.influence, maxInfluence)}
            stroke={isSelected ? 'var(--color-text-primary)' : 'none'}
            strokeWidth={isSelected ? 1.5 : 0}
            className="cursor-pointer"
            tabIndex={0}
            role="button"
            aria-label={describe(node)}
            aria-pressed={isSelected}
            onClick={(event) => { event.stopPropagation(); onSelect(isSelected ? null : node.id) }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelect(isSelected ? null : node.id)
              }
            }}
          />
        )
      })}
    </svg>
  )
}
