import type { Trace } from '@/lib/api-types'
import { mostInteresting } from '@/lib/highlights'
import { visibleToken } from './format'

/**
 * The one place the app tells the reader where to look before they ask.
 * Three findings computed from the trace, each a link that takes the shared
 * selection to the cell it describes. Anything the trace cannot back —
 * an undecided token, an unlabelled feature, a runner-up — is omitted rather
 * than padded into a claim.
 */
export function Highlights({ trace, onSelect }: {
  trace: Trace
  onSelect: (layer: number, position: number) => void
}) {
  const highlight = mostInteresting(trace)
  if (!highlight || (highlight.decisionLayer === null && highlight.concepts.length === 0)) {
    return null
  }

  const token = visibleToken(highlight.text)
  return (
    <nav aria-label="Where to look first" className="flex shrink-0 flex-wrap items-baseline gap-x-4 gap-y-1 text-[12px]">
      <span className="text-text-tertiary">Start here:</span>
      {highlight.decisionLayer !== null ? (
        <button type="button" onClick={() => onSelect(highlight.decisionLayer!, highlight.position)}
          className="text-fn rounded-xs hover:bg-fn/10 px-1 py-0.5 text-left">
          “{token}” — decided at layer {highlight.decisionLayer} of {trace.n_layers}
        </button>
      ) : null}
      {highlight.concepts.length > 0 ? (
        <button type="button" onClick={() => onSelect(highlight.concepts[0].layer, highlight.position)}
          className="text-fn rounded-xs hover:bg-fn/10 px-1 py-0.5 text-left">
          Noticed: {highlight.concepts.map((concept) => `“${concept.label}”`).join(', ')}
        </button>
      ) : null}
      {highlight.alternative ? (
        <button type="button" onClick={() => onSelect(highlight.decisionLayer ?? trace.n_layers - 1, highlight.position)}
          className="text-fn rounded-xs hover:bg-fn/10 px-1 py-0.5 text-left">
          Considered instead: “{visibleToken(highlight.alternative.text)}” {(highlight.alternative.prob * 100).toFixed(1)}%
        </button>
      ) : null}
    </nav>
  )
}
