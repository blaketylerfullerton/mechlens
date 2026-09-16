import { useEffect, useState } from 'react'

import { getFeatureEvidence, type CircuitNode, type FeatureEvidence as Evidence } from '@/lib/circuits-api'
import { Bar, Facts, SectionLabel } from '@/components/trace/primitives'

/**
 * What is known about the selected node.
 *
 * For a transcoder feature that is: where it fired, how hard, and the contexts
 * it fires hardest on. It is deliberately *not* a name. The transcoder artifact
 * ships activation examples and no written label, so the heading is the feature
 * id and the unlabeled state is stated rather than papered over with a label
 * borrowed from the residual SAEs, which index a different space entirely.
 */

function ExampleLine({ tokens, activations, peak }: { tokens: string[]; activations: number[]; peak: number }) {
  const max = Math.max(...activations, 1e-6)
  // A window around the peak: the stored contexts run long, and the tokens far
  // from the peak carry no activation worth the horizontal space.
  const from = Math.max(0, peak - 12)
  const to = Math.min(tokens.length, peak + 6)
  return (
    <p className="font-mono text-[11px] leading-5 [overflow-wrap:anywhere]">
      {from > 0 ? <span className="text-text-disabled">… </span> : null}
      {tokens.slice(from, to).map((token, index) => {
        const at = from + index
        const strength = activations[at] / max
        return (
          <span
            key={at}
            // Intensity is the activation, so the highlight is a measurement
            // rather than a marker someone placed by hand.
            style={strength > 0.02 ? { backgroundColor: `rgb(130 170 255 / ${(strength * 0.32).toFixed(3)})` } : undefined}
            className={at === peak ? 'text-fn' : 'text-text-secondary'}
          >
            {token}
          </span>
        )
      })}
      {to < tokens.length ? <span className="text-text-disabled"> …</span> : null}
    </p>
  )
}

export function FeatureEvidencePanel({ node }: { node: CircuitNode }) {
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (node.kind !== 'feature' || node.layer == null || node.feature_idx == null) {
      setEvidence(null)
      return
    }
    let current = true
    setLoading(true)
    setError(null)
    getFeatureEvidence(node.layer, node.feature_idx)
      .then((data) => { if (current) setEvidence(data) })
      .catch((err) => { if (current) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [node])

  const facts: [string, string][] = [
    ['Kind', node.kind],
    ...(node.layer != null ? ([['Layer', `L${node.layer}`]] as [string, string][]) : []),
    ...(node.position != null ? ([['Token position', `#${node.position}`]] as [string, string][]) : []),
    ...(node.activation != null ? ([['Activation here', node.activation.toFixed(2)]] as [string, string][]) : []),
    ['Influence on target', node.influence.toFixed(4)],
  ]

  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-mono text-[13px]">{node.id}</h3>
        {node.kind === 'feature' ? (
          <p className="text-text-tertiary mt-1 text-[11px]">
            Unlabeled — this dictionary ships activation examples, not names.
          </p>
        ) : node.kind === 'error' ? (
          <p className="text-text-tertiary mt-1 text-[11px]">
            Computation the transcoders do not reconstruct at this layer and position. Kept in the
            graph because leaving it out would make the explanation look complete.
          </p>
        ) : node.kind === 'embed' ? (
          <p className="text-text-tertiary mt-1 text-[11px]">An input token, not a computed feature.</p>
        ) : (
          <p className="text-text-tertiary mt-1 text-[11px]">The token being explained.</p>
        )}
      </div>

      <Facts rows={facts} />

      {node.kind !== 'feature' ? null : loading ? (
        <div aria-hidden="true" className="space-y-2 opacity-40">
          {Array.from({ length: 4 }, (_, row) => (
            <div className="bg-bg-surface h-5 rounded-xs" key={row} />
          ))}
        </div>
      ) : error ? (
        <div>
          <SectionLabel>Evidence</SectionLabel>
          <p className="text-text-secondary text-[12px]">
            Feature examples are unavailable ({error}). The graph above is unaffected — it is saved
            and needs no network to redraw.
          </p>
        </div>
      ) : evidence ? (
        <>
          <div>
            <SectionLabel note={`${evidence.n_examples_available} stored`}>
              Fires hardest on
            </SectionLabel>
            <div className="space-y-2">
              {evidence.examples.map((example, index) => (
                <div key={index}>
                  <div className="text-text-tertiary flex items-baseline gap-2 font-mono text-[10px] tabular-nums">
                    <Bar accent ratio={example.activation / (evidence.act_max || 1)} />
                    <span>{example.activation.toFixed(1)}</span>
                  </div>
                  <ExampleLine
                    tokens={example.tokens}
                    activations={example.activations}
                    peak={example.peak_index}
                  />
                </div>
              ))}
            </div>
            <p className="text-text-tertiary mt-2 text-[11px]">
              Activation range {evidence.act_min?.toFixed(2)} – {evidence.act_max?.toFixed(2)} across
              the stored corpus.
            </p>
          </div>

          {evidence.top_logits?.length ? (
            <div>
              <SectionLabel>Writes toward</SectionLabel>
              <p className="text-text-secondary font-mono text-[11px] [overflow-wrap:anywhere]">
                {evidence.top_logits.slice(0, 6).map((t) => JSON.stringify(t)).join('  ')}
              </p>
              <p className="text-text-tertiary mt-1 text-[11px]">
                Unembed directions, not a description. Frequently noisy — read the examples above
                first.
              </p>
            </div>
          ) : null}

          <p className="text-text-tertiary border-border-subtle border-t pt-2 font-mono text-[10px] [overflow-wrap:anywhere]">
            {evidence.transcoder_set}
            {evidence.transcoder_revision ? `@${evidence.transcoder_revision.slice(0, 8)}` : ' (revision unknown)'}
          </p>
        </>
      ) : null}
    </div>
  )
}
