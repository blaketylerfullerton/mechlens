import type { Feature, LayerState, TokenStep, TopToken } from '@/lib/api-types'

import { formatNumber, visibleToken } from './format'
import { Bar, Facts, Panel, SectionLabel } from './primitives'

/**
 * A ranked distribution: rank, token, the probability drawn to scale, the
 * probability as a number. The accent marks rank 1 and nothing else.
 */
function Distribution({ tokens }: { tokens: TopToken[] }) {
  return (
    <ol className="divide-border-subtle divide-y font-mono text-[12px]">
      {tokens.slice(0, 5).map((token, index) => (
        <li className="flex items-center gap-2 py-1.5" key={token.token_id}>
          <span className="text-text-disabled w-3 shrink-0 text-right tabular-nums">
            {index + 1}
          </span>
          <code className={`min-w-0 flex-1 truncate ${index === 0 ? 'text-fn' : 'text-text-primary'}`}>
            {visibleToken(token.text)}
          </code>
          <Bar accent={index === 0} ratio={token.prob} />
          <span className="text-text-secondary w-12 shrink-0 text-right tabular-nums">
            {(token.prob * 100).toFixed(1)}%
          </span>
        </li>
      ))}
    </ol>
  )
}

function FeatureList({ features }: { features: Feature[] }) {
  const maximum = Math.max(...features.map((feature) => feature.activation), 0)
  return (
    <ol className="divide-border-subtle divide-y font-mono text-[12px]">
      {features.slice(0, 8).map((feature) => (
        <li className="flex items-center gap-2 py-1.5" key={feature.index}>
          <span className="text-text-primary w-16 shrink-0">#{feature.index}</span>
          <Bar ratio={maximum > 0 ? feature.activation / maximum : 0} />
          <span className="text-text-secondary flex-1 text-right tabular-nums">
            {formatNumber(feature.activation)}
          </span>
        </li>
      ))}
    </ol>
  )
}

/** The lens pass can be skipped per layer; say so rather than showing nothing. */
function NoReadout({ running }: { running: boolean }) {
  return (
    <section className="border-border-subtle bg-bg-surface rounded-[2px] border border-dashed p-3">
      <SectionLabel>Layer readout</SectionLabel>
      <p className="text-text-secondary text-[13px] leading-[1.55]">
        {running ? 'Layer predictions are being computed for this position.' : 'No layer predictions are available for this position.'}
      </p>
    </section>
  )
}

/**
 * Everything about the one (layer, token) cell the reader has selected, plus
 * the position's own next-token distribution. The right-hand column.
 */
export function Inspector({ state, step, running = false }: { state: LayerState; step: TokenStep; running?: boolean }) {
  return (
    <aside className="space-y-3">
      <section className="border-fn/30 bg-fn/[0.05] rounded-[2px] border p-3">
        <SectionLabel>Selected state</SectionLabel>
        <Facts
          rows={[
            ['layer', `L${state.layer}`],
            ['token', `${step.step} · ${visibleToken(step.token.text)}`],
            ['residual L2 norm', formatNumber(state.resid_norm)],
            ['sae l0', state.l0 === null ? 'not computed' : formatNumber(state.l0)],
          ]}
        />
      </section>

      {state.logit_lens ? (
        <Panel note={`H ${formatNumber(state.logit_lens.entropy)}`} title="Layer readout">
          <Distribution tokens={state.logit_lens.top_k} />
        </Panel>
      ) : (
        <NoReadout running={running} />
      )}

      {state.features.length > 0 ? (
        <Panel note={`${state.features.length} active`} title="SAE features">
          <FeatureList features={state.features} />
        </Panel>
      ) : null}

      <Panel note={`H ${formatNumber(step.logits.entropy)}`} title="Next-token distribution">
        <Distribution tokens={step.logits.top_k} />
      </Panel>
    </aside>
  )
}
