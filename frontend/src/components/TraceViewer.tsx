import { useMemo, type ReactNode } from 'react'

import {
  CodeBlock,
  CodeBlockCopyButton,
  CodeBlockFilename,
  CodeBlockHeader,
  CodeBlockTitle,
} from '@/components/ai-elements/code-block'
import type { RunState } from '@/hooks/useTrace'
import { API_BASE_URL } from '@/lib/api-client'
import type { Feature, LayerState, TokenStep, TopToken, Trace } from '@/lib/api-types'

type TraceViewerProps = {
  trace: Trace | null
  status: RunState
  error: string | null
  /**
   * The shared selection, resolved by `App` — including the default for a
   * newly arrived trace. Null only when there is no trace to select in, which
   * is also when this component renders its empty state.
   */
  selection: { layer: number; position: number } | null
  onSelectCell: (layer: number, position: number) => void
  onSelectPosition: (position: number) => void
  /**
   * The prompt composer, rendered inside the empty state directly under the
   * facts block. Passed in rather than imported so this component keeps
   * knowing nothing about how a run is started — and so it goes away with the
   * empty state the moment a trace exists.
   */
  composer?: ReactNode
}

function visibleToken(text: string): string {
  const token = text.replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥')
  return token || '∅'
}

function formatNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

/**
 * The residual-magnitude ramp: one hue, `syntax-func`, from the code surface
 * up to the accent itself.
 *
 * Single-hue on purpose. A rainbow ramp invents colours the palette does not
 * contain and implies category boundaries where there is only a continuous L2
 * norm. The exact endpoints are printed beside the map, because a colour a
 * reader cannot convert back to a number is decoration.
 */
function heatColor(value: number, maximum: number): { backgroundColor: string } {
  const ratio = maximum > 0 ? Math.min(value / maximum, 1) : 0
  return {
    backgroundColor: `hsl(220.8 ${(10 + ratio * 90).toFixed(1)}% ${(9 + ratio * 66.5).toFixed(1)}%)`,
  }
}

/**
 * A quantity drawn to scale in the character cell: a 1px rule inside a
 * `ch`-measured track.
 *
 * Not block glyphs (`█▉▊`) — every partial cell caps differently and the run
 * stacks into a chunky rectangle that reads as a rendering artefact. The width
 * is computed from the real value with a small floor, so a near-zero row still
 * prints a mark instead of vanishing.
 */
function Bar({ accent = false, ratio }: { accent?: boolean; ratio: number }) {
  const width = Math.max(Math.min(ratio, 1), 0.02) * 100
  return (
    <span aria-hidden="true" className="inline-block w-[8ch] shrink-0 align-middle">
      <span
        className={`block h-px ${accent ? 'bg-fn' : 'bg-rule'}`}
        style={{ width: `${width.toFixed(1)}%` }}
      />
    </span>
  )
}

/** Term left, value right, hairline between. The shape a card grid replaced. */
function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="divide-border-subtle divide-y">
      {rows.map(([term, value]) => (
        <div className="flex items-baseline justify-between gap-4 py-1.5" key={term}>
          <dt className="text-text-tertiary text-[12px]">{term}</dt>
          <dd className="text-text-primary font-mono text-[12px] tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

const REQUEST_SNIPPET = (base: string) => `# The request this page makes. \`passes\` is what paints the brain: without
# "lens" you get the capture and no per-layer readouts to colour it with.
curl -sS ${base}/trace \\
  -H 'content-type: application/json' \\
  -d '{"prompt":"The capital of France is","max_tokens":20,"passes":["lens"]}'

# -> {"job_id":"..."}  — the job runs in the background; poll it until
#    "status" is "done", then read "trace.steps[].layers[].logit_lens".
curl -sS ${base}/trace/"$JOB_ID"`

function StatusLine({ status, error }: Pick<TraceViewerProps, 'status' | 'error'>) {
  if (error) {
    return (
      <div className="border-err/40 bg-err/[0.06] rounded-[10px] border p-3" role="alert">
        <p className="text-err text-[13px] font-medium">The trace service did not return a run.</p>
        <p className="text-text-secondary mt-1 text-[13px] leading-6">
          {error} Start the backend with <span className="text-text-primary font-mono">make dev</span>{' '}
          and check that it is bound to{' '}
          <span className="text-text-primary font-mono">{API_BASE_URL}</span>.
        </p>
      </div>
    )
  }

  if (status === 'warming' || status === 'pending' || status === 'running') {
    const copy = {
      // No job exists yet in this one — the service is still answering 503.
      warming: 'Loading gemma-2-2b — the first load pulls its weights and is slow.',
      pending: 'Trace queued.',
      running: 'Running the model and capturing every layer.',
    }[status]
    return (
      <p className="text-text-secondary text-[13px]" role="status">
        <span aria-hidden="true" className="bg-const mr-2 inline-block size-1.5 rounded-full" />
        {copy}
      </p>
    )
  }

  return null
}

/**
 * The shape of the grid that is coming, at the size it will be, so nothing
 * shifts when it arrives. Hairline blocks, not a spinner over an empty region.
 */
function GridSkeleton() {
  return (
    <div aria-hidden="true" className="space-y-1 opacity-40">
      {Array.from({ length: 8 }, (_, row) => (
        <div className="flex gap-1" key={row}>
          <div className="bg-bg-surface h-4 w-10 rounded-xs" />
          {Array.from({ length: 12 }, (_, cell) => (
            <div className="bg-bg-surface h-4 flex-1 rounded-xs" key={cell} />
          ))}
        </div>
      ))}
    </div>
  )
}

/**
 * What is not here, why, and the command that changes it — in a block that can
 * actually be copied and run. No illustration, no fabricated dashboard, and no
 * claim the page cannot currently back with data.
 */
function EmptyState({
  status,
  error,
  composer,
}: Pick<TraceViewerProps, 'status' | 'error' | 'composer'>) {
  const isWorking = status === 'warming' || status === 'pending' || status === 'running'

  return (
    <main className="enter flex min-h-full items-start justify-center py-6">
      <section className="w-full max-w-2xl">
        <p className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
          mechlens · gemma-2-2b · 26 layers
        </p>
        <h1 className="text-text-primary mt-3 text-[28px] leading-[1.1] font-semibold sm:text-[36px]">
          No trace loaded.
        </h1>
        <p className="text-text-secondary mt-4 max-w-[70ch] text-[15px] leading-[1.6]">
          Send a prompt below and the service runs it through gemma-2-2b, keeping the residual
          stream at every layer and decoding each layer with the logit lens. What comes back is the
          token sequence, the next-token distribution at each step, and the L2 norm of the residual
          at all 26 layers × every position.
        </p>

        <div className="mt-8">
          <Facts
            rows={[
              ['model', 'gemma-2-2b'],
              ['layers captured', '26'],
              ['enrichment passes', 'lens'],
              ['api', API_BASE_URL],
            ]}
          />
        </div>

        {composer ? <div className="mt-6">{composer}</div> : null}

        {/* Reserved either way, so the block below never pushes the page when a
            status arrives. */}
        <div className="mt-6 min-h-6">
          <StatusLine error={error} status={status} />
        </div>

        {isWorking ? (
          <div className="mt-6">
            <GridSkeleton />
          </div>
        ) : null}
      </section>
    </main>
  )
}

function TokenStrip({
  steps,
  selectedPosition,
  onSelect,
}: {
  steps: TokenStep[]
  selectedPosition: number
  onSelect: (position: number) => void
}) {
  return (
    <div className="mask-fade-r overflow-x-auto pb-2 [-webkit-overflow-scrolling:touch]">
      <div className="flex min-w-max gap-1">
        {steps.map((step) => {
          const isSelected = step.step === selectedPosition
          const tone = isSelected
            ? 'border-fn bg-fn/[0.08] text-text-primary'
            : step.token.source === 'generated'
              ? 'border-kw/30 text-kw hover:border-kw/60'
              : 'border-border-subtle text-text-secondary hover:border-border-strong'

          return (
            <button
              aria-label={`Select token ${step.step}: ${step.token.text || 'empty token'}`}
              aria-pressed={isSelected}
              className={`rounded-sm border px-2 py-1.5 font-mono text-[12px] transition-colors duration-150 ${tone}`}
              key={step.step}
              onClick={() => onSelect(step.step)}
              type="button"
            >
              <span className="text-text-disabled mr-1.5 text-[10px] tabular-nums">
                {step.step}
              </span>
              {visibleToken(step.token.text)}
            </button>
          )
        })}
      </div>
    </div>
  )
}

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

function Panel({
  children,
  note,
  title,
}: {
  children: React.ReactNode
  note?: string
  title: string
}) {
  return (
    <section className="border-border-subtle bg-bg-surface rounded-[12px] border p-3">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
          {title}
        </h3>
        {note ? (
          <span className="text-text-tertiary font-mono text-[11px] tabular-nums">{note}</span>
        ) : null}
      </div>
      {children}
    </section>
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

function SelectedCell({ state, step }: { state: LayerState; step: TokenStep }) {
  return (
    <div className="space-y-3">
      <section className="border-fn/30 bg-fn/[0.05] rounded-[12px] border p-3">
        <h3 className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
          Selected state
        </h3>
        <div className="mt-2">
          <Facts
            rows={[
              ['layer', `L${state.layer}`],
              ['token', `${step.step} · ${visibleToken(step.token.text)}`],
              ['residual L2 norm', formatNumber(state.resid_norm)],
              ['sae l0', state.l0 === null ? 'not computed' : formatNumber(state.l0)],
            ]}
          />
        </div>
      </section>

      {state.logit_lens ? (
        <Panel note={`H ${formatNumber(state.logit_lens.entropy)}`} title="Layer readout">
          <Distribution tokens={state.logit_lens.top_k} />
        </Panel>
      ) : (
        <section className="border-border-subtle bg-bg-surface rounded-[12px] border border-dashed p-3">
          <h3 className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
            Layer readout
          </h3>
          <p className="text-text-secondary mt-2 text-[13px] leading-[1.55]">
            This layer has no logit-lens readout, so there is nothing to decode here. The lens pass
            did not run for it — request it with{' '}
            <code className="border-border-subtle bg-bg-elevated text-text-primary rounded-xs border px-1 py-0.5 font-mono text-[12px]">
              "passes": ["lens"]
            </code>
            .
          </p>
        </section>
      )}

      {state.features.length > 0 ? (
        <Panel note={`${state.features.length} active`} title="SAE features">
          <FeatureList features={state.features} />
        </Panel>
      ) : null}
    </div>
  )
}

export function TraceViewer({
  trace,
  status,
  error,
  selection,
  onSelectCell,
  onSelectPosition,
  composer,
}: TraceViewerProps) {
  const maximumResidualNorm = useMemo(() => {
    if (!trace) return 0
    return Math.max(...trace.steps.flatMap((step) => step.layers.map((layer) => layer.resid_norm)))
  }, [trace])

  const minimumResidualNorm = useMemo(() => {
    if (!trace) return 0
    return Math.min(...trace.steps.flatMap((step) => step.layers.map((layer) => layer.resid_norm)))
  }, [trace])

  if (!trace || trace.steps.length === 0 || selection === null) {
    return <EmptyState composer={composer} error={error} status={status} />
  }

  const currentSelection = selection
  const selectedStep = trace.steps[currentSelection.position]
  const selectedState = selectedStep?.layers[currentSelection.layer]
  const gridColumns = `4rem repeat(${trace.steps.length}, minmax(2rem, 1fr))`

  if (!selectedStep || !selectedState) {
    return (
      <EmptyState
        composer={composer}
        error="The selected trace state is unavailable."
        status="error"
      />
    )
  }

  return (
    <main className="enter min-h-full">
      <div className="space-y-4">
        <header className="border-border-subtle flex flex-col justify-between gap-3 border-b pb-4 sm:flex-row sm:items-end">
          <div>
            <p className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
              Trace explorer
            </p>
            <h1 className="text-text-primary mt-1 text-[22px] leading-[1.2] font-semibold">
              {trace.model}
            </h1>
            <p className="text-text-tertiary mt-1 font-mono text-[11px]">
              {trace.trace_id} · {trace.device} · {trace.dtype}
            </p>
          </div>
          <dl className="text-text-tertiary flex flex-wrap gap-x-5 gap-y-1 text-[12px]">
            {(
              [
                ['tokens', String(trace.steps.length)],
                ['layers', String(trace.n_layers)],
                ['d_model', String(trace.d_model)],
                ['generated', String(trace.n_generated_tokens)],
                ['capture', `${formatNumber(trace.elapsed_s)}s`],
              ] as [string, string][]
            ).map(([term, value]) => (
              <div key={term}>
                <dt className="inline">{term} </dt>
                <dd className="text-text-primary inline font-mono tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        </header>

        <section>
          <div className="mb-2 flex items-baseline justify-between gap-3">
            <h2 className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
              Token sequence
            </h2>
            <span className="text-text-tertiary flex items-center gap-1.5 text-[11px]">
              <span aria-hidden="true" className="bg-kw size-1.5 rounded-full" />
              generated
            </span>
          </div>
          <TokenStrip
            onSelect={onSelectPosition}
            selectedPosition={currentSelection.position}
            steps={trace.steps}
          />
        </section>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
          {/* The same frame the brain gets: outer 16, inner 12, 4px gap. It
              wraps what the reader looks into and nothing else. */}
          <div className="border-border-subtle min-w-0 rounded-[16px] border bg-[#0D0E11] p-1">
            <section className="border-border-subtle bg-bg-surface overflow-hidden rounded-[12px] border">
              <div className="border-border-subtle flex flex-col justify-between gap-2 border-b p-3 sm:flex-row sm:items-center">
                <div>
                  <h2 className="text-text-primary text-[13px] font-medium">Residual-stream map</h2>
                  <p className="text-text-tertiary mt-1 text-[12px]">
                    L2 norm of the residual at each layer and token, as captured.
                  </p>
                </div>
                {/* The ramp with its endpoints as numbers: a colour nobody can
                    convert back to a value is decoration. */}
                <div className="text-text-tertiary flex shrink-0 items-center gap-2 font-mono text-[11px] tabular-nums">
                  <span>{formatNumber(minimumResidualNorm)}</span>
                  <span aria-hidden="true" className="flex h-2.5 w-24">
                    {Array.from({ length: 12 }, (_, step) => (
                      <span
                        className="flex-1"
                        key={step}
                        style={heatColor(step / 11, 1)}
                      />
                    ))}
                  </span>
                  <span>{formatNumber(maximumResidualNorm)}</span>
                </div>
              </div>
              <div className="mask-fade-b max-h-[58vh] overflow-auto p-3">
                <div className="min-w-max" style={{ display: 'grid', gridTemplateColumns: gridColumns }}>
                  <div className="bg-bg-surface text-text-tertiary sticky left-0 z-10 px-2 py-2 text-right text-[10px] font-medium tracking-[0.04em] uppercase">
                    layer
                  </div>
                  {trace.steps.map((step) => (
                    <button
                      aria-label={`Select token ${step.step}`}
                      className={`border-border-subtle border-b px-1 py-2 font-mono text-[10px] tabular-nums transition-colors duration-150 ${
                        step.step === currentSelection.position
                          ? 'text-fn'
                          : 'text-text-disabled hover:text-text-secondary'
                      }`}
                      key={step.step}
                      onClick={() => onSelectPosition(step.step)}
                      type="button"
                    >
                      {step.step}
                    </button>
                  ))}

                  {Array.from({ length: trace.n_layers }, (_, layer) => (
                    <div className="contents" key={layer}>
                      <div className="bg-bg-surface border-border-subtle text-text-disabled sticky left-0 z-10 border-b px-2 py-1 text-right font-mono text-[10px] tabular-nums">
                        L{layer}
                      </div>
                      {trace.steps.map((step) => {
                        const state = step.layers[layer]
                        const isSelected =
                          currentSelection.layer === layer &&
                          currentSelection.position === step.step
                        return (
                          <button
                            aria-label={`Layer ${layer}, token ${step.step}, residual norm ${formatNumber(state.resid_norm)}`}
                            aria-pressed={isSelected}
                            className={`m-px min-h-7 rounded-xs border transition-colors duration-150 ${
                              isSelected
                                ? 'border-text-primary z-10'
                                : 'hover:border-border-strong border-transparent'
                            }`}
                            key={`${layer}-${step.step}`}
                            onClick={() => onSelectCell(layer, step.step)}
                            style={heatColor(state.resid_norm, maximumResidualNorm)}
                            type="button"
                          >
                            <span className="sr-only">{formatNumber(state.resid_norm)}</span>
                          </button>
                        )
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </div>

          <aside className="space-y-3">
            <SelectedCell state={selectedState} step={selectedStep} />
            <Panel
              note={`H ${formatNumber(selectedStep.logits.entropy)}`}
              title="Next-token distribution"
            >
              <Distribution tokens={selectedStep.logits.top_k} />
            </Panel>
          </aside>
        </div>

        <p className="text-text-tertiary max-w-[70ch] text-[12px] leading-[1.5]">
          This view shows observed model states. Residual magnitude is not a semantic score or a
          causal explanation.
        </p>
      </div>
    </main>
  )
}
