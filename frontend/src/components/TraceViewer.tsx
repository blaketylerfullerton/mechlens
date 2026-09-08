import { useMemo, useState } from 'react'

import type { JobStatus, LayerState, TokenStep, Trace } from '@/lib/api-types'

type Selection = { layer: number; position: number; traceId: string }

type TraceViewerProps = {
  trace: Trace | null
  status: JobStatus | 'idle'
  error: string | null
}

function visibleToken(text: string): string {
  const token = text.replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥')
  return token || '∅'
}

function formatNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function heatColor(value: number, maximum: number): { backgroundColor: string; color: string } {
  const ratio = maximum > 0 ? Math.min(value / maximum, 1) : 0
  return {
    backgroundColor: `hsl(${233 - ratio * 173} 72% ${13 + ratio * 40}%)`,
    color: ratio > 0.56 ? 'white' : 'hsl(220 30% 88%)',
  }
}

function StatusCopy({ status, error }: Pick<TraceViewerProps, 'status' | 'error'>) {
  if (error) return <p className="text-sm text-rose-300">Could not load a trace: {error}</p>
  if (status === 'pending' || status === 'running') {
    return (
      <p className="text-sm text-cyan-100/80" role="status">
        {status === 'pending' ? 'Trace queued…' : 'Running Gemma and capturing every layer…'}
      </p>
    )
  }
  return null
}

function EmptyState({ status, error }: Pick<TraceViewerProps, 'status' | 'error'>) {
  return (
    <main className="flex min-h-full items-center justify-center px-6 pb-36 pt-16">
      <section className="max-w-2xl text-center">
        <p className="mb-4 text-xs font-medium tracking-[0.26em] text-cyan-200/70 uppercase">
          MechLens · Gemma 2 2B
        </p>
        <h1 className="font-heading text-4xl leading-tight text-white sm:text-6xl">
          Watch a model run take shape.
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-base leading-7 text-slate-300 sm:text-lg">
          Send a prompt to inspect its token sequence, next-token predictions, and residual-stream
          magnitude across all transformer layers.
        </p>
        <div className="mx-auto mt-10 grid max-w-lg grid-cols-3 gap-2 text-left text-xs text-slate-300">
          {['tokens', '26 layers', 'residual stream'].map((label) => (
            <div key={label} className="rounded-xl border border-white/10 bg-white/5 p-3">
              {label}
            </div>
          ))}
        </div>
        <div className="mt-7 min-h-5">
          <StatusCopy error={error} status={status} />
        </div>
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
    <div className="overflow-x-auto pb-2">
      <div className="flex min-w-max gap-1">
        {steps.map((step) => {
          const isSelected = step.step === selectedPosition
          const tone = isSelected
            ? 'border-cyan-300 bg-cyan-300/20 text-cyan-50'
            : step.token.source === 'generated'
              ? 'border-violet-300/30 bg-violet-300/10 text-violet-100 hover:bg-violet-300/20'
              : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10'

          return (
            <button
              aria-label={`Select token ${step.step}: ${step.token.text || 'empty token'}`}
              aria-pressed={isSelected}
              className={`rounded-lg border px-2 py-1.5 font-mono text-xs transition ${tone}`}
              key={step.step}
              onClick={() => onSelect(step.step)}
              type="button"
            >
              <span className="mr-1 text-[10px] text-current/60">{step.step}</span>
              {visibleToken(step.token.text)}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function PredictionList({ title, step }: { title: string; step: TokenStep }) {
  return (
    <section className="rounded-xl border border-white/10 bg-slate-950/50 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-xs font-medium tracking-[0.16em] text-slate-400 uppercase">{title}</h3>
        <span className="text-xs text-slate-400">H {formatNumber(step.logits.entropy)}</span>
      </div>
      <ol className="space-y-2">
        {step.logits.top_k.slice(0, 5).map((token, index) => (
          <li className="flex items-center gap-3 text-sm" key={token.token_id}>
            <span className="w-4 text-right font-mono text-xs text-slate-500">{index + 1}</span>
            <code className="min-w-0 flex-1 truncate text-cyan-100">{visibleToken(token.text)}</code>
            <span className="font-mono text-xs text-slate-300">{(token.prob * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </section>
  )
}

function SelectedCell({ state, step }: { state: LayerState; step: TokenStep }) {
  return (
    <div className="space-y-3">
      <section className="rounded-xl border border-cyan-300/20 bg-cyan-300/5 p-4">
        <p className="text-xs font-medium tracking-[0.16em] text-cyan-100/70 uppercase">Selected state</p>
        <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
          <div>
            <p className="text-xs text-slate-400">Layer</p>
            <p className="mt-1 font-mono text-white">{state.layer}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Token</p>
            <p className="mt-1 font-mono text-white">{step.step}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Residual norm</p>
            <p className="mt-1 font-mono text-white">{formatNumber(state.resid_norm)}</p>
          </div>
        </div>
      </section>

      {state.logit_lens ? (
        <section className="rounded-xl border border-white/10 bg-slate-950/50 p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-xs font-medium tracking-[0.16em] text-slate-400 uppercase">Layer readout</h3>
            <span className="text-xs text-slate-400">H {formatNumber(state.logit_lens.entropy)}</span>
          </div>
          <ol className="space-y-2">
            {state.logit_lens.top_k.slice(0, 5).map((token, index) => (
              <li className="flex items-center gap-3 text-sm" key={token.token_id}>
                <span className="w-4 text-right font-mono text-xs text-slate-500">{index + 1}</span>
                <code className="min-w-0 flex-1 truncate text-cyan-100">{visibleToken(token.text)}</code>
                <span className="font-mono text-xs text-slate-300">{(token.prob * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ol>
        </section>
      ) : (
        <section className="rounded-xl border border-dashed border-white/15 bg-white/[0.02] p-4 text-sm leading-6 text-slate-400">
          Layer readouts arrive when the API runs the logit-lens enrichment pass. This connected
          view is showing the capture data currently returned by <code className="text-slate-300">POST /trace</code>.
        </section>
      )}

      {state.features.length > 0 ? (
        <section className="rounded-xl border border-white/10 bg-slate-950/50 p-4">
          <h3 className="mb-3 text-xs font-medium tracking-[0.16em] text-slate-400 uppercase">
            Active SAE features
          </h3>
          <ol className="space-y-2">
            {state.features.slice(0, 8).map((feature) => (
              <li className="flex justify-between gap-3 font-mono text-xs" key={feature.index}>
                <span className="text-cyan-100">#{feature.index}</span>
                <span className="text-slate-300">{formatNumber(feature.activation)}</span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}
    </div>
  )
}

export function TraceViewer({ trace, status, error }: TraceViewerProps) {
  const [selection, setSelection] = useState<Selection | null>(null)

  const maximumResidualNorm = useMemo(() => {
    if (!trace) return 0
    return Math.max(...trace.steps.flatMap((step) => step.layers.map((layer) => layer.resid_norm)))
  }, [trace])

  if (!trace || trace.steps.length === 0) return <EmptyState error={error} status={status} />

  const currentSelection =
    selection?.traceId === trace.trace_id
      ? selection
      : { layer: trace.n_layers - 1, position: trace.steps.length - 1, traceId: trace.trace_id }

  const selectedStep = trace.steps[currentSelection.position]
  const selectedState = selectedStep?.layers[currentSelection.layer]
  const gridColumns = `4.75rem repeat(${trace.steps.length}, minmax(2.5rem, 1fr))`

  if (!selectedStep || !selectedState) {
    return <EmptyState error="The selected trace state is unavailable." status="error" />
  }

  return (
    <main className="min-h-full px-4 pb-36 pt-5 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1600px] space-y-4">
        <header className="flex flex-col justify-between gap-3 rounded-2xl border border-white/10 bg-slate-950/60 p-4 backdrop-blur sm:flex-row sm:items-center">
          <div>
            <p className="text-xs font-medium tracking-[0.2em] text-cyan-200/70 uppercase">Trace explorer</p>
            <h1 className="mt-1 text-lg font-medium text-white">{trace.model}</h1>
          </div>
          <dl className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-400">
            <div><dt className="inline">tokens </dt><dd className="inline font-mono text-slate-200">{trace.steps.length}</dd></div>
            <div><dt className="inline">layers </dt><dd className="inline font-mono text-slate-200">{trace.n_layers}</dd></div>
            <div><dt className="inline">generated </dt><dd className="inline font-mono text-slate-200">{trace.n_generated_tokens}</dd></div>
            <div><dt className="inline">capture </dt><dd className="inline font-mono text-slate-200">{formatNumber(trace.elapsed_s)}s</dd></div>
          </dl>
        </header>

        <section className="rounded-2xl border border-white/10 bg-slate-950/60 p-4 backdrop-blur">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-xs font-medium tracking-[0.16em] text-slate-400 uppercase">Token sequence</h2>
            <span className="text-xs text-slate-500">violet = generated</span>
          </div>
          <TokenStrip
            onSelect={(position) =>
              setSelection((current) => ({
                layer: current?.traceId === trace.trace_id ? current.layer : 0,
                position,
                traceId: trace.trace_id,
              }))
            }
            selectedPosition={currentSelection.position}
            steps={trace.steps}
          />
        </section>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <section className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/60 backdrop-blur">
            <div className="flex flex-col justify-between gap-1 border-b border-white/10 p-4 sm:flex-row sm:items-center">
              <div>
                <h2 className="text-sm font-medium text-white">Residual-stream map</h2>
                <p className="mt-1 text-xs text-slate-400">Colour intensity is the captured L2 norm at each layer and token.</p>
              </div>
              <span className="text-xs text-slate-500">input → output</span>
            </div>
            <div className="max-h-[58vh] overflow-auto p-3">
              <div className="min-w-max" style={{ display: 'grid', gridTemplateColumns: gridColumns }}>
                <div className="sticky left-0 z-10 bg-slate-950 px-2 py-2 text-right text-[10px] font-medium tracking-wide text-slate-500 uppercase">layer</div>
                {trace.steps.map((step) => (
                  <button
                    aria-label={`Select token ${step.step}`}
                    className={`border-b border-white/5 px-1 py-2 font-mono text-[10px] ${step.step === currentSelection.position ? 'bg-cyan-300/10 text-cyan-100' : 'text-slate-500'}`}
                    key={step.step}
                    onClick={() =>
                      setSelection((current) => ({
                        layer: current?.traceId === trace.trace_id ? current.layer : 0,
                        position: step.step,
                        traceId: trace.trace_id,
                      }))
                    }
                    type="button"
                  >
                    {step.step}
                  </button>
                ))}

                {Array.from({ length: trace.n_layers }, (_, layer) => (
                  <div className="contents" key={layer}>
                    <div className="sticky left-0 z-10 border-b border-white/5 bg-slate-950 px-2 py-1 text-right font-mono text-[10px] text-slate-500">L{layer}</div>
                    {trace.steps.map((step) => {
                      const state = step.layers[layer]
                      const isSelected = currentSelection.layer === layer && currentSelection.position === step.step
                      return (
                        <button
                          aria-label={`Layer ${layer}, token ${step.step}, residual norm ${formatNumber(state.resid_norm)}`}
                          aria-pressed={isSelected}
                          className={`m-px min-h-8 rounded-sm border transition hover:scale-110 hover:border-white/80 focus-visible:z-20 focus-visible:outline-2 focus-visible:outline-cyan-200 ${isSelected ? 'z-10 border-cyan-100 ring-2 ring-cyan-300/70' : 'border-transparent'}`}
                          key={`${layer}-${step.step}`}
                          onClick={() => setSelection({ layer, position: step.step, traceId: trace.trace_id })}
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

          <aside className="space-y-4">
            <SelectedCell state={selectedState} step={selectedStep} />
            <PredictionList step={selectedStep} title="Actual next-token distribution" />
          </aside>
        </div>

        <p className="px-1 text-xs leading-5 text-slate-500">
          This view shows observed model states. Residual magnitude is not a semantic score or a causal explanation.
        </p>
      </div>
    </main>
  )
}
