import type { RunState } from '@/hooks/useTrace'
import type { JobProgress, Trace } from '@/lib/api-types'
import { formatNumber } from './format'

export function TraceHeader({ trace, status, progress }: {
  trace: Trace
  status: RunState
  progress: JobProgress | null
}) {
  const state = status === 'running'
    ? progress?.phase === 'generating' ? 'Generating' : 'Analyzing'
    : status === 'error' ? 'Interrupted' : 'Complete'
  const facts = [
    ['Trace', trace.trace_id],
    ['Device', trace.device],
    ['Precision', trace.dtype],
    ['Tokens captured', String(trace.steps.length)],
    ['Tokens generated', String(trace.n_generated_tokens)],
    ['Layers', String(trace.n_layers)],
    ['Dimensions', String(trace.d_model)],
    ['Capture time', `${formatNumber(trace.elapsed_s)} s`],
  ]

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2">
      <h1 className="text-text-primary text-[13px] font-medium">{trace.model}</h1>
      <span role="status" className="text-text-secondary text-[12px]">{state}</span>
      <details className="group relative text-[12px]">
        <summary className="text-text-tertiary hover:text-text-primary cursor-pointer rounded-xs">Run details</summary>
        <dl className="border-border-strong bg-bg-elevated absolute left-0 top-7 z-30 w-64 space-y-2 rounded-xs border p-3">
          {facts.map(([term, value]) => (
            <div className="flex justify-between gap-3" key={term}>
              <dt className="text-text-secondary">{term}</dt>
              <dd className="text-text-primary break-all text-right font-mono tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  )
}
