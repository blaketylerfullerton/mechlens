import type { Trace } from '@/lib/api-types'

import { formatNumber } from './format'

/**
 * Identity and shape of the run, on one line above the stage.
 *
 * This used to be a page header — an uppercase eyebrow, a 22px `<h1>` and a
 * five-term definition list — living in the right-hand column. In a 26rem
 * column it was a page title for a page that was not there, and it pushed the
 * one thing that column exists for below the fold. A trace's identity is a
 * fact about what is in the frame, so it sits on the frame, at the size a fact
 * gets.
 */
export function TraceHeader({ trace }: { trace: Trace }) {
  const facts: [string, string][] = [
    ['tok', String(trace.steps.length)],
    ['gen', String(trace.n_generated_tokens)],
    ['layers', String(trace.n_layers)],
    ['d_model', String(trace.d_model)],
    ['capture', `${formatNumber(trace.elapsed_s)}s`],
  ]

  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-x-4 gap-y-1">
      <h1 className="text-text-primary text-[13px] font-medium">{trace.model}</h1>
      <p className="text-text-disabled truncate font-mono text-[11px]">
        {trace.trace_id} · {trace.device} · {trace.dtype}
      </p>
      <dl className="text-text-tertiary flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
        {facts.map(([term, value]) => (
          <div key={term}>
            <dt className="inline">{term} </dt>
            <dd className="text-text-secondary inline font-mono tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
