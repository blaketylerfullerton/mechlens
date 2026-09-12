import type { Trace } from '@/lib/api-types'

import { formatNumber } from './format'

/** Identity of the run, then its shape, as a definition list rather than cards. */
export function TraceHeader({ trace }: { trace: Trace }) {
  const facts: [string, string][] = [
    ['tokens', String(trace.steps.length)],
    ['layers', String(trace.n_layers)],
    ['d_model', String(trace.d_model)],
    ['generated', String(trace.n_generated_tokens)],
    ['capture', `${formatNumber(trace.elapsed_s)}s`],
  ]

  return (
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
        {facts.map(([term, value]) => (
          <div key={term}>
            <dt className="inline">{term} </dt>
            <dd className="text-text-primary inline font-mono tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </header>
  )
}
