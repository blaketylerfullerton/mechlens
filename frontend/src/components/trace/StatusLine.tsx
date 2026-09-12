import type { RunState } from '@/hooks/useTrace'
import { API_BASE_URL } from '@/lib/api-client'

/** The states that have something running behind them, and what to say about it. */
const WORKING_COPY: Partial<Record<RunState, string>> = {
  // No job exists yet in this one — the service is still answering 503.
  warming: 'Loading gemma-2-2b — the first load pulls its weights and is slow.',
  pending: 'Trace queued.',
  running: 'Running the model and capturing every layer.',
}

export function StatusLine({ status, error }: { status: RunState; error: string | null }) {
  if (error) {
    return (
      <div className="border-err/40 bg-err/[0.06] rounded-[2px] border p-3" role="alert">
        <p className="text-err text-[13px] font-medium">The trace service did not return a run.</p>
        <p className="text-text-secondary mt-1 text-[13px] leading-6">
          {error} Start the backend with <span className="text-text-primary font-mono">make dev</span>{' '}
          and check that it is bound to{' '}
          <span className="text-text-primary font-mono">{API_BASE_URL}</span>.
        </p>
      </div>
    )
  }

  const copy = WORKING_COPY[status]
  if (!copy) return null

  return (
    <p className="text-text-secondary text-[13px]" role="status">
      <span aria-hidden="true" className="bg-const mr-2 inline-block size-1.5 rounded-full" />
      {copy}
    </p>
  )
}
