import type { RunState } from '@/hooks/useTrace'
import { useErrorToast } from '@/hooks/useErrorToast'
import { API_BASE_URL } from '@/lib/api-client'

/** The states that have something running behind them, and what to say about it. */
const WORKING_COPY: Partial<Record<RunState, string>> = {
  // No job exists yet in this one — the service is still answering 503.
  warming: 'Loading gemma-2-2b — the first load pulls its weights and is slow.',
  pending: 'Trace queued.',
  running: 'Running the model and capturing every layer.',
}

export function StatusLine({ status, error }: { status: RunState; error: string | null }) {
  useErrorToast(
    error &&
      `The trace service did not return a run. ${error} Start the backend with \`make dev\` and check that it is bound to ${API_BASE_URL}.`,
  )
  if (error) return null

  const copy = WORKING_COPY[status]
  if (!copy) return null

  return (
    <p className="text-text-secondary text-[13px]" role="status">
      <span aria-hidden="true" className="bg-const mr-2 inline-block size-1.5 rounded-full" />
      {copy}
    </p>
  )
}
