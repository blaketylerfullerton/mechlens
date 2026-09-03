import { useCallback, useEffect, useRef, useState } from 'react'
import { getTraceJob, postTrace } from '@/lib/api-client'
import type { JobStatus, Trace } from '@/lib/api-types'

const POLL_INTERVAL_MS = 500

export interface UseTraceResult {
  status: JobStatus | 'idle'
  trace: Trace | null
  error: string | null
  run: (prompt: string, maxTokens: number) => void
}

// Submits a prompt to POST /trace and polls GET /trace/{job_id} until the
// job resolves. `trace` follows backend/app/schema.py's Trace shape exactly.
export function useTrace(): UseTraceResult {
  const [status, setStatus] = useState<JobStatus | 'idle'>('idle')
  const [trace, setTrace] = useState<Trace | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const generationRef = useRef(0)

  useEffect(
    () => () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    },
    [],
  )

  const poll = useCallback((jobId: string, generation: number) => {
    getTraceJob(jobId)
      .then((job) => {
        if (generation !== generationRef.current) return
        setStatus(job.status)
        if (job.status === 'done') {
          setTrace(job.trace)
        } else if (job.status === 'error') {
          setError(job.error ?? 'trace job failed')
        } else {
          timeoutRef.current = setTimeout(() => poll(jobId, generation), POLL_INTERVAL_MS)
        }
      })
      .catch((err: unknown) => {
        if (generation !== generationRef.current) return
        setError(err instanceof Error ? err.message : String(err))
      })
  }, [])

  const run = useCallback(
    (prompt: string, maxTokens: number) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      const generation = ++generationRef.current
      setTrace(null)
      setError(null)
      setStatus('pending')
      postTrace({ prompt, max_tokens: maxTokens })
        .then((res) => poll(res.job_id, generation))
        .catch((err: unknown) => {
          if (generation !== generationRef.current) return
          setError(err instanceof Error ? err.message : String(err))
          setStatus('error')
        })
    },
    [poll],
  )

  return { status, trace, error, run }
}
