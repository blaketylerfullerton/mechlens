import { useCallback, useEffect, useRef, useState } from 'react'
import { getTraceJob, postSteer } from '@/lib/api-client'
import type { JobStatus, Trace } from '@/lib/api-types'

const POLL_INTERVAL_MS = 500

export interface SteerParams {
  prompt: string
  maxTokens: number
  layer: number
  featureIdx: number
  coefficient: number
}

export interface UseSteerResult {
  status: JobStatus | 'idle'
  trace: Trace | null
  error: string | null
  run: (params: SteerParams) => void
}

// Same job/poll shape as useTrace, for POST /steer — a trace generated with
// a feature-activation intervention applied at `layer`/`featureIdx`.
export function useSteer(): UseSteerResult {
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
          setError(job.error ?? 'steer job failed')
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
    ({ prompt, maxTokens, layer, featureIdx, coefficient }: SteerParams) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      const generation = ++generationRef.current
      setTrace(null)
      setError(null)
      setStatus('pending')
      postSteer({
        prompt,
        max_tokens: maxTokens,
        layer,
        feature_idx: featureIdx,
        coefficient,
      })
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
