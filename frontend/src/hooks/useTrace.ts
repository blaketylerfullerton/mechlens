import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, getTraceJob, postTrace } from '@/lib/api-client'
import type { JobProgress, JobStatus, TracePass, Trace } from '@/lib/api-types'
import { atlasLayers, loadAtlas } from '@/lib/atlas'

// Fast enough to resolve the lens phase, which decodes a layer roughly every
// 90ms on gemma-2-2b: at 500ms the brain's depth sweep jumped in blocks of
// five or six layers. This is a localhost GET returning four small fields, and
// it only runs while a job is in flight.
const POLL_INTERVAL_MS = 150
// The model loads on a warm-up thread after the server starts, so an early
// submit gets a 503 rather than a job id. Wait it out instead of making the
// user press Run again — the first load pulls gemma's weights and is slow.
const WARMUP_RETRY_MS = 2000

// The passes the trace job runs. The service writes no residual sidecar to
// enrich from later, so every pass has to run inside the trace job or not at
// all — there is no second chance at these.
//
// `sae` fills the features the brain draws as nodes; `labels` names them;
// `lens` fills the per-(token, layer) readout the grid classifies. `labels`
// depends on `sae` and the service rejects the pair the other way round.
const REQUESTED_PASSES: TracePass[] = ['sae', 'labels', 'lens']

// `warming` is not a job state — no job exists yet, because /trace is still
// answering 503 while gemma loads. Kept distinct from `pending` so the UI can
// say "the model is loading" rather than implying a trace is being computed.
export type RunState = 'idle' | 'warming' | JobStatus

export interface UseTraceResult {
  status: RunState
  trace: Trace | null
  error: string | null
  progress: JobProgress | null
  run: (prompt: string, maxTokens: number) => void
}

// Submits a prompt to POST /trace and polls GET /trace/{job_id} until the
// job resolves. `trace` follows backend/app/schema.py's Trace shape exactly,
// and `progress` follows JobStatusResponse.progress: null until the job
// reports its first reading.
export function useTrace(): UseTraceResult {
  const [status, setStatus] = useState<RunState>('idle')
  const [trace, setTrace] = useState<Trace | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobProgress | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const generationRef = useRef(0)

  // Which layers to run the SAE pass on: the ones the atlas can actually
  // place. Derived from the atlas rather than hardcoded, so a trace never
  // reports features the map has nowhere to draw — and so this widens on its
  // own when a fuller atlas is built, with nothing here to update.
  //
  // Null means "no atlas loaded", and then the request omits `sae_layers`
  // entirely and the service runs every layer, which is the right default
  // when there is no map to agree with.
  const saeLayersRef = useRef<number[] | null>(null)

  useEffect(() => {
    let cancelled = false
    loadAtlas().then((atlas) => {
      if (!cancelled && atlas) saeLayersRef.current = atlasLayers(atlas)
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(
    () => () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    },
    [],
  )

  const poll = useCallback(
    function poll(jobId: string, generation: number) {
      getTraceJob(jobId)
        .then((job) => {
          if (generation !== generationRef.current) return
          setStatus(job.status)
          // Assigned straight through rather than merged: the service already
          // guarantees a reading never goes backwards within a phase, so the
          // latest response is always the one to show.
          setProgress(job.progress)
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
          setStatus('error')
        })
    },
    [],
  )

  const submit = useCallback(
    function submit(prompt: string, maxTokens: number, generation: number) {
      postTrace({
        prompt,
        max_tokens: maxTokens,
        passes: REQUESTED_PASSES,
        sae_layers: saeLayersRef.current,
      })
        .then((res) => {
          if (generation !== generationRef.current) return
          // A job exists now, so leave `warming` behind even before the first
          // poll answers.
          setStatus('pending')
          poll(res.job_id, generation)
        })
        .catch((err: unknown) => {
          if (generation !== generationRef.current) return
          if (err instanceof ApiError && err.status === 503) {
            setStatus('warming')
            timeoutRef.current = setTimeout(
              () => submit(prompt, maxTokens, generation),
              WARMUP_RETRY_MS,
            )
            return
          }
          setError(err instanceof Error ? err.message : String(err))
          setStatus('error')
        })
    },
    [poll],
  )

  const run = useCallback(
    (prompt: string, maxTokens: number) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      const generation = ++generationRef.current
      setTrace(null)
      setError(null)
      // Cleared with the trace: a new run must not briefly animate against the
      // previous one's counters.
      setProgress(null)
      setStatus('pending')
      submit(prompt, maxTokens, generation)
    },
    [submit],
  )

  return { status, trace, error, progress, run }
}
