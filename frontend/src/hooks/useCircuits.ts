import { useCallback, useEffect, useRef, useState } from 'react'

import {
  createAnalysis,
  getAnalysis,
  getGraph,
  type AnalysisStatus,
  type AnalysisSummary,
  type CircuitGraphResponse,
  type CircuitSource,
} from '@/lib/circuits-api'

const POLL_MS = 700

export interface CircuitsState {
  status: AnalysisStatus | 'idle'
  analysisId: string | null
  summary: AnalysisSummary | null
  graph: CircuitGraphResponse | null
  error: string | null
}

/**
 * Runs one analysis and holds its result.
 *
 * Attribution takes a handful of seconds and the backend has no cancellation to
 * offer — the upstream call runs to completion once entered — so this reports
 * honest queued/running states and never shows a stop control it cannot honour.
 *
 * A `runId` counter invalidates in-flight polls after a new run or a reset, so
 * a late reply from an abandoned analysis cannot overwrite the current one.
 */
export function useCircuits() {
  const [state, setState] = useState<CircuitsState>({
    status: 'idle', analysisId: null, summary: null, graph: null, error: null,
  })
  const runId = useRef(0)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const stop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = null
  }, [])

  useEffect(() => () => stop(), [stop])

  /**
   * Poll one analysis to completion, then fetch its graph.
   *
   * A named function expression so the retry can refer to itself without the
   * callback capturing its own binding mid-initialization.
   */
  const load = useCallback(async function poll(analysisId: string, mine: number): Promise<void> {
    try {
      const summary = await getAnalysis(analysisId)
      if (runId.current !== mine) return

      if (summary.status === 'done') {
        const graph = await getGraph(analysisId)
        if (runId.current !== mine) return
        setState({ status: 'done', analysisId, summary, graph, error: null })
        return
      }
      if (summary.status === 'error' || summary.status === 'interrupted') {
        setState({ status: summary.status, analysisId, summary, graph: null, error: summary.error })
        return
      }
      setState((s) => ({ ...s, status: summary.status, analysisId, summary }))
      timer.current = setTimeout(() => void poll(analysisId, mine), POLL_MS)
    } catch (err) {
      if (runId.current !== mine) return
      setState((s) => ({ ...s, status: 'error', error: err instanceof Error ? err.message : String(err) }))
    }
  }, [])

  const run = useCallback(async (source: CircuitSource) => {
    stop()
    const mine = ++runId.current
    setState({ status: 'queued', analysisId: null, summary: null, graph: null, error: null })
    try {
      const { analysis_id } = await createAnalysis(source)
      if (runId.current !== mine) return
      setState((s) => ({ ...s, analysisId: analysis_id }))
      void load(analysis_id, mine)
    } catch (err) {
      if (runId.current !== mine) return
      setState((s) => ({ ...s, status: 'error', error: err instanceof Error ? err.message : String(err) }))
    }
  }, [load, stop])

  /** Reopen a saved analysis. Reads stored data; runs no compute. */
  const open = useCallback((analysisId: string) => {
    stop()
    const mine = ++runId.current
    setState({ status: 'queued', analysisId, summary: null, graph: null, error: null })
    void load(analysisId, mine)
  }, [load, stop])

  const reset = useCallback(() => {
    stop()
    runId.current += 1
    setState({ status: 'idle', analysisId: null, summary: null, graph: null, error: null })
  }, [stop])

  return { ...state, run, open, reset }
}
