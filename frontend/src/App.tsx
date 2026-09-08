import { useCallback, useMemo, useState } from 'react'

import { Brain } from '@/components/Brain'
import { ChatPanel } from '@/components/ChatPanel'
import { TraceViewer } from '@/components/TraceViewer'
import { useTrace } from '@/hooks/useTrace'

/**
 * The selected (layer, token) cell, tagged with the trace it belongs to.
 *
 * Owned here rather than inside `TraceViewer` because the brain and the grid
 * both read it and both change it — two copies of this would let the two
 * surfaces disagree about what is selected, which is the one thing a shared
 * selection exists to prevent. The `traceId` tag is what makes a stale
 * selection detectable when a new trace lands.
 */
export interface Selection {
  layer: number
  position: number
  traceId: string
}

function App() {
  const { error, progress, run, status, trace } = useTrace()
  const [selection, setSelection] = useState<Selection | null>(null)

  // Resolved once, here, so both surfaces get the *same* default for a new
  // trace instead of each falling back on its own.
  const currentSelection = useMemo<Selection | null>(() => {
    if (!trace || trace.steps.length === 0) return null
    if (selection?.traceId === trace.trace_id) return selection
    return {
      layer: trace.n_layers - 1,
      position: trace.steps.length - 1,
      traceId: trace.trace_id,
    }
  }, [trace, selection])

  const selectCell = useCallback(
    (layer: number, position: number) => {
      if (!trace) return
      setSelection({ layer, position, traceId: trace.trace_id })
    },
    [trace],
  )

  // Picking a token keeps the depth you were looking at, and picking a band
  // keeps the token — so moving along one axis never silently resets the other.
  const selectPosition = useCallback(
    (position: number) => {
      if (!trace || !currentSelection) return
      setSelection({ layer: currentSelection.layer, position, traceId: trace.trace_id })
    },
    [trace, currentSelection],
  )

  const selectLayer = useCallback(
    (layer: number) => {
      if (!trace || !currentSelection) return
      setSelection({ layer, position: currentSelection.position, traceId: trace.trace_id })
    },
    [trace, currentSelection],
  )

  return (
    <div className="min-h-svh bg-[radial-gradient(circle_at_top,_#17304a_0%,_#0a1220_34%,_#060912_72%)] text-slate-100">
      {/* Stacks under lg so neither surface gets clipped on a narrow screen;
          the grid keeps its own scroll container either way. */}
      <div className="mx-auto flex min-h-svh max-w-[1800px] flex-col gap-4 px-4 pb-36 pt-5 sm:px-6 lg:flex-row lg:gap-5 lg:px-8">
        <div className="h-[42svh] shrink-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-950/40 backdrop-blur lg:h-auto lg:w-1/2 lg:min-w-[20rem]">
          <Brain
            onSelectLayer={selectLayer}
            progress={progress}
            selection={currentSelection}
            status={status}
            trace={trace}
          />
        </div>

        <div className="min-w-0 flex-1">
          <TraceViewer
            error={error}
            onSelectCell={selectCell}
            onSelectPosition={selectPosition}
            selection={currentSelection}
            status={status}
            trace={trace}
          />
        </div>
      </div>

      <ChatPanel error={error} onTraceRequest={run} status={status} />
    </div>
  )
}

export default App
