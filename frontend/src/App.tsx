import { useCallback, useMemo, useState } from 'react'

import type { SelectionVia } from '@/components/Brain'
import { ChatPanel } from '@/components/ChatPanel'
import { Stage } from '@/components/Stage'
import { TraceViewer } from '@/components/TraceViewer'
import { useTrace } from '@/hooks/useTrace'

/**
 * The selected (layer, token) cell, tagged with the trace it belongs to.
 *
 * Owned here rather than inside `Stage` because the brain, the grid and the
 * inspector rail all read it and the first two both change it — two copies of
 * this would let the surfaces disagree about what is selected, which is the
 * one thing a shared selection exists to prevent. The `traceId` tag is what
 * makes a stale selection detectable when a new trace lands.
 */
export interface Selection {
  layer: number
  position: number
  traceId: string
  /**
   * What moved the selection here.
   *
   * The pair alone is not enough for the brain: clicking one grid cell and
   * arriving at the same cell by default mean different things about what
   * should be lit. Carried on the selection rather than kept beside it, so
   * the two can never disagree about which is which.
   */
  via: SelectionVia
}

function App() {
  const { error, storageNotice, progress, run, reset, status, trace } = useTrace()
  const [selection, setSelection] = useState<Selection | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)

  // Resolved once, here, so every surface gets the *same* default for a new
  // trace instead of each falling back on its own.
  const currentSelection = useMemo<Selection | null>(() => {
    if (!trace || trace.steps.length === 0) return null
    if (selection?.traceId === trace.trace_id) return selection
    return {
      layer: trace.n_layers - 1,
      position: trace.steps.length - 1,
      traceId: trace.trace_id,
      via: status === 'running' ? 'token' : 'default',
    }
  }, [trace, selection, status])

  const selectCell = useCallback(
    (layer: number, position: number) => {
      if (!trace) return
      setInspectorOpen(true)
      setSelection({ layer, position, traceId: trace.trace_id, via: 'cell' })
    },
    [trace],
  )

  // Picking a token keeps the layer you were looking at — so moving along one
  // axis never silently resets the other.
  const selectPosition = useCallback(
    (position: number) => {
      if (!trace || !currentSelection) return
      setInspectorOpen(true)
      setSelection({
        layer: currentSelection.layer,
        position,
        traceId: trace.trace_id,
        via: 'token',
      })
    },
    [trace, currentSelection],
  )

  // The brain's transport, moving the shared selection's depth.
  const selectLayer = useCallback(
    (layer: number) => {
      if (!trace || !currentSelection) return
      setSelection({
        layer,
        position: currentSelection.position,
        traceId: trace.trace_id,
        via: 'layer',
      })
    },
    [trace, currentSelection],
  )


  // Clearing the trace clears everything that was an index into it. A
  // selection or an open inspector outliving its trace would point at a run
  // that no longer exists.
  const startOver = useCallback(() => {
    reset()
    setSelection(null)
    setInspectorOpen(false)
  }, [reset])

  return (
    <div className="text-text-primary bg-bg-base min-h-svh">
      <div className="mx-auto flex min-h-svh max-w-[1800px] flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:px-8">
        <main className="flex min-h-[38rem] min-w-0 flex-col lg:sticky lg:top-5 lg:h-[calc(100svh-2.5rem)] lg:flex-1">
          <Stage
            composer={<ChatPanel error={error} onTraceRequest={run} status={status} />}
            inspectorOpen={inspectorOpen}
            onToggleInspector={() => setInspectorOpen((open) => !open)}
            onReset={startOver}
            onFollowLatest={() => setSelection(null)}
            followingLatest={selection?.traceId !== trace?.trace_id}
            onSelectCell={selectCell}
            onSelectLayer={selectLayer}
            onSelectPosition={selectPosition}
            progress={progress}
            selection={currentSelection}
            status={status}
            trace={trace}
          />
          {error ? <p role="alert" className="text-err mt-2 text-[13px]">{error}</p> : null}
          {storageNotice ? <p role="status" className="text-text-secondary mt-2 text-[12px]">{storageNotice}</p> : null}
        </main>

        {trace && inspectorOpen ? (
          <section id="trace-inspector" aria-label="Inspector"
            className="border-border-subtle min-w-0 border-t pt-4 lg:sticky lg:top-5 lg:h-[calc(100svh-2.5rem)] lg:w-80 lg:shrink-0 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-5">
            <div className="flex h-full min-h-0 flex-col">
                <div className="mb-4 flex shrink-0 items-center justify-between">
                  <h2 className="text-[13px] font-medium">Inspector</h2>
                  <button type="button" aria-label="Close inspector"
                    onClick={() => setInspectorOpen(false)}
                    className="text-text-secondary hover:text-text-primary rounded-xs px-2 py-1 text-[13px]">
                    Close
                  </button>
                </div>
              <div className="min-h-0 flex-1">
                <TraceViewer
                  error={null}
                  selection={currentSelection}
                  status={status}
                  trace={trace}
                />
              </div>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  )
}

export default App
