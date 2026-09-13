import { useCallback, useMemo, useState } from 'react'

import type { SelectionVia } from '@/components/Brain'
import { ChatPanel } from '@/components/ChatPanel'
import { Stage } from '@/components/Stage'
import { TraceViewer } from '@/components/TraceViewer'
import { useTrace } from '@/hooks/useTrace'
import { TrainingPage } from '@/components/TrainingPage'

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
  const [page, setPage] = useState<'viewer' | 'training'>('viewer')
  const [trainingRunId, setTrainingRunId] = useState<string | null>(null)
  const { error, storageNotice, progress, run, reset, status, trace } = useTrace(trainingRunId)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)

  // Resolved once, here, so every surface gets the *same* default for a new
  // trace instead of each falling back on its own.
  const currentSelection = useMemo<Selection | null>(() => {
    if (!trace || trace.steps.length === 0) return null
    if (selection?.traceId === trace.trace_id) return selection
    const dictionary = trace.passes.find((pass) => pass.name === 'sae' && String(pass.params.release).startsWith('local/'))
    const trainedLayer = dictionary ? Number(String(dictionary.params.layers).split(',')[0]) : null
    return {
      layer: trainedLayer !== null && Number.isInteger(trainedLayer) && trainedLayer >= 0 && trainedLayer < trace.n_layers ? trainedLayer : trace.n_layers - 1,
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
    <div className="text-text-primary bg-bg-base flex h-svh flex-col overflow-hidden">
      <nav aria-label="Workspace" className="border-border-subtle mx-auto flex w-full max-w-[1800px] shrink-0 items-center gap-6 border-b px-8 py-4 text-sm">
        <span className="mr-4 font-medium tracking-tight">mechlens</span>
        {(['viewer', 'training'] as const).map((item) => <button key={item} aria-current={page === item ? 'page' : undefined}
          className={page === item ? 'text-text-primary' : 'text-text-tertiary'} onClick={() => setPage(item)}>{item === 'viewer' ? 'Explore' : 'Training'}</button>)}
      </nav>
      {page === 'training' ? <TrainingPage onInspect={(id) => { reset(); setTrainingRunId(id); setPage('viewer'); setInspectorOpen(true) }} /> : <>
      {trainingRunId ? <div className="mx-auto flex w-full max-w-[1800px] shrink-0 items-center justify-between gap-4 px-8 pt-4 text-sm">
        <p>Custom SAE · {trainingRunId.slice(0, 8)} · Unlabeled features. Enter a prompt to inspect its saved checkpoint.</p>
        <button className="text-fn shrink-0" onClick={() => { reset(); setTrainingRunId(null) }}>Use Gemma Scope</button>
      </div> : null}
      <div className="mx-auto flex w-full min-h-0 max-w-[1800px] flex-1 flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:px-8">
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Stage
            customDictionary={trainingRunId !== null}
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
            className="border-border-subtle min-h-0 min-w-0 border-t pt-4 lg:h-full lg:w-80 lg:shrink-0 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-5">
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
      </>}
    </div>
  )
}

export default App
