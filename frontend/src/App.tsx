import { CLOUD_VIEWER } from '@/lib/api-client'
import { useCallback, useMemo, useState } from 'react'
import { Toaster } from 'sonner'

import type { SelectionVia } from '@/components/Brain'
import { ChatPanel } from '@/components/ChatPanel'
import { Stage } from '@/components/Stage'
import { TraceViewer } from '@/components/TraceViewer'
import { useTrace } from '@/hooks/useTrace'
import { TrainingPage } from '@/components/TrainingPage'
import { DictionariesPage } from '@/components/DictionariesPage'
import { CircuitsPage } from '@/components/CircuitsPage'
import { sourceForStep, type CircuitSource } from '@/lib/circuits-api'

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
  const [page, setPage] = useState<'viewer' | 'training' | 'dictionaries' | 'circuits'>('viewer')
  // Keyed so asking to explain the same token twice is a second request rather
  // than a no-op, and tagged with its source token so returning to Explore
  // lands on the token the graph is about.
  const [circuitsRequest, setCircuitsRequest] = useState<{ source: CircuitSource; key: number } | null>(null)
  const [trainingNavigation, setTrainingNavigation] = useState<{ id: string | null; key: number; newRun?: boolean }>({ id: null, key: 0 })
  const [dictionaryFocus, setDictionaryFocus] = useState<{ id: string; key: number } | null>(null)
  const [trainingFeatures, setTrainingFeatures] = useState<Record<string, number>>({})
  const [trainingVisited, setTrainingVisited] = useState(false)
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

  /**
   * Explain the token generated at `position`.
   *
   * The snapshot is built here, from the trace this app already holds, because
   * service trace jobs live in memory and die with the process — the browser is
   * what still has the trace afterwards. Token ids travel; text never gets
   * re-tokenized on the way.
   */
  function explainPrediction(position: number) {
    if (!trace) return
    const source = sourceForStep(trace, position)
    if (!source) return
    setCircuitsRequest((current) => ({ source, key: (current?.key ?? 0) + 1 }))
    setPage('circuits')
  }

  function backToTrace(position: number) {
    setPage('viewer')
    setInspectorOpen(true)
    if (trace) setSelection({ layer: currentSelection?.layer ?? trace.n_layers - 1, position, traceId: trace.trace_id, via: 'token' })
  }

  function openTraining(id: string | null = null, newRun = false) {
    setTrainingNavigation((current) => ({ id, newRun, key: current.key + 1 }))
    setTrainingVisited(true); setPage('training')
  }
  function inspectDictionary(id: string) {
    reset(); setTrainingRunId(id); setPage('viewer'); setInspectorOpen(true)
  }
  function openDictionaries(id: string) {
    setDictionaryFocus((current) => ({ id, key: (current?.key ?? 0) + 1 })); setPage('dictionaries')
  }
  function deletedDictionary(id: string) {
    if (trainingRunId === id) { reset(); setTrainingRunId(null) }
    setTrainingFeatures((current) => { const next = { ...current }; delete next[id]; return next })
    setTrainingNavigation((current) => current.id === id ? { id: null, key: current.key + 1 } : current)
  }

  return (
    <div className="text-text-primary bg-bg-base flex h-svh flex-col overflow-hidden">
      <Toaster richColors theme="dark" position="bottom-right" />
      <nav aria-label="Workspace" className="border-border-subtle mx-auto flex w-full max-w-[1800px] shrink-0 items-center gap-4 border-b px-4 py-4 text-sm sm:gap-6 sm:px-8">
        <span className="font-medium tracking-tight sm:mr-4">mechlens</span>
        {(CLOUD_VIEWER ? ['viewer'] as const : ['viewer', 'circuits', 'training', 'dictionaries'] as const).map((item) => <button key={item} aria-current={page === item ? 'page' : undefined}
          className={page === item ? 'text-text-primary' : 'text-text-tertiary'} onClick={() => { if (item === 'training') openTraining(); else setPage(item) }}>{item === 'viewer' ? 'Explore' : item === 'circuits' ? 'Circuits' : item === 'training' ? 'Training' : 'Dictionaries'}</button>)}
      </nav>
      {trainingVisited ? <div hidden={page !== 'training'} className={page === 'training' ? 'min-h-0 flex-1 overflow-y-auto' : 'hidden'}><TrainingPage active={page === 'training'} navigation={trainingNavigation} onInspect={inspectDictionary} onDictionaries={openDictionaries} /></div> : null}
      <div hidden={page !== 'circuits'} className={page === 'circuits' ? 'min-h-0 flex-1 overflow-hidden' : 'hidden'}><CircuitsPage active={page === 'circuits'} request={circuitsRequest} onBackToExplore={backToTrace} /></div>
      <div hidden={page !== 'dictionaries'} className={page === 'dictionaries' ? 'min-h-0 flex-1 overflow-y-auto' : 'hidden'}><DictionariesPage active={page === 'dictionaries'} focusRequest={dictionaryFocus} onSelect={(id) => openTraining(id)} onInspect={inspectDictionary} onDeleted={deletedDictionary} onNew={() => openTraining(null, true)} /></div>
      {page === 'viewer' ? <>
      {trainingRunId ? <div className="mx-auto flex w-full max-w-[1800px] shrink-0 items-center justify-between gap-4 px-8 pt-4 text-sm">
        <div><button className="text-fn mb-2 text-xs" onClick={() => openTraining(trainingRunId)}>Back to training run</button><p>Dictionary {trainingRunId.slice(0, 8)} · Enter a prompt, then select an active feature to find examples.</p></div>
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
            onExplain={CLOUD_VIEWER ? undefined : explainPrediction}
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
                  trainingRunId={trainingRunId}
                  selectedFeature={trainingRunId ? trainingFeatures[trainingRunId] ?? null : null}
                  onSelectFeature={(feature) => { if (trainingRunId) setTrainingFeatures((current) => ({ ...current, [trainingRunId]: feature })) }}
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
      </> : null}
    </div>
  )
}

export default App
