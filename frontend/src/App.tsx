import { useCallback, useMemo, useState } from 'react'

import { Brain, type SelectionVia } from '@/components/Brain'
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
      via: 'default',
    }
  }, [trace, selection])

  const selectCell = useCallback(
    (layer: number, position: number) => {
      if (!trace) return
      setSelection({ layer, position, traceId: trace.trace_id, via: 'cell' })
    },
    [trace],
  )

  // Picking a token keeps the layer you were looking at — so moving along one
  // axis never silently resets the other.
  const selectPosition = useCallback(
    (position: number) => {
      if (!trace || !currentSelection) return
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

  return (
    <div className="text-text-primary bg-bg-base min-h-svh">
      {/* Stacks under lg so neither surface gets clipped on a narrow screen;
          the grid keeps its own scroll container either way. */}
      <div className="mx-auto flex min-h-svh max-w-[1800px] flex-col gap-4 px-4 pt-5 pb-8 sm:px-6 lg:flex-row lg:gap-5 lg:px-8">
        {/* The one frame treatment, used here and on the residual map: an outer
            frame holding an inner surface, hairline on both, 4px gap, radii
            concentric (16 − 4 = 12). It wraps what the reader looks *into* and
            nothing else. */}
        {/* Sticky and viewport-height on a wide screen, rather than growing
            with the trace beside it. The brain anchors its own overlays — the
            legend, the lit-feature panel, the hover detail — to its bottom
            edge, and a column that grew to the height of a long trace put
            that edge below the fold: the detail panel for whatever you were
            pointing at rendered somewhere you could not see. */}
        <div className="border-border-subtle flex h-[42svh] shrink-0 rounded-[16px] border bg-[#0D0E11] p-1 lg:sticky lg:top-5 lg:h-[calc(100svh-3.25rem)] lg:w-1/2 lg:min-w-[20rem]">
          <div className="border-border-subtle bg-bg-surface min-w-0 flex-1 overflow-hidden rounded-[12px] border">
            <Brain
              onSelectLayer={selectLayer}
              progress={progress}
              selection={currentSelection}
              status={status}
              trace={trace}
            />
          </div>
        </div>

        <div className="min-w-0 flex-1">
          {/* The composer lives inside the empty state, under the facts
              block — so it is there when there is nothing to look at, and
              gone the moment a prompt is submitted and the trace takes the
              page over. */}
          <TraceViewer
            composer={<ChatPanel error={error} onTraceRequest={run} status={status} />}
            error={error}
            onSelectCell={selectCell}
            onSelectPosition={selectPosition}
            selection={currentSelection}
            status={status}
            trace={trace}
          />
        </div>
      </div>
    </div>
  )
}

export default App
