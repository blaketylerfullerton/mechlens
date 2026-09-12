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
  const { error, progress, run, status, trace } = useTrace()
  const [selection, setSelection] = useState<Selection | null>(null)

  // Resolved once, here, so every surface gets the *same* default for a new
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

  // The rail is collapsible, but only once there is a trace to inspect. With
  // no trace the right column holds the composer and the empty state — the one
  // action at rest — and collapsing that would hide the only thing to do.
  const [railOpen, setRailOpen] = useState(true)
  const collapsed = railOpen === false && trace !== null

  // Two widths, not four.
  //
  // At rest the right column *is* the product: what this is, and the three
  // prompts that start it. That wants half the window — squeezing it into a
  // quarter put a 36px headline in a 26rem column.
  //
  // With a trace it is the inspector: one vertical stack of panels, whose
  // natural width is the width of a panel. A fixed 24rem, so it stops changing
  // size as the stage grows, and the stage takes everything left over — which
  // is the whole point of moving the grid onto it.
  const resting = trace === null

  return (
    <div className="text-text-primary bg-bg-base min-h-svh">
      {/* Stacks under lg so neither surface gets clipped on a narrow screen.
          Collapsing is an lg-and-up affordance: on a stacked column a 2.5rem
          rail is not a smaller version of the inspector, it is a bar in the
          way. */}
      <div className="mx-auto flex min-h-svh max-w-[1800px] flex-col gap-4 px-4 pt-5 pb-8 sm:px-6 lg:flex-row lg:gap-5 lg:px-8">
        {/* Sticky and viewport-height on a wide screen, rather than growing
            with whatever is beside it. The brain anchors its own overlays — the
            legend, the lit-feature panel, the hover detail — to its own edges,
            and a column that grew put those below the fold. */}
        <div
          className={`flex h-[52svh] min-w-0 shrink-0 flex-col lg:sticky lg:top-5 lg:h-[calc(100svh-3.25rem)] lg:min-w-[22rem] ${
            resting ? 'lg:w-1/2' : 'lg:flex-1'
          }`}
        >
          <Stage
            onSelectCell={selectCell}
            onSelectLayer={selectLayer}
            onSelectPosition={selectPosition}
            progress={progress}
            selection={currentSelection}
            status={status}
            trace={trace}
          />
        </div>

        <div
          className={`min-w-0 lg:sticky lg:top-5 lg:h-[calc(100svh-3.25rem)] ${
            collapsed
              ? 'flex-1 lg:w-10 lg:flex-none'
              : resting
                ? 'flex-1'
                : 'flex-1 lg:w-96 lg:flex-none'
          }`}
        >
          {/* The rail the inspector collapses to. Only at lg, and only with a
              trace — it is the same control as the chevron above, wearing the
              width it has left. */}
          {collapsed ? (
            <button
              aria-expanded="false"
              aria-label="Show the inspector"
              className="border-border-subtle bg-bg-surface text-text-tertiary hover:text-text-primary hover:border-border-strong hidden h-full w-10 flex-col items-center gap-3 rounded-[6px] border transition-colors duration-150 lg:flex"
              onClick={() => setRailOpen(true)}
              type="button"
            >
              <Chevron className="mt-3 rotate-180" />
              <span className="[writing-mode:vertical-rl] font-mono text-[11px] tracking-[0.04em]">
                inspector
              </span>
            </button>
          ) : null}

          {/* The composer lives inside the empty state, under the facts
              block — so it is there when there is nothing to look at, and gone
              the moment a prompt is submitted and the trace takes the page
              over. */}
          <div className={`flex h-full min-h-0 flex-col ${collapsed ? 'lg:hidden' : ''}`}>
            {trace ? (
              <div className="mb-2 hidden shrink-0 justify-end lg:flex">
                <button
                  aria-expanded="true"
                  aria-label="Hide the inspector"
                  className="text-text-tertiary hover:text-text-primary inline-flex items-center gap-1.5 rounded-xs px-2 py-1 font-mono text-[11px] transition-colors duration-150"
                  onClick={() => setRailOpen(false)}
                  type="button"
                >
                  Hide
                  <Chevron />
                </button>
              </div>
            ) : null}

            <div className="min-h-0 flex-1">
              <TraceViewer
                composer={<ChatPanel error={error} onTraceRequest={run} status={status} />}
                error={error}
                selection={currentSelection}
                status={status}
                trace={trace}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Stroke, monochrome, 16px — the one icon this layout needs. */
function Chevron({ className = '' }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={`size-4 shrink-0 ${className}`}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}

export default App
