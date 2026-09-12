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

  // The grid is collapsible, but only once there is a trace in it. With no
  // trace the right column holds the composer and the empty state — the one
  // action at rest — and collapsing that would hide the only thing to do.
  const [gridOpen, setGridOpen] = useState(true)
  const collapsed = gridOpen === false && trace !== null

  // The brain dominates once there is something lit in it. Before that the
  // right column holds the whole first-run experience — what this is, and the
  // three prompts that start it — and squeezing that into a quarter of the
  // window put a 36px headline in a 26rem column and left the largest object
  // on screen as the least explained one. So the split follows the content:
  // 60/40 at rest, 75/25 with a trace, and the full width when the grid is
  // folded away. Animated, because it moves on its own rather than on a click.
  const brainWidth = collapsed
    ? 'lg:w-[calc(100%-3.75rem)]'
    : trace === null
      ? 'lg:w-[60%]'
      : 'lg:w-[75%]'

  return (
    <div className="text-text-primary bg-bg-base min-h-svh">
      {/* Stacks under lg so neither surface gets clipped on a narrow screen;
          the grid keeps its own scroll container either way. Collapsing is an
          lg-and-up affordance: on a stacked column a 2.5rem rail is not a
          smaller version of the grid, it is just a bar in the way. */}
      <div className="mx-auto flex min-h-svh max-w-[1800px] flex-col gap-4 px-4 pt-5 pb-8 sm:px-6 lg:flex-row lg:gap-5 lg:px-8">
        {/* The one frame treatment, used here and on the residual map: an outer
            frame holding an inner surface, hairline on both, 4px gap, radii
            concentric (6 − 4 = 2). It wraps what the reader looks *into* and
            nothing else. */}
        {/* Sticky and viewport-height on a wide screen, rather than growing
            with the trace beside it. The brain anchors its own overlays — the
            legend, the lit-feature panel, the hover detail — to its bottom
            edge, and a column that grew to the height of a long trace put
            that edge below the fold: the detail panel for whatever you were
            pointing at rendered somewhere you could not see. */}
        <div
          className={`border-border-subtle flex h-[42svh] shrink-0 rounded-[6px] border bg-[#0D0E11] p-1 transition-[width] duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] lg:sticky lg:top-5 lg:h-[calc(100svh-3.25rem)] lg:min-w-[22rem] ${brainWidth}`}
        >
          <div className="border-border-subtle bg-bg-surface min-w-0 flex-1 overflow-hidden rounded-[2px] border">
            <Brain
              onSelectLayer={selectLayer}
              progress={progress}
              selection={currentSelection}
              status={status}
              trace={trace}
            />
          </div>
        </div>

        <div
          className={`min-w-0 ${collapsed ? 'flex-1 lg:w-10 lg:flex-none' : 'flex-1'}`}
        >
          {/* The rail the grid collapses to. Only at lg, and only with a
              trace — it is the same control as the chevron above, wearing the
              width it has left. */}
          {collapsed ? (
            <button
              aria-expanded="false"
              aria-label="Show the trace grid"
              className="border-border-subtle bg-bg-surface text-text-tertiary hover:text-text-primary hover:border-border-strong sticky top-5 hidden h-[calc(100svh-3.25rem)] w-10 flex-col items-center gap-3 rounded-[6px] border transition-colors duration-150 lg:flex"
              onClick={() => setGridOpen(true)}
              type="button"
            >
              <Chevron className="mt-3 rotate-180" />
              <span className="[writing-mode:vertical-rl] font-mono text-[11px] tracking-[0.04em]">
                trace
              </span>
            </button>
          ) : null}

          {/* The composer lives inside the empty state, under the facts
              block — so it is there when there is nothing to look at, and
              gone the moment a prompt is submitted and the trace takes the
              page over. */}
          <div className={collapsed ? 'lg:hidden' : ''}>
            {/* Hidden rather than unmounted, so collapsing and reopening does
                not throw away the grid's scroll position. */}
            {trace ? (
              <div className="mb-2 hidden justify-end lg:flex">
                <button
                  aria-expanded="true"
                  aria-label="Hide the trace grid"
                  className="text-text-tertiary hover:text-text-primary inline-flex items-center gap-1.5 rounded-xs px-2 py-1 font-mono text-[11px] transition-colors duration-150"
                  onClick={() => setGridOpen(false)}
                  type="button"
                >
                  <Chevron />
                  Hide
                </button>
              </div>
            ) : null}

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
