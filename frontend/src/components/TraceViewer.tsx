import type { ReactNode } from 'react'

import type { RunState } from '@/hooks/useTrace'
import type { Trace } from '@/lib/api-types'

import { EmptyState } from './trace/EmptyState'
import { Inspector } from './trace/Inspector'

type TraceViewerProps = {
  trace: Trace | null
  status: RunState
  error: string | null
  /**
   * The shared selection, resolved by `App` — including the default for a
   * newly arrived trace. Null only when there is no trace to select in, which
   * is also when this component renders its empty state.
   */
  selection: { layer: number; position: number } | null
  /**
   * The prompt composer, rendered inside the empty state directly under the
   * facts block. Passed in rather than imported so this component keeps
   * knowing nothing about how a run is started — and so it goes away with the
   * empty state the moment a trace exists.
   */
  composer?: ReactNode
}

/**
 * The right-hand rail: everything about the one cell the reader has selected,
 * and nothing else.
 *
 * It used to be a whole second page in a quarter-width column — a page header,
 * the token strip, the residual grid, this inspector and the map's caveats,
 * stacked. The grid and the strip are axes of the primary object and have
 * moved onto it (`Stage`); the header became a line above it. What is left is
 * a single vertical stack of panels, which is a shape that is happy at a fixed
 * 24rem and was never happy sharing one.
 */
export function TraceViewer({ trace, status, error, selection, composer }: TraceViewerProps) {
  if (!trace || trace.steps.length === 0 || selection === null) {
    return <EmptyState composer={composer} error={error} status={status} />
  }

  const selectedStep = trace.steps[selection.position]
  const selectedState = selectedStep?.layers[selection.layer]

  if (!selectedStep || !selectedState) {
    return (
      <EmptyState
        composer={composer}
        error="The selected trace state is unavailable."
        status="error"
      />
    )
  }

  return (
    <div className="enter flex h-full min-h-0 flex-col">
      {error ? <p role="alert" className="text-err mb-2 text-[13px]">{error}</p> : null}
      {/* Its own scroll, so a long feature list never drags the stage beside it
          taller than the window. */}
      <div className="mask-fade-b min-h-0 flex-1 overflow-y-auto">
        <Inspector state={selectedState} step={selectedStep} running={status === 'running'} />
      </div>
    </div>
  )
}
