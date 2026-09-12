import type { ReactNode } from 'react'

import type { RunState } from '@/hooks/useTrace'
import type { Trace } from '@/lib/api-types'

import { EmptyState } from './trace/EmptyState'
import { Inspector } from './trace/Inspector'
import { MapNotes, ResidualMap } from './trace/ResidualMap'
import { TokenStrip } from './trace/TokenStrip'
import { TraceHeader } from './trace/TraceHeader'
import { useResidualMap } from './trace/useResidualMap'

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
  onSelectCell: (layer: number, position: number) => void
  onSelectPosition: (position: number) => void
  /**
   * The prompt composer, rendered inside the empty state directly under the
   * facts block. Passed in rather than imported so this component keeps
   * knowing nothing about how a run is started — and so it goes away with the
   * empty state the moment a trace exists.
   */
  composer?: ReactNode
}

/**
 * The trace page, and nothing but its composition: header, token strip, the
 * residual map beside the inspector, the caveats under both. Every piece lives
 * in `./trace`; what stays here is the arrangement and the two conditions
 * under which there is no page to arrange.
 */
export function TraceViewer({
  trace,
  status,
  error,
  selection,
  onSelectCell,
  onSelectPosition,
  composer,
}: TraceViewerProps) {
  const map = useResidualMap(trace)

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
    <main className="enter min-h-full">
      <div className="space-y-4">
        <TraceHeader trace={trace} />

        <TokenStrip
          onSelect={onSelectPosition}
          selectedPosition={selection.position}
          steps={trace.steps}
        />

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <ResidualMap
            map={map}
            onSelectCell={onSelectCell}
            onSelectPosition={onSelectPosition}
            selection={selection}
            trace={trace}
          />
          <Inspector state={selectedState} step={selectedStep} />
        </div>

        <MapNotes crossover={map.crossover} mode={map.mode} />
      </div>
    </main>
  )
}
