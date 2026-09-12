import { useState } from 'react'

import { Brain } from '@/components/Brain'
import { ResidualMap } from '@/components/trace/ResidualMap'
import { TokenStrip } from '@/components/trace/TokenStrip'
import { TraceHeader } from '@/components/trace/TraceHeader'
import { useResidualMap } from '@/components/trace/useResidualMap'
import type { RunState } from '@/hooks/useTrace'
import type { JobProgress, Trace } from '@/lib/api-types'

type View = 'brain' | 'grid'

type Selection = { layer: number; position: number }

/**
 * Two views of one object, and the control that names which is on screen.
 *
 * They are the same tensor — layers down, positions across — so having both
 * permanently on screen was two primary objects competing, and the grid always
 * lost: it lived in a quarter-width column where its own token axis could not
 * fit. One at a time, both at full width.
 */
function ViewToggle({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return (
    <div
      aria-label="Stage view"
      className="border-border-subtle bg-bg-elevated flex shrink-0 rounded-[2px] border text-[11px]"
      role="group"
    >
      {(['brain', 'grid'] as const).map((option) => (
        <button
          aria-pressed={option === view}
          className={`px-2 py-1 font-mono transition-colors duration-150 first:rounded-l-[2px] last:rounded-r-[2px] ${
            option === view
              ? 'bg-fn/[0.10] text-text-primary'
              : 'text-text-tertiary hover:bg-white/[0.02]'
          }`}
          key={option}
          onClick={() => onChange(option)}
          type="button"
        >
          {option}
        </button>
      ))}
    </div>
  )
}

type StageProps = {
  followingLatest: boolean
  onFollowLatest: () => void
  trace: Trace | null
  selection: Selection | null
  status: RunState
  progress: JobProgress | null
  onSelectLayer: (layer: number) => void
  onSelectCell: (layer: number, position: number) => void
  onSelectPosition: (position: number) => void
}

/**
 * The primary object, and everything that is an axis of it: the trace's
 * identity above, the view itself in the frame, the token sequence below.
 *
 * All three appear only with a trace. Before that the frame holds the brain at
 * rest and nothing else, because a view toggle with one view in it, a header
 * for no run, and a token strip with no tokens are all controls arriving
 * before their object.
 */
export function Stage({
  followingLatest,
  onFollowLatest,
  trace,
  selection,
  status,
  progress,
  onSelectLayer,
  onSelectCell,
  onSelectPosition,
}: StageProps) {
  const map = useResidualMap(trace)
  const [view, setView] = useState<View>('brain')

  const loaded = trace !== null && trace.steps.length > 0 && selection !== null
  const showGrid = loaded && view === 'grid'

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {loaded ? (
        <div className="flex items-baseline justify-between gap-4">
          <TraceHeader trace={trace} />
          <ViewToggle onChange={setView} view={view} />
        </div>
      ) : null}

      {loaded ? (
        <div className="border-border-subtle rounded-sm border p-3">
          <div className="text-text-tertiary mb-2 flex items-center justify-between text-[11px]">
            <span role="status">
              {status === 'running'
                ? progress?.phase === 'generating' ? 'Live · generating and analyzing' : 'Finalizing trace measurements'
                : status === 'error' ? 'Interrupted · showing available data' : 'Completion'}
            </span>
            {status === 'running' ? (
              <button type="button" onClick={onFollowLatest} aria-pressed={followingLatest}
                className="text-fn px-2 py-1">
                {followingLatest ? 'Following latest' : 'Follow latest'}
              </button>
            ) : null}
          </div>
          <p className="text-text-primary max-h-24 overflow-y-auto whitespace-pre-wrap font-mono text-[13px]">
            {trace.completion || 'Waiting for the first token…'}
          </p>
          {status === 'running' ? (
            <p className="text-text-tertiary mt-2 text-[11px]">
              Select a token to inspect it. Its activations arrive on the next model step; analysis fills in afterward.
            </p>
          ) : null}
        </div>
      ) : null}

      {/* The one frame treatment: an outer frame holding an inner surface,
          hairline on both, 4px gap, radii concentric (6 − 4 = 2). It wraps what
          the reader looks *into*, and now there is exactly one of those. */}
      <div className="border-border-subtle min-h-0 flex-1 rounded-[6px] border bg-[#0D0E11] p-1">
        <div className="border-border-subtle bg-bg-surface relative h-full min-w-0 overflow-hidden rounded-[2px] border">
          {/* Hidden rather than unmounted, and absolutely placed at the same
              size so the renderer never sees a resize. The camera is a held
              state — orbiting to a cluster, checking the grid and coming back
              to a reset view would make the toggle cost something. */}
          <div
            className={showGrid ? 'invisible absolute inset-0' : 'h-full w-full'}
            inert={showGrid ? true : undefined}
          >
            <Brain
              onSelectLayer={onSelectLayer}
              paused={showGrid}
              progress={progress}
              selection={selection}
              status={status}
              trace={trace}
            />
          </div>

          {showGrid ? (
            <ResidualMap
              map={map}
              onSelectCell={onSelectCell}
              onSelectPosition={onSelectPosition}
              selection={selection}
              trace={trace}
            />
          ) : null}
        </div>
      </div>

      {loaded ? (
        <TokenStrip
          onSelect={onSelectPosition}
          selectedPosition={selection.position}
          steps={trace.steps}
        />
      ) : null}
    </div>
  )
}
