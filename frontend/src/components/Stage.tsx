import { useState, type ReactNode } from 'react'

import { Brain, type SelectionVia } from '@/components/Brain'
import { ResourceMeter } from '@/components/ResourceMeter'
import { ResidualMap } from '@/components/trace/ResidualMap'
import { TraceResponse } from '@/components/trace/TraceResponse'
import { TraceHeader } from '@/components/trace/TraceHeader'
import { useResidualMap } from '@/components/trace/useResidualMap'
import type { RunState } from '@/hooks/useTrace'
import type { JobProgress, Trace } from '@/lib/api-types'

type View = 'brain' | 'grid'

type Selection = { layer: number; position: number; via?: SelectionVia }

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
          className={`px-2 py-1 transition-colors duration-150 first:rounded-l-[2px] last:rounded-r-[2px] ${
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
  customDictionary?: boolean
  composer: ReactNode
  inspectorOpen: boolean
  onToggleInspector: () => void
  onReset: () => void
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
  customDictionary = false,
  composer,
  inspectorOpen,
  onToggleInspector,
  onReset,
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
  const customSAE = customDictionary || (trace?.passes.some((pass) => pass.name === 'sae' && String(pass.params.release).startsWith('local/')) ?? false)
  const showGrid = loaded && (view === 'grid' || customSAE)

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {loaded ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <TraceHeader trace={trace} status={status} progress={progress} />
          <div className="flex items-center gap-3">
            <ResourceMeter />
            {customSAE ? <span className="text-text-tertiary text-xs">Custom SAE · grid · no atlas or labels</span> : <ViewToggle onChange={setView} view={view} />}
            <button type="button" aria-expanded={inspectorOpen} aria-controls="trace-inspector"
              onClick={onToggleInspector}
              className="text-text-secondary hover:text-text-primary rounded-xs px-2 py-1 text-[12px]">
              Inspector
            </button>
            {/* Last in the row, and only ever beside a finished trace: it
                discards the primary object, so it sits furthest from the
                controls that merely look at it. Not destructive enough for
                `err` — the trace is a reproducible run of a saved prompt, not
                authored work — so it stays in the grey tier with the rest of
                the chrome. */}
            <button type="button" onClick={onReset}
              className="text-text-secondary hover:text-text-primary rounded-xs px-2 py-1 text-[12px]">
              New trace
            </button>
          </div>
        </div>
      ) : (
        <div className="flex min-h-7 flex-wrap items-center gap-x-4 gap-y-1">
          <h1 className="text-text-primary text-[13px] font-medium">mechlens</h1>
          <span role="status" className="text-text-secondary text-[12px]">
            {status === 'warming' ? 'Loading model' : status === 'pending' ? 'Trace queued' : status === 'running' ? 'Generating' : status === 'error' ? 'Ready to retry' : 'Ready'}
          </span>
          {/* Furthest right, and present before a trace is: the weights load
              is the longest wait in the product, and it is the window where
              "is this machine doing anything" is least answerable from the
              rest of the screen. */}
          <div className="ml-auto"><ResourceMeter /></div>
        </div>
      )}

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
            {customSAE ? <div className="text-text-secondary flex h-full items-center justify-center p-8 text-center text-sm">Enter a prompt to inspect this dictionary’s features in the layer grid. No atlas or labels have been generated for it.</div> : <Brain
              onSelectLayer={onSelectLayer}
              paused={showGrid}
              progress={progress}
              selection={selection}
              status={status}
              trace={trace}
            />}
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
        <TraceResponse
          trace={trace}
          running={status === 'running'}
          followingLatest={followingLatest}
          onFollowLatest={onFollowLatest}
          onSelect={onSelectPosition}
          selection={selection}
        />
      ) : (
        <div className="border-border-subtle flex shrink-0 items-center border-t pt-3">
          <p className="text-text-secondary text-[13px]">Enter a prompt to watch the response form and explore what happens inside the model.</p>
        </div>
      )}
      <div className="shrink-0">{composer}</div>
    </div>
  )
}
