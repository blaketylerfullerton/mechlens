import type { Trace } from '@/lib/api-types'

import { type ColorMode, describeClassification, heatColor, lensColor } from './colors'
import { formatNumber } from './format'
import { MapLegend } from './MapLegend'
import type { ResidualMapModel } from './useResidualMap'

type Selection = { layer: number; position: number }

/**
 * The left edge of the grid. It carries the crossover marker, the one
 * trace-level number the map states outright.
 */
function LayerAxisCell({ layer, crossover }: { layer: number; crossover: number | null }) {
  const isCrossover = layer === crossover
  return (
    <div
      className={`bg-bg-surface border-border-subtle sticky left-0 z-10 border-b px-2 py-1 text-right font-mono text-[10px] tabular-nums ${
        isCrossover ? 'text-text-primary' : 'text-text-disabled'
      }`}
    >
      L{layer}
      {/* The crossover layer, marked on the axis it belongs to. One trace-level
          number, drawn once, not a class per cell — see the note under the map. */}
      {isCrossover ? (
        <span className="text-fn ml-1" title="The answer settles here">
          <span aria-hidden="true">◀</span>
          <span className="sr-only"> the answer settles here</span>
        </span>
      ) : null}
    </div>
  )
}

/**
 * The residual grid: layers down, token positions across, colour carrying
 * whichever quantity the mode names. Every cell is a button — the selection it
 * sets is what the inspector column reads.
 */
export function ResidualMap({
  trace,
  selection,
  onSelectCell,
  onSelectPosition,
  map,
}: {
  trace: Trace
  selection: Selection
  onSelectCell: (layer: number, position: number) => void
  onSelectPosition: (position: number) => void
  map: ResidualMapModel
}) {
  const { mode, classifications, crossover, maximumResidualNorm } = map
  const gridColumns = `4rem repeat(${trace.steps.length}, minmax(2rem, 1fr))`

  return (
    /* No frame of its own any more: this fills the stage's frame, which is the
       same frame the brain fills, because it is the same object seen another
       way. Two framed grids side by side were two primary objects, and the
       narrower of them was never wide enough to read. */
    <section className="flex h-full min-h-0 flex-col">
      <div className="shrink-0">
        <MapLegend
          lensAvailable={map.lensAvailable}
          maximumResidualNorm={map.maximumResidualNorm}
          minimumResidualNorm={map.minimumResidualNorm}
          mode={mode}
          onChangeMode={map.setMode}
        />
      </div>

      {/* The grid and the notes under it scroll together: the caveats say what
          the colours above mean, and a caveat parked in another column is a
          caveat nobody read. */}
      <div className="mask-fade-b min-h-0 flex-1 overflow-auto p-3">
        <div className="min-w-max" style={{ display: 'grid', gridTemplateColumns: gridColumns }}>
          <div className="bg-bg-surface text-text-tertiary sticky left-0 z-10 px-2 py-2 text-right text-[10px] font-medium tracking-[0.04em] uppercase">
            layer
          </div>
          {trace.steps.map((step) => (
            <button
              aria-label={`Select token ${step.step}`}
              className={`border-border-subtle border-b px-1 py-2 font-mono text-[10px] tabular-nums transition-colors duration-150 ${
                step.step === selection.position
                  ? 'text-fn'
                  : 'text-text-disabled hover:text-text-secondary'
              }`}
              key={step.step}
              onClick={() => onSelectPosition(step.step)}
              type="button"
            >
              {step.step}
            </button>
          ))}

            {Array.from({ length: trace.n_layers }, (_, layer) => (
              <div className="contents" key={layer}>
                <LayerAxisCell crossover={crossover} layer={layer} />
                {trace.steps.map((step, position) => {
                  const state = step.layers[layer]
                  const isSelected =
                    selection.layer === layer && selection.position === step.step
                  const classification = classifications[position]?.[layer] ?? null
                  const reading =
                    mode === 'lens'
                      ? classification === null
                        ? 'no lens readout'
                        : describeClassification(classification)
                      : `residual norm ${formatNumber(state.resid_norm)}`

                  return (
                    <button
                      aria-label={`Layer ${layer}, token ${step.step}, ${reading}`}
                      aria-pressed={isSelected}
                      className={`m-px min-h-7 rounded-xs border transition-colors duration-150 ${
                        isSelected
                          ? 'border-text-primary z-10'
                          : 'hover:border-border-strong border-transparent'
                      }`}
                      key={`${layer}-${step.step}`}
                      onClick={() => onSelectCell(layer, step.step)}
                      style={
                        mode === 'lens'
                          ? lensColor(classification)
                          : heatColor(state.resid_norm, maximumResidualNorm)
                      }
                      type="button"
                    >
                      <span className="sr-only">{reading}</span>
                    </button>
                  )
                })}
              </div>
            ))}
        </div>

        <div className="mt-4">
          <MapNotes crossover={crossover} mode={mode} />
        </div>
      </div>
    </section>
  )
}

/**
 * What the map does and does not claim, in whichever mode is on screen. It
 * lives under the grid because a caveat a reader meets before there is
 * anything to apply it to is not read.
 */
export function MapNotes({ mode, crossover }: { mode: ColorMode; crossover: number | null }) {
  if (mode !== 'lens') {
    return (
      <p className="text-text-tertiary max-w-[70ch] text-[12px] leading-[1.5]">
        This view shows observed model states. Residual magnitude is not a semantic score or a
        causal explanation.
      </p>
    )
  }

  return (
    <div className="text-text-tertiary max-w-[70ch] space-y-1.5 text-[12px] leading-[1.5]">
      <p>
        Each cell is that layer&apos;s logit-lens top-1 token, classified: the token the model
        finally emits here, the token already at this position, or neither. Colour intensity is
        the readout&apos;s own probability.
      </p>
      <p>
        {/* The one reading error this classification exists to prevent. */}
        An <span className="text-text-secondary font-mono">echo</span> is not early certainty.
        Early layers hold the current token because the residual is still mostly that
        token&apos;s embedding, so a wall of echo near the input is the expected behaviour of the
        residual stream, not the model knowing an answer.
      </p>
      <p>
        {crossover === null
          ? 'No crossover layer is marked: the lens pass recorded none, so the answer never reached half the positions.'
          : `The marker on the axis is L${crossover}, the first layer where at least half the positions already hold the final answer. It is one number for the whole trace, not a per-cell claim.`}{' '}
        A lens readout is a decode of an intermediate state, not a decision the model made at
        that layer.
      </p>
    </div>
  )
}
