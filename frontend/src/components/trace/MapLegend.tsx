import { NEUTRAL, classColor, cssColor } from '@/lib/lens'

import { CLASSES, CLASS_COPY, type ColorMode, heatColor } from './colors'
import { formatNumber } from './format'

/**
 * The ramp with its endpoints as numbers: a colour nobody can convert back to
 * a value is decoration.
 */
function ResidualRamp({ minimum, maximum }: { minimum: number; maximum: number }) {
  return (
    <div className="text-text-tertiary flex items-center gap-2 font-mono text-[11px] tabular-nums">
      <span>{formatNumber(minimum)}</span>
      <span aria-hidden="true" className="flex h-2.5 w-24">
        {Array.from({ length: 12 }, (_, step) => (
          <span className="flex-1" key={step} style={heatColor(step / 11, 1)} />
        ))}
      </span>
      <span>{formatNumber(maximum)}</span>
    </div>
  )
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="size-2 shrink-0 rounded-full"
      style={{ backgroundColor: color }}
    />
  )
}

/**
 * Three classes, each colour printed beside its own name — the colour never
 * carries the meaning on its own.
 */
function LensKey() {
  return (
    <ul className="text-text-tertiary flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
      {CLASSES.map((klass) => (
        <li className="flex items-center gap-1.5" key={klass}>
          <Swatch color={cssColor(classColor(klass, 1))} />
          <span className="text-text-secondary font-mono">{klass}</span>
          <span>{CLASS_COPY[klass]}</span>
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <Swatch color={cssColor(NEUTRAL)} />
        <span>no readout</span>
      </li>
    </ul>
  )
}

/**
 * Two quantities, one grid. The control names which is on screen rather than
 * leaving the colours to be guessed at.
 */
function ModeToggle({
  mode,
  lensAvailable,
  onChange,
}: {
  mode: ColorMode
  lensAvailable: boolean
  onChange: (mode: ColorMode) => void
}) {
  return (
    <div className="border-border-subtle bg-bg-elevated flex rounded-[2px] border text-[11px]">
      {(['residual', 'lens'] as const).map((option) => {
        const disabled = option === 'lens' && !lensAvailable
        return (
          <button
            className={`px-2 py-1 font-mono transition-colors duration-150 first:rounded-l-[2px] last:rounded-r-[2px] ${
              option === mode
                ? 'bg-fn/[0.10] text-text-primary'
                : disabled
                  ? 'text-text-disabled cursor-not-allowed'
                  : 'text-text-tertiary hover:bg-white/[0.02]'
            }`}
            disabled={disabled}
            key={option}
            onClick={() => onChange(option)}
            title={
              disabled
                ? 'This trace has no logit-lens readouts — the lens pass did not run.'
                : undefined
            }
            type="button"
          >
            {option}
          </button>
        )
      })}
    </div>
  )
}

/** The map's header: what it is, what the colour means, and how to switch it. */
export function MapLegend({
  mode,
  lensAvailable,
  onChangeMode,
  minimumResidualNorm,
  maximumResidualNorm,
}: {
  mode: ColorMode
  lensAvailable: boolean
  onChangeMode: (mode: ColorMode) => void
  minimumResidualNorm: number
  maximumResidualNorm: number
}) {
  return (
    <div className="border-border-subtle space-y-2 border-b p-3">
      <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start">
        <div className="min-w-0">
          <h2 className="text-text-primary text-[13px] font-medium">Residual-stream map</h2>
          <p className="text-text-tertiary mt-1 text-[12px]">
            {mode === 'residual'
              ? 'L2 norm of the residual at each layer and token, as captured.'
              : 'What each layer’s logit-lens readout holds at each token.'}
          </p>
        </div>
        <ModeToggle lensAvailable={lensAvailable} mode={mode} onChange={onChangeMode} />
      </div>

      {mode === 'residual' ? (
        <ResidualRamp maximum={maximumResidualNorm} minimum={minimumResidualNorm} />
      ) : (
        <LensKey />
      )}
    </div>
  )
}
