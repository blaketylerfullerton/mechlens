import { useMemo, useState } from 'react'

import type { Trace } from '@/lib/api-types'
import type { LayerClassification } from '@/lib/lens'
import { classifyLayer, crossoverLayer, hasLensData } from '@/lib/lens'

import type { ColorMode } from './colors'

export type ResidualMapModel = {
  mode: ColorMode
  setMode: (mode: ColorMode) => void
  lensAvailable: boolean
  /** Every cell's class, indexed [position][layer] — the grid's own nesting. */
  classifications: (LayerClassification | null)[][]
  /** The layer where the answer settles, or null when the lens pass did not run. */
  crossover: number | null
  minimumResidualNorm: number
  maximumResidualNorm: number
}

/**
 * Everything the map derives from a trace, computed once per trace rather than
 * once per render of a grid that can be hundreds of cells wide.
 *
 * Called unconditionally from `TraceViewer`, before its empty-state return, so
 * the hook order never depends on whether a trace has arrived.
 */
export function useResidualMap(trace: Trace | null): ResidualMapModel {
  // Residual magnitude is the default because every trace has it; the lens
  // classification exists only when the lens pass ran.
  const [colorMode, setMode] = useState<ColorMode>('residual')

  const lensAvailable = trace !== null && hasLensData(trace)

  const [minimumResidualNorm, maximumResidualNorm] = useMemo(() => {
    if (!trace) return [0, 0]
    const norms = trace.steps.flatMap((step) => step.layers.map((layer) => layer.resid_norm))
    return [Math.min(...norms), Math.max(...norms)]
  }, [trace])

  const classifications = useMemo<(LayerClassification | null)[][]>(() => {
    if (!trace || !lensAvailable) return []
    return trace.steps.map((step) =>
      Array.from({ length: trace.n_layers }, (_, layer) => classifyLayer(step, layer)),
    )
  }, [trace, lensAvailable])

  const crossover = useMemo(() => (trace === null ? null : crossoverLayer(trace)), [trace])

  return {
    // A trace with no lens data cannot be in lens mode, whatever the toggle was
    // left on by the trace before it.
    mode: lensAvailable ? colorMode : 'residual',
    setMode,
    lensAvailable,
    classifications,
    crossover,
    minimumResidualNorm,
    maximumResidualNorm,
  }
}
