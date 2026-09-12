import type { LayerClassification, LensClass } from '@/lib/lens'
import { NEUTRAL, classColor, cssColor } from '@/lib/lens'

/**
 * What the map's colour encodes. Two different quantities, never mixed: a
 * continuous L2 norm, or a three-way classification of the lens readout. They
 * share no scale, so they are modes rather than layers of one picture.
 */
export type ColorMode = 'residual' | 'lens'

/**
 * What each lens class means, spelled out wherever its colour appears.
 *
 * `echo` earns the longest gloss because it is the trap this classification
 * exists to expose: early layers on gemma-2-2b "predict" the token already at
 * the position, because the residual is still mostly that token's embedding.
 * A reader who takes that for early certainty has learnt the opposite of what
 * the trace shows.
 */
export const CLASS_COPY: Record<LensClass, string> = {
  answer: 'holds the final answer',
  echo: 'echoes the token here',
  other: 'neither',
}

export const CLASSES: LensClass[] = ['answer', 'echo', 'other']

/**
 * The residual-magnitude ramp: one hue, `syntax-func`, from the code surface
 * up to the accent itself.
 *
 * Single-hue on purpose. A rainbow ramp invents colours the palette does not
 * contain and implies category boundaries where there is only a continuous L2
 * norm. The exact endpoints are printed beside the map, because a colour a
 * reader cannot convert back to a number is decoration.
 */
export function heatColor(value: number, maximum: number): { backgroundColor: string } {
  const ratio = maximum > 0 ? Math.min(value / maximum, 1) : 0
  return {
    backgroundColor: `hsl(220.8 ${(10 + ratio * 90).toFixed(1)}% ${(9 + ratio * 66.5).toFixed(1)}%)`,
  }
}

/** The cell's fill in lens mode. A cell with no readout is neutral, not a class. */
export function lensColor(classification: LayerClassification | null): { backgroundColor: string } {
  return {
    backgroundColor: cssColor(
      classification === null ? NEUTRAL : classColor(classification.klass, classification.confidence),
    ),
  }
}

/** The screen-reader sentence for one cell, in whichever mode is on screen. */
export function describeClassification(classification: LayerClassification): string {
  const confidence =
    classification.confidence < 0.005 ? 'under 1' : (classification.confidence * 100).toFixed(0)
  return `${classification.klass}, ${CLASS_COPY[classification.klass]}, at ${confidence}%`
}
