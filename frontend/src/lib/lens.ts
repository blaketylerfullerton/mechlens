// The one place the logit-lens classification and its colours are defined.
//
// The brain and the grid both colour things by "what is this layer holding at
// this position" — and if they each computed that themselves they would drift,
// which would make a colour mean two things in one screen. So both read this
// module. The residual-magnitude heatmap in TraceViewer is deliberately *not*
// routed through here: it encodes a different quantity (an L2 norm, not a
// classification) and shares no scale with these colours.
//
// Semantics follow backend/app/passes/lens.py, which computes the same three
// populations as per-layer curves (`top1_agreement_by_layer`, `echo_by_layer`).
// This module answers the per-(position, layer) version of that question.
//
// Deliberately free of runtime imports — types only — so the pure helpers can
// be executed directly by `node --experimental-strip-types` for verification.

import type { PassRecord, TokenStep, Trace } from './api-types'

/**
 * What a layer's logit-lens readout is holding at one token position.
 *
 * - `answer` — its top-1 is the token the model finally predicts here. The
 *   crystallisation the lens exists to show.
 * - `echo` — its top-1 is the token already *at* this position. Early
 *   residuals are dominated by the token embedding, so layers 0-9 "predict"
 *   the current token with high confidence on gemma-2-2b. Reading that as
 *   early certainty about an answer is exactly the mistake `echo_by_layer` was
 *   added to prevent.
 * - `other` — neither. Mid-depth churn, usually.
 */
export type LensClass = 'answer' | 'echo' | 'other'

export interface LayerClassification {
  layer: number
  klass: LensClass
  /** The lens top-1's probability at this layer: how strongly it holds it. */
  confidence: number
}

/**
 * Classify one (position, layer). `null` when that layer has no lens readout —
 * the pass did not run, or ran with `layers=[...]` and skipped this one. Never
 * a fabricated class: a missing readout is not evidence of anything.
 *
 * `answer` wins when a token is somehow both the final answer and the token
 * sitting here, because "this layer already holds the answer" is the stronger
 * and more useful reading. lens.py counts its two curves independently and so
 * does not need this precedence; a single per-cell class does.
 */
export function classifyLayer(step: TokenStep, layer: number): LayerClassification | null {
  const state = step.layers[layer]
  const lens = state?.logit_lens
  if (!lens || lens.top_k.length === 0) return null

  const top = lens.top_k[0]
  const finalAnswer = step.logits.top_k[0]?.token_id
  const tokenHere = step.token.token_id

  const klass: LensClass =
    top.token_id === finalAnswer ? 'answer' : top.token_id === tokenHere ? 'echo' : 'other'

  return { layer, klass, confidence: top.prob }
}

// -- bands ----------------------------------------------------------------

/** A contiguous run of layers drawn as one region. */
export interface Band {
  index: number
  startLayer: number
  endLayer: number
  layers: number[]
}

/**
 * Default band count. 26 layers as 26 bands would be wafer-thin stripes on a
 * 1.4-radius shell — unreadable and unclickable — so layers are binned, the
 * same call the archived brain-view design made for its rings.
 */
export const DEFAULT_BAND_COUNT = 7

/**
 * Partition `0..nLayers-1` into contiguous bands, every layer in exactly one,
 * no band empty. When the count does not divide evenly the remainder goes to
 * the earliest bands (so band sizes decrease by at most one, front to back),
 * and asking for more bands than layers yields one band per layer rather than
 * empty ones.
 */
export function bandLayers(nLayers: number, bandCount: number = DEFAULT_BAND_COUNT): Band[] {
  if (nLayers <= 0 || bandCount <= 0) return []

  const count = Math.min(bandCount, nLayers)
  const base = Math.floor(nLayers / count)
  const remainder = nLayers % count

  const bands: Band[] = []
  let next = 0
  for (let index = 0; index < count; index++) {
    const size = base + (index < remainder ? 1 : 0)
    const startLayer = next
    const endLayer = next + size - 1
    bands.push({
      index,
      startLayer,
      endLayer,
      layers: Array.from({ length: size }, (_, offset) => startLayer + offset),
    })
    next += size
  }
  return bands
}

export interface BandState {
  band: Band
  /** `null` when no layer in the band had a lens readout to classify. */
  klass: LensClass | null
  /** Mean confidence of the layers matching `klass`; 0 when `klass` is null. */
  confidence: number
  /** How many layers voted for each class. Zero-count classes are included. */
  counts: Record<LensClass, number>
  /** Per-layer detail, for the hover disclosure. Excludes undecoded layers. */
  layers: LayerClassification[]
  /** Layers in this band with no lens readout at all. */
  undecoded: number[]
}

const CLASSES: LensClass[] = ['answer', 'echo', 'other']

/**
 * Blend one band's layers into a single class and intensity.
 *
 * Majority class by layer count, ties broken toward the class held by the
 * higher layer — the more settled reading, and the same rule the archived
 * design chose. Intensity is the mean confidence of the layers that voted for
 * the winner, not of the whole band, so a confident majority is not dimmed by
 * the layers it outvoted.
 *
 * The blend is lossy by construction, which is why `counts`, `layers` and
 * `undecoded` come back with it: the caller is expected to disclose the
 * breakdown rather than let a mixed band's colour stand as the whole story.
 */
export function blendBand(step: TokenStep, band: Band): BandState {
  const layers: LayerClassification[] = []
  const undecoded: number[] = []

  for (const layer of band.layers) {
    const classification = classifyLayer(step, layer)
    if (classification === null) undecoded.push(layer)
    else layers.push(classification)
  }

  const counts: Record<LensClass, number> = { answer: 0, echo: 0, other: 0 }
  for (const { klass } of layers) counts[klass] += 1

  if (layers.length === 0) {
    return { band, klass: null, confidence: 0, counts, layers, undecoded }
  }

  // Ties toward the higher layer: scanning the band's own layers from the top
  // down, the first class holding the maximum count wins.
  const maximum = Math.max(...CLASSES.map((klass) => counts[klass]))
  let winner: LensClass = layers[layers.length - 1].klass
  for (let i = layers.length - 1; i >= 0; i--) {
    if (counts[layers[i].klass] === maximum) {
      winner = layers[i].klass
      break
    }
  }

  const matching = layers.filter((l) => l.klass === winner)
  const confidence = matching.reduce((sum, l) => sum + l.confidence, 0) / matching.length

  return { band, klass: winner, confidence, counts, layers, undecoded }
}

// -- colour ---------------------------------------------------------------

/**
 * Hue per class, in turns (0-1) so THREE's `Color.setHSL` takes them directly.
 *
 * Every hue is taken from the syntax palette the rest of the interface is
 * built from — no colour is invented outside it, so the blue that means
 * "settled" here is the same blue that marks a function in a code block and
 * rings a focused input.
 *
 * `answer` is `syntax-func` #82AAFF, the leading accent: this is the thing the
 * lens exists to show. `echo` is `syntax-const` #FFCB6B, the palette's
 * attention tone, and not `syntax-keyword` violet — violet marks a generated
 * token in the strip and cannot mean two things on one screen. `other` is a
 * near-grey that reads as "nothing to see here" without going invisible.
 *
 * `answer` and `other` sit in the same hue family, so they are told apart by
 * saturation, not hue alone. That is deliberate but not sufficient on its own:
 * every surface that paints a class also prints its name.
 */
export const CLASS_HUE: Record<LensClass, number> = {
  answer: 220.8 / 360,
  echo: 38.9 / 360,
  other: 217 / 360,
}

export interface Hsl {
  h: number
  s: number
  l: number
}

/** The neutral fill for a band with no lens data. Not on any class ramp. */
export const NEUTRAL: Hsl = { h: 221.5 / 360, s: 0.13, l: 0.24 }

/**
 * One ramp for every class: confidence drives saturation and lightness, hue
 * carries the class. So hue answers "what is this layer holding" and
 * brightness answers "how strongly", independently — which is what lets the
 * same colour mean the same thing on the brain and in the grid.
 *
 * Each ramp lands exactly on its palette colour at full confidence (#82AAFF is
 * hsl(220.8 100% 75.5%), #FFCB6B is hsl(38.9 100% 71%)), so a fully confident
 * band is the accent itself rather than something near it.
 */
export function classColor(klass: LensClass | null, confidence: number): Hsl {
  if (klass === null) return NEUTRAL

  const t = Math.min(Math.max(confidence, 0), 1)
  if (klass === 'other') {
    return { h: CLASS_HUE.other, s: 0.06 + t * 0.05, l: 0.26 + t * 0.1 }
  }
  const top = klass === 'answer' ? 0.755 : 0.71
  return {
    h: CLASS_HUE[klass],
    s: 0.55 + t * 0.45,
    l: 0.4 + t * (top - 0.4),
  }
}

export function cssColor({ h, s, l }: Hsl): string {
  return `hsl(${(h * 360).toFixed(1)} ${(s * 100).toFixed(1)}% ${(l * 100).toFixed(1)}%)`
}

// -- trace-level lens facts -----------------------------------------------

export function lensPass(trace: Trace): PassRecord | null {
  return trace.passes.find((pass) => pass.name === 'lens') ?? null
}

/**
 * True when there is at least one lens readout to draw. Checked against the
 * steps rather than the pass list, because that is what the renderers actually
 * consume — a pass record with no readouts would still colour nothing.
 */
export function hasLensData(trace: Trace): boolean {
  return trace.steps.some((step) => step.layers.some((layer) => layer.logit_lens !== null))
}

/**
 * The depth where the answer settles for the trace as a whole: the first layer
 * at which at least half of positions already hold the model's final answer.
 *
 * `null` rather than a number when the lens pass did not run, or when it
 * recorded `-1` — its way of saying the answer never crosses that line. A
 * marker drawn there anyway would be an invention.
 */
export function crossoverLayer(trace: Trace): number | null {
  const stat = lensPass(trace)?.stats.crossover_layer
  if (typeof stat !== 'number' || stat < 0 || stat >= trace.n_layers) return null
  return Math.round(stat)
}

/** Which band contains `layer`, or -1. */
export function bandOfLayer(bands: Band[], layer: number): number {
  return bands.findIndex((band) => layer >= band.startLayer && layer <= band.endLayer)
}
