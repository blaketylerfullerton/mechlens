// The one place the logit-lens classification and its colours are defined.
//
// The brain used to colour layer bands from this module; it no longer reads
// lens data at all — it shows SAE features, which nothing else displays, and
// the bands are gone. What stays here is the per-(position, layer) question
// "what is this layer holding here", defined once so that whichever surface
// renders it cannot drift from any other. The trace grid does not colour its
// cells by it today — that grid encodes residual L2 magnitude, a different
// quantity on a scale these colours do not share — so the classification
// currently has no renderer.
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

/** The neutral fill for a cell with no lens readout. Not on any class ramp. */
export const NEUTRAL: Hsl = { h: 221.5 / 360, s: 0.13, l: 0.24 }

/**
 * One ramp for every class: confidence drives saturation and lightness, hue
 * carries the class. So hue answers "what is this layer holding" and
 * brightness answers "how strongly", independently — which is what lets the
 * same colour mean the same thing wherever it is drawn.
 *
 * Each ramp lands exactly on its palette colour at full confidence (#82AAFF is
 * hsl(220.8 100% 75.5%), #FFCB6B is hsl(38.9 100% 71%)), so a fully confident
 * cell is the accent itself rather than something near it.
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
