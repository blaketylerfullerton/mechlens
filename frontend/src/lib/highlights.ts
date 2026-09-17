import type { Trace } from '@/lib/api-types'

/**
 * What the highlights strip surfaces: the generated token the model decided
 * latest, what else it considered, and which named concepts fired there.
 *
 * Everything here is computed from data the trace already holds — the
 * strip's job is to put the interesting cells in front of a reader who does
 * not yet know where to look. When a measurement is missing (the lens pass
 * has not run, no features are labelled), that highlight is absent rather
 * than approximated.
 */
export interface TraceHighlights {
  /**
   * The first layer at which this token's top-1 guess equals the token that
   * was actually generated and stays equal through the final layer. `null`
   * when the model never converged on it (it was sampled below rank 1) or
   * when the lens pass did not reach this position.
   */
  decisionLayer: number | null
  position: number
  text: string
  /** The strongest generated-token alternative to what it wrote, if any. */
  alternative: { text: string; prob: number } | null
  /** Labels of the strongest features firing at (decisionLayer, position). */
  concepts: { index: number; label: string; layer: number }[]
}

/**
 * For one generated step, the earliest layer whose top-1 lens readout is the
 * generated token and never flips back. The answer a "when did it make up its
 * mind" question actually wants is the *converged* layer, not the first flash
 * of the right answer that a deeper layer could still override.
 */
export function decisionLayer(trace: Trace, position: number): number | null {
  const step = trace.steps[position]
  if (!step) return null
  const wanted = step.token.token_id
  // Layers without a lens readout are skipped over — the pass runs a subset,
  // so absent means "not measured", not "did not match".
  const readouts = step.layers.filter((state) => state.logit_lens !== null)
  // Walk back past the correct tail: the first layer of that run is where the
  // guess settled and never flipped again. A mid-sequence flip just makes the
  // tail shorter; never converging leaves nothing to return.
  let i = readouts.length
  while (i > 0 && readouts[i - 1].logit_lens?.top_k[0]?.token_id === wanted) i--
  return i < readouts.length ? readouts[i].layer : null
}

/**
 * The generated token whose decision came latest — the one place in this
 * trace where the model was still choosing deepest into the network. Ties go
 * to the earlier position. Falls back to the last generated token when the
 * lens pass produced nothing usable.
 */
export function mostInteresting(trace: Trace): TraceHighlights | null {
  const generated = trace.steps.filter(
    (step) => step.token.source === 'generated' && step.step > 0,
  )
  if (generated.length === 0) return null

  let best = generated[generated.length - 1]
  let bestLayer = -1
  for (const step of generated) {
    const layer = decisionLayer(trace, step.step)
    if (layer !== null && layer > bestLayer) {
      bestLayer = layer
      best = step
    }
  }

  const layer = decisionLayer(trace, best.step)
  const top = best.logits.top_k
  const runnerUp =
    top[0]?.token_id === best.token.token_id ? top[1] : top[0]

  const concepts: TraceHighlights['concepts'] = []
  if (layer !== null) {
    const state = best.layers.find((candidate) => candidate.layer === layer)
    for (const feature of state?.features.slice(0, 8) ?? []) {
      const label = trace.labels[`${layer}/${feature.index}`]?.text
      if (label) concepts.push({ index: feature.index, label, layer })
      if (concepts.length === 3) break
    }
  }

  return {
    decisionLayer: layer,
    position: best.step,
    text: best.token.text,
    alternative: runnerUp ? { text: runnerUp.text, prob: runnerUp.prob } : null,
    concepts,
  }
}
