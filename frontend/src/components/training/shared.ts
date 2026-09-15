export const button = 'border-border-strong rounded border px-4 py-2.5 text-sm transition-colors hover:border-fn disabled:cursor-not-allowed disabled:opacity-50'
export const primary = `${button} bg-text-primary text-bg-base font-medium`
export const field = 'border-border-strong bg-bg-base text-text-primary mt-2 w-full rounded border px-3 py-2.5 text-sm focus:border-fn'
export const defaults = { layer: 12, features: 4096, training_tokens: 131072, dataset: 'tiny-stories', training_text: '', evaluation_text: '', batch_size: 256, context_size: 128, learning_rate: 0.0003, l1_coefficient: 0.1, seed: 42, evaluation_sequences: 64, compare_huggingface: false }
export type TrainingDraft = typeof defaults


export function number(value: number | undefined, digits = 2) {
  return value === undefined || !Number.isFinite(value) ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

