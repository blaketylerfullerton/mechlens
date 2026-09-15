import { request } from '@/lib/api-client'
import type { Options } from './types'

// Older running backends report the TransformerLens name but not its repository.
// Match only known names; an unknown model must never count as the selected one.
const legacyRepositories: Record<string, string> = {
  'gemma-2-2b': 'google/gemma-2-2b',
  gpt2: 'openai-community/gpt2',
  'tiny-stories-1M': 'roneneldan/TinyStories-1M',
}

export async function getTrainingOptions(): Promise<Options> {
  const available = await request<Options>('/training/options')
  return {
    ...available,
    model_repository: available.readiness !== 'ready' ? null
      : available.model_repository !== undefined ? available.model_repository
      : legacyRepositories[available.model] ?? null,
  }
}
