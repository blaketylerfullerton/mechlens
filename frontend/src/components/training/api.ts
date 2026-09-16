import { useEffect, useState } from 'react'
import { request } from '@/lib/api-client'
import type { ModelDownloads, Options } from './types'

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


/** Base model weights already in the Hugging Face cache, or null while loading.
 *
 * Read-only, and deliberately forgiving: a backend too old to serve this route
 * simply leaves the size hints off rather than failing the page around it.
 */
export function useModelDownloads(active: boolean, refresh = 0): ModelDownloads | null {
  const [downloads, setDownloads] = useState<ModelDownloads | null>(null)
  useEffect(() => {
    if (!active) return
    let stopped = false
    void request<ModelDownloads>('/training/models')
      .then((result) => { if (!stopped) setDownloads(result) })
      .catch(() => { if (!stopped) setDownloads(null) })
    return () => { stopped = true }
  }, [active, refresh])
  return downloads
}
