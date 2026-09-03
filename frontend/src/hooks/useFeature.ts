import { useEffect, useState } from 'react'
import { getFeature } from '@/lib/api-client'
import type { FeatureResponse } from '@/lib/api-types'

export interface UseFeatureResult {
  feature: FeatureResponse | null
  loading: boolean
  error: string | null
}

// One-shot fetch of GET /feature/{layer}/{idx}. Pass null for either arg to
// skip the request (e.g. nothing selected yet).
export function useFeature(layer: number | null, idx: number | null): UseFeatureResult {
  const [feature, setFeature] = useState<FeatureResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (layer === null || idx === null) {
      setFeature(null)
      setError(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    getFeature(layer, idx)
      .then((res) => {
        if (!cancelled) setFeature(res)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [layer, idx])

  return { feature, loading, error }
}
