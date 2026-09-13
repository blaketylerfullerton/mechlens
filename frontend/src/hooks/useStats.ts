import { useEffect, useState } from 'react'

import { getStats } from '@/lib/api-client'
import type { StatsResponse } from '@/lib/api-types'

// Slow enough that the numbers are readable rather than flickering, fast
// enough to move visibly while a trace runs. The route is a psutil read plus
// a cached nvidia-smi, so the cost of the poll is the round trip.
const POLL_MS = 2000

/**
 * Polls GET /stats for as long as the tab is visible.
 *
 * Returns null rather than zeros when the backend is unreachable: the readout
 * has to be able to say "no reading" instead of drawing an empty machine,
 * which is the one wrong answer available here. The error is swallowed
 * because this is furniture — the backend being down is already reported,
 * loudly, by the surfaces that matter.
 */
export function useStats(): StatsResponse | null {
  const [stats, setStats] = useState<StatsResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const tick = () => {
      // A hidden tab polls nothing: this is a background reading, and a
      // window left open overnight should not keep a request loop alive.
      if (document.visibilityState !== 'visible') return
      getStats()
        .then((res) => {
          if (!cancelled) setStats(res)
        })
        .catch(() => {
          if (!cancelled) setStats(null)
        })
    }

    tick()
    timer = window.setInterval(tick, POLL_MS)
    document.addEventListener('visibilitychange', tick)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [])

  return stats
}
