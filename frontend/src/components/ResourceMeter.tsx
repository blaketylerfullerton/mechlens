import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'

import { useStats } from '@/hooks/useStats'
import type { StatsResponse } from '@/lib/api-types'

const GB = 1024 ** 3

// How much of each new GPU reading to believe. nvidia-smi reports the share of
// a ~1s window that had work on it, so a GPU doing steady batched work still
// reads 0 whenever the poll lands between kernel launches — the raw figure
// swings 90 to 0 and back with nothing wrong. At 0.35 over a 2s poll the line
// settles in about five seconds, which is slow enough to read and fast enough
// to show a trace starting.
const GPU_SMOOTHING = 0.35

function gb(bytes: number): string {
  return (bytes / GB).toFixed(1)
}

/**
 * A quantity drawn to scale, sized for the header.
 *
 * The same construction as `trace/primitives.tsx`'s `Bar` — a 1px rule in a
 * `ch`-measured track — at a narrower track, and not shared with it because
 * that one is a figure in a table and this is chrome beside a title. The
 * unfilled remainder is drawn too, in `rule`, so the mark reads as a fraction
 * of a known pool rather than a bar of arbitrary length. Colour is by tier,
 * not category: the rule goes accent until the pool is nearly gone, and the
 * number beside it is the actual signal either way, so nothing here depends
 * on the colour being seen.
 */
function Track({ ratio }: { ratio: number }) {
  const pressed = ratio >= 0.9
  const clamped = Math.max(Math.min(ratio, 1), 0)
  return (
    <span aria-hidden="true" className="bg-rule/40 inline-block h-px w-[5ch] shrink-0 align-middle">
      <span
        className={`block h-px ${pressed ? 'bg-err' : 'bg-fn'}`}
        style={{ width: `${(Math.max(clamped, 0.02) * 100).toFixed(1)}%` }}
      />
    </span>
  )
}

function Field({ children, label }: { children: ReactNode; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-text-tertiary">{label}</span>
      {children}
    </span>
  )
}

/**
 * The GPU figure, averaged rather than sampled.
 *
 * An exponential moving average, and labelled as one at the call site: it is a
 * derived number, not the instantaneous reading, and §10 forbids printing a
 * figure nobody computed under a name that implies a raw measurement. Keyed on
 * the whole stats object rather than the percent, because two identical
 * readings in a row are still two readings and the average has to keep
 * converging through them.
 *
 * Resets to null the moment the backend stops answering. Decaying a stale
 * average toward zero would draw an idle GPU on a machine we cannot see.
 */
function useSmoothedGpu(stats: StatsResponse | null): number | null {
  const [smoothed, setSmoothed] = useState<number | null>(null)

  useEffect(() => {
    const next = stats?.gpu_util_pct ?? null
    setSmoothed((prev) => {
      if (next === null) return null
      if (prev === null) return next
      return prev + GPU_SMOOTHING * (next - prev)
    })
  }, [stats])

  return smoothed
}

/**
 * Host memory and GPU load, as a single mono line beside the title.
 *
 * Three figures, because three different things go wrong:
 *
 * - **mem** — the whole machine's pool. What runs out.
 * - **torch** — what *this* process is holding of it. The figure that says
 *   whether a trace that died was mechlens's fault or the browser's.
 * - **gpu** — utilisation, the only one of the three that answers "is it
 *   actually working right now".
 *
 * On unified-memory parts (GB10/Jetson) the CPU and GPU address one physical
 * pool, so the backend measures that and this draws one gauge labelled as
 * such. Drawing the usual RAM-and-VRAM pair there would put 257 GB on screen
 * for a 128 GB machine — a fabricated number, which §10 forbids outright.
 *
 * Renders nothing at all when the backend has not answered: an unreachable
 * service is already reported by the surfaces that matter, and a meter
 * showing zeros would claim an idle machine rather than no reading.
 */
export function ResourceMeter() {
  const stats = useStats()
  const gpu = useSmoothedGpu(stats)
  if (!stats) return null

  const { ram_total_bytes: total, ram_used_bytes: used } = stats
  const ratio = total > 0 ? used / total : 0
  const torch = stats.torch_allocated_bytes

  return (
    <div
      className="text-text-secondary flex items-center gap-3 font-mono text-[11px] tabular-nums"
      role="status"
      aria-label={
        `Memory ${gb(used)} of ${gb(total)} gigabytes used` +
        (stats.unified ? ', one unified CPU and GPU pool' : '') +
        (torch !== null ? `. This process holds ${gb(torch)} gigabytes` : '') +
        (gpu !== null ? `. GPU ${Math.round(gpu)} percent utilised, five-second average` : '')
      }
    >
      <Field label={stats.unified ? 'mem·unified' : 'mem'}>
        <Track ratio={ratio} />
        <span className="text-text-primary">
          {gb(used)}/{gb(total)}
        </span>
        <span className="text-text-tertiary">GB</span>
      </Field>

      {torch !== null ? (
        <Field label="torch">
          <Track ratio={total > 0 ? torch / total : 0} />
          <span className="text-text-primary">{gb(torch)}</span>
          <span className="text-text-tertiary">GB</span>
        </Field>
      ) : null}

      {gpu !== null ? (
        <Field label="gpu">
          <Track ratio={gpu / 100} />
          <span className="text-text-primary">{Math.round(gpu)}</span>
          <span className="text-text-tertiary">%</span>
        </Field>
      ) : null}
    </div>
  )
}
