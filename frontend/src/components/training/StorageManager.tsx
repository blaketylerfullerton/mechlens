import { useEffect, useState } from 'react'
import { request } from '@/lib/api-client'
import { button, number } from './shared'
import type { Storage } from './types'

function bytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${number(value / 1024, 1)} KB`
  if (value < 1024 ** 3) return `${number(value / 1024 ** 2, 1)} MB`
  return `${number(value / 1024 ** 3, 2)} GB`
}

export function StorageManager({ active }: { active: boolean }) {
  const [storage, setStorage] = useState<Storage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  async function refresh() {
    try {
      setStorage(await request<Storage>('/training/storage'))
    } catch (err) { setError(String(err)) }
  }

  useEffect(() => { if (active) void refresh() }, [active])

  async function remove(id: string) {
    setBusy(id); setError(null)
    try {
      await request(`/training/runs/${id}`, { method: 'DELETE' })
      await refresh()
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }

  if (!storage) return null

  return <section aria-label="Saved dictionaries" className="mt-3">
    {error ? <p role="alert" className="text-err mb-3 text-sm">{error}</p> : null}
    <p className="text-text-secondary mb-3 text-xs">
      {storage.runs.length} dictionaries · {bytes(storage.total_bytes)} used · {bytes(storage.free_bytes)} free on disk
    </p>
    {storage.runs.length ? <div className="grid gap-2">
      {storage.runs.map((run) => <div key={run.id} className="border-border-subtle flex flex-wrap items-center justify-between gap-3 rounded border px-3 py-2 text-sm">
        <span>
          <span className="font-mono text-xs">{run.id.slice(0, 8)}</span>
          {' · '}{run.model ?? 'unknown model'} · layer {run.layer} · {number(run.features, 0)} features
          {' · '}<span className="text-text-secondary">{run.status}</span>
          {!run.has_checkpoint ? <span className="text-text-tertiary"> · no checkpoint</span> : null}
        </span>
        <span className="flex items-center gap-3">
          <span className="text-text-secondary text-xs">{bytes(run.bytes)}</span>
          <button
            className={button}
            disabled={!!run.delete_blocked_reason || busy === run.id}
            title={run.delete_blocked_reason ?? undefined}
            onClick={() => remove(run.id)}
          >
            {busy === run.id ? 'Deleting…' : 'Delete'}
          </button>
        </span>
      </div>)}
    </div> : <p className="text-text-secondary text-sm">No dictionaries saved yet.</p>}
  </section>
}
