import { useEffect, useRef, useState } from 'react'
import { ApiError, request } from '@/lib/api-client'
import { button, number, primary, storageSize } from './training/shared'
import { useModelDownloads } from './training/api'
import type { Run } from './training/types'

type StoredRun = { id: string; model: string | null; layer: number; features: number; status: string; created_at: number; has_checkpoint: boolean; bytes: number; delete_blocked_reason: string | null }
type Storage = { runs: StoredRun[]; total_bytes: number; free_bytes: number }
type Entry = StoredRun & { batch_id?: string; batch_size?: number; resume_supported?: boolean }
type Filter = 'All' | 'In progress' | 'Ready' | 'Needs attention'
const filters: Filter[] = ['All', 'In progress', 'Ready', 'Needs attention']
const terminal = new Set(['completed', 'cancelled', 'failed', 'interrupted'])
function matches(run: Entry, filter: Filter) {
  if (filter === 'All') return true
  if (filter === 'In progress') return !terminal.has(run.status)
  if (filter === 'Ready') return run.status === 'completed' && run.has_checkpoint
  return ['failed', 'interrupted', 'cancelled'].includes(run.status) || (run.status === 'completed' && !run.has_checkpoint)
}
function statusLabel(run: Entry) {
  return run.status === 'completed' && run.has_checkpoint ? 'Ready' : run.status.replaceAll('_', ' ')
}

export function DictionariesPage({ active, focusRequest, onSelect, onInspect, onDeleted, onNew }: {
  active: boolean; focusRequest: { id: string; key: number } | null;
  onSelect: (id: string) => void; onInspect: (id: string) => void; onDeleted: (id: string) => void; onNew: () => void;
}) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [storage, setStorage] = useState<Storage | null>(null)
  const [filter, setFilter] = useState<Filter>('All')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  const downloads = useModelDownloads(active, refresh)
  const handledFocus = useRef(-1)
  const heading = useRef<HTMLHeadingElement>(null)
  useEffect(() => { if (active) heading.current?.focus() }, [active])
  useEffect(() => {
    if (!active) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const [result, runs] = await Promise.all([request<Storage>('/training/storage'), request<Run[]>('/training/runs')])
        if (stopped) return
        const metadata = new Map(runs.map((run) => [run.id, run]))
        setStorage(result)
        setEntries(result.runs.map((run) => ({ ...run, batch_id: metadata.get(run.id)?.batch_id, batch_size: metadata.get(run.id)?.batch_size, resume_supported: metadata.get(run.id)?.resume_supported })))
        setError(null)
      } catch (err) {
        if (!stopped) setError(err instanceof ApiError && err.status === 404 ? 'Restart the backend to enable dictionary management.' : String(err))
      }
      if (!stopped) timer = setTimeout(poll, 5000)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [active, refresh])
  useEffect(() => {
    if (!active || !focusRequest || handledFocus.current === focusRequest.key) return
    const entry = entries.find((run) => run.id === focusRequest.id)
    if (!entry) return
    handledFocus.current = focusRequest.key
    const id = entry.batch_id ?? entry.id
    setFilter('All'); setExpanded((current) => new Set([...current, id]))
    const frame = requestAnimationFrame(() => {
      const row = document.getElementById(`dictionary-${id}`)
      row?.scrollIntoView({ block: 'center' }); row?.focus({ preventScroll: true })
    })
    return () => cancelAnimationFrame(frame)
  }, [active, focusRequest, entries])
  const groups = new Map<string, Entry[]>()
  for (const entry of entries) {
    const id = entry.batch_id ?? entry.id
    groups.set(id, [...groups.get(id) ?? [], entry])
  }
  const visible = [...groups].filter(([, rows]) => rows.some((run) => matches(run, filter)))
  function toggle(id: string) {
    setExpanded((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next })
  }
  async function remove(run: Entry) {
    if (deleting) return
    setDeleting(run.id); setError(null); setNotice(null)
    try {
      await request(`/training/runs/${run.id}`, { method: 'DELETE' })
      setEntries((current) => current.filter((item) => item.id !== run.id))
      setStorage((current) => current ? { ...current, total_bytes: Math.max(0, current.total_bytes - run.bytes) } : current)
      setConfirm(null); onDeleted(run.id); setNotice(`Deleted layer ${run.layer} run ${run.id.slice(0, 8)}.`); setRefresh((value) => value + 1)
    } catch (err) { setError(String(err)) }
    finally { setDeleting(null) }
  }
  function actions(run: Entry) {
    return <div className="flex items-center gap-2">
      {run.status === 'completed' && run.has_checkpoint ? <button className={button} onClick={() => onInspect(run.id)}>Explore</button> : <button className={button} onClick={() => onSelect(run.id)}>{!terminal.has(run.status) ? 'View progress' : run.resume_supported ? 'Review and resume' : 'View run'}</button>}
      <details className="relative"><summary aria-label={`More actions for layer ${run.layer}, run ${run.id.slice(0, 8)}`} className={`${button} list-none px-3 text-lg leading-5 [&::-webkit-details-marker]:hidden`}>···</summary><div className="border-border-strong bg-bg-elevated absolute right-0 z-20 mt-2 w-64 rounded border p-2 shadow-lg">
        <button className="hover:bg-fn/10 block w-full rounded px-3 py-2 text-left text-sm" onClick={(event) => { event.currentTarget.closest('details')?.removeAttribute('open'); onSelect(run.id) }}>View run details</button>
        <button className="text-err hover:bg-err/10 block w-full rounded px-3 py-2 text-left text-sm disabled:opacity-50" disabled={!!deleting || !!run.delete_blocked_reason} onClick={(event) => { event.currentTarget.closest('details')?.removeAttribute('open'); setConfirm(run.id); setNotice(null) }}>Delete run…</button>
        {run.delete_blocked_reason ? <p className="text-text-secondary px-3 py-2 text-xs leading-5">{run.delete_blocked_reason}.</p> : null}
      </div></details>
    </div>
  }
  function confirmation(run: Entry) {
    return confirm === run.id ? <div className="border-err/40 mt-4 rounded border p-4" role="group" aria-label={`Confirm deletion of run ${run.id.slice(0, 8)}`}><p className="text-sm font-medium">Delete layer {run.layer} and its saved run?</p><p className="text-text-secondary mt-2 text-sm leading-6">Removes this dictionary, checkpoints, examples, labels, and run history ({storageSize(run.bytes)} of files). This cannot be undone. Other layers and the shared base model stay available.</p><div className="mt-4 flex flex-wrap gap-3"><button className={`${button} border-err/60 text-err`} disabled={!!deleting || !!run.delete_blocked_reason} onClick={() => remove(run)}>{deleting === run.id ? 'Deleting…' : 'Delete run and files'}</button><button className={button} disabled={!!deleting} onClick={() => setConfirm(null)}>Keep run</button></div></div> : null
  }
  return <main className="mx-auto w-full max-w-[1280px] px-4 py-6 sm:px-8 lg:py-8">
    <header className="mb-8 flex flex-wrap items-start justify-between gap-4"><div><h1 ref={heading} tabIndex={-1} className="text-3xl font-medium tracking-tight outline-none">Dictionaries</h1><p className="text-text-secondary mt-3 text-sm">Your training runs and learned features, in one place.</p></div><button className={primary} onClick={onNew}>New training run</button></header>
    <div className="border-border-subtle mb-6 flex flex-wrap items-center justify-between gap-4 border-b pb-4"><div className="flex flex-wrap gap-2" aria-label="Filter dictionaries">{filters.map((value) => <button key={value} aria-pressed={filter === value} className={`${button} px-3 py-2 ${filter === value ? 'border-fn bg-fn/10 text-fn' : 'text-text-secondary'}`} onClick={() => { setFilter(value); setConfirm(null) }}>{value}</button>)}</div><p className="text-text-secondary text-xs">{storage ? `${storageSize(storage.total_bytes)} used · ${storageSize(storage.free_bytes)} free on disk` : 'Reading storage…'}</p></div>
    {error ? <p role="alert" className="text-err mb-5 text-sm">{error}</p> : null}
    {notice ? <p role="status" className="text-text-secondary mb-5 text-sm">{notice}</p> : null}
    {!storage && !error ? <p role="status" className="text-text-secondary py-10 text-sm">Loading your dictionaries…</p> : null}
    {storage && !entries.length ? <section className="border-border-subtle rounded border border-dashed p-8 text-center"><h2 className="text-lg font-medium">Your first dictionary starts with a training run</h2><p className="text-text-secondary mt-3 text-sm">Come back here to explore it, check its progress, or manage its storage.</p><button className={`${primary} mt-6`} onClick={onNew}>Set up training</button></section> : storage && !visible.length ? <p className="text-text-secondary py-10 text-sm">No runs match this filter.</p> : null}
    {downloads?.models.length ? <section aria-labelledby="base-models" className="border-border-subtle mb-8 rounded border p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="base-models" className="text-base font-medium">Base models</h2>
        <p className="text-text-secondary text-xs">{storageSize(downloads.total_bytes)} downloaded from Hugging Face</p>
      </div>
      <p className="text-text-secondary mt-2 text-sm leading-6">The language models your dictionaries are trained on. These are downloaded once and shared by every run.</p>
      <ul className="border-border-subtle mt-4 divide-y border-t">{downloads.models.map((model) => <li key={model.repository} className="flex flex-wrap items-center justify-between gap-3 py-3">
        <span className="min-w-0"><span className="break-all font-mono text-sm">{model.repository}</span>{model.loaded ? <span className="text-fn ml-3 text-xs">Loaded</span> : null}</span>
        <span className={`text-sm ${model.downloaded ? 'text-text-secondary' : 'text-text-tertiary'}`}>{model.downloaded ? storageSize(model.bytes ?? undefined) : 'Not downloaded'}</span>
      </li>)}</ul>
      <p className="text-text-tertiary mt-4 text-xs leading-5">Stored in the shared Hugging Face cache, outside this app. Remove them with <span className="font-mono">huggingface-cli delete-cache</span>.</p>
    </section> : null}
    <div className="space-y-4">{visible.map(([id, rows]) => {
      const first = rows[0]
      const batch = !!first.batch_id
      const completed = rows.filter((run) => run.status === 'completed' && run.has_checkpoint).length
      const activeCount = rows.filter((run) => !terminal.has(run.status)).length
      const attention = rows.filter((run) => matches(run, 'Needs attention')).length
      const totalSize = rows.reduce((sum, run) => sum + run.bytes, 0)
      const title = `${first.model ?? 'Training run'} · ${batch ? `${rows.length} layers` : `Layer ${first.layer}`}`
      return <article id={`dictionary-${id}`} tabIndex={-1} key={id} className="border-border-subtle bg-bg-surface rounded border p-5 focus-visible:border-fn">
        <div className="flex flex-wrap items-center justify-between gap-4"><div className="min-w-0"><h2 className="break-words text-base font-medium">{title}</h2><p className="text-text-secondary mt-2 text-xs">{new Date(first.created_at * 1000).toLocaleString()} · {number(first.features, 0)} features{batch ? ' per layer' : ''}</p><p className={`mt-2 text-sm ${attention ? 'text-err' : 'text-text-secondary'}`}>{batch ? `${completed} ready${activeCount ? ` · ${activeCount} in progress` : ''}${attention ? ` · ${attention} need attention` : ''}` : statusLabel(first)}</p></div><div className="flex flex-wrap items-center gap-4"><span className="text-text-secondary font-mono text-sm">{storageSize(totalSize)}</span>{batch ? <button className={button} aria-expanded={expanded.has(id)} aria-controls={`layers-${id}`} onClick={() => toggle(id)}>{expanded.has(id) ? 'Hide layers' : 'View layers'}</button> : actions(first)}</div></div>
        {!batch ? confirmation(first) : expanded.has(id) ? <div id={`layers-${id}`} className="border-border-subtle mt-5 divide-y border-t">{[...rows].sort((a, b) => a.layer - b.layer).map((run) => <div key={run.id} className="py-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-medium">Layer {run.layer}</h3><p className="text-text-secondary mt-1 text-xs">{statusLabel(run)} · {storageSize(run.bytes)} · {run.id.slice(0, 8)}</p></div>{actions(run)}</div>{confirmation(run)}</div>)}</div> : null}
      </article>
    })}</div>
    {entries.length ? <p className="text-text-tertiary mt-6 text-xs leading-5">Storage includes checkpoints, resume state, and examples. Base model downloads are counted separately, above. Use a run’s actions menu to delete it.</p> : null}
  </main>
}
