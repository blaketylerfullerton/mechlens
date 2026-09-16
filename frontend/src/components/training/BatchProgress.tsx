import type { Run } from './types'
import { button } from './shared'
const terminal = new Set(['completed', 'cancelled', 'failed', 'interrupted'])

export function BatchProgress({ runs, selected, following, onFollow, onSelect, onStop, busy }: {
  runs: Run[]; selected: string; following: boolean; onFollow: (value: boolean) => void;
  onSelect: (id: string) => void; onStop: () => void; busy: boolean;
}) {
  const ordered = [...runs].sort((a, b) => (a.batch_index ?? 0) - (b.batch_index ?? 0))
  const total = ordered[0]?.batch_size ?? ordered.length
  const completed = ordered.filter((run) => run.status === 'completed').length
  const stopped = ordered.filter((run) => terminal.has(run.status) && run.status !== 'completed').length
  const active = ordered.find((run) => !terminal.has(run.status))
  const removed = total - ordered.length
  return <section className="border-fn/30 bg-fn/5 mb-6 rounded border p-5" aria-label="Multi-layer progress">
    <div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="font-medium">{active ? `Layer ${(active.batch_index ?? 0) + 1} of ${total} · model layer ${active.config.layer}` : 'Multi-layer run finished'}</h2><p role="status" className="text-text-secondary mt-2 text-sm">{completed} of {total} layers completed{stopped ? ` · ${stopped} stopped or failed` : ''}{removed ? ` · ${removed} deleted` : ''}</p></div>{active ? <button className={button} disabled={busy} onClick={onStop}>Stop all remaining layers</button> : null}</div>
    <progress className="mt-4 h-2 w-full accent-[var(--color-fn)]" value={completed} max={total} aria-label="Layers completed" />
    <div className="mt-4 flex flex-wrap gap-2">{ordered.map((run) => <button key={run.id} title={`Layer ${run.config.layer}: ${run.status}`} aria-pressed={run.id === selected} className={`${button} px-3 py-2 ${run.id === selected ? 'border-fn bg-fn/10' : ''}`} onClick={() => onSelect(run.id)}><span>Layer {run.config.layer}</span><span className={`ml-2 text-xs ${run.status === 'failed' ? 'text-err' : 'text-text-secondary'}`}>{run.status}</span></button>)}</div>
    {active ? <label className="text-text-secondary mt-4 flex items-center gap-2 text-sm"><input type="checkbox" checked={following} onChange={(event) => onFollow(event.target.checked)} />Follow the active layer automatically</label> : null}
  </section>
}
