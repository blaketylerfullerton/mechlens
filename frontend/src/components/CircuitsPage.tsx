import { useEffect, useMemo, useState } from 'react'

import { CircuitGraph } from '@/components/circuits/CircuitGraph'
import { FeatureEvidencePanel } from '@/components/circuits/FeatureEvidence'
import { SectionLabel } from '@/components/trace/primitives'
import { visibleToken } from '@/components/trace/format'
import { useCircuits } from '@/hooks/useCircuits'
import { deleteAnalysis, listAnalyses, type AnalysisSummary, type CircuitSource } from '@/lib/circuits-api'

/**
 * "Why this token?" — an attribution graph rooted in one predicted token.
 *
 * The graph is a partial, approximate account of the computation behind one
 * prediction on one prefix. It is not a transcript of thoughts, and a path
 * through it is a selection the reader made, not a sequence the model ran. The
 * coverage line under the graph is what keeps that honest and is why it sits
 * beside the picture rather than behind a disclosure.
 */

const PRESETS = [
  { label: 'Overview', value: 0.02 },
  { label: 'Detail', value: 0.005 },
  { label: 'Everything retained', value: 0 },
]

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div>
      <div className="text-text-tertiary text-[10px] tracking-[0.04em] uppercase">{label}</div>
      <div className="font-mono text-[13px] tabular-nums">{value}</div>
      {note ? <div className="text-text-tertiary text-[10px]">{note}</div> : null}
    </div>
  )
}

export function CircuitsPage({
  active, request, onBackToExplore,
}: {
  active: boolean
  request: { source: CircuitSource; key: number } | null
  onBackToExplore: (position: number) => void
}) {
  const { status, summary, graph, error, run, open, reset } = useCircuits()
  const [threshold, setThreshold] = useState(PRESETS[0].value)
  const [selected, setSelected] = useState<string | null>(null)
  const [saved, setSaved] = useState<AnalysisSummary[]>([])

  // A new request from Explore starts a new analysis; the key makes asking for
  // the same token twice a real second request rather than a no-op.
  useEffect(() => {
    if (!request) return
    setSelected(null)
    void run(request.source)
  }, [request?.key])

  const refreshSaved = () => { void listAnalyses(20).then((r) => setSaved(r.analyses)).catch(() => {}) }
  useEffect(() => { if (active) refreshSaved() }, [active, status])

  const filtered = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] }
    // The explained token always stays. Influence is measured *towards* the
    // logit, so the logit's own influence is 0 by construction and any
    // threshold above zero would filter away the root of the graph.
    const nodes = graph.nodes.filter((n) => n.kind === 'logit' || n.influence >= threshold)
    const ids = new Set(nodes.map((n) => n.id))
    return { nodes, edges: graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target)) }
  }, [graph, threshold])

  const selectedNode = useMemo(
    () => filtered.nodes.find((n) => n.id === selected) ?? null, [filtered.nodes, selected],
  )

  const source = graph?.source ?? summary?.source ?? request?.source ?? null
  const tokens = source?.prefix_texts ?? source?.prefix_token_ids.map(String) ?? []

  if (status === 'idle' && !request) {
    return (
      <div className="mx-auto w-full max-w-[1800px] px-4 py-10 sm:px-8">
        <div className="max-w-2xl">
          <h1 className="text-[15px] font-medium">Circuits</h1>
          <p className="text-text-secondary mt-2 text-[13px] leading-6">
            Pick a generated token in Explore and choose <span className="text-text-primary">Explain
            this prediction</span>. Circuits traces that one token backward through the transcoder
            features that fed it, to the input tokens.
          </p>
          <p className="text-text-tertiary mt-3 text-[12px] leading-5">
            One supported setup: gemma-2-2b with per-layer GemmaScope transcoders, short prefixes,
            one output token per analysis. A graph is a partial approximation of the computation
            behind a single prediction — not a record of what the model was thinking.
          </p>
        </div>
        {saved.length ? (
          <div className="mt-8 max-w-2xl">
            <SectionLabel note={`${saved.length}`}>Saved analyses</SectionLabel>
            <ul className="divide-border-subtle divide-y">
              {saved.map((item) => (
                <li className="flex items-center justify-between gap-4 py-2" key={item.analysis_id}>
                  <button
                    className="min-w-0 text-left"
                    onClick={() => { setSelected(null); open(item.analysis_id) }}
                  >
                    <span className="font-mono text-[12px]">
                      {visibleToken(item.source.target_text ?? String(item.source.target_token_id))}
                    </span>
                    <span className="text-text-tertiary ml-2 text-[11px]">{item.source.prompt}</span>
                  </button>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="text-text-tertiary font-mono text-[11px] tabular-nums">
                      {item.status === 'done' ? `${item.coverage?.retained_nodes ?? 0} nodes` : item.status}
                    </span>
                    <button
                      className="text-text-tertiary hover:text-err text-[11px]"
                      onClick={() => void deleteAnalysis(item.analysis_id).then(refreshSaved)}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-[1800px] flex-col px-4 py-5 sm:px-8">
      <header className="border-border-subtle flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 border-b pb-3">
        <div className="min-w-0">
          <h1 className="text-[13px] font-medium">
            Why{' '}
            <span className="text-fn font-mono">
              {visibleToken(source?.target_text ?? String(source?.target_token_id ?? ''))}
            </span>
            ?
          </h1>
          {source?.prompt ? (
            <p className="text-text-tertiary mt-1 truncate font-mono text-[11px]">
              {source.prompt}
              {source.prefix_texts ? source.prefix_texts.slice(source.prefix_texts.length - 1).join('') : ''}
            </p>
          ) : null}
        </div>
        <div className="flex items-center gap-4 text-[12px]">
          {source ? (
            <button className="text-fn" onClick={() => onBackToExplore(source.target_position)}>
              Back to the trace
            </button>
          ) : null}
          <button className="text-text-tertiary hover:text-text-primary" onClick={() => { reset(); setSelected(null) }}>
            Close
          </button>
        </div>
      </header>

      {status === 'queued' || status === 'running' ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="text-center">
            <p className="text-text-secondary text-[13px]">
              {status === 'queued' ? 'Queued…' : 'Attributing…'}
            </p>
            <p className="text-text-tertiary mt-1 text-[11px]">
              A few seconds. This cannot be cancelled once it starts.
            </p>
          </div>
        </div>
      ) : status === 'error' || status === 'interrupted' ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="max-w-md text-center">
            <p className="text-err text-[13px]">
              {status === 'interrupted' ? 'Interrupted by a service restart.' : 'Attribution failed.'}
            </p>
            {error ? <p className="text-text-tertiary mt-2 font-mono text-[11px]">{error}</p> : null}
          </div>
        </div>
      ) : graph ? (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row lg:gap-6">
          <main className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 py-3">
              <div className="flex items-center gap-1 text-[12px]">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    aria-pressed={threshold === preset.value}
                    onClick={() => setThreshold(preset.value)}
                    className={`rounded-xs px-2 py-1 ${threshold === preset.value ? 'text-text-primary bg-bg-elevated' : 'text-text-tertiary hover:text-text-primary'}`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-4 font-mono text-[11px] tabular-nums">
                <span className="text-text-tertiary">
                  <span className="bg-fn mr-1 inline-block size-2 align-middle" /> feature
                </span>
                <span className="text-text-tertiary">
                  <span className="bg-rule mr-1 inline-block size-2 align-middle" /> error
                </span>
                <span className="text-text-tertiary">
                  <span className="bg-text-tertiary mr-1 inline-block size-2 align-middle" /> input
                </span>
                <span className="text-text-tertiary">solid + / dashed −</span>
              </div>
            </div>

            <div className="border-border-subtle bg-bg-surface min-h-0 flex-1 overflow-auto rounded-[2px] border p-3">
              <CircuitGraph
                nodes={filtered.nodes}
                edges={filtered.edges}
                nLayers={graph.coverage.n_layers}
                nPositions={graph.coverage.n_positions}
                tokens={tokens}
                selected={selected}
                onSelect={setSelected}
              />
            </div>

            {/* The caveat belongs beside the claim, not behind a disclosure: the
                picture above is a small fraction of the computation, and a reader
                who cannot see that is being misled by it. */}
            <div className="border-border-subtle mt-3 grid shrink-0 grid-cols-2 gap-x-6 gap-y-3 border-t pt-3 sm:grid-cols-4">
              <Metric
                label="Shown"
                value={`${filtered.nodes.length} of ${graph.coverage.retained_nodes}`}
                note={`retained from ${graph.coverage.total_nodes} computed`}
              />
              <Metric
                label="Edges"
                value={`${filtered.edges.length} of ${graph.coverage.sent_edges}`}
                note={`${graph.coverage.total_edges_nonzero.toLocaleString()} nonzero before pruning`}
              />
              <Metric
                label="Unexplained"
                value={`${filtered.nodes.filter((n) => n.kind === 'error').length} error nodes`}
                note={`${graph.coverage.n_error_nodes} in the computation`}
              />
              <Metric
                label="Target probability"
                value={graph.target_probability != null ? graph.target_probability.toFixed(4) : '—'}
                note={summary?.elapsed_s != null ? `${summary.elapsed_s.toFixed(1)}s to attribute` : undefined}
              />
            </div>
            <p className="text-text-tertiary mt-2 shrink-0 text-[11px] leading-5">
              Omitted nodes and edges fall below the retention thresholds — they are not zero
              effects. A highlighted path is a selection you made, not a sequence the model ran.
            </p>
          </main>

          <aside className="border-border-subtle mt-4 min-h-0 shrink-0 overflow-y-auto border-t pt-4 lg:mt-0 lg:w-80 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
            {selectedNode ? (
              <FeatureEvidencePanel node={selectedNode} />
            ) : (
              <div>
                <SectionLabel>Inspector</SectionLabel>
                <p className="text-text-secondary text-[12px] leading-5">
                  Select a node to see where it fired and what it fires on elsewhere. Selecting one
                  dims everything not wired directly to it.
                </p>
              </div>
            )}
          </aside>
        </div>
      ) : null}
    </div>
  )
}
