import { useState } from 'react'
import type { RunState } from '@/hooks/useTrace'
import { API_BASE_URL } from '@/lib/api-client'

const DRAFT_KEY = `mechlens-prompt:${API_BASE_URL}`
const EXAMPLES = [
  'The Golden Gate Bridge is located in the city of',
  'The capital of France is',
  '2 + 2 =',
]

export function ChatPanel({ onTraceRequest, status }: {
  error: string | null
  onTraceRequest: (prompt: string, maxTokens: number) => void
  status: RunState
}) {
  const [prompt, setPrompt] = useState(() => {
    try { return localStorage.getItem(DRAFT_KEY) ?? '' } catch { return '' }
  })
  const [examplesOpen, setExamplesOpen] = useState(false)
  const busy = status === 'warming' || status === 'pending' || status === 'running'
  const updatePrompt = (value: string) => {
    setPrompt(value)
    try { localStorage.setItem(DRAFT_KEY, value) } catch { /* Draft still works in memory. */ }
  }
  const submit = () => {
    if (busy || !prompt.trim()) return
    setExamplesOpen(false)
    onTraceRequest(prompt.trim(), 20)
  }

  return (
    <form aria-label="Run a trace" onSubmit={(event) => { event.preventDefault(); submit() }}
      className="border-border-strong bg-bg-surface relative rounded-xs border p-3">
      <div className="flex items-end gap-3">
        <label className="min-w-0 flex-1">
          <span className="sr-only">Prompt</span>
          <textarea value={prompt} onChange={(event) => updatePrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                event.preventDefault()
                submit()
              }
            }}
            disabled={busy} rows={2} placeholder="Enter a prompt…"
            className="text-text-primary placeholder:text-text-tertiary block max-h-24 w-full resize-none rounded-xs bg-transparent font-mono text-[13px] leading-5 disabled:opacity-60" />
        </label>
        <button type="button" aria-expanded={examplesOpen} aria-controls="prompt-examples"
          disabled={busy} onClick={() => setExamplesOpen((open) => !open)}
          className="text-text-secondary hover:text-text-primary rounded-xs px-2 py-2 text-[12px] disabled:opacity-50">Examples</button>
        <button type="submit" disabled={busy || !prompt.trim()}
          className="bg-text-primary text-bg-base rounded-xs px-4 py-2 text-[12px] font-medium disabled:opacity-50">
          {busy ? 'Running…' : 'Run'}
        </button>
      </div>
      {examplesOpen ? (
        <div id="prompt-examples" className="border-border-strong bg-bg-elevated absolute right-0 bottom-full z-30 mb-2 w-full max-w-lg rounded-xs border p-2">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => { updatePrompt(example); setExamplesOpen(false) }}
              className="text-text-primary hover:bg-fn/10 block w-full rounded-xs px-2 py-2 text-left font-mono text-[12px]">
              {example}
            </button>
          ))}
        </div>
      ) : null}
    </form>
  )
}
