import type { RunState } from '@/hooks/useTrace'
import type { ChatStatus } from 'ai'
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  type PromptInputMessage,
} from '@/components/ai-elements/prompt-input'
import { Suggestion, Suggestions } from '@/components/ai-elements/suggestion'

const STARTER_PROMPTS = [
  'The Golden Gate Bridge is located in the city of',
  'The capital of France is',
  '2 + 2 =',
]

type ChatPanelProps = {
  error: string | null
  onTraceRequest: (prompt: string, maxTokens: number) => void
  status: RunState
}

// `warming` counts as busy: no job exists yet, but the service is answering
// 503 while gemma loads and submitting again would only queue another retry.
function isRunning(status: RunState): boolean {
  return status === 'warming' || status === 'pending' || status === 'running'
}

function inputStatus(status: RunState, error: string | null): ChatStatus {
  if (error) return 'error'
  if (isRunning(status)) return 'submitted'
  return 'ready'
}

// Says which of the two waits this is. They look identical from the outside —
// nothing is happening on screen either way — but only one of them is the
// model actually running the prompt.
function statusCopy(status: RunState): string {
  if (status === 'warming') return 'Loading the model — this is slow the first time…'
  if (status === 'pending') return 'Queued…'
  if (status === 'running') return 'Capturing the model run…'
  return 'Run a prompt through the local trace service.'
}

export function ChatPanel({ error, onTraceRequest, status }: ChatPanelProps) {
  const isBusy = isRunning(status)

  const handleSubmit = (message: PromptInputMessage, event: { preventDefault: () => void }) => {
    event.preventDefault()
    const prompt = message.text.trim()
    if (!prompt || isBusy) return
    onTraceRequest(prompt, 20)
  }

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-20 flex justify-center px-4">
      <div className="pointer-events-auto w-full max-w-2xl rounded-2xl border border-white/15 bg-slate-950/90 p-3 shadow-2xl shadow-black/40 backdrop-blur">
        <div className="mb-2 flex items-center justify-between gap-3 px-1 text-xs text-slate-400">
          <span>{statusCopy(status)}</span>
          <span className="font-mono text-[10px] text-cyan-100/70">20 generated tokens</span>
        </div>
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea disabled={isBusy} placeholder="Enter a prompt to trace…" />
          </PromptInputBody>
          <PromptInputFooter>
            <span className="px-2 text-[11px] text-slate-500">Gemma 2 2B · residual capture</span>
            <PromptInputSubmit disabled={isBusy} status={inputStatus(status, error)} />
          </PromptInputFooter>
        </PromptInput>
        <Suggestions className="mt-2">
          {STARTER_PROMPTS.map((prompt) => (
            <Suggestion
              disabled={isBusy}
              key={prompt}
              onClick={() => onTraceRequest(prompt, 20)}
              suggestion={prompt}
            />
          ))}
        </Suggestions>
      </div>
    </div>
  )
}
