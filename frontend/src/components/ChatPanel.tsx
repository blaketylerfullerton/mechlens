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

const MAX_TOKENS = 20

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
  if (status === 'warming') return 'Loading the model — this is slow the first time'
  if (status === 'pending') return 'Queued'
  if (status === 'running') return 'Capturing the model run'
  return 'Run a prompt through the local trace service'
}

// Colour never carries the state on its own: the dot always sits beside the
// word it is colouring.
function statusTone(status: RunState, error: string | null): string {
  if (error) return 'bg-err'
  if (status === 'running') return 'bg-fn'
  if (isRunning(status)) return 'bg-const'
  return 'bg-rule'
}

export function ChatPanel({ error, onTraceRequest, status }: ChatPanelProps) {
  const isBusy = isRunning(status)

  const handleSubmit = (message: PromptInputMessage, event: { preventDefault: () => void }) => {
    event.preventDefault()
    const prompt = message.text.trim()
    if (!prompt || isBusy) return
    onTraceRequest(prompt, MAX_TOKENS)
  }

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-20 flex justify-center px-4">
      {/* The highest layer, so it is lighter than what it sits over and edged
          with the strong hairline. No shadow: on #0A0B0D a shadow is either
          invisible or reads as grime. */}
      <div className="border-border-strong bg-bg-elevated pointer-events-auto w-full max-w-2xl rounded-[16px] border p-3">
        <div className="text-text-tertiary mb-2 flex items-center justify-between gap-3 px-1 text-[12px]">
          <span className="flex min-w-0 items-center gap-2">
            <span
              aria-hidden="true"
              className={`size-1.5 shrink-0 rounded-full ${statusTone(status, error)}`}
            />
            <span className="truncate">{statusCopy(status)}</span>
          </span>
          <span className="shrink-0 font-mono text-[11px] tabular-nums">
            max_tokens={MAX_TOKENS}
          </span>
        </div>
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputBody>
            <PromptInputTextarea disabled={isBusy} placeholder="Enter a prompt to trace…" />
          </PromptInputBody>
          <PromptInputFooter>
            <span className="text-text-tertiary px-2 text-[11px]">
              <span className="font-mono">gemma-2-2b</span> · residual capture
            </span>
            <PromptInputSubmit disabled={isBusy} status={inputStatus(status, error)} />
          </PromptInputFooter>
        </PromptInput>
        <Suggestions className="mt-2">
          {STARTER_PROMPTS.map((prompt) => (
            <Suggestion
              className="border-border-subtle text-text-secondary hover:border-border-strong hover:text-text-primary rounded-md px-3 font-sans"
              disabled={isBusy}
              key={prompt}
              onClick={() => onTraceRequest(prompt, MAX_TOKENS)}
              suggestion={prompt}
            />
          ))}
        </Suggestions>
      </div>
    </div>
  )
}
