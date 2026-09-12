import { useState } from 'react'

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

/**
 * The three prompts, and why these three.
 *
 * Each one produces a trace worth looking at, which is the whole job of a
 * starter: a first run that resolves into nothing teaches the reader that the
 * tool shows nothing. Golden Gate is the one the README walks through, France
 * is the shortest fact that still crosses layers, and `2 + 2 =` is the case
 * where the answer appears late and the early layers are visibly wrong.
 */
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

/**
 * The one action at rest.
 *
 * Three prompts anyone can run with one click, and nothing else at rung 1.
 * Composing a prompt is a decision, and a first-time reader has no basis for
 * making it — they do not yet know what a good prompt for this tool looks
 * like, because they have not seen a trace. So the blank textarea is rung 3,
 * behind "write your own", and the three rows are the thing on screen.
 *
 * No status line here: `TraceViewer`'s own `StatusLine` sits directly below
 * this in the empty state and already says which wait a wait is. Two status
 * readouts one above the other were saying the same thing twice.
 */
export function ChatPanel({ error, onTraceRequest, status }: ChatPanelProps) {
  const isBusy = isRunning(status)
  const [ownOpen, setOwnOpen] = useState(false)

  const handleSubmit = (message: PromptInputMessage, event: { preventDefault: () => void }) => {
    event.preventDefault()
    const prompt = message.text.trim()
    if (!prompt || isBusy) return
    onTraceRequest(prompt, MAX_TOKENS)
  }

  return (
    <div className="w-full">
      {/* Hairline rows, not cards and not chips. The prompt is set in mono
          because it is text the reader could type; the row around it is the
          control, so its affordance is sans. */}
      <ul className="border-border-subtle divide-border-subtle divide-y border-y">
        {STARTER_PROMPTS.map((prompt) => (
          <li key={prompt}>
            <button
              className="group hover:bg-fn/[0.04] focus-visible:bg-fn/[0.04] flex w-full items-center gap-3 px-1 py-3 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isBusy}
              onClick={() => onTraceRequest(prompt, MAX_TOKENS)}
              type="button"
            >
              <span className="text-text-primary min-w-0 flex-1 font-mono text-[13px] leading-5">
                {prompt}
              </span>
              <ArrowRight className="text-text-disabled group-hover:text-fn shrink-0 transition-colors duration-150" />
            </button>
          </li>
        ))}
      </ul>

      {/* Rung 3: reachable, not present. */}
      <div className="mt-4">
        <button
          aria-expanded={ownOpen}
          className="text-text-tertiary hover:text-text-primary inline-flex items-center gap-1.5 text-[13px] transition-colors duration-150"
          onClick={() => setOwnOpen((open) => !open)}
          type="button"
        >
          <Chevron className={ownOpen ? 'rotate-90' : ''} />
          Write your own
        </button>

        {ownOpen ? (
          <div className="border-border-strong bg-bg-elevated mt-3 rounded-[6px] border p-3">
            <PromptInput onSubmit={handleSubmit}>
              <PromptInputBody>
                <PromptInputTextarea disabled={isBusy} placeholder="Enter a prompt to trace…" />
              </PromptInputBody>
              <PromptInputFooter>
                <span className="text-text-tertiary px-2 font-mono text-[11px] tabular-nums">
                  gemma-2-2b · max_tokens={MAX_TOKENS}
                </span>
                <PromptInputSubmit disabled={isBusy} status={inputStatus(status, error)} />
              </PromptInputFooter>
            </PromptInput>
          </div>
        ) : null}
      </div>
    </div>
  )
}

/** Stroke, monochrome, 16px. */
function ArrowRight({ className = '' }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={`size-4 ${className}`}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  )
}

function Chevron({ className = '' }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={`size-3.5 shrink-0 transition-transform duration-150 ${className}`}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}
