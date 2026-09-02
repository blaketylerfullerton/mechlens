import { useEffect, useRef, useState } from 'react'
import type { ChatStatus } from 'ai'
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from '@/components/ai-elements/prompt-input'
import { Suggestion, Suggestions } from '@/components/ai-elements/suggestion'
import { BrainIcon } from 'lucide-react'

const STARTER_PROMPTS = [
  'What does the motor cortex do?',
  'Highlight the hippocampus',
  'Explain the limbic system',
]

// Mocked chain-of-thought text, streamed word-by-word into the monologue bar
// while "Think" is on — stands in until this is wired to a real reasoning stream.
const MOCK_MONOLOGUE =
  "Parsing the question... locating the relevant cortex region... cross-checking known lobe boundaries... drafting a concise answer."

export function ChatPanel() {
  const [status, setStatus] = useState<ChatStatus>('ready')
  const [thinkEnabled, setThinkEnabled] = useState(true)
  const [monologue, setMonologue] = useState('')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(
    () => () => {
      if (timerRef.current) clearInterval(timerRef.current)
    },
    [],
  )

  const streamText = (text: string, onDone: () => void) => {
    setMonologue('')
    const words = text.split(' ')
    let revealed = 0
    timerRef.current = setInterval(() => {
      revealed += 1
      setMonologue(words.slice(0, revealed).join(' '))
      if (revealed >= words.length) {
        if (timerRef.current) clearInterval(timerRef.current)
        onDone()
      }
    }, 70)
  }

  const sendMessage = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || status !== 'ready') return
    setStatus('submitted')

    // TODO: wire up to the real backend once it streams live reasoning + answers.
    const answer = `I don't have a live answer yet, but I heard: "${trimmed}"`
    const finish = () => setStatus('ready')

    if (thinkEnabled) {
      streamText(MOCK_MONOLOGUE, () => streamText(answer, finish))
    } else {
      streamText(answer, finish)
    }
  }

  const handleSubmit = (message: PromptInputMessage, event: { preventDefault: () => void }) => {
    event.preventDefault()
    sendMessage(message.text)
  }

  return (
    <>
      <div className="pointer-events-none absolute inset-x-0 top-6 z-10 flex justify-center px-6">
        <div className="max-w-2xl truncate rounded-full border border-border bg-background/90 px-4 py-2 text-sm text-muted-foreground shadow-lg backdrop-blur">
          {monologue || "Ask a question to see the model's reasoning stream here."}
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-6 z-10 flex justify-center px-4">
        <div className="pointer-events-auto w-full max-w-xl space-y-2 rounded-2xl border border-border bg-background/90 p-3 shadow-lg backdrop-blur">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputBody>
              <PromptInputTextarea placeholder="Ask about the brain..." />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputButton
                  aria-pressed={thinkEnabled}
                  variant={thinkEnabled ? 'secondary' : 'ghost'}
                  onClick={() => setThinkEnabled((prev) => !prev)}
                >
                  <BrainIcon className="size-4" />
                  Think
                </PromptInputButton>
              </PromptInputTools>
              <PromptInputSubmit status={status} />
            </PromptInputFooter>
          </PromptInput>
          <Suggestions>
            {STARTER_PROMPTS.map((prompt) => (
              <Suggestion key={prompt} suggestion={prompt} onClick={sendMessage} />
            ))}
          </Suggestions>
        </div>
      </div>
    </>
  )
}
