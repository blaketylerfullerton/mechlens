import { useEffect, useRef, useState } from 'react'

import type { Trace } from '@/lib/api-types'
import { TokenStrip } from './TokenStrip'
import { visibleToken } from './format'

/** One response serves both reading and inspection. Raw token notation is opt-in. */
export function TraceResponse({ trace, running, followingLatest, onFollowLatest, onSelect, selection }: {
  trace: Trace
  running: boolean
  followingLatest: boolean
  onFollowLatest: () => void
  onSelect: (position: number) => void
  selection: { position: number; layer: number; via?: string }
}) {
  const [detailsOpen, setDetailsOpen] = useState(false)
  const responseRef = useRef<HTMLDivElement>(null)
  const generated = trace.steps.filter((step) => step.token.source === 'generated')
  const measuredText = generated.map((step) => step.token.text).join('')
  // Some tokenizers decode byte fragments differently in isolation. Never
  // replace the model's exact completion with a lossy concatenation of tokens.
  const aligned = trace.completion.startsWith(measuredText)
  const pendingText = aligned ? trace.completion.slice(measuredText.length) : ''
  const step = trace.steps[selection.position]
  const prediction = step?.logits.chosen ?? step?.logits.top_k[0]
  const analysisReady = step?.layers[selection.layer]?.logit_lens != null
  // A token is tinted only when the reader put the selection there. The
  // resting selection is the last token, so tinting on position alone left the
  // final word lit after the run finished — the same blue that means "look
  // here" reading as "still going".
  const highlighted = selection.via && selection.via !== 'default' ? selection.position : null

  useEffect(() => {
    const element = responseRef.current
    if (followingLatest && element) element.scrollTop = element.scrollHeight
  }, [trace.completion, followingLatest])

  return (
    <section aria-label="Model response" className="border-border-subtle shrink-0 border-t pt-4">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <h2 className="text-text-secondary text-[12px] font-medium">Response</h2>
          {running ? (
            <span role="status" className="text-text-tertiary font-mono text-[11px]">generating…</span>
          ) : trace.completion ? (
            <span role="status" className="text-str font-mono text-[11px] tabular-nums">
              <span aria-hidden="true" className="bg-str mr-1.5 inline-block size-1.5 rounded-full align-middle" />
              done · {trace.n_generated_tokens} tokens ·{' '}
              {trace.stop_reason === 'eos' ? 'model stopped' : 'hit the token budget'}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-3 text-[12px]">
          {running && !followingLatest ? (
            <button type="button" onClick={onFollowLatest} className="text-fn rounded-xs px-1 py-1">
              Follow latest
            </button>
          ) : null}
          <button type="button" aria-expanded={detailsOpen} aria-controls="token-details"
            onClick={() => setDetailsOpen((open) => !open)}
            className="text-text-tertiary hover:text-text-primary rounded-xs px-1 py-1">
            Token details
          </button>
        </div>
      </div>

      <div ref={responseRef} className="max-h-36 overflow-y-auto py-1 font-mono text-[15px] leading-7 whitespace-pre-wrap [overflow-wrap:anywhere]">
        {!trace.completion ? (
          <span className="text-text-secondary">{running ? 'Waiting for the first token…' : 'No text generated.'}</span>
        ) : aligned ? (
          <>
            {generated.map((item) => (
              <button type="button" key={item.step}
                aria-label={`Inspect token ${item.step}: ${visibleToken(item.token.text)}`}
                aria-pressed={highlighted === item.step}
                title={`Token ${item.step} · click to inspect`}
                onClick={() => onSelect(item.step)}
                className={`inline cursor-pointer rounded-xs p-0 text-left align-baseline whitespace-pre-wrap hover:bg-fn/10 hover:outline hover:outline-fn/50 ${highlighted === item.step ? 'bg-fn/10 text-fn' : 'text-text-primary'}`}>
                {item.token.text}
              </button>
            ))}
            {pendingText ? <span title="Activations available after the next model step" className="text-text-secondary">{pendingText}</span> : null}
          </>
        ) : trace.completion}
      </div>

      <div className="text-text-secondary mt-3 flex min-h-5 flex-wrap items-baseline gap-x-4 gap-y-1 text-[12px]">
        {prediction ? (
          <span>
            Next token <span className="text-text-primary font-mono">{visibleToken(prediction.text)}</span>
            {' · '}<span className="font-mono tabular-nums">{(prediction.prob * 100).toFixed(1)}%</span>
          </span>
        ) : null}
        <span className="text-text-tertiary">
          <span className="font-mono">L{selection.layer} · #{selection.position}</span>
          {' · '}{analysisReady ? 'Layer data ready' : running ? 'Layer data pending' : 'Layer data unavailable'}
        </span>
      </div>
      {pendingText ? <p className="text-text-tertiary mt-1 text-[12px]">The newest token’s activations arrive on the next model step.</p> : null}
      {!aligned ? <p className="text-text-tertiary mt-1 text-[12px]">Open Token details to inspect exact token boundaries for this text.</p> : null}

      {detailsOpen ? (
        <div id="token-details" className="border-border-subtle mt-4 max-h-48 overflow-auto border-t pt-3">
          <p className="text-text-secondary mb-1 text-[12px]">Prompt</p>
          <p className="text-text-primary mb-3 whitespace-pre-wrap font-mono text-[13px]">{trace.prompt}</p>
          <TokenStrip steps={trace.steps} selectedPosition={selection.position} onSelect={onSelect} />
        </div>
      ) : null}
    </section>
  )
}
