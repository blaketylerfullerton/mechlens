import type { ReactNode } from 'react'

import type { RunState } from '@/hooks/useTrace'

import { StatusLine } from './StatusLine'

/**
 * The shape of the grid that is coming, at the size it will be, so nothing
 * shifts when it arrives. Hairline blocks, not a spinner over an empty region.
 */
function GridSkeleton() {
  return (
    <div aria-hidden="true" className="space-y-1 opacity-40">
      {Array.from({ length: 8 }, (_, row) => (
        <div className="flex gap-1" key={row}>
          <div className="bg-bg-surface h-4 w-10 rounded-xs" />
          {Array.from({ length: 12 }, (_, cell) => (
            <div className="bg-bg-surface h-4 flex-1 rounded-xs" key={cell} />
          ))}
        </div>
      ))}
    </div>
  )
}

/**
 * What this is, and the one thing to do about it.
 *
 * The version this replaced opened on "No trace loaded." over a paragraph
 * about residual streams, L2 norms and the logit lens, above a four-row facts
 * table that included the API base URL. All of it true, and all of it the
 * wrong rung: it spent the first line of the product on an absence, and the
 * next on vocabulary a first-time reader cannot have yet. They learn what a
 * residual stream is by running one and looking at the grid — which is why
 * that prose now lives in the legend and the caveats, one click away, where
 * there is something on screen for it to describe.
 *
 * So: what the tool shows, in a sentence with no jargon in it, and three
 * prompts that run on one click. No illustration, no fabricated dashboard,
 * and no claim the page cannot currently back with data.
 */
export function EmptyState({
  status,
  error,
  composer,
}: {
  status: RunState
  error: string | null
  composer?: ReactNode
}) {
  return (
    <main className="enter flex min-h-full items-start justify-center py-6">
      <section className="w-full max-w-2xl">
        <p className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
          mechlens · gemma-2-2b
        </p>
        <h1 className="text-text-primary mt-3 text-[28px] leading-[1.1] font-semibold sm:text-[36px]">
          See which features fire inside a language model.
        </h1>
        <p className="text-text-secondary mt-4 max-w-[54ch] text-[15px] leading-[1.6]">
          Pick a prompt. It runs on your machine, and mechlens shows you what happened
          inside — which of the model&apos;s features fired, and how the answer took shape
          layer by layer.
        </p>

        {composer ? <div className="mt-8">{composer}</div> : null}

        {/* Reserved either way, so the block below never pushes the page when a
            status arrives. */}
        <div className="mt-6 min-h-6">
          <StatusLine error={error} status={status} />
        </div>

        {status === 'warming' || status === 'pending' || status === 'running' ? (
          <div className="mt-6">
            <GridSkeleton />
          </div>
        ) : null}
      </section>
    </main>
  )
}
