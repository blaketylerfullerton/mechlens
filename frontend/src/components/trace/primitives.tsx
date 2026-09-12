import type { ReactNode } from 'react'

/**
 * A quantity drawn to scale in the character cell: a 1px rule inside a
 * `ch`-measured track.
 *
 * Not block glyphs (`█▉▊`) — every partial cell caps differently and the run
 * stacks into a chunky rectangle that reads as a rendering artefact. The width
 * is computed from the real value with a small floor, so a near-zero row still
 * prints a mark instead of vanishing.
 */
export function Bar({ accent = false, ratio }: { accent?: boolean; ratio: number }) {
  const width = Math.max(Math.min(ratio, 1), 0.02) * 100
  return (
    <span aria-hidden="true" className="inline-block w-[8ch] shrink-0 align-middle">
      <span
        className={`block h-px ${accent ? 'bg-fn' : 'bg-rule'}`}
        style={{ width: `${width.toFixed(1)}%` }}
      />
    </span>
  )
}

/** Term left, value right, hairline between. The shape a card grid replaced. */
export function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="divide-border-subtle divide-y">
      {rows.map(([term, value]) => (
        <div className="flex items-baseline justify-between gap-4 py-1.5" key={term}>
          <dt className="text-text-tertiary text-[12px]">{term}</dt>
          <dd className="text-text-primary font-mono text-[12px] tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** The one section chrome: uppercase label, optional right-aligned note, body. */
export function Panel({
  children,
  note,
  title,
}: {
  children: ReactNode
  note?: string
  title: string
}) {
  return (
    <section className="border-border-subtle bg-bg-surface rounded-[2px] border p-3">
      <SectionLabel
        note={
          note ? (
            <span className="text-text-tertiary font-mono text-[11px] tabular-nums">{note}</span>
          ) : null
        }
      >
        {title}
      </SectionLabel>
      {children}
    </section>
  )
}

/**
 * The label above a section. Extracted because the same three type settings
 * appeared on every heading in this view and drifted between them.
 */
export function SectionLabel({
  children,
  as: Tag = 'h3',
  note,
}: {
  children: ReactNode
  as?: 'h2' | 'h3'
  note?: ReactNode
}) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-3">
      <Tag className="text-text-tertiary text-[11px] font-medium tracking-[0.04em] uppercase">
        {children}
      </Tag>
      {note}
    </div>
  )
}
