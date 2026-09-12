import type { TokenStep } from '@/lib/api-types'

import { visibleToken } from './format'
import { SectionLabel } from './primitives'

export function TokenStrip({
  steps,
  selectedPosition,
  onSelect,
}: {
  steps: TokenStep[]
  selectedPosition: number
  onSelect: (position: number) => void
}) {
  return (
    <section>
      <SectionLabel
        as="h2"
        note={
          <span className="text-text-tertiary flex items-center gap-1.5 text-[11px]">
            <span aria-hidden="true" className="bg-kw size-1.5 rounded-full" />
            generated
          </span>
        }
      >
        Token sequence
      </SectionLabel>
      <div className="mask-fade-r overflow-x-auto pb-2 [-webkit-overflow-scrolling:touch]">
        <div className="flex min-w-max gap-1">
          {steps.map((step) => {
            const isSelected = step.step === selectedPosition
            const tone = isSelected
              ? 'border-fn bg-fn/[0.08] text-text-primary'
              : step.token.source === 'generated'
                ? 'border-kw/30 text-kw hover:border-kw/60'
                : 'border-border-subtle text-text-secondary hover:border-border-strong'

            return (
              <button
                aria-label={`Select token ${step.step}: ${step.token.text || 'empty token'}`}
                aria-pressed={isSelected}
                className={`rounded-sm border px-2 py-1.5 font-mono text-[12px] transition-colors duration-150 ${tone}`}
                key={step.step}
                onClick={() => onSelect(step.step)}
                type="button"
              >
                <span className="text-text-disabled mr-1.5 text-[10px] tabular-nums">
                  {step.step}
                </span>
                {visibleToken(step.token.text)}
              </button>
            )
          })}
        </div>
      </div>
    </section>
  )
}
