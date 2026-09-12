/** Rendering helpers shared by every part of the trace view. */

/**
 * Tokens are whitespace-bearing and often empty. Printing them raw makes a
 * space indistinguishable from an empty string, so the invisible characters
 * get visible stand-ins.
 */
export function visibleToken(text: string): string {
  const token = text.replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥')
  return token || '∅'
}

export function formatNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}
