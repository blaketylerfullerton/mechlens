import type { ThemeRegistrationRaw } from 'shiki'

/**
 * The one Shiki theme, built from the same seven syntax colours the rest of
 * the interface is coloured with.
 *
 * It exists so that highlighting is real tokenisation rather than hand-painted
 * words, and so that a keyword in a code block is exactly the purple a keyword
 * is everywhere else. `github-dark` — Shiki's default here — carries its own
 * background (#0d1117) and its own accent set, which would put a second,
 * slightly-off colour system on screen next to this one.
 *
 * `comment` is #7A828F rather than the #5C6370 the palette names: measured on
 * the #101216 code surface that value is 3.10:1, under the 4.5:1 floor, and an
 * unreadable comment is worse than an off-palette one.
 */
export const terminalDark: ThemeRegistrationRaw = {
  name: 'terminal-dark',
  type: 'dark',
  colors: {
    'editor.background': '#101216',
    'editor.foreground': '#E6E8EB',
  },
  settings: [
    { settings: { background: '#101216', foreground: '#E6E8EB' } },
    {
      scope: ['comment', 'punctuation.definition.comment', 'string.comment'],
      settings: { foreground: '#7A828F', fontStyle: 'italic' },
    },
    {
      scope: [
        'keyword',
        'storage',
        'storage.type',
        'keyword.control',
        'keyword.operator.new',
        'keyword.operator.expression',
        'variable.language',
        'entity.name.tag',
        'markup.bold',
      ],
      settings: { foreground: '#C792EA' },
    },
    {
      scope: [
        'string',
        'string.quoted',
        'string.template',
        'punctuation.definition.string',
        'markup.inserted',
        'meta.attribute-selector',
      ],
      settings: { foreground: '#C3E88D' },
    },
    {
      scope: [
        'entity.name.function',
        'support.function',
        'meta.function-call',
        'entity.name.class',
        'entity.name.type',
        'support.class',
        'markup.heading',
      ],
      settings: { foreground: '#82AAFF' },
    },
    {
      scope: [
        'constant.numeric',
        'constant.language',
        'constant.character',
        'keyword.other.unit',
        'variable.parameter',
      ],
      settings: { foreground: '#F78C6C' },
    },
    {
      scope: [
        'constant',
        'support.constant',
        'variable.other.constant',
        'entity.other.attribute-name',
        'support.type.property-name',
        'meta.object-literal.key',
      ],
      settings: { foreground: '#FFCB6B' },
    },
    {
      scope: ['invalid', 'invalid.illegal', 'markup.deleted', 'message.error'],
      settings: { foreground: '#F07178' },
    },
    {
      scope: ['punctuation', 'meta.brace', 'keyword.operator'],
      settings: { foreground: '#9BA1AC' },
    },
    {
      scope: ['variable', 'variable.other', 'meta.definition.variable.name'],
      settings: { foreground: '#E6E8EB' },
    },
  ],
}

export const TERMINAL_DARK_THEME_NAME = 'terminal-dark'
