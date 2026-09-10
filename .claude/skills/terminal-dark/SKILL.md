---
name: terminal-dark
description: "A visual direction for developer-facing interfaces: dark-first surfaces, hairline borders, layered near-blacks instead of shadows, syntax colours as the entire accent system, monospace used as a material rather than a costume, and real code and data treated as the primary visual. Use when the user is building or restyling anything developer-adjacent — a tool, console, CLI site, editor, dashboard, docs, internal app — or mentions dark-first design, terminals, syntax highlighting, or wanting something that reads as software rather than marketing. This skill sets look and feel only; it prescribes no page or app structure."
---

# Terminal Dark

A visual direction for interfaces developers look at. The tool lives in a terminal, an
editor, a dark IDE, so the interface is dark, precise, monospaced where it counts, and
built around real data. It assumes technical literacy. Nothing is dumbed
down, nothing glows, nothing is fake.

**This is direction, not scaffolding.** It says nothing about what sections a page has,
what a shell layout looks like, or what order anything goes in. Bring your own
structure; apply this to it.

The failure mode you are guarding against is the "developer" interface that is a light
SaaS template with `dark:` variants bolted on: a glowing purple-to-cyan gradient, a
code sample that does not compile, body copy set in monospace because monospace "looks
technical", pastel cards with soft shadows, and a fabricated dashboard nobody asked
for. Someone who reads code for a living stops trusting that in two seconds.

Design dark-first. Light mode is optional and secondary, and it is a re-derivation, not
an inversion. Read the anti-patterns (§10) before you write a line, then build the
colour system first; every other rule depends on it.

## 1. The three commitments

> Show the real thing. Respect the reader. Design in the dark.

1. **Real data is the visual.** The most persuasive element is a block
   that compiles, or a table with values someone would act on. It replaces the
   illustration, the mockup, and the hero image entirely.
2. **Dark is the design, not a toggle.** Surfaces, borders, text tiers and syntax
   colours are chosen together for a dark canvas.
3. **Density is respect.** Developers scan fast and resent padding. Tight leading, real
   specifics, no three words floating in 90vh of nothing.

## 2. Colour

A dark layered system plus a syntax theme. Depth comes from layers, never shadows.
Every value is exact.

```
Dark (primary design)
  bg-base        #0A0B0D   canvas: the deepest layer
  bg-surface     #101216   raised surfaces ( panels, table headers)
  bg-elevated    #16181D   menus, popovers, drawers, the highest layer
  border-subtle  #1E2127   hairlines between rows, panes, sections
  border-strong  #2A2E37   focused inputs, active tabs, emphasised dividers
  text-primary   #E6E8EB   headings, values, primary copy
  text-secondary #9BA1AC   body copy, descriptions
  text-tertiary  #6B7280   captions, metadata, column headers, inactive
  text-disabled  #454B54   disabled, placeholder, line numbers

Syntax palette (the ONLY accent colours, used in UI)
  syntax-keyword  #C792EA   keywords, primary accent option
  syntax-string   #C3E88D   strings, success / healthy / passing
  syntax-func     #82AAFF   functions, links, selection, focus
  syntax-number   #F78C6C   numbers, warning / degraded
  syntax-comment  #5C6370   comments, muted annotation (verify 4.5:1)
  syntax-const    #FFCB6B   constants, pending / attention
  syntax-error    #F07178   errors, destructive, failing
```

- **Three near-blacks `~6px` apart in lightness.** A raised element is *lighter* than
  its parent and edged with a `border-subtle` hairline. Do not add `shadow-lg`. A scrim
  behind a dialog (`bg-black/60`) is a dim, not a shadow, and is fine.
- **The syntax palette is the entire accent system.** The purple that colours keywords
  is the same purple on a button. Status colours come from the same theme. No brand
  colour invented outside it. This coherence is the whole point.
- **Never `#000000`, never `#FFFFFF` for large surfaces.** Pure black on pure white
  vibrates and reads as unstyled.
- **One accent leads.** Pick one syntax colour (usually `syntax-func` or
  `syntax-keyword`) for links, active tabs, focus rings, selection, and the one marked
  row in a figure. 
- **Solid primary buttons are light, not accent-filled:** `bg-[#E6E8EB]` on
  `text-[#0A0B0D]`. A large accent rectangle is the loudest thing on a dark canvas, and
  once it exists the accent can no longer do quiet work elsewhere without the interface
  reading as two competing signals. Secondary is a hairline border with `text-primary`;
  destructive is `text-[#F07178]` on a hairline border, filled only inside a confirm.
- **No gradients.** The only exceptions are `mask-image` fades (§6.2, §11), which paint
  nothing.

## 3. Monospace: material, not decoration

Using monospace everywhere is the single loudest "fake developer page" signal.

**Use it for:** config, queries, payloads, terminal commands and output; machine
values (IDs, hashes, paths, package names, env vars, hosts, ports, status codes,
durations, byte sizes, versions); timestamps and key bindings; numeric table columns
where fixed-width alignment carries meaning.

**Never use it for:** body copy, descriptions, help text, empty-state prose; headlines
and titles; buttons, nav items, menu and form labels.

The discipline: **monospace marks something the reader could type or that the machine
produced.** If a human wrote it as prose, it is set in the sans face.

## 4. Type

Two families: a sans UI face for everything human-written, a mono face for everything
machine-adjacent. Never a third. Scale down the whole column for dense app-like
surfaces; the ratios hold either way.

| Token | Marketing-scale | App-scale | Family | Weight | Leading |
|---|---|---|---|---|---|
| `h1` | `clamp(36px, 5vw, 60px)` | `28px` | sans | 600 | `1.05–1.2` |
| `h2` | `clamp(26px, 3vw, 38px)` | `22px` | sans | 600 | `1.15–1.25` |
| `h3` | `20px` | `16px` | sans | 500 | `1.3` |
| `body` | `16px` | `14px` | sans | 400 | `1.55–1.6` |
| `small` | `14px` | `13px` | sans | 400 | `1.45` |
| `micro` | — | `11px` uppercase, `tracking-[0.04em]` | sans | 500 | `1.4` |
| `code-sm` | `13px` | `12px` | mono | 400 | `1.5` |

- **Sans:** Inter, Geist, or IBM Plex Sans. **Mono:** JetBrains Mono, Geist Mono, IBM
  Plex Mono, or Berkeley Mono. One of each, never more.
- **`font-semibold` (600) is the ceiling.** Developer aesthetics are precise, not loud.
- **Code sits at `leading-[1.6]`.** Never `leading-tight`; dense text needs vertical air.
- Prose caps at `max-w-[70ch]`. Code is as wide as the code demands and scrolls (§11).
- Numerals in data columns use mono or `tabular-nums` so digits align down the column.

## 5. Terminal, and data as the visual

Get these right and the interface is credible; get them wrong and nothing else saves it.

**Framing:** `bg-surface`, `1px border-subtle`, `rounded-lg` (`8px`), no drop shadow. A
slim header bar carrying a filename or resource label and a format tag in
`code-sm text-tertiary`. Prefer a filename tab to window dots; dots belong only on an
actual terminal.

**Non-negotiables:**
- **A copy button on every block**, icon swapping to a check, `text-tertiary` hover
  `text-secondary`. A block a developer cannot copy is broken.
- **Line numbers `select-none`** in `text-disabled`, right-aligned in a small gutter, so
  copying grabs only the content. Same for timestamp gutters in logs.
- **Real tokenisation** (Shiki, CodeMirror, Monaco) using the §2 palette. Never
  hand-coloured words or a runtime regex; wrong tokenisation is spotted instantly.
- **Real content.** Code compiles and shows the actual API; data is actual data. One
  comment that adds insight (`// streams tokens as they arrive`), never one that
  narrates the obvious. `8–16` lines when it is a showcase: not a wall, not a toy.
- **Highlight a changed or matched line** with `border-l-2 border-syntax-func` plus a
  `bg-syntax-func/5` row tint. Never a bright full-width band.

**Terminal and stream output:** a real prompt glyph (`$`, `>`, or `user@host`) in
`text-tertiary`; typed input in `text-primary`; output in `text-secondary`; success
lines may take `syntax-string`, errors `syntax-error`. Never one flat green — that is a
Hollywood terminal. Real flags, plausible output, a real exit. A one-time type-on when
a block enters view is acceptable; an infinite loop is a toy.

**Tables and values:** hairline rows (`divide-y divide-[#1E2127]`), a `bg-surface`
header in `micro text-tertiary`, no vertical rules, no zebra striping, no card per row.
Hover `bg-white/[0.02]`; selection `bg-[accent]/[0.06]` plus a `2px` accent left rule.
Status is a `6px` dot plus a word in the status colour, never a filled pill. A
hairline definition list — term left, value right in `code-sm` — beats a card grid for
attributes, and is a shape a template never produces.

## 6. Borders and elevation

Shadows are a light-UI device. On `#0A0B0D` they are invisible, or, made visible, they
look like grime.

- **Separation comes from a lighter surface, a hairline, and negative space**, in that
  order. Reach for a border before a shadow every time.
- **Hairlines are `1px` at `#1E2127`.** They are the workhorse of dark-UI depth.
- **Elevation is lightness.** A popover or drawer is `bg-elevated` with a
  `border-strong` edge; that step *is* the elevation cue.
- **The one permitted glow:** a `1px` accent ring (`ring-1 ring-syntax-func/40`) on a
  focused input or active tab. A focus cue, not decoration, and the only place a syntax
  colour touches a border.
- **Prefer a vertical hairline rule to a boxed card** whenever things are parts of one
  idea rather than genuinely independent objects. Boxes are what the template reaches
  for.

### 6.1 The double border

Use exactly **one** structural motif and repeat it: an outer frame holding an inner
surface, hairline on both, `4px` gap.

```html
<div class="rounded-[16px] border border-[#1E2127] bg-[#0D0E11] p-1">
  <div class="rounded-[12px] border border-[#1E2127] bg-[#101216] overflow-hidden">
    <!--terminal, table, diff, chart -->
  </div>
</div>
```

- **Radii are concentric: `inner = outer − gap`.** `16 − 4 = 12`. Eyeballing it produces
  a visible wobble where the arcs disagree. Scale: `16` outer frame, `12` inner surface
  and dialogs, `10` nested panel, `8` buttons and inputs, `6` tabs and chips, `2` inline
  code. Never `rounded-3xl` on a technical surface; `rounded-full` only on avatars and
  status dots.
- **The frame wraps what the reader looks *into*** — the editor, terminal, diff, table,
  chart. Prose, toolbars and forms never get a frame.
- **One frame treatment throughout.** Two read as two different products.

### 6.2 Mask out, do not cut off

A surface that continues past its bounds fades rather than ending on a hard edge:

```css
mask-image: linear-gradient(to bottom, #000 92%, transparent 100%);
```

`mask-image`, not `background-image`: nothing is painted, so the no-gradients rule
holds and it works on any background. Apply it only while the region is genuinely
scrollable.

## 7. Motion

Precise and functional, matching the aesthetic of a well-built tool.

| Interaction | Duration | Easing |
|---|---|---|
| Fade + `12px` rise on view, once | `400ms` | `cubic-bezier(0.16, 1, 0.3, 1)` |
| Menu / popover fade | `140ms` | `ease-out` |
| Drawer or pane slide | `200ms` | `cubic-bezier(0.16, 1, 0.3, 1)` |
| Copy / toggle feedback | `150ms` | `linear` |
| Tab or content cross-fade | `120–150ms` | `ease-out` |
| Live-value flash decay | `400ms` | `ease-out` |

- **Copy feedback is the most-used interaction anywhere in this style.** Instant icon
  swap, never blocked behind a toast.
- **Loading is a skeleton at the shape of the content** — hairline blocks on
  `bg-surface` at `~40%` opacity with reserved height — not a spinner over an empty
  region.
- **Reserve height before anything switches or arrives** so nothing shifts.
- Only animate `transform` and `opacity`.
- **Never:** pulsing glows, aurora or mesh backgrounds, particles, looping fake typing,
  gradient shimmer across text, count-up numbers, parallax, cursor trails. A developer
  tool that sparkles reads as unserious.

## 8. Imagery and figures

- **The primary image is text**: real highlighted, selectable, copyable data. A
  PNG of a  editor is the amateur move.
- **Icons are stroke, monochrome, `16px`,** `text-tertiary` brightening to
  `text-secondary`. Never filled coloured tiles, never mixed icon families, never emoji
  as UI iconography.
- **Screenshots, if any, are real, dark, uncluttered, and straight-on** in a hairline
  `bg-surface` frame. Never a fabricated dashboard, never a light-mode screenshot on a
  dark canvas, never a 3D-tilted browser or floating device mockup.
- **Charts are line and hairline work.** One accent series, the rest in grey tiers, axis
  labels in `micro text-tertiary`, horizontal gridlines at `border-subtle` only. No
  gradient fills, no shadows, no 3D, no donut where a value would do. Every value must
  be readable exactly, with units.
- Diagrams are line drawings in `text-secondary` strokes on the canvas.

### 8.1 Draw quantities in the character cell

The strongest small figure is the system's own output: a `<pre>` whose columns are real
fields, with quantities drawn to scale beside them.

**Draw the quantity as a 1px rule, not block glyphs.** `█▉▊▋` is the obvious answer and
a trap: every partial cell caps differently, joins notch, and at real line-height the
run stacks into a chunky rectangle that reads as a rendering artefact. Use a hairline in
a `ch`-measured track, which keeps the bar on the character grid without going through
the font.

```html
<span style="display:inline-block;width:12ch;vertical-align:middle">
  <span style="display:block;width:64%;height:1px;background:#6E7681"></span>
</span>
```

```
OFFSET     EXTENT         SEGMENT   STATE
0x000200   ___________    seg.0000  sealed
0x04a180   _______        seg.0001  sealed
0x0c9a80   __             seg.0004  open      <- accent, 1px, syntax-func
```

- **Compute widths from real values** (`v / max`) with a small floor so near-zero rows
  still print a mark. Invented proportions read as a drawing.
- **Colour by tier, not category:** rules in mid grey, labels `text-tertiary`, accent on
  exactly the one row that is the point.
- **Sibling figures get equal line counts** and a fixed container height so captions
  share a baseline.
- Set the grid at `11px / 1.8`.
- Avoid isometric line art, stacked plates and floating cubes: the house style of
  several large developer-tool sites, and instantly recognisable as borrowed.

## 9. Copy and content tone

- Say the specific thing. "Type-safe" is worthless; two lines showing the inferred type
  in a comment is proof.
- No marketing verbs — "supercharge", "seamless", "all-in-one", "effortlessly".
- Empty states are text: what is not here, why, and the command or action that changes
  it, with the command in a real copyable block. Never a cartoon illustration.
- Errors name the failure, carry the identifier (request ID, trace ID, exit code) in
  `code-sm` with a copy button, and say the next step. Never "Something went wrong".
- Numbers carry their units, window, and source.

## 10. Anti-patterns

Each alone is enough to break the style. Most are agent defaults.

**Colour and surface**
1. A glowing purple-to-cyan gradient anywhere.
2. Pure `#000000` canvas or pure `#FFFFFF` surfaces.
3. `dark:` variants bolted onto a light template instead of a dark-first design.
4. Drop shadows for elevation on a dark canvas.
5. A brand accent invented outside the syntax palette.
6. More than one leading accent; rainbow icons in coloured tiles.
7. Filled accent buttons that outshout links, selection and status.
8. Status as filled pills in six saturated colours.
9. Aurora, mesh, or animated-blob backgrounds.

**Code and data**
10. Code that does not compile, pseudocode, or lorem-ipsum-in-code.
11. A screenshot PNG of code instead of real selectable text.
12. Hand-coloured or regex highlighting with wrong tokenisation.
13. A code, log, or payload block with no copy button.
14. Line numbers or timestamps that copy along with the content.
15. A flat all-green terminal with no input/output distinction.
16. An infinitely looping fake typing animation.
17. A 40-line wall of code as a showcase, or a 3-line toy.
18. Fabricated, rounded, or placeholder data presented as real.
19. Charts whose exact values cannot be read.

**Typography**
20. Body copy, descriptions, or empty-state prose in monospace.
21. A monospace headline or title.
22. Monospace buttons, nav items, or form labels.
23. Numeric columns in proportional figures, so digits fail to align.
24. `font-extrabold` display headings.
25. Code at `leading-tight`.
26. Three or more type families.

**Composition**
27. Feature or stat cards with pastel backgrounds and soft shadows.
28. A fabricated dashboard or 3D-tilted browser mockup.
29. Boxed cards where a vertical hairline rule would have done the same job.
30. Two different frame treatments, or an inner radius that is not `outer − gap`.
31. A surface ending at a hard edge mid-content instead of masking out.
32. Isometric line art, stacked plates, floating cubes.
33. Figures whose proportions were drawn rather than computed.
34. Bars built from block glyphs (`█▉▊`) instead of a 1px rule in a `ch` track.

**Motion**
35. Count-up animations on numbers.
36. Gradient shimmer, parallax, particles, cursor trails.
37. Centred spinners over empty regions instead of shape-matched skeletons.

## 11. Code on narrow widths

Code does not reflow, and forcing it to wrap destroys meaning. Solve this explicitly.

- **Never soft-wrap code by default.** Wrapped code changes indentation and reads as
  broken. Scroll horizontally inside the frame (`overflow-x-auto`, plus
  `-webkit-overflow-scrolling: touch`), with wrap as an explicit toggle that persists.
- **Show the scroll affordance:** a `24px` right-edge fade, via `mask-image`.
- **Drop code one step on narrow screens** (`13px` → `12px`), no smaller.
- **Keep line numbers**, narrowing the gutter; they anchor horizontal scrolling.
- Consider collapsing a long showcase block to its most important `6–8` lines with a
  "show full" affordance rather than a giant scroll region.
- Touch targets `44×44px` minimum, especially the copy button. Verify at 375 that a real
  sample scrolls cleanly without breaking layout, and in a `~420px` side pane.

## 12. Accessibility

- **Dark contrast is the main risk.** `text-secondary #9BA1AC` on `#0A0B0D` is `7.4:1`
  (AAA); `text-tertiary #6B7280` on `#0A0B0D` is `4.6:1`, which passes at `16px`+ only.
  If you use tertiary at `11–13px` — common in dense UI — lighten it to `~#7D848F` or
  move it onto `bg-surface`.
- **Syntax colours must clear 4.5:1 on `bg-surface`.** `syntax-comment #5C6370` is the
  danger case; lighten it rather than shipping unreadable code.
- **Never convey status by colour alone.** Pair every status colour with a word, shape,
  or icon.
- **Focus rings stay visible:** `ring-2 ring-syntax-func` offset `2px`. Never removed
  "because they clash" on dark.
- **Real semantics:** `<pre><code>` with a `lang` attribute, real `<table>`, buttons that
  are buttons, `aria-label` on icon-only controls, `aria-live="polite"` on copy
  confirmation and background updates.
- **Full `prefers-reduced-motion`:** type-on becomes instant full text, fades become
  static, value flashes become a static marker.
- **Light mode, if it exists, is re-derived:** `#F7F8FA` surfaces, `#1A1C20` text, syntax
  colours re-picked to pass on light. Never a pure inversion.

## 13. Performance

- **Highlight ahead of time, not on every render.** Pre-tokenise at build time where the
  content is static; use a worker or streamed chunks for large dynamic input. A runtime
  highlighter re-tokenising a 5k-line file is the most common freeze in this style.
- **Virtualise anything unbounded** — long logs, large payloads, big tables. Nothing
  that can exceed a few hundred rows mounts fully.
- **Reserve height** rows and charts so hydration and tab switches cost
  zero layout shift.
- **Subset both fonts** (mono + sans, needed weights, `< 120KB` combined),
  `font-display: swap` with metric-matched fallbacks, so code does not reflow on font
  load and shift every line number.
- Animate only `transform` and `opacity`; type-on uses `opacity`/`clip`, not width.

## 14. Tokens

```css
@theme {
  --color-bg-base:      #0A0B0D;
  --color-bg-surface:   #101216;
  --color-bg-elevated:  #16181D;
  --color-border-subtle:#1E2127;
  --color-border-strong:#2A2E37;
  --color-text-primary: #E6E8EB;
  --color-text-secondary:#9BA1AC;
  --color-text-tertiary:#6B7280;

  --color-kw:    #C792EA;
  --color-str:   #C3E88D;
  --color-fn:    #82AAFF;
  --color-num:   #F78C6C;
  --color-const: #FFCB6B;
  --color-err:   #F07178;

  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;
}
```

If a component library is already in the project, prefer headless primitives (Radix,
Ark, CodeMirror, TanStack Table, Shiki) styled to §2 and §6 over a themed kit with its
own opinions. Strip default shadows, filled accent buttons, pill tabs and pastel status
colours before use, and never let a pre-built block reintroduce a gradient, fake code,
or monospace body copy.

## 15. Self-verification loop

Re-read the rendered output and check every item. If any fails, fix it and run the loop
again. Do not report completion with known failures.

**Colour and surface**
- [ ] Canvas is `#0A0B0D`, not `#000000`; text is `#E6E8EB`, not `#FFFFFF`.
- [ ] Depth comes from layered surfaces and hairlines, not shadows.
- [ ] The only accents are syntax-palette colours; one leads; status uses the same theme.
- [ ] Computed `background-image` gradients are zero; any fade is `mask-image`.

**Code and data**
- [ ] Highlighting is correctly tokenised using the §2 palette.
- [ ] Every block has a working copy button; gutters are `select-none`.
- [ ] Terminal and log output distinguishes input from output; nothing loops.
- [ ] Every chart value is readable exactly, with units.

**Typography**
- [ ] Prose, titles, buttons and labels are sans; monospace only marks machine content.
- [ ] Numeric columns align via mono or `tabular-nums`.

**Form**
- [ ] Every frame uses the same outer/inner pair; inner radius equals `outer − gap`.
- [ ] Hairline rules and definition lists are used where boxed cards were the reflex.
- [ ] Surfaces that continue past their bounds mask out rather than cutting off.
- [ ] Figures are computed from real numbers; quantities are 1px rules in `ch` tracks.
      Zoom to 400% and confirm no notched or stepped edges.
- [ ] No fabricated screenshot, tilted mockup, or isometric line art.

**Motion**
- [ ] Nothing loops, shimmers, pulses, or counts up. Copy feedback is instant.
- [ ] Loading uses shape-matched skeletons with reserved height; zero layout shift.
- [ ] `prefers-reduced-motion` renders fully static and complete.

**Anti-patterns (§10)**
- [ ] Re-read the full list. Zero hits.
- [ ] Specifically confirm: no purple-cyan gradient, no fake data, no monospace prose,
      no shadow-based elevation, no pastel cards.

**Responsive, a11y, performance**
- [ ] Every text tier passes 4.5:1 at the size it is actually used; tertiary at small
      sizes was checked and lightened where needed.
- [ ] Status never depends on colour alone; focus rings visible everywhere.
- [ ] Highlighting is not blocking the main thread; unbounded lists are virtualised.

**Smell test**
- [ ] Would an engineer who reads code for a living trust this, or close it in two
      seconds? If the code is fake or the monospace is everywhere, they close it.