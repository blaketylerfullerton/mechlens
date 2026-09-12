---
name: mechlens-design
description: "The mechlens visual direction: dark-only surfaces, a Swiss/International-Style composition on a near-black canvas, the seven-colour syntax palette as the entire accent system with `fn` (#82AAFF) leading, two radii, Inter for human text and JetBrains Mono for machine text, real traces and real numbers as the primary visual, and the rule that outranks the rest — a measurement that qualifies a visible claim is never hidden. Its lineage is Bauhaus (form follows function, ornament is a lie), Swiss/International Style (the grid is the argument) and Apple/Rams (progressive disclosure), synthesised as: hide decisions, never hide facts — complexity is ordered, not removed. The tokens live in `frontend/src/index.css`, which is the source of truth this skill describes; `site/src/index.css` is byte-identical on purpose. Use when building or restyling anything in `frontend/` or `site/` — the trace viewer, the residual map, the chat panel, a shadcn component being vendored in, or the landing page — or when picking a colour, a radius, a font, a figure, or deciding what to show at rest versus behind a disclosure."
---

# mechlens design

The direction for both surfaces in this repo: the app (`frontend/`) and the landing
page (`site/`). They share one ground and one token file.

mechlens shows a language model's internals to someone who reads code for a living.
The interface is dark, precise, monospaced where it counts, and built out of the
model's actual output. Nothing glows, nothing is fake, nothing is dumbed down.

**This is direction, not scaffolding.** It says nothing about what sections a page
has or what order anything goes in. Bring your own structure; apply this to it.

## Lineage: Bauhaus, Swiss, Apple

Three debts, and it is worth being explicit about which part of each we are taking,
because two of them are frequently taken wrongly.

**Bauhaus — form follows function, and ornament is a lie.** Every element on screen
earns its place by doing a job. A border separates, a rule measures, a colour encodes.
Nothing is there to look designed. If an element carries no information, it is not
minimal, it is decoration that has not been deleted yet.

**Swiss / International Style — the grid is the argument.** Müller-Brockmann's claim
was that objective, systematic composition communicates *more* than expressive
composition, not less. One modular grid, flush-left type, numbered sections, hairline
rules, a strict type scale. The structure does the work that a designer's flourish
would otherwise be doing badly. This is why §6 allows two radii and no more, and why
§5 allows two type families and no more: a system with fewer decisions in it is read
faster than one with more.

**Apple — progressive disclosure, by way of Dieter Rams.** "As little design as
possible" never meant as little capability as possible. The Braun radio and the
iPhone both hold a lot of function; what they do is arrange it so the user meets each
piece at the moment it becomes relevant. Complexity is **ordered**, not removed. That
ordering is exactly the disclosure ladder in §3.

### What we take, and what we refuse

mechlens has genuinely irreducible complexity: 98,210 SAE features, 26 layers, a logit
lens, attribution, steering. Pretending otherwise would make the tool useless. So the
synthesis is narrow and specific:

> **Hide decisions. Never hide facts.**

A user should never have to work out what to do next — that is the Apple debt. But the
data itself is the product, and it stays legible, exact, and on screen — that is the
Bauhaus and Swiss debt, and it is the half of Apple's approach we explicitly refuse.

Apple's weakness is that hidden complexity shades into hidden truth: the battery that
cannot be checked, the setting that cannot be found, the confident number with no error
bar. An interpretability tool that did that would be worse than useless, because it
would be *persuasive* and wrong. So we take the arrangement and leave the opacity.
§3.1 is where that refusal is written down, and it is the rule that outranks the rest.

## 1. Source of truth

`frontend/src/index.css` holds every token. **Read it before choosing any value.** It
is commented with the reasoning behind each deviation, and it is more authoritative
than this file — if they disagree, the CSS wins and this file is stale.

`site/src/index.css` is byte-identical below its header comment. If you change a token
in one, change it in the other. One project, one ground: the landing page is dark
because the app is dark, and the click from "Set it up" into the tool must not flash
white.

**There is no light mode.** `color-scheme: dark`, `:root` and `.dark` carry the same
values, and dark is not a `dark:` variant bolted onto a light theme. Do not add one.

## 2. The four commitments

> Show the real thing. Respect the reader. Design in the dark. Ask nothing at rest.

1. **Real data is the visual.** A trace, a token strip, a table of feature activations
   with values someone would act on. This replaces the illustration, the mockup and
   the hero image entirely. Never fabricate a screenshot, a feature label, or a number.
2. **Dark is the design, not a toggle.**
3. **Density is respect.** Tight leading, real specifics, no three words floating in
   90vh of nothing.
4. **Calm at rest (§3).** Density is how tightly you set what you show — not a licence
   to show everything.

## 3. The disclosure ladder

Reduce **decisions**, not facts. This is the Apple/Rams debt from the lineage section,
and the ladder is the mechanism: complexity stays, ordered by when the user needs it.

Every element sits on exactly one rung. Lower than it belongs is clutter; higher is
hiding.

| Rung | What lives here | Persistence |
|---|---|---|
| **1 · Rest** | The primary object, and the single action that starts the work | Always |
| **2 · Pointing** | Detail about the thing hovered or selected | Transient |
| **3 · Asked for** | Panels, settings, alternate modes the user opened | Until closed |
| **4 · Deep** | Raw numbers, methodology, diagnostics, provenance | One click from 3 |

- **One primary object.** In the app that is the trace. Everything else is furniture
  and is sized like furniture.
- **A control appears with its object, not before it.** A legend for colours not on
  screen, a layer selector with no trace loaded: rung 3 material sitting on rung 1.
- **Earn a permanent control.** A setting most users never change is rung 3. A setting
  nobody changes twice is a default.
- **Empty states carry rung 1.** `EmptyState.tsx` says what this shows and the one
  action that fills it. It is the highest-value copy in the product.
- **Disclosure is not animation.** Revealing means it was not rendered before — not
  faded to 40%, not behind a hover that vanishes on touch.

### 3.1 The floor, and it is not negotiable

**Never hide a measurement that qualifies a claim already on screen.** Trustworthiness
is the thing mechlens exists to add to interpretability visualisations, so this rule
outranks everything else here.

If the claim is at rung 1, its caveat is at rung 1 or 2:

- neighbourhood preservation (29% of a feature's nearest neighbours survive the 3D
  projection)
- region coherence against a random baseline (0.465 vs 0.233)
- how many of the fired features are actually drawn (strongest 16 of a mean 78 per cell)
- colour-ramp endpoints printed beside the map — a colour a reader cannot convert back
  to a number is decoration

Controls, methodology and raw tables hide. Caveats do not. A picture nobody measured is
decoration; a measurement the interface buried is worse, because it knew better.

## 4. Colour

Three near-blacks and a hairline. Depth is layered surfaces and 1px borders — **never a
shadow**, which on `#0A0B0D` is either invisible or looks like grime.

The **seven syntax colours are the entire accent system.** The blue that colours a
function in a code block is the same blue on a focus ring and on a selected cell.
Nothing is coloured from outside the palette.

**Seven syntax colours, but not seven UI colours.** In chrome you get `fn` (#82AAFF,
the lead), `err` (#F07178, destructive only) and the greys. `kw`, `str`, `num` and
`const` never leave a code block or a legend. Solid primary buttons are light-on-dark
(`#E6E8EB` on `#0A0B0D`), never accent-filled — one large accent rectangle and the
accent can no longer do quiet work on links, selection and focus.

**Ramps are single-hue.** `heatColor()` in `components/trace/colors.ts` runs one hue
(220.8°, the `fn` hue) from the code surface up to the accent. A rainbow ramp invents
colours the palette does not contain and implies category boundaries where there is
only a continuous L2 norm. The lens-class colours in `lib/lens.ts` are the exception
and are a genuine three-way classification, not a scale — the two never mix on one map.

**Contrast beats palette fidelity.** Several tokens are deliberately lifted off the
canonical values because the canonical ones fail at the size they are used:

- `--color-text-tertiary` is `#7D848F`, not `#6B7280` (4.07:1 on base, and every
  tertiary string here is 10–12px)
- `--color-comment` is `#7A828F`, not `#5C6370` (3.10:1 on the code surface), in the
  CSS **and** in `lib/shiki-theme.ts`

If a palette value fails at its real size, lift it and leave a comment saying why.

**No gradients.** Computed `background-image` gradients are zero. A fade is
`mask-image` — see the `mask-fade-r` / `mask-fade-b` utilities, which paint nothing and
work over any background. Use them only while a region is genuinely scrollable.

## 5. Type

Two families and no third. **Inter** for anything a human wrote; **JetBrains Mono** for
anything the machine produced or the reader could type — tokens, layer indices, feature
IDs, code, log output, numeric columns. Monospace is a material, not a costume: body
copy set in mono because mono "looks technical" is the failure mode.

`--font-heading` exists only as an alias onto the sans face so vendored shadcn parts
keep working. A third family is forbidden; a display serif most of all.

Numeric columns align via mono or `tabular-nums`. Flush-left, numbered sections,
hairline rules and definition lists where a boxed card was the reflex.

## 6. Form

**Two radii, and that is all.** `2px` for everything; `6px` (`--radius-frame`) only for
the outer half of the double-border frame, where `6 − 4 = 2` keeps the arcs concentric
across the 4px gap. Every shadcn alias collapses onto 2px so a vendored component
cannot reintroduce a radius from a different system. Grep for `rounded-[…px]` that
drifted off the scale.

A control never gets a softer corner than the surface it sits on.

## 7. Figures

The strongest small figure is the system's own output: a `<pre>` whose columns are real
fields, with quantities drawn to scale beside them. `Bar` in
`components/trace/primitives.tsx` is the implementation.

**Draw a quantity as a 1px rule in a `ch`-measured track, not block glyphs.** `█▉▊▋`
notches at every partial cell and stacks into a rectangle that reads as a rendering
artefact.

- Compute widths from real values (`v / max`) with a small floor so near-zero rows still
  print a mark. Invented proportions read as a drawing.
- Colour by tier, not category: rules in `--color-rule`, labels `text-tertiary`, accent
  on exactly the one row that is the point.
- Sibling figures get equal line counts and a fixed container height so captions share
  a baseline. Grid at `11px / 1.8`.
- Every chart value must be readable exactly, with units.
- No isometric line art, stacked plates, floating cubes, or tilted mockups.

## 8. Motion

One entrance: the `enter` utility — fade plus a 12px rise, once, on mount. That is the
motion budget.

Nothing loops, shimmers, pulses, or counts up. Copy feedback is instant. Loading uses
shape-matched skeletons with reserved height and zero layout shift.
`prefers-reduced-motion` renders **fully static and complete** — not merely shortened.

## 9. Accessibility

- Every text tier passes 4.5:1 **at the size it is actually used**, not at 16px.
- Graphical elements (rules, borders carrying meaning) pass 3:1.
- Status never depends on colour alone. `describeClassification()` in
  `components/trace/colors.ts` is the pattern: every colour-coded cell has a
  screen-reader sentence that states the class, its gloss, and its confidence.
- Focus rings are visible everywhere and are never removed "because it clashes".
- Hairline scrollbars — an OS-default light scrollbar gives away a dark surface as an
  afterthought.

## 10. Anti-patterns

Zero hits, every time:

- A light SaaS template with `dark:` variants bolted on
- Purple-to-cyan gradients, glow, shader backgrounds, animated noise, vignettes
- Fabricated data: a code sample that does not compile, an invented feature label, a
  plausible-looking number nobody computed
- Monospace body copy
- Shadow-based elevation; pastel cards
- A caveat demoted below rung 2 (§3.1)
- A rainbow ramp over a continuous quantity
- Pill tabs, filled accent buttons, or pastel status colours carried in from a themed
  component kit

When adding a library, prefer headless primitives (Radix, Shiki, CodeMirror, TanStack
Table) styled to §4 and §6 over a themed kit with its own opinions. Strip default
shadows, filled accent buttons, pill tabs and pastel status colours before use.

## 11. Self-verification

Re-read the rendered output. If any item fails, fix it and run the loop again. Do not
report completion with known failures.

- [ ] Canvas `#0A0B0D`, not `#000000`; text `#E6E8EB`, not `#FFFFFF`.
- [ ] Depth is layered surfaces and hairlines. Zero shadows.
- [ ] Accents are syntax-palette only; `fn` leads; `kw`/`str`/`num`/`const` never left a
      code block or legend.
- [ ] Computed gradients: zero. Fades are `mask-image`.
- [ ] Exactly two radii in the build.
- [ ] Prose, titles, buttons and labels are sans; mono marks machine content only.
- [ ] Numeric columns align.
- [ ] Every figure is computed from real values; quantities are 1px rules in `ch` tracks.
      Zoom to 400% and confirm no notched edges.
- [ ] One primary object and one obvious action at rest; a first-time user knows what to
      do in two seconds.
- [ ] Every control's object exists before the control renders.
- [ ] **Every caveat on a rung-1 claim is still at rung 1 or 2.** Check this last and
      check it honestly — it is the rule this design is most likely to have eaten.
- [ ] Nothing loops. `prefers-reduced-motion` is static and complete.
- [ ] Contrast checked at real sizes; status never colour-alone; focus visible.
- [ ] Token change applied to **both** `frontend/src/index.css` and `site/src/index.css`.
- [ ] Anti-patterns (§10): re-read the list, zero hits.
- [ ] Smell test: would an engineer who reads code for a living trust this, or close it
      in two seconds?
