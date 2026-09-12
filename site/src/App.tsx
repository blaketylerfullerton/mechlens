import { useState } from 'react'
import { IconBrandGithub, IconCheck, IconCopy } from '@tabler/icons-react'

const REPO = 'https://github.com/blaketylerfullerton/mechlens'

/**
 * The mechlens landing page.
 *
 * Deliberately one file and plain markup. The composition is Swiss: one
 * modular grid, a hairline rule between every section, a micro label in the
 * left column and the content flush-left in the right, and exactly one accent
 * (`fn`) against the greys. Nothing is centred and nothing is decorated — the
 * page is set, not styled.
 *
 * Swiss on a dark ground, which is not a contradiction: the grid, the rules,
 * the flush-left type and the single accent are the style, and none of them
 * need paper. The ground is dark because the app is dark — this page exists to
 * show the app, so the two cannot disagree about what the product looks like.
 *
 * Every number on this page is measured and comes from the README. There is no
 * illustration, because the figures *are* the illustration: a real logit-lens
 * readout and a real measurement table carry the argument that a rendered
 * graphic would only gesture at.
 *
 * Tokens are the app's own, byte-identical (`src/index.css`) — so anything
 * built for `frontend/` renders here unchanged, and vice versa.
 */
function App() {
  return (
    <div className="bg-bg-base text-text-primary min-h-screen">
      <Nav />
      <main>
        <Hero />
        <Lens />
        <Measured />
        <Setup />
      </main>
      <Footer />
    </div>
  )
}

function Nav() {
  return (
    <nav className="border-border-subtle bg-bg-base/80 sticky top-0 z-10 border-b backdrop-blur-md">
      <Container className="flex h-14 items-center justify-between">
        {/* The wordmark is the one place mono stands in for a logotype. Nav
            items beside it are sans — mono on a nav item is the costume. */}
        <span className="inline-flex items-center gap-2 font-mono text-sm tracking-tight">
          <img src="/logo.svg" alt="" width={24} height={24} />
          mechlens
        </span>
        <a
          className="text-text-secondary hover:text-fn inline-flex items-center gap-2 text-sm transition-colors duration-150"
          href={REPO}
        >
          <IconBrandGithub className="size-4" stroke={1.5} />
          GitHub
        </a>
      </Container>
    </nav>
  )
}

function Hero() {
  return (
    <Section label="01" title="Mechlens" titleHidden>
      <div className="enter">
        <h1 className="max-w-[22ch] text-4xl leading-[1.05] font-semibold tracking-tight sm:text-5xl lg:text-6xl">
          Look inside a language model on your own machine.
        </h1>

        <p className="text-text-secondary mt-8 max-w-[62ch] text-base leading-relaxed">
          Send a prompt and see which of the model&rsquo;s features actually fired &mdash;
          not neurons, but the sparse-autoencoder features that have human-readable
          names. What each one is called, where it sits among the other{' '}
          <Val>98,210</Val> the atlas covers, and how the answer took shape layer by
          layer. The model, the features, the labels and the layout all run locally.
          Nothing leaves the machine.
        </p>

        <p className="text-text-secondary mt-5 max-w-[62ch] text-base leading-relaxed">
          It also tells you how much to trust the picture, which is the part most
          interpretability tools leave out. Every claim on screen carries its
          measurement.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <a
            className="bg-primary text-primary-foreground rounded-xs px-4 py-2 text-sm font-medium transition-opacity duration-150 hover:opacity-90"
            href="#setup"
          >
            Set it up
          </a>
          <a
            className="border-border-strong text-text-secondary hover:border-fn hover:text-text-primary inline-flex items-center gap-2 rounded-xs border px-4 py-2 text-sm font-medium transition-colors duration-150"
            href={REPO}
          >
            <IconBrandGithub className="size-4" stroke={1.5} />
            Clone the repo
          </a>
        </div>

        <p className="text-text-tertiary mt-6 font-mono text-xs">
          gemma-2-2b · 26 layers · Gemma Scope 16k · CUDA · Apache-2.0
        </p>
      </div>
    </Section>
  )
}

/**
 * The logit lens, as the tool actually prints it.
 *
 * The quantities are 1px rules in a `ch`-measured track, computed from the
 * probabilities rather than drawn — block glyphs (`█▉▊`) cap differently per
 * partial cell and stack into something that reads as a rendering artefact.
 * One row takes the accent: layer 20, where the answer locks in.
 */
const LENS_ROWS: {
  layer: number
  token: string
  prob: number
  entropy: number
  answer: boolean
  echo: boolean
}[] = [
  { layer: 3, token: "' of'", prob: 71.81, entropy: 1.76, answer: false, echo: true },
  { layer: 7, token: "'بوابة'", prob: 37.97, entropy: 3.94, answer: false, echo: false },
  { layer: 17, token: "' city'", prob: 40.87, entropy: 2.6, answer: false, echo: false },
  { layer: 19, token: "' Paris'", prob: 22.47, entropy: 3.15, answer: true, echo: false },
  { layer: 20, token: "' Paris'", prob: 92.66, entropy: 0.46, answer: true, echo: false },
  { layer: 25, token: "' Paris'", prob: 92.59, entropy: 0.6, answer: true, echo: false },
]

const ACCENT_LAYER = 20

function Lens() {
  return (
    <Section label="02" title="The logit lens">
      <p className="text-text-secondary max-w-[62ch] text-base leading-relaxed">
        Ask the model&rsquo;s own output head what it would answer at layer L instead of
        at layer 25, at every depth, and you can watch a fact resolve. Below: token 8 of{' '}
        <Code>The Golden Gate Bridge is located in the city of</Code>, where the model
        answers <Val>&nbsp;Paris</Val> at <Val>92.6%</Val>.
      </p>

      {/* The one frame treatment: outer frame, inner surface, hairline on
          both, 4px gap, radii concentric (6 − 4 = 2). It wraps what the reader
          looks into and nothing else. */}
      <div className="border-border-subtle bg-bg-elevated mt-8 rounded-[6px] border p-1">
        <div className="border-border-subtle bg-bg-surface overflow-hidden rounded-xs border">
          <div className="border-border-subtle text-text-tertiary flex items-center justify-between border-b px-3 py-2">
            <span className="font-mono text-xs">trace 8e2c · token 8 · lens</span>
            <span className="text-[11px] font-medium tracking-[0.04em] uppercase">
              26 layers
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] border-collapse font-mono text-xs tabular-nums">
              <caption className="sr-only">
                Logit-lens readout by layer: the top predicted token, its probability,
                and the distribution&rsquo;s entropy.
              </caption>
              <thead>
                <tr className="text-text-tertiary text-left">
                  <Th>Layer</Th>
                  <Th>Top token</Th>
                  <Th align="right">Prob</Th>
                  <Th align="right">Entropy</Th>
                  <Th>Reading</Th>
                </tr>
              </thead>
              <tbody className="divide-border-subtle divide-y">
                {LENS_ROWS.map((row) => {
                  const accent = row.layer === ACCENT_LAYER
                  return (
                    <tr
                      key={row.layer}
                      className={accent ? 'border-l-fn bg-fn/[0.04] border-l-2' : ''}
                    >
                      <Td className={accent ? 'text-fn pl-[calc(0.75rem-2px)]' : ''}>
                        L{row.layer}
                      </Td>
                      <Td className="text-text-primary">{row.token}</Td>
                      <Td align="right">{row.prob.toFixed(2)}%</Td>
                      <Td align="right" className="text-text-tertiary">
                        {row.entropy.toFixed(2)}
                      </Td>
                      <Td>
                        <span className="flex items-center gap-2">
                          <Bar accent={accent} ratio={row.prob / 100} />
                          <span className="text-text-tertiary whitespace-nowrap">
                            {row.answer ? 'final answer' : row.echo ? 'echo' : 'neither'}
                          </span>
                        </span>
                      </Td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <p className="text-text-tertiary mt-4 max-w-[62ch] text-sm leading-relaxed">
        Layer 25 is <Code>resid_post</Code> of the last block &mdash; precisely what{' '}
        <Code>forward</Code> hands to <Code>ln_final</Code>. So the last layer&rsquo;s
        lens is not an approximation of the model&rsquo;s output, it <em>is</em> the
        output, recomputed: on all five saved traces it reproduces the model at every
        position, with a max-probability delta of <Val>0.0</Val>. An early layer that
        &ldquo;predicts&rdquo; the token already at the position is an echo of that
        token&rsquo;s embedding, not early certainty &mdash; which is why the reading is
        printed rather than left to the colour.
      </p>
    </Section>
  )
}

/**
 * The measurements, as a hairline definition list: term left, value right.
 *
 * A card grid is the reflex here and it is the wrong shape — these are rows of
 * one table, not eight independent objects, and a rule says so more cheaply
 * than eight boxes.
 */
const MEASUREMENTS: [string, string, string][] = [
  ['Atlas', '98,210', 'features over 6 layers, reproducible from seed'],
  ['kNN preservation', '0.292', "share of a feature's neighbours surviving 3D"],
  ['Cluster coherence', '0.465', 'against a random baseline of 0.233'],
  ['Named regions', '27 / 32', 'clusters earning a name at that coherence'],
  ['SAE sparsity', '78.3', 'mean L0 — features firing per token, per layer'],
  ['Explained variance', '0.880', 'mean across layers, range 0.83–0.96'],
  ['Label coverage', '6234 / 6234', 'distinct features on golden-gate, no network'],
  ['Lens agreement', '1.0', 'layer 25 reproduces the model bit-identically'],
  ['Attribution gap', '≤1.3e-2', 'reconstruction error across all five traces'],
  ['Tests', '316 passed', '1 skipped, ~79s'],
]

function Measured() {
  return (
    <Section label="03" title="Measured">
      <p className="text-text-secondary max-w-[62ch] text-base leading-relaxed">
        A picture nobody measured is decoration. These are the numbers behind the ones on
        screen, taken from the traces in <Code>backend/traces/</Code>.
      </p>

      <dl className="divide-border-subtle mt-8 divide-y">
        {MEASUREMENTS.map(([term, value, note]) => (
          <div
            key={term}
            className="grid grid-cols-1 gap-x-6 gap-y-1 py-3 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)_auto] sm:items-baseline"
          >
            <dt className="text-text-primary text-sm font-medium">{term}</dt>
            <dd className="text-text-tertiary order-3 text-sm leading-snug sm:order-none">
              {note}
            </dd>
            <dd className="text-text-primary font-mono text-sm tabular-nums sm:text-right">
              {value}
            </dd>
          </div>
        ))}
      </dl>

      <p className="text-text-tertiary mt-6 max-w-[62ch] text-sm leading-relaxed">
        The atlas figures are a 6-layer pilot (0, 5, 10, 16, 20, 25), not the full 26.
        The decoder-source atlas is reported too, and it fails: kNN preservation{' '}
        <Val>0.143</Val>, <Val>0</Val> clusters earning a name. Both are in the README
        because only publishing the one that worked is how a measurement becomes
        marketing.
      </p>
    </Section>
  )
}

const SETUP = `python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # accept the license at hf.co/google/gemma-2-2b

cd backend
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels --lens`

function Setup() {
  return (
    <Section label="04" title="Setup" last>
      <p className="text-text-secondary max-w-[62ch] text-base leading-relaxed">
        Today this is a repo you clone, not a tool you install. Gemma is gated, so the
        login is not optional. <Code>--sae</Code> and <Code>--labels</Code> never load
        the model; <Code>--lens</Code> does.
      </p>

      <CodeBlock code={SETUP} lang="bash" name="setup.sh" />

      <p className="text-text-tertiary mt-4 max-w-[62ch] text-sm leading-relaxed">
        26 resident 16k SAEs come to ~7.9GB &mdash; fine on a unified-memory box, not
        fine on a 16GB discrete GPU. <Code>sae_layers</Code> restricts the pass to a
        subset.
      </p>
    </Section>
  )
}

function Footer() {
  return (
    <footer className="border-border-subtle border-t">
      <Container className="text-text-tertiary flex h-16 items-center justify-between text-xs">
        <span>
          mechlens · <span className="font-mono">Apache-2.0</span>
        </span>
        <a className="hover:text-fn transition-colors duration-150" href={REPO}>
          GitHub
        </a>
      </Container>
    </footer>
  )
}

/* -------------------------------------------------------------------------
   The grid, and the small parts that sit on it.
   ------------------------------------------------------------------------- */

/** The one thing keeping the page from sprawling on a wide screen. */
function Container({
  children,
  className = '',
}: {
  children: React.ReactNode
  className?: string
}) {
  return <div className={`mx-auto w-full max-w-5xl px-6 ${className}`}>{children}</div>
}

/**
 * One band of the page: a hairline above, a numbered micro label in the left
 * column, content flush-left in the right.
 *
 * The asymmetry is the Swiss part — a label and its content in two unequal
 * columns, aligned to one grid, rather than a centred stack of headings. Under
 * `md` the label sits above the content instead of beside it, because a 4rem
 * gutter on a phone is just lost width.
 */
function Section({
  children,
  label,
  last = false,
  title,
  titleHidden = false,
}: {
  children: React.ReactNode
  label: string
  last?: boolean
  title: string
  titleHidden?: boolean
}) {
  return (
    <section
      className={`border-border-subtle ${last ? '' : 'border-b'}`}
      id={title.toLowerCase().replace(/\s+/g, '-')}
    >
      <Container className="grid grid-cols-1 gap-x-10 gap-y-6 py-20 md:grid-cols-[minmax(0,7rem)_minmax(0,1fr)] md:py-24">
        <div className="md:pt-2">
          <span className="text-text-tertiary font-mono text-xs tabular-nums">
            {label}
          </span>
          {titleHidden ? null : (
            <h2 className="text-text-primary mt-2 text-[11px] font-medium tracking-[0.08em] uppercase">
              {title}
            </h2>
          )}
        </div>
        <div className="min-w-0">{children}</div>
      </Container>
    </section>
  )
}

/** A machine-produced value inside prose. Mono marks it; nothing else does. */
function Val({ children }: { children: React.ReactNode }) {
  return <span className="text-text-primary font-mono tabular-nums">{children}</span>
}

/** Something the reader could type, inline. */
function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-bg-elevated border-border-subtle text-text-primary rounded-xs border px-1 py-px font-mono text-[0.9em]">
      {children}
    </code>
  )
}

function Th({
  align = 'left',
  children,
}: {
  align?: 'left' | 'right'
  children: React.ReactNode
}) {
  return (
    <th
      className={`px-3 py-2 text-[11px] font-medium tracking-[0.04em] uppercase ${
        align === 'right' ? 'text-right' : 'text-left'
      }`}
      scope="col"
    >
      {children}
    </th>
  )
}

function Td({
  align = 'left',
  children,
  className = '',
}: {
  align?: 'left' | 'right'
  children: React.ReactNode
  className?: string
}) {
  return (
    <td
      className={`text-text-secondary px-3 py-2 ${
        align === 'right' ? 'text-right' : 'text-left'
      } ${className}`}
    >
      {children}
    </td>
  )
}

/**
 * A quantity drawn to scale in the character cell: a 1px rule inside a
 * `ch`-measured track, computed from the real value with a small floor so a
 * near-zero row still prints a mark instead of vanishing.
 */
function Bar({ accent = false, ratio }: { accent?: boolean; ratio: number }) {
  const width = Math.max(Math.min(ratio, 1), 0.02) * 100
  return (
    <span aria-hidden="true" className="inline-block w-[10ch] shrink-0 align-middle">
      <span
        className={`block h-px ${accent ? 'bg-fn' : 'bg-rule'}`}
        style={{ width: `${width.toFixed(1)}%` }}
      />
    </span>
  )
}

/**
 * A code block with a filename tab and a copy button.
 *
 * The copy button is not optional: a block a developer cannot copy is broken.
 * Feedback is an instant icon swap, never a toast, and it is announced on a
 * live region so it is not a colour-only signal.
 */
function CodeBlock({ code, lang, name }: { code: string; lang: string; name: string }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      /* Clipboard denied — the text is selectable, which is the fallback. */
    }
  }

  return (
    <div className="border-border-subtle bg-bg-elevated mt-8 rounded-[6px] border p-1">
      <div className="border-border-subtle bg-bg-surface overflow-hidden rounded-xs border">
        <div className="border-border-subtle flex items-center justify-between border-b px-3 py-2">
          <span className="text-text-tertiary font-mono text-xs">{name}</span>
          <button
            aria-label={copied ? 'Copied' : 'Copy to clipboard'}
            className="text-text-tertiary hover:text-text-primary -m-2 inline-flex size-11 items-center justify-center transition-colors duration-150"
            onClick={copy}
            type="button"
          >
            {copied ? (
              <IconCheck className="size-4" stroke={1.5} />
            ) : (
              <IconCopy className="size-4" stroke={1.5} />
            )}
          </button>
        </div>
        <pre className="mask-fade-r overflow-x-auto px-3 py-3">
          <code className="text-text-secondary font-mono text-xs leading-[1.6]" lang={lang}>
            {code}
          </code>
        </pre>
      </div>
      <span aria-live="polite" className="sr-only">
        {copied ? 'Copied to clipboard' : ''}
      </span>
    </div>
  )
}

export default App
