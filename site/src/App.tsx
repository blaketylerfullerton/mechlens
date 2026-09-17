import { useState } from 'react'
import { IconArrowUpRight, IconCheck, IconCopy } from '@tabler/icons-react'
import './home.css'
import { navigate } from './lib/routing'
import { REPO, SiteFooter, SiteHeader } from './SiteChrome'

const N_LAYERS = 26

// Retain the existing homepage's six-row excerpt; no full trace is shipped.
const LENS_ROWS = [
  { layer: 3, token: "' of'", prob: 71.81, entropy: 1.76, reading: 'Input echo' },
  { layer: 7, token: "'بوابة'", prob: 37.97, entropy: 3.94, reading: 'Intermediate prediction' },
  { layer: 17, token: "' city'", prob: 40.87, entropy: 2.6, reading: 'Intermediate prediction' },
  { layer: 19, token: "' Paris'", prob: 22.47, entropy: 3.15, reading: 'Matches final prediction' },
  { layer: 20, token: "' Paris'", prob: 92.66, entropy: 0.46, reading: 'Matches final prediction' },
  { layer: 25, token: "' Paris'", prob: 92.59, entropy: 0.6, reading: 'Final-layer prediction' },
]

// What the pilot ran on. Every value is from the README.
const SPEC = [
  ['Model', 'gemma-2-2b'],
  ['Layers', '26'],
  ['d_model', '2304'],
  ['SAE', 'Gemma Scope 16k'],
  ['Labels', '425,679'],
]

// docs/manim/data/step2.json — golden-gate trace, layer 20, position 4 ' Bridge',
// l0 = 83. The six strongest of the sixteen saved features.
const SAE_ROWS = [
  { id: 3124, value: 104.65, label: 'San Francisco, Oakland, Bay Area' },
  { id: 6492, value: 81.45, label: 'bridge connection or circuitry' },
  { id: 1370, value: 66.19, label: 'bridge and crossing' },
  { id: 1466, value: 44.34, label: 'California legal and official mentions' },
  { id: 7182, value: 37.09, label: 'state or outcome description' },
  { id: 1692, value: 33.86, label: 'deficiencies' },
]

// README, "Attribution": L2 norms of the residual stream, the MLP term and the
// largest attention source at the final position, three of the 26 layers.
const ATTRIBUTION_ROWS = [
  { layer: 20, term: 'resid', value: 340.1 },
  { layer: 20, term: 'mlp', value: 117.1 },
  { layer: 20, term: 'attn←0', value: 28.7 },
  { layer: 21, term: 'resid', value: 387.7 },
  { layer: 21, term: 'mlp', value: 129.8 },
  { layer: 21, term: 'attn←29', value: 27.5 },
  { layer: 25, term: 'resid', value: 624.1 },
  { layer: 25, term: 'mlp', value: 236.3 },
  { layer: 25, term: 'attn←30', value: 149.3 },
]

// README, "The feature atlas": the four largest named areas of the shipping
// atlas, with coherence and its margin over the 0.233 random baseline.
const ATLAS_ROWS = [
  { size: 8320, coherence: 0.441, margin: 0.208, name: 'programming-related constructs and elements in code' },
  { size: 2296, coherence: 0.552, margin: 0.319, name: 'mathematical notations and expressions' },
  { size: 2136, coherence: 0.498, margin: 0.266, name: 'references to legal terminology and concepts' },
  { size: 1518, coherence: 0.517, margin: 0.283, name: 'numeric values and statistical references' },
]

// Measurements retained from the original homepage, scoped to its pilot.
const MEASUREMENTS = [
  ['Atlas', '98,210', 'features over 6 layers, reproducible from seed'],
  ['kNN preservation', '0.292', "share of a feature's neighbours surviving 3D"],
  ['Cluster coherence', '0.465', 'against a random baseline of 0.233'],
  ['Named regions', '27 / 32', 'clusters earning a name at that coherence'],
  ['SAE sparsity', '78.3', 'mean L0 — features firing per token, per layer'],
  ['Explained variance', '0.880', 'mean across layers, range 0.83–0.96'],
  ['Label coverage', '6234 / 6234', 'distinct features on golden-gate, no network'],
  ['Lens agreement', '1.0', 'layer 25 reproduces the model bit-identically'],
  ['Attribution gap', '≤1.3e-2', 'reconstruction error across all five traces'],
]

// README, "Two sources, because they answer different questions."
const SOURCES = [
  { source: 'labels', near: 'an explainer described two features similarly', knn: '0.292', areas: '32', named: '27' },
  { source: 'decoder', near: 'the model writes two features similarly', knn: '0.143', areas: '12', named: '0' },
]

const STEPS = [
  ['Create the environment', 'A Python venv with the backend requirements. Torch and TransformerLens are the heavy part.'],
  ['Accept the license', 'Gemma is gated. Accept it once on Hugging Face, then log in from the CLI.'],
  ['Trace a prompt', 'One command generates eight tokens and enriches every layer with features, labels and the lens.'],
]

const SETUP = `python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # gated: accept the Gemma license first

cd backend
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels --lens`

function App() {
  return (
    <div className="homepage">
      <a className="skip-link" href="#main">Skip to content</a>
      <SiteHeader />
      <main id="main">
        <section className="hero container" aria-labelledby="hero-title">
          <div className="hero-intro enter">
            <div className="hero-main">
              <p className="eyebrow">Local model interpretability</p>
              <h1 id="hero-title">Inside the<br />AI brain.</h1>
            </div>
            <div className="hero-copy">
              <p>See what activates and how predictions change as a language model builds an answer. Explore it on your own machine: no API, no upload, every number checkable.</p>
              <div className="actions">
                <a className="primary-action" href="#setup">Set it up <IconArrowUpRight size={16} stroke={1.75} aria-hidden="true" /></a>
                <a className="text-link" href="/how" onClick={navigate('/how')}>How it works →</a>
              </div>
            </div>
          </div>
          <dl className="spec-strip enter" aria-label="Pilot configuration">
            {SPEC.map(([term, value]) => (
              <div key={term}><dt>{term}</dt><dd className="machine">{value}</dd></div>
            ))}
          </dl>
          <Lens />
        </section>
        <Section number="01" label="Explore" title="An answer is only the surface." wide>
          <p className="section-lead">Follow what happens between your prompt and the model&rsquo;s next token.</p>
          <div className="figures">
            <MiniFigure label="SAE features" note="L20 · ' Bridge' · 6 of 16 saved" title="See what activated." body="Sparse-autoencoder features at a token and layer, with activation values and Neuronpedia labels. 83 fired here; the strongest sixteen are kept.">
              {SAE_ROWS.map(row => (
                <div className="fig-row" key={row.id}>
                  <span className="fig-key">#{row.id}</span>
                  <span className="fig-bar"><Bar ratio={row.value / SAE_ROWS[0].value} accent={row.id === SAE_ROWS[0].id} /></span>
                  <span className="fig-value">{row.value.toFixed(2)}</span>
                  <span className="fig-label">{row.label}</span>
                </div>
              ))}
            </MiniFigure>
            <MiniFigure label="Attribution" note="‖·‖₂ · 3 of 26 layers" title="Attribute the change." body="resid_post = resid_pre + attn + mlp is the literal structure of the residual stream, so each layer decomposes into its terms without gradients or patching. attn←n is the largest attention source, by position.">
              {ATTRIBUTION_ROWS.map(row => (
                <div className="fig-row" key={`${row.layer}-${row.term}`}>
                  <span className="fig-key">L{row.layer} <span className="fig-term">{row.term}</span></span>
                  <span className="fig-bar"><Bar ratio={row.value / 624.1} accent={row.term !== 'resid'} /></span>
                  <span className="fig-value">{row.value.toFixed(1)}</span>
                </div>
              ))}
            </MiniFigure>
            <MiniFigure label="Atlas areas" note="coherence · 4 of 32 areas" title="Place them in space." body="98,210 features, one position each. An area is named only when its members agree more than a random group of the same size — the margin over that 0.233 baseline is printed beside it.">
              {ATLAS_ROWS.map(row => (
                <div className="fig-row" key={row.name}>
                  <span className="fig-key">{row.size.toLocaleString('en-US')}</span>
                  <span className="fig-bar"><Bar ratio={row.coherence} /></span>
                  <span className="fig-value">{row.coherence.toFixed(3)} <span className="fig-term">+{row.margin.toFixed(3)}</span></span>
                  <span className="fig-label">{row.name}</span>
                </div>
              ))}
            </MiniFigure>
          </div>
        </Section>
        <Section number="02" label="Trust" title="A clear view. Including the limits.">
          <p className="section-lead">Feature labels are interpretations. A projection loses information. Mechlens keeps those limits beside the picture they qualify.</p>
          <dl className="measurements" aria-label="Pilot measurements">
            {MEASUREMENTS.map(([term, value, note]) => (
              <div key={term}><dt>{term}</dt><dd className="measurement-value">{value}</dd><dd className="note">{note}</dd></div>
            ))}
          </dl>
          <p className="note measurements-note">Reported from five saved traces. The atlas is a six-layer pilot (0, 5, 10, 16, 20, 25), not all 26 layers.</p>
          <div className="sources">
            <div className="sources-head"><span>Atlas source</span><span>What &ldquo;near&rdquo; means</span><span>kNN kept</span><span>Areas</span><span>Named</span></div>
            {SOURCES.map(row => (
              <div className={`sources-row ${row.source === 'labels' ? 'shipping' : ''}`} key={row.source}>
                <span className="machine">{row.source}{row.source === 'labels' && <span className="tag">default</span>}</span>
                <span>{row.near}</span>
                <span className="machine" data-term="kNN kept">{row.knn}</span>
                <span className="machine" data-term="areas">{row.areas}</span>
                <span className="machine" data-term="named">{row.named}</span>
              </div>
            ))}
          </div>
          <p className="note measurements-note">The decoder-source atlas is built and kept, and reports its failed result: at 0.143 preservation no cluster earns a name. Under an absolute threshold it would have named plausible-sounding areas — which is what made that threshold dangerous.</p>
          <a className="text-link" href="/how" onClick={navigate('/how')}>Read the method, with the formulas →</a>
        </Section>
        <Section number="03" label="Setup" title="Your model. Your machine." id="setup">
          <p className="section-lead">Start with the repository. Today, Mechlens needs a Python environment, CUDA, and access to the gated Gemma model.</p>
          <div className="setup-grid">
            <ol className="steps">
              {STEPS.map(([title, body], i) => (
                <li key={title}><span className="machine step-index">{String(i + 1).padStart(2, '0')}</span><div><strong>{title}</strong><p>{body}</p></div></li>
              ))}
            </ol>
            <div className="setup-code">
              <CodeBlock />
              <p className="note">Run from the repository root. This is the CLI trace; the browser interface adds <code>make dev</code>. <code>--sae</code> and <code>--labels</code> never load the model; <code>--lens</code> does.</p>
            </div>
          </div>
          <div className="actions setup-actions">
            <a className="primary-action" href={REPO}>Get the repository <IconArrowUpRight size={16} stroke={1.75} aria-hidden="true" /></a>
            <a className="text-link" href={`${REPO}#setup`}>Full setup in the README →</a>
          </div>
          <aside className="callout">
            <strong>Hardware matters.</strong> All 26 resident 16k SAEs take about 7.9 GB, before the model and runtime. Use <code>sae_layers</code> to work with a subset on a smaller GPU.
          </aside>
        </Section>
      </main>
      <SiteFooter />
    </div>
  )
}

function Lens() {
  const [selected, setSelected] = useState(20)
  const row = LENS_ROWS.find(item => item.layer === selected)!
  const recorded = new Set(LENS_ROWS.map(item => item.layer))
  return (
    <figure className="lens-figure" aria-labelledby="lens-title">
      <div className="figure-heading"><h2 id="lens-title">A prediction, layer by layer.</h2><span className="note">Recorded excerpt · 6 of 26 layers</span></div>
      <div className="trace-frame">
        <div className="trace-surface">
          <div className="trace-toolbar"><span>Logit lens</span><span className="machine">golden-gate · token 8</span></div>
          <div className="trace-prompt"><span className="small-label">Prompt</span><p>The Golden Gate Bridge is located in the city of</p></div>
          <div className="depth" role="group" aria-label="Layer depth">
            <span className="small-label">Depth</span>
            <ol className="depth-ruler">
              {Array.from({ length: N_LAYERS }, (_, layer) => (
                <li key={layer} className={`${recorded.has(layer) ? 'recorded' : ''} ${selected === layer ? 'selected' : ''}`}>
                  {recorded.has(layer)
                    ? <button type="button" aria-label={`Layer ${layer}`} aria-pressed={selected === layer} onClick={() => setSelected(layer)} />
                    : <span aria-hidden="true" />}
                </li>
              ))}
            </ol>
            <span className="machine depth-ends"><span>L00</span><span>L25</span></span>
          </div>
          <div className="trace-grid">
            <div className="layer-list">
              <div className="layer-head"><span>Layer</span><span>Top token</span><span>Probability</span></div>
              {LENS_ROWS.map(item => (
                <button key={item.layer} type="button" className={`layer-row ${selected === item.layer ? 'selected' : ''}`} aria-pressed={selected === item.layer} aria-controls="layer-detail" aria-label={`Layer ${item.layer}: ${item.token}, ${item.prob.toFixed(2)} percent. ${item.reading}`} onClick={() => setSelected(item.layer)}>
                  <span className="layer-index">L{String(item.layer).padStart(2, '0')}</span>
                  <span className="predicted-token" dir="auto">{item.token}</span>
                  <span className="probability"><Bar ratio={item.prob / 100} /><span>{item.prob.toFixed(2)}%</span></span>
                </button>
              ))}
              <p className="note interaction-hint">Select a layer to inspect its readout.</p>
            </div>
            <div id="layer-detail" className="layer-detail" aria-live="polite" aria-atomic="true">
              <div className="detail-heading"><span className="small-label">Selected layer</span><span className="machine">L{row.layer}</span></div>
              <p className="detail-token" dir="auto">{row.token}</p>
              <p className="reading">{row.reading}</p>
              <dl className="detail-stats"><div><dt>Probability</dt><dd>{row.prob.toFixed(2)}%</dd></div><div><dt>Entropy</dt><dd>{row.entropy.toFixed(2)} <span>nats</span></dd></div></dl>
              <p className="note">{row.reading === 'Input echo' ? 'This layer reads back the input token. A high probability here is not early certainty about the answer.' : 'The output head reads the residual stream at this layer. This is a prediction about the next token, not an explanation of its cause.'}</p>
            </div>
          </div>
        </div>
      </div>
      <figcaption className="figure-caption"><span className="caption-label">Probability ≠ accuracy</span><p>This excerpt predicts &ldquo;Paris&rdquo; despite the Golden Gate prompt. High model probability does not establish factual correctness. Early layers can echo input tokens; intermediate layers are omitted here.</p></figcaption>
      <details className="disclosure lens-methodology">
        <summary>How to read the logit lens <span aria-hidden="true">+</span></summary>
        <div className="disclosure-body note">The model&rsquo;s output head decodes the residual stream at each selected layer. The last layer uses <code>resid_post</code> before <code>ln_final</code>, matching the model&rsquo;s output path. The original five-trace check reports a maximum probability delta of <code>0.0</code> at the final layer. Earlier readouts are diagnostic views, not independent model outputs.</div>
      </details>
    </figure>
  )
}

function Section({ number, label, title, id, wide, children }: { number: string; label: string; title: string; id?: string; wide?: boolean; children: React.ReactNode }) {
  return (
    <section className={`section ${wide ? 'section-wide' : ''}`} id={id} aria-labelledby={`section-${number}`}>
      <div className="container section-grid">
        <div className="section-index"><span className="machine">{number}</span><span>{label}</span></div>
        <div className="section-content"><h2 id={`section-${number}`}>{title}</h2>{children}</div>
      </div>
    </section>
  )
}

function MiniFigure({ label, note, title, body, children }: { label: string; note: string; title: string; body: string; children: React.ReactNode }) {
  return (
    <figure className="mini-figure">
      <div className="mini-head"><span className="small-label">{label}</span><span className="machine">{note}</span></div>
      <div className="fig-rows">{children}</div>
      <figcaption><strong>{title}</strong><p>{body}</p></figcaption>
    </figure>
  )
}

function Bar({ ratio, accent = false }: { ratio: number; accent?: boolean }) {
  return <span className={`bar-track ${accent ? 'accent' : ''}`} aria-hidden="true"><span style={{ width: `${Math.max(Math.min(ratio, 1), 0.02) * 100}%` }} /></span>
}

function CodeBlock() {
  const [feedback, setFeedback] = useState('')
  async function copy() {
    try { await navigator.clipboard.writeText(SETUP); setFeedback('Copied') }
    catch { setFeedback('Select the commands below to copy them.') }
  }
  return <div className="code-block"><div className="code-toolbar"><span className="machine">setup.sh</span><button type="button" onClick={copy} aria-label="Copy setup commands">{feedback === 'Copied' ? <IconCheck size={16} /> : <IconCopy size={16} />}<span>{feedback === 'Copied' ? 'Copied' : 'Copy'}</span></button></div><pre tabIndex={0} aria-label="Setup commands"><code>{SETUP}</code></pre><span className="note" role="status">{feedback && feedback !== 'Copied' ? feedback : <span className="sr-only">{feedback}</span>}</span></div>
}

export default App
