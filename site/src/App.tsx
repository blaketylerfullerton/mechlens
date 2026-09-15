import { useState } from 'react'
import { IconBrandGithub, IconCheck, IconCopy } from '@tabler/icons-react'
import './home.css'
import { navigate } from './lib/routing'

const REPO = 'https://github.com/blaketylerfullerton/mechlens'

// Retain the existing homepage's six-row excerpt; no full trace is shipped.
const LENS_ROWS = [
  { layer: 3, token: "' of'", prob: 71.81, entropy: 1.76, reading: 'Input echo' },
  { layer: 7, token: "'بوابة'", prob: 37.97, entropy: 3.94, reading: 'Intermediate prediction' },
  { layer: 17, token: "' city'", prob: 40.87, entropy: 2.6, reading: 'Intermediate prediction' },
  { layer: 19, token: "' Paris'", prob: 22.47, entropy: 3.15, reading: 'Matches final prediction' },
  { layer: 20, token: "' Paris'", prob: 92.66, entropy: 0.46, reading: 'Matches final prediction' },
  { layer: 25, token: "' Paris'", prob: 92.59, entropy: 0.6, reading: 'Final-layer prediction' },
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

const SETUP = `python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
huggingface-cli login   # accept the license at hf.co/google/gemma-2-2b

cd backend
python -m app.cli trace -p "2 + 2 =" -n 8 --sae --labels --lens`

function App() {
  return (
    <div className="homepage">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <nav className="container nav" aria-label="Main navigation">
          <a href="#main" className="wordmark" aria-label="Mechlens home">
            <img src="/logo.svg" alt="" width={26} height={26} />mechlens
          </a>
          <div className="nav-links">
            <a href="/how" onClick={navigate('/how')}>How</a>
            <a href="#setup">Setup</a>
            <a href={REPO}><IconBrandGithub size={16} stroke={1.5} />GitHub</a>
          </div>
        </nav>
      </header>
      <main id="main">
        <section className="hero container" aria-labelledby="hero-title">
          <div className="enter">
            <p className="eyebrow">Local model interpretability</p>
            <div className="hero-intro">
              <h1 id="hero-title">Inside the<br />AI brain.</h1>
              <div className="hero-copy">
                <p>See what activates and how predictions change as a language model builds an answer. Explore it on your own machine.</p>
                <a className="primary-action" href="#setup">Set it up <span aria-hidden="true">↗</span></a>
              </div>
            </div>
            <p className="model-meta">gemma-2-2b <span>/</span> 26 layers <span>/</span> Gemma Scope 16k</p>
          </div>
          <Lens />
        </section>
        <Section number="01" label="Explore" title="An answer is only the surface.">
          <p className="section-lead">Follow what happens between your prompt and the model&rsquo;s next token.</p>
          <dl className="capabilities">
            <div><dt>See what activated.</dt><dd>Inspect sparse-autoencoder features at a token and layer, with activation values and labels where available.</dd></div>
            <div><dt>Follow the prediction.</dt><dd>Compare the model&rsquo;s readout across layers. See when a prediction changes, and when early layers are echoing the input.</dd></div>
            <div><dt>Keep the evidence in view.</dt><dd>Read the measurements alongside the result, from feature coverage to the limits of a projected map.</dd></div>
          </dl>
        </Section>
        <Section number="02" label="Trust" title="A clear view. Including the limits.">
          <p className="section-lead">Feature labels are interpretations. A projection loses information. Mechlens keeps those limits beside the picture they qualify.</p>
          <details className="disclosure">
            <summary>Read the pilot measurements <span aria-hidden="true">+</span></summary>
            <div className="disclosure-body">
              <p className="note">Reported measurements from five saved traces. The atlas is a six-layer pilot (0, 5, 10, 16, 20, 25), not all 26 layers.</p>
              <dl className="measurements">
                {MEASUREMENTS.map(([term, value, note]) => (
                  <div key={term}><dt>{term}</dt><dd className="measurement-value">{value}</dd><dd className="note">{note}</dd></div>
                ))}
              </dl>
              <p className="note">The decoder-source atlas also reports its failed result: kNN preservation <code>0.143</code>, with <code>0</code> clusters earning a name.</p>
              <a className="text-link" href="/how" onClick={navigate('/how')}>Read the method, with the formulas →</a>
            </div>
          </details>
        </Section>
        <Section number="03" label="Setup" title="Your model. Your machine." id="setup">
          <p className="section-lead">Start with the repository. Today, Mechlens needs a Python environment, CUDA, and access to the gated Gemma model.</p>
          <a className="primary-action" href={REPO}>Get the repository <span aria-hidden="true">↗</span></a>
          <details className="disclosure">
            <summary>Generate your first trace <span aria-hidden="true">+</span></summary>
            <div className="disclosure-body">
              <p className="note">Clone the repository and run these commands from its root. Accept Gemma&rsquo;s license on Hugging Face before logging in.</p>
              <CodeBlock />
              <p className="note">This generates a CLI trace. See the repository for the browser interface setup. Enrichment with <code>--sae</code> and <code>--labels</code> does not load the model; <code>--lens</code> does.</p>
            </div>
          </details>
          <p className="note setup-note">Hardware matters: all 26 resident 16k SAEs take about 7.9 GB, before the model and runtime. Use <code>sae_layers</code> to work with a subset on a smaller GPU.</p>
        </Section>
      </main>
      <footer className="container footer"><span>mechlens <span className="footer-divider">/</span> Open source, open to inspection.</span><a href={`${REPO}/blob/main/LICENSE`}>Apache-2.0</a></footer>
    </div>
  )
}

function Lens() {
  const [selected, setSelected] = useState(20)
  const row = LENS_ROWS.find(item => item.layer === selected)!
  return (
    <figure className="lens-figure" aria-labelledby="lens-title">
      <div className="figure-heading"><h2 id="lens-title">A prediction, layer by layer.</h2><span className="note">Recorded excerpt · 6 of 26 layers</span></div>
      <div className="trace-frame">
        <div className="trace-surface">
          <div className="trace-toolbar"><span>Logit lens</span><span className="machine">token 8</span></div>
          <div className="trace-prompt"><span className="small-label">Prompt</span><p>The Golden Gate Bridge is located in the city of</p></div>
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

function Section({ number, label, title, id, children }: { number: string; label: string; title: string; id?: string; children: React.ReactNode }) {
  return <section className="section" id={id} aria-labelledby={`section-${number}`}><div className="container section-grid"><div className="section-index"><span className="machine">{number}</span><span>{label}</span></div><div className="section-content"><h2 id={`section-${number}`}>{title}</h2>{children}</div></div></section>
}

function Bar({ ratio }: { ratio: number }) {
  return <span className="bar-track" aria-hidden="true"><span style={{ width: `${Math.max(ratio, 0.02) * 100}%` }} /></span>
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
