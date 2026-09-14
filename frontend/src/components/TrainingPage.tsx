import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { API_BASE_URL, request } from '@/lib/api-client'
import { ResourceMeter } from '@/components/ResourceMeter'
import { useErrorToast } from '@/hooks/useErrorToast'

type Options = { model: string; layers: number | null; device: string; readiness: string; estimate_note: string; supported_models: string[] }
type Check = { status: string; reason?: string; definition?: string; max_loss_delta?: number }
type Evaluation = { [key: string]: unknown; identity_substitution?: Check }
type Run = {
  id: string; status: string; phase: string; created_at: number; tokens: number; error: string | null;
  config: { layer: number; features: number; training_tokens: number; dataset: string; compare_huggingface?: boolean };
  checkpoint: { artifact_id: string; tokens: number } | null;
  evaluation: Evaluation | null;
  validation?: { model_agreement?: Check; checkpoint_agreement?: Check; activation_identity?: Check; identity_substitution?: Check };
  validation_progress?: { completed: number; total: number } | null;
  resume_supported: boolean;
  examples?: { sequences_scanned: number; tokens_scanned: number } | null;
}
type InterpJob = { id: string; run_id: string; artifact_id: string; feature_ids: number[]; endpoint: string; model: string; status: string; phase: string; completed: number; total: number; error: string | null; created_at: number }
type ExampleReport = { artifact_id: string; sequences_scanned: number; tokens_scanned: number; examples: Record<string, { activation: number; context: string; token_position: number }[]> }
type Metric = { seq: number; tokens: number; loss: number; mse: number; explained_variance: number;
  l0: number; dead_fraction: number; tokens_per_second: number; eta_s: number; elapsed_s: number }
const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted'])
const field = 'border-border-subtle bg-bg-base text-text-primary mt-2 w-full rounded border px-3 py-2.5 text-sm outline-none focus:border-fn'
const button = 'border-border-subtle rounded border px-4 py-2 text-sm transition-colors hover:border-fn disabled:cursor-not-allowed disabled:opacity-40'

function number(value: number | undefined, digits = 2) {
  return value === undefined || !Number.isFinite(value) ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function Chart({ metrics, metric, label }: { metrics: Metric[]; metric: 'loss' | 'explained_variance' | 'l0' | 'dead_fraction'; label: string }) {
  const points = metrics.filter((m) => Number.isFinite(m[metric]))
  const values = points.map((m) => m[metric])
  const low = Math.min(...values), high = Math.max(...values)
  const first = points[0]?.tokens ?? 0, last = points.at(-1)?.tokens ?? 1
  const line = points.map((m) => `${12 + 376 * (m.tokens - first) / Math.max(1, last - first)},${100 - 80 * (m[metric] - low) / Math.max(1e-9, high - low)}`).join(' ')
  return <section className="border-border-subtle bg-bg-surface rounded border p-4">
    <div className="flex items-center justify-between gap-2 text-sm"><h3 className="text-text-secondary">{label}</h3><span className="font-mono">{number(values.at(-1), 3)}</span></div>
    {points.length ? <><svg role="img" aria-label={`${label} by training tokens, latest ${number(values.at(-1), 3)}`} viewBox="0 0 400 120" className="mt-3 h-28 w-full text-fn">
      <line x1="12" y1="108" x2="388" y2="108" stroke="currentColor" opacity="0.15" />
      {points.length > 1 ? <polyline points={line} fill="none" stroke="currentColor" strokeWidth="2" /> : <circle cx="12" cy="100" r="3" fill="currentColor" />}
    </svg><div className="text-text-tertiary flex justify-between font-mono text-[10px]"><span>{number(first, 0)} tokens</span><span>{number(last, 0)} tokens</span></div></>
      : <p className="text-text-tertiary flex h-36 items-center justify-center text-xs">Waiting for the first measurement</p>}
  </section>
}

export function TrainingPage({ onInspect }: { onInspect: (id: string) => void }) {
  const [options, setOptions] = useState<Options | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [interpJobs, setInterpJobs] = useState<InterpJob[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [error, setError] = useState<string | null>(null)
  useErrorToast(error)
  const [busy, setBusy] = useState(false)
  const [layer, setLayer] = useState(12)
  const [features, setFeatures] = useState(4096)
  const [tokens, setTokens] = useState(32768)
  const [dataset, setDataset] = useState('tiny-stories')
  const [trainingText, setTrainingText] = useState('')
  const [evaluationText, setEvaluationText] = useState('')
  const [batchSize, setBatchSize] = useState(256)
  const [contextSize, setContextSize] = useState(128)
  const [learningRate, setLearningRate] = useState(0.0003)
  const [sparsity, setSparsity] = useState(0.1)
  const [seed, setSeed] = useState(42)
  const [evaluationSequences, setEvaluationSequences] = useState(64)
  const [compareHuggingface, setCompareHuggingface] = useState(false)
  const [repository, setRepository] = useState('google/gemma-2-2b')
  const [exampleFeature, setExampleFeature] = useState('0')
  const [exampleStatus, setExampleStatus] = useState<string | null>(null)
  const [exampleReport, setExampleReport] = useState<ExampleReport | null>(null)
  const [interpProfile, setInterpProfile] = useState('local')
  const run = runs.find((item) => item.id === selected)
  const latest = metrics.at(-1)

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const results = await Promise.allSettled([request<Options>('/training/options'), request<Run[]>('/training/runs'), request<InterpJob[]>('/training/interp-jobs')])
      if (stopped) return
      if (results[0].status === 'fulfilled') {
        const available = results[0].value
        setOptions(available)
        if (available.layers) setLayer((current) => Math.min(current, available.layers! - 1))
      }
      if (results[1].status === 'fulfilled') {
        const history = results[1].value
        setRuns(history)
        setSelected((id) => id ?? history[0]?.id ?? null)
      }
      if (results[2].status === 'fulfilled') setInterpJobs(results[2].value)
      const failure = results.find((result) => result.status === 'rejected')
      if (failure?.status === 'rejected') setError(String(failure.reason))
      timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [])

  useEffect(() => {
    setMetrics([])
    if (!selected) return
    let stopped = false
    let cursor = 0
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const values = await request<Metric[]>(`/training/runs/${selected}/metrics?after=${cursor}`)
        if (stopped) return
        if (values.length) {
          cursor = values.at(-1)!.seq
          setMetrics((previous) => [...previous, ...values].slice(-2000))
        }
      } catch (err) { if (!stopped) setError(String(err)) }
      if (!stopped) timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [selected])

  async function start(event: FormEvent) {
    event.preventDefault()
    setBusy(true); setError(null)
    try {
      const result = await request<Run>('/training/runs', { method: 'POST', body: JSON.stringify({
        layer, features, training_tokens: tokens, dataset, training_text: trainingText, evaluation_text: evaluationText,
        batch_size: batchSize, context_size: contextSize, learning_rate: learningRate, l1_coefficient: sparsity, seed,
        evaluation_sequences: evaluationSequences, compare_huggingface: compareHuggingface,
      }) })
      setRuns((previous) => [result, ...previous]); setSelected(result.id)
    } catch (err) { setError(String(err)) }
    finally { setBusy(false) }
  }

  async function control(id: string, action: 'cancel' | 'resume') {
    setBusy(true); setError(null)
    try {
      const result = await request<Run>(`/training/runs/${id}/${action}`, { method: 'POST' })
      setRuns((previous) => previous.map((item) => item.id === id ? result : item))
    } catch (err) { setError(String(err)) }
    finally { setBusy(false) }
  }

  async function prepareModel() {
    setBusy(true); setError(null)
    try {
      await request('/training/model', { method: 'POST', body: JSON.stringify({ repository }) })
      setOptions((current) => current ? { ...current, readiness: 'Preparing model weights…' } : current)
    } catch (err) { setError(String(err)) }
    finally { setBusy(false) }
  }

  async function collectExamples() {
    if (!run || !Number.isInteger(Number(exampleFeature))) return
    const runId = run.id
    setError(null); setExampleReport(null); setExampleStatus('Queueing example scan…')
    try {
      const submission = await request<{ job_id: string }>(`/training/runs/${runId}/examples`, {
        method: 'POST', body: JSON.stringify({ feature_ids: [Number(exampleFeature)] }),
      })
      async function poll() {
        const job = await request<{ status: string; error: string | null; progress: { done: number; total: number } | null }>(`/training/runs/${runId}/examples/jobs/${submission.job_id}`)
        if (job.status === 'done') {
          const report = await request<ExampleReport>(`/training/runs/${runId}/examples`)
          setExampleReport(report); setExampleStatus(`Saved ${report.sequences_scanned} scanned chunks.`)
        } else if (job.status === 'error') {
          setExampleStatus(null); setError(job.error ?? 'Example collection failed')
        } else {
          setExampleStatus(job.progress ? `Scanning ${job.progress.done} / ${job.progress.total} chunks…` : 'Waiting for the compute queue…')
          window.setTimeout(() => { void poll() }, 800)
        }
      }
      void poll()
    } catch (err) {
      setExampleStatus(null); setError(String(err))
    }
  }

  async function createInterpJob() {
    if (!run || !Number.isInteger(Number(exampleFeature))) return
    setError(null)
    try {
      await request(`/training/runs/${run.id}/interp-jobs`, { method: 'POST',
        body: JSON.stringify({ feature_ids: [Number(exampleFeature)], profile: interpProfile }) })
    } catch (err) { setError(String(err)) }
  }

  return <main className="mx-auto flex w-full min-h-0 flex-1 max-w-[1500px] flex-col px-6 py-6 lg:px-10">
    <header className="mb-6 flex shrink-0 flex-wrap items-start justify-between gap-4">
      <div><p className="text-fn mb-3 font-mono text-xs uppercase tracking-[0.16em]">SAE workspace</p>
        <h1 className="text-3xl font-medium tracking-tight">Train a dictionary. Look inside it.</h1>
        <p className="text-text-secondary mt-3 max-w-2xl text-sm leading-6">Learn sparse features from a frozen language model. Choose a layer, feed it text, and watch the measurements arrive.</p>
        <ol className="text-text-tertiary mt-4 flex flex-wrap gap-2 text-xs" aria-label="Dictionary workflow"><li className="border-fn bg-fn/10 text-fn rounded border px-2 py-1">1 · Set up</li><li className="border-border-subtle rounded border px-2 py-1">2 · Train</li><li className="border-border-subtle rounded border px-2 py-1">3 · Inspect</li><li className="border-border-subtle rounded border px-2 py-1">4 · Interpret</li></ol></div>
      <div className="text-text-tertiary space-y-2 text-xs"><p>Backend resources · {options?.device ?? 'connecting'}</p><ResourceMeter /></div>
    </header>
    <div className="grid min-h-0 flex-1 gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="min-h-0 space-y-7 overflow-y-auto pr-1 lg:pb-6">
        <form onSubmit={start} className="border-border-subtle bg-bg-surface space-y-5 rounded border p-5">
          <h2 className="text-lg font-medium">What are you trying to do?</h2><p className="text-text-tertiary text-xs">Start simple. You can refine the training plan later.</p><div className="grid grid-cols-2 gap-2"><button className={`${button} w-full`} type="button" onClick={() => { setLayer(Math.floor((options?.layers ?? 26) / 2)); setFeatures(4096); setTokens(32768); setDataset("tiny-stories"); setBatchSize(256); setContextSize(128); setLearningRate(0.0003); setSparsity(0.1); setSeed(42); setEvaluationSequences(64) }}>Quick check</button><button className={`${button} border-fn/50 bg-fn/5 text-fn w-full`} type="button" onClick={() => { setLayer(Math.floor((options?.layers ?? 26) / 2)); setFeatures(4096); setTokens(131072); setDataset("tiny-stories"); setBatchSize(256); setContextSize(128); setLearningRate(0.0003); setSparsity(0.1); setSeed(42); setEvaluationSequences(64) }}>Useful pilot</button></div>
          <p className="text-text-tertiary text-xs">Useful pilot is the recommended start for feature inspection. Quick check only validates the workflow.</p>
          <label className="text-text-secondary block text-xs">Hugging Face model<select className={field} value={repository} onChange={(e) => setRepository(e.target.value)}>{(options?.supported_models ?? ['google/gemma-2-2b']).map((id) => <option key={id} value={id}>{id}</option>)}</select></label>
          <button className={`${button} w-full`} type="button" disabled={busy} onClick={prepareModel}>Download / load model</button>
          <div className="border-border-subtle rounded border p-3"><p className="text-text-tertiary text-xs">Loaded model</p><p className="mt-1 break-all text-sm">{options?.model ?? 'Connecting…'}</p>
            <p role="status" className="text-text-secondary mt-2 text-xs">{options?.readiness === 'ready' ? 'Weights ready · base model stays frozen' : options?.readiness ?? 'Checking backend'}</p>
            {options && options.readiness !== 'ready' ? <p className="text-text-tertiary mt-2 text-xs">The backend prepares model weights. For gated models, accept the model license on Hugging Face and sign in on the backend host.</p> : null}</div>
          <label className="text-text-secondary block text-xs">Layer<input className={field} type="number" min="0" max={(options?.layers ?? 26) - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} required /></label>
          <details className="text-text-secondary text-xs"><summary className="cursor-pointer">Training settings</summary><div className="mt-4 space-y-4">
            <label className="block">Dictionary features<select className={field} value={features} onChange={(e) => setFeatures(Number(e.target.value))}>{[256, 1024, 4096, 16384].map((n) => <option key={n} value={n}>{n.toLocaleString()}</option>)}</select></label>
          </div></details>
          <label className="text-text-secondary block text-xs">Dataset<select className={field} value={dataset} onChange={(e) => setDataset(e.target.value)}><option value="tiny-stories">TinyStories · train / validation</option><option value="text">My text · separate evaluation</option></select></label>
          {dataset === 'text' ? <><label className="text-text-secondary block text-xs">Training text<textarea className={field} rows={5} value={trainingText} onChange={(e) => setTrainingText(e.target.value)} required maxLength={2000000} /></label><label className="text-text-secondary block text-xs">Held-out evaluation text<textarea className={field} rows={3} value={evaluationText} onChange={(e) => setEvaluationText(e.target.value)} required maxLength={200000} /></label></> : null}
          <details className="text-text-secondary text-xs"><summary className="cursor-pointer">More training controls</summary><div className="mt-4 space-y-4">
          <label className="text-text-secondary block text-xs">Training token budget<input className={field} type="number" min={batchSize} max="100000000" step={batchSize} value={tokens} onChange={(e) => setTokens(Number(e.target.value))} required /></label>
          <label className="text-text-secondary block text-xs">Sparsity penalty · L1<input className={field} type="number" min="0" max="100" step="any" value={sparsity} onChange={(e) => setSparsity(Number(e.target.value))} required /></label>
          <label className="text-text-secondary block text-xs">Held-out evaluation sequences<input className={field} type="number" min="1" max="2048" value={evaluationSequences} onChange={(e) => setEvaluationSequences(Number(e.target.value))} required /></label>
          <label className="text-text-secondary block text-xs leading-5"><input className="mr-2" type="checkbox" checked={compareHuggingface} onChange={(e) => setCompareHuggingface(e.target.checked)} />Check Hugging Face agreement · optional
            <span className="text-text-tertiary mt-1 block">Loads a second copy of the model on the backend CPU. Adds RAM use and evaluation time; enable on Spark when ready.</span></label>
          </div></details>
          <details className="text-text-secondary text-xs"><summary className="cursor-pointer">Advanced settings</summary><div className="mt-4 space-y-4">
            <label className="block">Batch tokens<input className={field} type="number" min="16" max="8192" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} required /></label>
            <label className="block">Context tokens<input className={field} type="number" min="8" max="2048" value={contextSize} onChange={(e) => setContextSize(Number(e.target.value))} required /></label>
            <label className="block">Learning rate<input className={field} type="number" min="0.00000001" max="0.01" step="any" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} required /></label>
            <label className="block">Seed<input className={field} type="number" min="0" max="4294967295" value={seed} onChange={(e) => setSeed(Number(e.target.value))} required /></label>
          </div></details>
          <p className="text-text-tertiary text-xs leading-5">{options?.estimate_note} Small runs check the workflow; feature quality needs evaluation. Training and inference share one queue.</p>
          <button className={`${button} bg-fn/10 text-fn w-full`} disabled={busy || options?.readiness !== 'ready'} type="submit">{busy ? 'Working…' : 'Train dictionary'}</button>
        </form>
        <section><h2 className="mb-3 text-sm font-medium">Run history</h2><div className="space-y-2">
          {runs.length === 0 ? <p className="text-text-tertiary text-xs">Your runs will be saved here.</p> : runs.map((item) => <button className={`w-full rounded border p-3 text-left ${selected === item.id ? 'border-fn/50 bg-fn/5' : 'border-border-subtle'}`} key={item.id} onClick={() => setSelected(item.id)}>
            <span className="flex justify-between gap-2 text-sm"><span>Layer {item.config.layer} · {number(item.config.features, 0)} features</span><span className="text-text-tertiary text-xs">{item.status}</span></span><span className="text-text-tertiary mt-2 block text-xs">{new Date(item.created_at * 1000).toLocaleString()} · {item.id.slice(0, 8)}</span>
          </button>)}
        </div></section>
        <section className="border-border-subtle rounded border p-4"><div className="flex items-center justify-between gap-3"><h2 className="text-sm font-medium">Auto-interp jobs</h2><span className="text-text-tertiary text-xs">self-hosted</span></div>
          <p className="text-text-tertiary mt-2 text-xs leading-5">Candidate-label jobs will appear here with their checkpoint, local endpoint, model, and evaluation status. The labeling worker is not configured yet.</p>
          <div className="mt-4 space-y-2">{interpJobs.length === 0 ? <p className="text-text-tertiary text-xs">No auto-interp jobs yet.</p> : interpJobs.map((job) => <article className="bg-bg-base rounded p-3" key={job.id}><div className="flex justify-between gap-3 text-xs"><span className="font-mono">{job.id.slice(0, 8)}</span><span className={job.status === "failed" ? "text-err" : "text-text-secondary"}>{job.status}</span></div><p className="text-text-secondary mt-2 text-xs">{job.feature_ids.length} feature{job.feature_ids.length === 1 ? "" : "s"} · {job.model}</p><p className="text-text-tertiary mt-1 break-all text-xs">{job.endpoint} · {job.artifact_id.slice(0, 12)}</p>{job.total ? <progress className="mt-2 h-1 w-full accent-[var(--color-fn)]" value={job.completed} max={job.total} aria-label={`Auto-interp job  progress`} /> : null}{job.error ? <p className="text-err mt-2 text-xs">{job.error}</p> : null}</article>)}</div>
        </section>
      </aside>
      <section className="min-h-0 min-w-0 overflow-y-auto pr-1 lg:pb-6">
        <div className="border-border-subtle mb-6 border-b pb-6">
          <div className="flex flex-wrap items-center justify-between gap-4"><div><p className="text-text-tertiary mb-2 font-mono text-xs">{run ? run.id.slice(0, 8) : 'NO RUN SELECTED'}</p><h2 className="text-xl font-medium">{run ? `Layer ${run.config.layer} training` : 'A clear view of every run'}</h2></div>
            {run && !terminal.has(run.status) ? <button className={button} disabled={busy || run.status === 'cancelling'} onClick={() => control(run.id, 'cancel')}>{run.status === 'cancelling' ? 'Saving and stopping…' : 'Cancel run'}</button> : null}
            {run?.resume_supported && ['interrupted', 'cancelled', 'failed'].includes(run.status) ? <button className={button} disabled={busy || options?.readiness !== 'ready'} onClick={() => control(run.id, 'resume')}>Resume checkpoint</button> : null}</div>
          <p role="status" className="text-text-secondary mt-3 text-sm">{run ? `${run.status} · ${run.phase}${run.status === 'queued' ? ' · waiting for the compute queue' : ''}` : 'Start a run to see reconstruction quality, sparsity, and progress.'}</p>
          {run ? <><progress className="mt-5 h-1.5 w-full accent-[var(--color-fn)]" value={run.tokens} max={run.config.training_tokens} aria-label="Training tokens processed" /><div className="text-text-tertiary mt-2 flex flex-wrap justify-between gap-2 font-mono text-xs"><span>{number(run.tokens, 0)} / {number(run.config.training_tokens, 0)} tokens</span><span>{number(latest?.tokens_per_second, 0)} tokens/s · {run.status === 'running' && run.phase === 'training' && latest ? `${number(latest.eta_s / 60, 1)} min training remaining` : 'ETA —'}</span></div></> : null}
          {run?.error ? <p className="text-err mt-4 text-sm">{run.error}</p> : null}
        </div>
        <div className="grid gap-4 sm:grid-cols-2"><Chart metrics={metrics} metric="loss" label="Training loss" /><Chart metrics={metrics} metric="explained_variance" label="Explained variance" /><Chart metrics={metrics} metric="l0" label="Active features per token · L0" /><Chart metrics={metrics} metric="dead_fraction" label="Dead feature fraction · 1,000 steps" /></div>
        <p className="text-text-tertiary mt-3 text-xs leading-5">Training-batch measurements; most recent 2,000 samples retained. Dead features have been inactive for more than 1,000 optimizer steps. Low loss alone does not establish interpretability.</p>
        {run ? <section className="border-border-subtle mt-7 rounded border p-5">
          <div className="flex flex-wrap items-center justify-between gap-3"><h3 className="font-medium">Validation report</h3>
            <a className="text-text-secondary text-xs underline underline-offset-4" href={`${API_BASE_URL}/training/runs/${run.id}/report`}>Download report JSON</a></div>
          <p className="text-text-tertiary mt-3 text-xs leading-5">Correctness checks validate the pipeline. Quality measurements describe the learned dictionary; passing checks does not establish interpretability.</p>
          <div role="status" aria-live="polite" className="text-text-secondary mt-4 text-sm">
            {run.status === 'cancelling' ? 'Stopping validation and training…' : !terminal.has(run.status) ?
              run.phase === 'evaluating' ? <><span className="mr-2 inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none" aria-hidden="true" />Validation running · {run.validation_progress?.completed ?? 0} / {run.validation_progress?.total ?? '—'} held-out sequences checked</> :
              run.phase === 'checking Hugging Face agreement' ? <><span className="mr-2 inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none" aria-hidden="true" />Hugging Face comparison running · loading and checking the CPU reference model</> :
              'Validation pending · checkpoint and held-out checks run after training.' :
              run.status === 'completed' ? 'Run finished · individual check results below.' : 'Run stopped · unfinished checks did not complete.'}
          </div>
          {run.phase === 'evaluating' && !terminal.has(run.status) && run.validation_progress ? <progress className="mt-3 h-1.5 w-full accent-[var(--color-fn)]" value={run.validation_progress.completed} max={run.validation_progress.total} aria-label="Held-out validation sequences checked" /> : null}
          <dl className="mt-4 space-y-3">{([
            ['Hugging Face model agreement', run.config.compare_huggingface === false ? {
              status: 'skipped', reason: 'Optional check was disabled for this run. Enable “Check Hugging Face agreement” before starting a new run. Other checks still run.',
            } : run.validation?.model_agreement],
            ['Saved checkpoint agreement', run.validation?.checkpoint_agreement],
            ['Model, hook and dictionary identity', run.validation?.activation_identity],
            ['Unchanged-activation control', run.evaluation?.identity_substitution ?? run.validation?.identity_substitution],
          ] as [string, Check | undefined][]).map(([label, check]) => <div key={label}>
            <div className="flex justify-between gap-3 text-sm"><dt>{label}</dt><dd className={check?.status === 'failed' ? 'text-err' : 'text-text-secondary'}>{['pending', 'running'].includes(check?.status ?? '') && terminal.has(run.status) ? 'not completed' : check?.status?.replaceAll('_', ' ') ?? (terminal.has(run.status) ? 'not run' : 'pending')}</dd></div>
            {check?.reason || check?.definition ? <p className="text-text-tertiary mt-1 text-xs leading-5">{check.reason ?? check.definition}</p> : null}
          </div>)}</dl>
          {run.evaluation ? <><h4 className="mb-4 mt-6 font-medium">Held-out quality</h4>
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">{[
              ['mse', 'Reconstruction error · MSE'], ['explained_variance', 'Explained variance'],
              ['l0', 'Active features per token'], ['baseline_loss', 'Original model loss'],
              ['reconstruction_loss', 'SAE substitution loss'], ['ablation_loss', 'Zero-ablation loss'],
              ['loss_increase', 'Loss increase with SAE'], ['loss_recovered', 'Loss recovered · ratio'],
            ].map(([key, label]) => <div key={key}><dt className="text-text-tertiary text-xs">{label}</dt><dd className="mt-1 font-mono text-sm">{typeof run.evaluation![key] === 'number' ? number(run.evaluation![key] as number, 4) : '—'}</dd></div>)}</dl>
            <p className="text-text-tertiary mt-4 text-xs leading-5">{String(run.evaluation.tokens)} activation tokens · {String(run.evaluation.sequences ?? 'unknown')} sequences{run.evaluation.requested_sequences ? ` / ${run.evaluation.requested_sequences} requested` : ''}. {run.evaluation.sample_limit_reached === false ? 'Evaluation data ended before the requested sample count.' : ''}</p>
            <p className="text-text-tertiary mt-2 text-xs leading-5">{String(run.evaluation.definition)}</p>
            <p className="text-text-tertiary mt-2 text-xs leading-5">Lower SAE loss increase means better preservation of model predictions. Loss recovered compares the SAE with zero ablation; it is undefined when zero ablation does not increase loss.</p>
          </> : <p className="text-text-tertiary mt-5 text-xs">Held-out measurements appear after training and evaluation complete. Older runs may not have all checks.</p>}
        </section> : null}
        {run?.checkpoint ? <section className="border-border-subtle mt-7 rounded border p-5"><h3 className="font-medium">Saved dictionary</h3><p className="text-text-secondary mt-2 text-sm">Checkpoint at {number(run.checkpoint.tokens, 0)} tokens. Features are unlabeled and have their own dictionary identity.</p><code className="text-text-tertiary mt-2 block break-all text-xs">{run.checkpoint.artifact_id}</code>
          <div className="mt-5 flex flex-wrap items-center gap-3"><button className={`${button} text-fn`} onClick={() => onInspect(run.id)}>Open in Explore</button>{['cfg.json', 'sae_weights.safetensors', 'manifest.json'].map((name) => <a className="text-text-secondary text-xs underline underline-offset-4" key={name} href={`${API_BASE_URL}/training/runs/${run.id}/files/${name}`}>{name}</a>)}</div>
          <div className="border-border-subtle mt-5 rounded border p-4"><h4 className="text-sm font-medium">Activation examples</h4>
            <p className="text-text-tertiary mt-1 text-xs">Scan the held-out corpus for this dictionary’s strongest positive activations. Results are saved under this checkpoint only.</p>
            <div className="mt-3 flex flex-wrap items-end gap-3"><label className="text-text-secondary text-xs">Feature ID<input className={`${field} mt-1 w-28`} type="number" min="0" max={run.config.features - 1} value={exampleFeature} onChange={(e) => setExampleFeature(e.target.value)} /></label>
              <button className={button} disabled={Boolean(exampleStatus && !exampleStatus.startsWith('Saved'))} onClick={collectExamples}>Collect examples</button></div>
            <div className="mt-3 flex flex-wrap items-end gap-3"><label className="text-text-secondary text-xs">Auto-interp profile<input className={`${field} mt-1 w-36`} value={interpProfile} onChange={(e) => setInterpProfile(e.target.value)} /></label>
              <button className={button} disabled={!exampleReport || Boolean(exampleStatus && !exampleStatus.startsWith('Saved'))} onClick={createInterpJob}>Generate candidate label</button></div>
            {exampleStatus ? <p role="status" className="text-text-secondary mt-3 text-xs">{exampleStatus}</p> : null}
            {exampleReport ? <div className="mt-4 space-y-3">{(exampleReport.examples[exampleFeature] ?? []).map((row, index) => <article className="bg-bg-base rounded p-3" key={`${row.token_position}-${index}`}><p className="text-text-tertiary font-mono text-xs">activation {number(row.activation, 4)} · token {row.token_position}</p><p className="mt-2 whitespace-pre-wrap text-xs leading-5">{row.context}</p></article>)}
              {(exampleReport.examples[exampleFeature] ?? []).length === 0 ? <p className="text-text-tertiary text-xs">No positive activations appeared in this bounded scan. Try a larger scan or another feature.</p> : null}</div> : null}
          </div>
          <p className="text-text-tertiary mt-4 text-xs">Resume restores optimizer and random state, then replays the recorded dataset to the checkpoint position. It requires the same model and training package versions.</p>
        </section> : null}
      </section>
    </div>
  </main>
}
