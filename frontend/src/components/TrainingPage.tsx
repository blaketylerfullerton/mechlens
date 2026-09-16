import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { API_BASE_URL, request } from '@/lib/api-client'
import { ResourceMeter } from '@/components/ResourceMeter'
import { TrainingSetup } from './training/TrainingSetup'
import { button, defaults, primary, number, selectedLayers } from './training/shared'
import { RunMeasurements } from './training/RunMeasurements'
import type { Metric, Options, Run } from './training/types'
import { getTrainingOptions } from './training/api'
import type { LayerMode } from './training/shared'
import { BatchProgress } from './training/BatchProgress'

const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted'])
function runTitle(run: Run) {
  if (run.status === 'completed') return 'Your dictionary is ready to explore'
  if (run.status === 'failed') return 'This run needs attention'
  if (run.status === 'interrupted') return 'Training was interrupted'
  if (run.status === 'cancelled') return 'Training stopped'
  if (run.status === 'cancelling') return 'Saving and stopping…'
  if (run.status === 'queued') return 'Waiting for compute'
  if (run.phase === 'checking Hugging Face agreement') return 'Checking model compatibility…'
  if (run.phase === 'evaluating') return 'Training finished. Checking the result…'
  if (run.phase === 'training') return 'Training your dictionary'
  return `Preparing your run · ${run.phase}`
}

export function TrainingPage({ onInspect, onDictionaries, active, navigation }: {
  onInspect: (id: string) => void; onDictionaries: (id: string) => void; active: boolean;
  navigation: { id: string | null; key: number; newRun?: boolean };
}) {
  const [options, setOptions] = useState<Options | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<'setup' | 'run'>('setup')
  const [layerMode, setLayerMode] = useState<LayerMode>('one')
  const [chosenLayers, setChosenLayers] = useState<number[]>([12])
  const [following, setFollowing] = useState(false)
  const followBatch = useRef<string | null>(null)
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [metricsRun, setMetricsRun] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [preparationJob, setPreparationJob] = useState<string | null>(null)
  const [draft, setDraft] = useState(defaults)
  const [repository, setRepository] = useState('google/gemma-2-2b')
  const modelChosen = useRef(false)
  const heading = useRef<HTMLHeadingElement>(null)
  const run = runs.find((item) => item.id === selected)
  const currentMetrics = metricsRun === selected ? metrics : []
  const latest = currentMetrics.at(-1)
  const isTerminal = run ? terminal.has(run.status) : false
  const checking = run?.phase === 'evaluating'
  const modelCheck = run?.phase === 'checking Hugging Face agreement'
  const stage = view === 'setup' ? 0 : run?.status === 'completed' ? 2 : 1

  useEffect(() => {
    if (!active) return
    let stopped = false
    let entering = true
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const results = await Promise.allSettled([getTrainingOptions(), request<Run[]>('/training/runs')])
      if (stopped) return
      if (results[0].status === 'fulfilled') {
        const available = results[0].value
        setOptions(available)
        if (!modelChosen.current && available.model_repository) {
          setRepository(available.model_repository)
          modelChosen.current = true
          if (available.layers) setDraft((current) => ({ ...current, layer: Math.floor(available.layers! / 2) }))
        }
        if (available.layers) setDraft((current) => current.layer >= available.layers! ? { ...current, layer: available.layers! - 1 } : current)
      }
      if (results[1].status === 'fulfilled') {
        const history = results[1].value
        setRuns(history)
        if (!entering && followBatch.current) {
          const batch = history.filter((item) => item.batch_id === followBatch.current).sort((a, b) => (a.batch_index ?? 0) - (b.batch_index ?? 0))
          const next = batch.find((item) => !terminal.has(item.status)) ?? batch.at(-1)
          if (next) setSelected(next.id)
        }
        if (entering) {
          entering = false
          const requested = navigation.id ? history.find((item) => item.id === navigation.id) : null
          const activeRun = history.find((item) => item.status === 'running' || item.status === 'cancelling') ?? [...history].reverse().find((item) => !terminal.has(item.status))
          const target = navigation.newRun ? null : navigation.id ? requested : activeRun
          setSelected(target?.id ?? null); setView(target ? 'run' : 'setup')
          followBatch.current = !navigation.id && target?.batch_id ? target.batch_id : null
          setFollowing(!!followBatch.current)
          setError(navigation.id && !requested ? 'This run is no longer available. Find your saved work in Dictionaries.' : null)
        }
      }
      const failure = results.find((result) => result.status === 'rejected')
      setConnectionError(failure?.status === 'rejected' ? String(failure.reason) : null)
      timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [active, navigation])

  useEffect(() => {
    if (!preparationJob) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const job = await request<{ status: string; error: string | null }>(`/trace/${preparationJob}`)
        if (stopped) return
        if (job.status === 'error') { setError(job.error ?? 'Model preparation failed. Try preparing it again.'); setPreparationJob(null) }
        else if (job.status === 'done') {
          const available = await getTrainingOptions()
          if (!stopped) {
            setOptions(available)
            if (available.layers) setDraft((current) => ({ ...current, layer: Math.min(current.layer, available.layers! - 1) }))
            setPreparationJob(null)
          }
        } else timer = setTimeout(poll, 1000)
      } catch (err) { if (!stopped) { setError(String(err)); setPreparationJob(null) } }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [preparationJob])

  useEffect(() => {
    if (!active || !selected || view !== 'run') return
    let stopped = false
    let cursor = 0
    let timer: ReturnType<typeof setTimeout>
    let samples: Metric[] = []
    async function poll() {
      try {
        const values = await request<Metric[]>(`/training/runs/${selected}/metrics?after=${cursor}`)
        if (stopped) return
        if (values.length) { cursor = values.at(-1)!.seq; samples = [...samples, ...values].slice(-2000) }
        setMetrics(samples); setMetricsRun(selected)
      } catch (err) { if (!stopped) setError(String(err)) }
      if (!stopped) timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [selected, active, view])

  useEffect(() => { if (active) heading.current?.focus() }, [view, selected, active])

  async function start(event: FormEvent) {
    event.preventDefault()
    if (busy || preparationJob) return
    setBusy('Starting training…'); setError(null)
    try {
      const available = await getTrainingOptions()
      setOptions(available)
      if (available.readiness !== 'ready' || available.model_repository !== repository) throw new Error('Prepare your selected model before starting training.')
      const layers = selectedLayers(layerMode, draft.layer, chosenLayers, available.layers)
      if (!layers.length) throw new Error('Choose at least one layer before starting.')
      if (layers.some((layer) => layer < 0 || layer >= (available.layers ?? 0))) throw new Error('Choose layers that belong to the selected model.')
      let created: Run[]
      if (layerMode !== 'one') {
        if (!available.multi_layer) throw new Error('Restart the backend to enable multi-layer training.')
        const batch = await request<{ batch_id: string; runs: Run[] }>('/training/batches', { method: 'POST', body: JSON.stringify({ config: draft, layers, model_repository: repository }) })
        created = batch.runs; followBatch.current = batch.batch_id; setFollowing(true)
      } else {
        created = [await request<Run>('/training/runs', { method: 'POST', body: JSON.stringify(draft) })]
        followBatch.current = null; setFollowing(false)
      }
      setRuns((previous) => [...created, ...previous]); setSelected(created[0].id); setView('run')
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }
  async function control(action: 'cancel' | 'resume') {
    if (!run || busy) return
    setBusy(action === 'cancel' ? 'Stopping training…' : 'Resuming training…'); setError(null)
    try {
      const result = await request<Run>(`/training/runs/${run.id}/${action}`, { method: 'POST' })
      setRuns((previous) => previous.map((item) => item.id === result.id ? result : item))
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }
  async function stopBatch() {
    if (!run?.batch_id || busy) return
    setBusy('Stopping remaining layers…'); setError(null)
    try {
      const changed = await request<Run[]>(`/training/batches/${run.batch_id}/cancel`, { method: 'POST' })
      setRuns((previous) => previous.map((item) => changed.find((result) => result.id === item.id) ?? item))
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }
  function selectRun(id: string) {
    followBatch.current = null; setFollowing(false)
    setSelected(id); setView('run'); setError(null)
  }
  async function prepareModel() {
    if (busy || preparationJob) return
    setBusy('Preparing model…'); setError(null)
    try {
      const result = await request<{ job_id: string }>('/training/model', { method: 'POST', body: JSON.stringify({ repository }) })
      setPreparationJob(result.job_id)
      setOptions((current) => current ? { ...current, readiness: 'Preparing model…', model_repository: null } : current)
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }

  return <main className="mx-auto w-full max-w-[1280px] px-4 py-6 sm:px-8 lg:py-8">
    <header className="mb-8 flex flex-wrap items-start justify-between gap-5">
      <div><h1 ref={heading} tabIndex={-1} className="max-w-5xl text-3xl font-medium tracking-tight outline-none">Training</h1><ol className="text-text-secondary mt-4 flex flex-wrap gap-4 text-sm" aria-label="Dictionary workflow">{['Set up', 'Train and check', 'Explore features'].map((label, index) => <li key={label} aria-current={stage === index ? 'step' : undefined} className={stage === index ? 'text-fn font-medium' : ''}><span className="mr-2 font-mono text-xs">{index + 1}</span>{label}</li>)}</ol></div>
      {view === 'run' ? <button className={button} onClick={() => { setView('setup'); setError(null); followBatch.current = null; setFollowing(false) }}>New run</button> : run && !terminal.has(run.status) ? <button className={button} onClick={() => setView('run')}>Back to active run</button> : null}
    </header>
    {error || connectionError ? <div role="alert" className="border-err/50 text-err mb-6 rounded border p-4 text-sm"><p>{error ?? connectionError}</p>{error ? <button className="mt-2 underline" onClick={() => setError(null)}>Dismiss</button> : <p className="mt-2">Reconnecting automatically. Check that the backend is running.</p>}</div> : null}
    <div hidden={view !== 'setup'}><TrainingSetup layerMode={layerMode} setLayerMode={setLayerMode} chosenLayers={chosenLayers} setChosenLayers={setChosenLayers} options={options} draft={draft} setDraft={setDraft} repository={repository} setRepository={(value) => { modelChosen.current = true; setRepository(value) }} busy={busy} preparing={!!preparationJob} onPrepare={prepareModel} onStart={start} /></div>
    {view === 'run' && run ? <section key={run.id} className="enter">
      {run.batch_id ? <BatchProgress runs={runs.filter((item) => item.batch_id === run.batch_id)} selected={run.id} following={following} onFollow={(value) => { setFollowing(value); followBatch.current = value ? run.batch_id! : null }} onSelect={selectRun} onStop={stopBatch} busy={!!busy} /> : null}
      <div className="border-border-subtle bg-bg-surface rounded border p-5 sm:p-7">
        <p className="text-text-secondary text-sm">Layer {run.config.layer} · {number(run.config.features, 0)} features · {run.config.dataset === 'text' ? 'Your text' : 'Sample stories'}</p>
        <h2 className="mt-3 text-2xl font-medium tracking-tight" role="status">{runTitle(run)}</h2>
        <p className="text-text-secondary mt-3 text-sm leading-6">{run.status === 'completed' ? 'Try a prompt, then select an activated feature to see examples. The features are still unlabeled.' : run.status === 'queued' ? 'Another operation may be using the model. Your run will start when compute is available.' : modelCheck && !isTerminal ? 'Comparing the loaded model with a CPU reference before training starts.' : checking && !isTerminal ? 'Checking the saved dictionary and how well it preserves the model’s predictions.' : isTerminal ? run.checkpoint ? `A checkpoint was saved at ${number(run.checkpoint.tokens, 0)} tokens. You can inspect this partial result${run.resume_supported ? ' or resume the run' : ''}.` : 'No checkpoint is available. Review the error and start a new run.' : 'Your run is saved in Dictionaries. You can leave this page and come back to check progress.'}</p>
        {run.storage_warning ? <p role="alert" className="text-err mt-4 text-sm">{run.storage_warning}</p> : null}
        {run.error ? <p role="alert" className="text-err mt-4 break-words text-sm">{run.error}</p> : null}
        {!isTerminal ? <div className="mt-6">
          <progress className="h-2 w-full accent-[var(--color-fn)]" value={modelCheck ? undefined : checking ? run.validation_progress?.completed ?? 0 : run.tokens} max={checking ? run.validation_progress?.total || 1 : run.config.training_tokens} aria-label={checking ? 'Evaluation progress' : 'Training tokens processed'} />
          <div className="text-text-secondary mt-3 flex flex-wrap justify-between gap-2 text-sm"><span>{modelCheck ? 'Comparing against the Hugging Face model' : checking ? `${number(run.validation_progress?.completed ?? 0, 0)} / ${number(run.validation_progress?.total, 0)} evaluation sequences` : `${number(run.tokens, 0)} / ${number(run.config.training_tokens, 0)} training tokens`}</span><span>{run.status === 'running' && run.phase === 'training' && latest ? `About ${number(latest.eta_s / 60, 1)} min of training remaining` : run.status === 'queued' ? 'Queued' : 'Time estimate unavailable'}</span></div>
          {!checking && !modelCheck ? <p className="text-text-tertiary mt-2 text-xs">Evaluation follows training.{latest ? ` ${number(latest.tokens_per_second, 0)} tokens/s.` : ''}</p> : null}
        </div> : null}
        <div className="mt-6 flex flex-wrap gap-3">{!isTerminal ? <button className={button} disabled={!!busy || run.status === 'cancelling'} onClick={() => control('cancel')}>{busy ?? (run.status === 'cancelling' ? 'Saving and stopping…' : 'Stop training')}</button> : <>
          {run.resume_supported && run.status !== 'completed' ? <button className={primary} disabled={!!busy || options?.readiness !== 'ready'} onClick={() => control('resume')}>{busy ?? 'Resume training'}</button> : null}
          {run.checkpoint ? <button className={run.status === 'completed' ? primary : button} onClick={() => onInspect(run.id)}>{run.status === 'completed' ? 'Explore features' : 'Explore saved checkpoint'}</button> : null}
          {run.status !== 'completed' ? <button className={button} onClick={() => setView('setup')}>Edit setup for a new run</button> : null}
        </>}
          {isTerminal ? <button className="text-fn px-2 py-2 text-sm" onClick={() => onDictionaries(run.id)}>View in Dictionaries</button> : null}
        </div>
      </div>
      <RunMeasurements run={run} metrics={currentMetrics} />
      {run.checkpoint ? <details className="border-border-subtle mt-6 rounded border p-5"><summary className="cursor-pointer text-sm font-medium">Saved dictionary and downloads</summary><p className="text-text-secondary mt-4 text-sm">Checkpoint at {number(run.checkpoint.tokens, 0)} tokens.</p><code className="text-text-tertiary mt-2 block break-all text-xs">{run.checkpoint.artifact_id}</code><div className="mt-4 flex flex-wrap gap-4">{['cfg.json', 'sae_weights.safetensors', 'manifest.json'].map((name) => <a className="text-fn text-sm underline underline-offset-4" key={name} href={`${API_BASE_URL}/training/runs/${run.id}/files/${name}`}>{name}</a>)}</div><p className="text-text-tertiary mt-4 text-xs leading-5">Resume restores saved training state and requires the same model and training package versions.</p></details> : null}
    </section> : null}
    <details className="border-border-subtle mt-8 border-t pt-4"><summary className="text-text-secondary cursor-pointer text-xs">Compute resources · {options?.device ?? 'connecting'}</summary><div className="mt-3"><ResourceMeter /></div></details>
  </main>
}
