import { useState } from 'react'
import type { FormEvent } from 'react'
import type { ModelDownload, Options } from './types'
import { useModelDownloads } from './api'
import { number, selectedLayers, storageSize } from './shared'

import { button, defaults, field, primary } from './shared'
import type { TrainingDraft, LayerMode } from './shared'

export function TrainingSetup({ options, draft, setDraft, repository, setRepository, busy, preparing, onPrepare, onStart, layerMode, setLayerMode, chosenLayers, setChosenLayers }: {
  layerMode: LayerMode; setLayerMode: (mode: LayerMode) => void; chosenLayers: number[]; setChosenLayers: (layers: number[]) => void;
  options: Options | null; draft: TrainingDraft; setDraft: (draft: TrainingDraft) => void;
  repository: string; setRepository: (value: string) => void; busy: string | null; preparing: boolean;
  onPrepare: () => void; onStart: (event: FormEvent) => void;
}) {
  const [changeModel, setChangeModel] = useState(false)
  const downloads = useModelDownloads(true)
  const byRepository = new Map((downloads?.models ?? []).map((entry) => [entry.repository, entry]))
  const selectedDownload = byRepository.get(repository)
  function downloadLabel(entry: ModelDownload | undefined) {
    if (!entry) return ''
    // A size is only known for weights already on disk; asking the hub for the
    // size of one that is not would be a network call this read-only view avoids.
    return entry.downloaded ? ` — ${storageSize(entry.bytes ?? undefined)} downloaded` : ' — not downloaded'
  }
  const ready = options?.readiness === 'ready' && options.model_repository === repository
  const layers = selectedLayers(layerMode, draft.layer, chosenLayers, ready ? options?.layers : null)
  const weights = ready && options?.d_in ? (2 * options.d_in * draft.features + options.d_in + draft.features) * 4 : undefined
  const retainedBytes = weights === undefined ? undefined : weights * 3 * layers.length
  const presetDefaults = draft.features === 4096 && draft.batch_size === 256 && draft.context_size === 128 && draft.learning_rate === 0.0003 && draft.l1_coefficient === 0.1 && draft.seed === 42 && draft.evaluation_sequences === 64 && !draft.compare_huggingface
  const preset = presetDefaults && draft.training_tokens === 32768 ? 'quick' : presetDefaults && draft.training_tokens === 131072 ? 'pilot' : 'custom'
  const update = <K extends keyof TrainingDraft>(key: K, value: TrainingDraft[K]) => setDraft({ ...draft, [key]: value })
  function size(tokens: number) {
    setDraft({ ...draft, features: defaults.features, training_tokens: tokens, batch_size: defaults.batch_size, context_size: defaults.context_size, learning_rate: defaults.learning_rate, l1_coefficient: defaults.l1_coefficient, seed: defaults.seed, evaluation_sequences: defaults.evaluation_sequences, compare_huggingface: false })
  }
  return <form onSubmit={onStart} className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_300px]">
    <div className="space-y-8">
      <section><h2 className="text-xl font-medium">Set up your training run</h2><p className="text-text-secondary mt-2 text-sm leading-6">Train a dictionary to find features in a language model. The language model itself stays unchanged.</p></section>
      <section className="border-border-subtle border-t pt-6">
        <div className="flex items-center justify-between gap-3"><h3 className="font-medium">Model</h3><button type="button" className="text-fn text-sm" onClick={() => setChangeModel(!changeModel)} aria-expanded={changeModel}>{changeModel ? 'Done' : 'Change'}</button></div>
        {changeModel ? <label className="text-text-secondary mt-3 block text-sm">Choose a model<select className={field} value={repository} disabled={!!busy || preparing} onChange={(event) => setRepository(event.target.value)}>{(options?.supported_models ?? [repository]).map((id) => <option key={id} value={id}>{id}{downloadLabel(byRepository.get(id))}</option>)}</select></label> : <p className="mt-3 break-all text-sm">{repository}</p>}
        {selectedDownload && !selectedDownload.downloaded && !preparing ? <p className="text-text-secondary mt-2 text-xs leading-5">These weights are not on this machine yet. Preparing the model downloads them from Hugging Face first, which can take a while on a slow connection. They are kept for later runs.</p> : null}
        <p role="status" className="text-text-secondary mt-2 text-sm">{preparing ? 'Preparing model. You can finish choosing your settings while it loads.' : ready ? 'Ready to train' : !options ? 'Connecting to the training backend…' : options.readiness === 'ready' ? `Preparation needed. Currently loaded: ${options.model}.` : options.readiness}</p>
      </section>
      <fieldset><legend className="mb-3 font-medium">Layers to train</legend>
        <div className="flex flex-wrap gap-2">{([['one', 'One layer'], ['choose', 'Choose layers'], ['all', 'All layers']] as const).map(([mode, label]) => <button key={mode} type="button" aria-pressed={layerMode === mode} disabled={mode !== 'one' && (!ready || !options?.multi_layer)} className={`${button} ${layerMode === mode ? 'border-fn bg-fn/10' : ''}`} onClick={() => setLayerMode(mode)}>{label}</button>)}</div>
        {layerMode === 'one' ? <label className="text-text-secondary mt-4 block text-sm">Layer<input className={`${field} max-w-32`} type="number" min="0" max={(options?.layers ?? 26) - 1} required value={draft.layer} onChange={(event) => update('layer', Number(event.target.value))} /></label> : layerMode === 'choose' ? <div className="mt-4 flex flex-wrap gap-2">{Array.from({ length: ready ? options?.layers ?? 0 : 0 }, (_, index) => <label key={index} className="border-border-subtle flex cursor-pointer items-center gap-2 rounded border px-3 py-2 text-sm"><input type="checkbox" checked={chosenLayers.includes(index)} onChange={(event) => setChosenLayers(event.target.checked ? [...chosenLayers, index] : chosenLayers.filter((layer) => layer !== index))} />Layer {index}</label>)}</div> : <p className="text-text-secondary mt-3 text-sm">{layers.length} layers, trained one at a time. Each gets its own saved dictionary.</p>}
        {layerMode !== 'one' ? <p className="text-text-secondary mt-3 text-xs">{layers.length ? `${layers.length} layers selected. The token budget below applies to each layer.` : 'Select at least one layer to continue.'}</p> : null}
        {ready && !options?.multi_layer ? <p className="text-text-secondary mt-3 text-xs">Restart the backend to enable multi-layer training and storage management.</p> : null}
      </fieldset>
      <section><label className="block font-medium" htmlFor="training-dataset">Training text</label><select id="training-dataset" className={field} value={draft.dataset} onChange={(event) => update('dataset', event.target.value)}><option value="tiny-stories">Use sample stories · TinyStories</option><option value="text">Use my text</option></select>
        {draft.dataset === 'tiny-stories' ? <p className="text-text-secondary mt-2 text-sm">A ready-to-use dataset with separate text for evaluation.</p> : <div className="mt-4 space-y-4"><div><label htmlFor="training-corpus" className="text-text-secondary block text-sm">Text to train on</label><textarea id="training-corpus" className={field} rows={5} required maxLength={2000000} value={draft.training_text} onChange={(event) => update('training_text', event.target.value)} /></div><div><label htmlFor="evaluation-corpus" className="text-text-secondary block text-sm">Separate text to check the result</label><textarea id="evaluation-corpus" aria-describedby="evaluation-help" className={field} rows={3} required maxLength={200000} value={draft.evaluation_text} onChange={(event) => update('evaluation_text', event.target.value)} /><p id="evaluation-help" className="text-text-secondary mt-2 text-xs">Use text that does not appear in your training data.</p></div></div>}
      </section>
      <fieldset><legend className="mb-3 font-medium">Run size {preset === 'custom' ? <span className="text-fn ml-2 text-xs">Custom settings</span> : null}</legend><div className="grid gap-3 sm:grid-cols-2">{[{ id: 'quick', tokens: 32768, title: 'Quick test', text: 'Check that the workflow works.' }, { id: 'pilot', tokens: 131072, title: 'Pilot run', text: 'A larger starting point for exploring features.' }].map((item) => <button type="button" key={item.id} aria-pressed={preset === item.id} onClick={() => size(item.tokens)} className={`${button} p-4 text-left ${preset === item.id ? 'border-fn bg-fn/10' : ''}`}><span className="block font-medium">{item.title}{item.id === 'pilot' ? <span className="text-text-secondary ml-2 text-xs">Default</span> : null}</span><span className="text-text-secondary mt-2 block text-sm leading-5">{item.text}</span><span className="text-text-tertiary mt-3 block text-xs">{number(item.tokens, 0)} tokens</span></button>)}</div><p className="text-text-secondary mt-3 text-xs">Both are small experiments. Feature quality needs evaluation.</p></fieldset>
      <details className="border-border-subtle rounded border p-4"><summary className="cursor-pointer text-sm font-medium">Advanced settings</summary><p className="text-text-secondary mt-3 text-xs">The default layer is a starting point. Change it to investigate a specific part of the model.</p><div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label className="text-text-secondary text-sm">Dictionary features<select className={field} value={draft.features} onChange={(event) => update('features', Number(event.target.value))}>{[256, 1024, 4096, 16384].map((n) => <option key={n} value={n}>{number(n, 0)}</option>)}</select></label>
        {([
          ['training_tokens', 'Training token budget', draft.batch_size, 100000000, draft.batch_size],
          ['batch_size', 'Batch tokens', 16, 8192, 1], ['context_size', 'Context tokens', 8, 2048, 1],
          ['learning_rate', 'Learning rate', 0.00000001, 0.01, 'any'], ['l1_coefficient', 'Sparsity penalty · L1', 0, 100, 'any'],
          ['evaluation_sequences', 'Evaluation sequences', 1, 2048, 1], ['seed', 'Random seed', 0, 4294967295, 1],
        ] as const).map(([key, label, min, max, step]) => <label className="text-text-secondary text-sm" key={key}>{label}<input className={field} type="number" min={min} max={max} step={step} required value={draft[key]} onChange={(event) => update(key, Number(event.target.value))} /></label>)}
      </div><label className="text-text-secondary mt-5 block text-sm"><input type="checkbox" className="mr-2" checked={draft.compare_huggingface} onChange={(event) => update('compare_huggingface', event.target.checked)} />Compare with Hugging Face model</label><p className="text-text-tertiary mt-2 text-xs">Optional correctness check. Loads a second model on CPU and adds memory use and evaluation time.</p></details>
    </div>
    <aside className="border-border-subtle bg-bg-surface rounded border p-5 lg:sticky lg:top-6"><h3 className="font-medium">Your training plan</h3><dl className="mt-5 space-y-4 text-sm">{[['Model', repository], ['Text', draft.dataset === 'text' ? 'Your text' : 'Sample stories'], ['Layers', layerMode === 'one' ? `Layer ${draft.layer}` : layerMode === 'all' ? `All ${layers.length} layers` : layers.join(', ') || 'None selected'], ['Features per layer', number(draft.features, 0)], ['Tokens per layer', number(draft.training_tokens, 0)], ['Compute', options?.device ?? 'Connecting…']].map(([label, value]) => <div className="flex justify-between gap-4" key={label}><dt className="text-text-secondary shrink-0">{label}</dt><dd className="break-all text-right">{value}</dd></div>)}</dl><p className="text-text-secondary mt-5 text-xs leading-5">{options?.estimate_note ?? 'Runtime estimate unavailable.'} Evaluation follows training.</p><p className="text-text-tertiary mt-3 text-xs leading-5">Training and exploration share one compute queue.</p>
      <div className="border-border-subtle mt-5 border-t pt-4"><p className="text-sm font-medium">Estimated storage: {storageSize(retainedBytes)}</p><p className="text-text-secondary mt-2 text-xs leading-5">Includes dictionary weights and resume state for {layers.length} layer{layers.length === 1 ? '' : 's'}. Only the latest checkpoint is kept. Allow extra room while a new checkpoint is being saved.</p><p className="text-text-tertiary mt-2 text-xs">Base model, dataset, and examples are additional.</p></div>
      {ready ? <button className={`${primary} mt-5 w-full`} type="submit" disabled={!!busy || layers.length === 0 || (layerMode !== 'one' && !options?.multi_layer)}>{busy ?? (layers.length > 1 ? `Train ${layers.length} layers` : 'Start training')}</button> : <button className={`${primary} mt-5 w-full`} type="button" disabled={!options || !!busy || preparing} onClick={onPrepare}>{busy ?? (preparing ? 'Preparing model…' : 'Prepare model')}</button>}
      {!ready ? <p className="text-text-secondary mt-3 text-xs">Once this model is ready, start training here.</p> : null}
    </aside>
  </form>
}
