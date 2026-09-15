import { number } from './shared'
import { API_BASE_URL } from '@/lib/api-client'
import type { Run, Metric, Check } from './types'
const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted'])

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

export function RunMeasurements({ run, metrics }: { run: Run; metrics: Metric[] }) {
  const checks = [
    run.config.compare_huggingface === false ? { status: 'skipped' } : run.validation?.model_agreement,
    run.validation?.checkpoint_agreement, run.validation?.activation_identity,
    run.evaluation?.identity_substitution ?? run.validation?.identity_substitution,
  ]
  const failed = checks.filter((check) => check?.status === 'failed')
  const passed = checks.filter((check) => check?.status === 'passed').length
  const skipped = checks.filter((check) => ['skipped', 'not_run'].includes(check?.status ?? '')).length
  const incomplete = checks.length - passed - skipped - failed.length
  return <>
    {failed.length > 0 ? <div role="alert" className="border-err/50 text-err mt-5 rounded border p-4"><h3 className="font-medium">Some correctness checks failed</h3>{failed.map((check, index) => <p className="mt-2 text-sm" key={index}>{check?.reason ?? check?.definition ?? 'See the validation report before relying on this result.'}</p>)}</div> : null}
    <details className="border-border-subtle mt-6 rounded border p-5" open={!terminal.has(run.status)}>
      <summary className="cursor-pointer font-medium">Training measurements</summary>
      <p className="text-text-secondary my-4 text-sm">Loss balances reconstruction error and sparsity. Explained variance measures how much activation variation is captured. Neither proves the features are interpretable.</p>
      <div className="grid gap-4 sm:grid-cols-2"><Chart metrics={metrics} metric="loss" label="Training loss" /><Chart metrics={metrics} metric="explained_variance" label="Explained variance" /><Chart metrics={metrics} metric="l0" label="Active features per token · L0" /><Chart metrics={metrics} metric="dead_fraction" label="Dead feature fraction" /></div>
      <p className="text-text-tertiary mt-3 text-xs">Training-batch measurements. Latest 2,000 samples shown. Dead features have been inactive for over 1,000 optimizer steps.</p>
    </details>
        <section className="border-border-subtle mt-7 rounded border p-5">
          <div className="flex flex-wrap items-center justify-between gap-3"><h3 className="font-medium">Validation report</h3>
            <a className="text-text-secondary text-xs underline underline-offset-4" href={`${API_BASE_URL}/training/runs/${run.id}/report`}>Download report JSON</a></div>
          <p className="text-text-tertiary mt-3 text-xs leading-5">Correctness checks validate the pipeline. Quality measurements describe the learned dictionary; passing checks does not establish interpretability.</p>
          <div role="status" aria-live="polite" className="text-text-secondary mt-4 text-sm">
            {run.status === 'cancelling' ? 'Stopping validation and training…' : !terminal.has(run.status) ?
              run.phase === 'evaluating' ? <><span className="mr-2 inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none" aria-hidden="true" />Validation running · {run.validation_progress?.completed ?? 0} / {run.validation_progress?.total ?? '—'} held-out sequences checked</> :
              run.phase === 'checking Hugging Face agreement' ? <><span className="mr-2 inline-block size-3 animate-spin rounded-full border-2 border-current border-t-transparent motion-reduce:animate-none" aria-hidden="true" />Hugging Face comparison running · loading and checking the CPU reference model</> :
              'Validation pending · checkpoint and held-out checks run after training.' :
              `${passed} checks passed · ${failed.length} failed · ${skipped} skipped${incomplete ? ` · ${incomplete} incomplete` : ''}`}
          </div>
          {run.phase === 'evaluating' && !terminal.has(run.status) && run.validation_progress ? <progress className="mt-3 h-1.5 w-full accent-[var(--color-fn)]" value={run.validation_progress.completed} max={run.validation_progress.total} aria-label="Held-out validation sequences checked" /> : null}
          <details className="mt-4"><summary className="cursor-pointer text-sm">Check details and quality measurements</summary><dl className="mt-4 space-y-3">{([
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
        </details></section>
  </>
}
