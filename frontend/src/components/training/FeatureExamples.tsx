import { useEffect, useState } from 'react'
import { API_BASE_URL, ApiError, request } from '@/lib/api-client'
import { button, field } from './shared'
import type { ExampleReport, Run } from './types'
import { number } from './shared'

type Profile = { name: string; model: string }
type LabelJob = { id: string; artifact_id: string; feature_ids: number[]; status: string; completed: number; total: number; error: string | null; records?: { feature_id: number; candidate: { label: string; summary: string; uncertainty: string } }[] }
type Scan = { status: string; error: string | null; progress: { done: number; total: number } | null }

// Remount by run and feature so a late result cannot overwrite another feature's panel.
export function FeatureExamples({ runId, featureId }: { runId: string; featureId: number }) {
  return <FeatureExamplesContent key={`${runId}:${featureId}`} runId={runId} featureId={featureId} />
}

function FeatureExamplesContent({ runId, featureId }: { runId: string; featureId: number }) {
  const [report, setReport] = useState<ExampleReport | null>(null)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [profile, setProfile] = useState('')
  const [jobs, setJobs] = useState<LabelJob[]>([])
  const [artifact, setArtifact] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const scanKey = `mechlens-example-scan:${API_BASE_URL}:${runId}:${featureId}`
  const [scanId, setScanId] = useState<string | null>(() => {
    try { return sessionStorage.getItem(scanKey) } catch { return null }
  })
  const [scanStatus, setScanStatus] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const base = `/training/runs/${runId}`
  const rows = report?.artifact_id === artifact ? report?.examples[String(featureId)] ?? [] : []

  useEffect(() => {
    let stopped = false
    async function load() {
      setLoading(true)
      try {
        const run = await request<Run>(base)
        const [available, saved] = await Promise.allSettled([
          request<Profile[]>('/training/interp-profiles'),
          run.examples ? request<ExampleReport>(`${base}/examples`) : Promise.resolve(null),
        ])
        if (stopped) return
        setArtifact(run.checkpoint?.artifact_id ?? null)
        if (saved.status === 'fulfilled') setReport(saved.value)
        else setError(String(saved.reason))
        if (available.status === 'fulfilled') {
          setProfiles(available.value)
          setProfile((current) => available.value.some((p) => p.name === current) ? current : available.value[0]?.name ?? '')
        } else setError(String(available.reason))
      } catch (err) { if (!stopped) setError(String(err)) }
      finally { if (!stopped) setLoading(false) }
    }
    void load()
    return () => { stopped = true }
  }, [base, reload])

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const result = await request<LabelJob[]>(`${base}/interp-jobs`)
        if (!stopped) setJobs(result)
      } catch (err) { if (!stopped) setError(String(err)) }
      if (!stopped) timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [base])

  useEffect(() => {
    try { if (scanId) sessionStorage.setItem(scanKey, scanId); else sessionStorage.removeItem(scanKey) } catch { /* Scan still works without browser storage. */ }
  }, [scanId, scanKey])

  useEffect(() => {
    if (!scanId) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const job = await request<Scan>(`${base}/examples/jobs/${scanId}`)
        if (stopped) return
        if (job.status === 'done') {
          setScanId(null); setScanStatus('Examples saved.'); setReload((value) => value + 1)
        } else if (job.status === 'error') {
          setScanId(null); setScanStatus(null); setError(job.error ?? 'Example collection failed. Try again.')
        } else {
          setScanStatus(job.progress ? `Finding examples · ${job.progress.done} / ${job.progress.total} text chunks` : 'Waiting for the compute queue…')
          timer = setTimeout(poll, 800)
        }
      } catch (err) {
        if (!stopped) { setScanId(null); setScanStatus(null); setError(String(err)) }
      }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [base, scanId])

  async function collect() {
    setBusy('Queueing examples…'); setError(null)
    try {
      const result = await request<{ job_id: string }>(`${base}/examples`, { method: 'POST', body: JSON.stringify({ feature_ids: [featureId] }) })
      setScanId(result.job_id)
    } catch (err) { setError(String(err)) }
    finally { setBusy(null) }
  }
  async function label() {
    setBusy('Starting label job…'); setError(null)
    try {
      const job = await request<LabelJob>(`${base}/interp-jobs`, { method: 'POST', body: JSON.stringify({ feature_ids: [featureId], profile }) })
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)])
    } catch (err) {
      setError(String(err))
      if (err instanceof ApiError && err.status === 422) setReload((value) => value + 1)
    } finally { setBusy(null) }
  }
  const relevantJobs = jobs.filter((job) => job.artifact_id === artifact && job.feature_ids.includes(featureId))
  const labeling = relevantJobs.some((job) => ['queued', 'running', 'pending'].includes(job.status))
  return <section className="border-border-subtle bg-bg-surface mt-4 rounded border p-4" aria-label={`Examples for feature ${featureId}`}>
    <h3 className="text-sm font-medium">Feature #{featureId}</h3><p className="text-text-secondary mt-2 text-xs leading-5">See passages that strongly activate this feature in your saved dictionary.</p>
    {error ? <div role="alert" className="text-err mt-3 text-xs"><p>{error}</p><button className="mt-2 underline" onClick={() => { setError(null); setReload((value) => value + 1) }}>Reload feature data</button></div> : null}
    {loading ? <p role="status" className="text-text-secondary mt-3 text-xs">Loading saved examples…</p> : rows.length ? <div className="mt-4 space-y-3">{rows.map((row, index) => <article key={index} className="border-border-subtle border-t pt-3"><p className="text-text-tertiary text-xs">Activation {number(row.activation, 3)}</p><p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6">{row.context}</p></article>)}</div> : <p className="text-text-secondary mt-3 text-xs">{report?.examples[String(featureId)] ? 'No positive activations appeared in the saved scan. Try another feature or scan again.' : 'No saved examples for this feature yet.'}</p>}
    <button className={`${button} mt-4 w-full`} disabled={loading || !!busy || !!scanId || !artifact} onClick={collect}>{busy === 'Queueing examples…' ? busy : scanId ? 'Finding examples…' : rows.length ? 'Scan for examples again' : 'Find examples'}</button>
    {scanStatus ? <p role="status" className="text-text-secondary mt-3 text-xs">{scanStatus}</p> : null}
    <div className="border-border-subtle mt-5 border-t pt-4"><h4 className="text-sm font-medium">Explain this feature</h4><p className="text-text-secondary mt-2 text-xs leading-5">Generate a candidate explanation from its examples. Labels are unverified.</p>
      {profiles.length ? <><label className="text-text-secondary mt-3 block text-xs">Labeling model<select className={field} value={profile} onChange={(event) => setProfile(event.target.value)}>{profiles.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.model}</option>)}</select></label><button className={`${button} mt-3 w-full`} disabled={!rows.length || !!busy || !!scanId || labeling || !profile} onClick={label}>{labeling ? 'Generating explanation…' : busy === 'Starting label job…' ? busy : 'Generate candidate label'}</button>{!rows.length ? <p className="text-text-tertiary mt-2 text-xs">Find examples first to enable labeling.</p> : null}</> : !loading ? <p className="text-text-secondary mt-3 text-xs">No labeling model is configured. You can still inspect examples.</p> : null}
      {relevantJobs.map((job) => <article key={job.id} className="mt-4 text-xs"><p className={job.error ? 'text-err' : 'text-text-secondary'}>{job.error ?? `Label job ${job.status}`}</p>{job.records?.filter((record) => record.feature_id === featureId).map((record) => <div key={record.feature_id} className="mt-2"><p className="text-sm font-medium">{record.candidate.label} <span className="text-text-tertiary text-xs">· Unverified</span></p><p className="mt-2 leading-5">{record.candidate.summary}</p><p className="text-text-secondary mt-2 leading-5">{record.candidate.uncertainty}</p></div>)}</article>)}
    </div>
  </section>
}
