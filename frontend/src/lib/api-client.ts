import type {
  FeatureResponse,
  JobResponse,
  JobStatusResponse,
  SteerRequest,
  TraceRequest,
} from './api-types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

// POST /trace — starts a trace job, returns immediately with a job id.
// Poll getTraceJob() until status is "done" or "error".
export function postTrace(body: TraceRequest): Promise<JobResponse> {
  return request<JobResponse>('/trace', { method: 'POST', body: JSON.stringify(body) })
}

export function getTraceJob(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/trace/${jobId}`)
}

// POST /steer — same job/poll shape as postTrace, with a feature intervention applied.
export function postSteer(body: SteerRequest): Promise<JobResponse> {
  return request<JobResponse>('/steer', { method: 'POST', body: JSON.stringify(body) })
}

// GET /feature/{layer}/{idx} — synchronous, no job/poll needed.
export function getFeature(layer: number, idx: number): Promise<FeatureResponse> {
  return request<FeatureResponse>(`/feature/${layer}/${idx}`)
}
