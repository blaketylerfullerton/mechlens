import type {
  FeatureResponse,
  HealthResponse,
  JobResponse,
  JobStatusResponse,
  SteerRequest,
  TraceRequest,
} from './api-types'

// Exported because the UI prints it: an empty state that tells you to call the
// API, or an error that says it could not be reached, has to name the same URL
// this client actually uses rather than a plausible-looking one.
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const BASE_URL = API_BASE_URL

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    // fetch only rejects when the request never got an HTTP reply at all —
    // backend down, still binding its port, or a different origin than the one
    // its CORS config allows. The browser's own message for this ("network
    // connection was lost", "access control checks") points at neither, so say
    // which URL failed and what to check.
    throw new ApiError(0, `cannot reach the mechlens API at ${BASE_URL} — is the backend running?`)
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

// POST /trace — starts a trace job, returns immediately with a job id.
// Poll getTraceJob() until status is "done" or "error". `body.passes` picks the
// enrichment passes the job runs; omitting it is capture-only.
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

// GET /health — is the model loaded? "loading" is normal right after start:
// the server answers before gemma is in memory, and the model-backed routes
// return 503 until it is.
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

// GET /feature/{layer}/{idx} — synchronous, no job/poll needed.
export function getFeature(layer: number, idx: number): Promise<FeatureResponse> {
  return request<FeatureResponse>(`/feature/${layer}/${idx}`)
}
