import type { Trace } from './api-types'
import { API_BASE_URL } from './api-client'

// IndexedDB avoids localStorage's small synchronous quota for multi-MB traces.
// Keep one completed trace per backend. Partial runs never replace it.
const DATABASE = 'mechlens-workspace'
const STORE = 'traces'

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(STORE)
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

export async function readLastTrace(): Promise<Trace | null> {
  const database = await openDatabase()
  try {
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE).objectStore(STORE).get(API_BASE_URL)
      request.onsuccess = () => {
        const trace = request.result as Trace | undefined
        // Ignore incompatible records instead of mounting a broken workspace.
        resolve(trace && typeof trace.trace_id === 'string' && Array.isArray(trace.steps)
          && trace.steps.length > 0 && Array.isArray(trace.passes)
          && typeof trace.completion === 'string' ? trace : null)
      }
      request.onerror = () => reject(request.error)
    })
  } finally {
    database.close()
  }
}

export async function saveLastTrace(trace: Trace): Promise<void> {
  const database = await openDatabase()
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE, 'readwrite')
      transaction.objectStore(STORE).put(trace, API_BASE_URL)
      transaction.oncomplete = () => resolve()
      transaction.onabort = () => reject(transaction.error)
      transaction.onerror = () => reject(transaction.error)
    })
  } finally {
    database.close()
  }
}

/** Drops the saved trace, so a reset workspace stays reset across a reload.
 * Resolves even when there was nothing stored: "already gone" is the state
 * the caller asked for. */
export async function clearLastTrace(): Promise<void> {
  const database = await openDatabase()
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE, 'readwrite')
      transaction.objectStore(STORE).delete(API_BASE_URL)
      transaction.oncomplete = () => resolve()
      transaction.onabort = () => reject(transaction.error)
      transaction.onerror = () => reject(transaction.error)
    })
  } finally {
    database.close()
  }
}
