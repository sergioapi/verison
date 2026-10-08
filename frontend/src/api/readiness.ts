import { getApiBaseUrl } from './config'
import { READINESS_ENDPOINT_PATH } from './endpoints'
import { createApiError, isApiError } from './errors'
import { fetchWithTimeout } from './request'

export const READINESS_TIMEOUT_MS = 15_000
export const READINESS_POLL_INTERVAL_MS = 5_000
export const READINESS_REQUEST_TIMEOUT_MS = 5_000

export type ReadinessStatus = 'pending' | 'ready' | 'failed' | 'unavailable'

type GetReadinessOptions = {
  fetchImpl?: typeof fetch
  signal?: AbortSignal
  timeoutMs?: number
}

type WaitUntilReadyOptions = GetReadinessOptions & {
  timeoutMs?: number
  pollIntervalMs?: number
  now?: () => number
  sleep?: (milliseconds: number, signal?: AbortSignal) => Promise<void>
}

export async function getReadiness(
  options: GetReadinessOptions = {},
): Promise<ReadinessStatus> {
  const response = await getReadinessResponse(options)
  const payload = await readReadinessPayload(response)

  if (!isReadinessStatus(payload.status)) {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The readiness API returned an unexpected response.',
    })
  }
  const expectedStatusCode = payload.status === 'ready' ? 200 : 503
  if (response.status !== expectedStatusCode) {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The readiness API returned an inconsistent response.',
    })
  }

  return payload.status
}

export async function waitUntilReady(
  options: WaitUntilReadyOptions = {},
): Promise<void> {
  const now = options.now ?? Date.now
  const timeoutMs = options.timeoutMs ?? READINESS_TIMEOUT_MS
  const pollIntervalMs = options.pollIntervalMs ?? READINESS_POLL_INTERVAL_MS
  const sleep = options.sleep ?? sleepFor
  const deadline = now() + timeoutMs

  while (true) {
    throwIfAborted(options.signal)
    const remainingMs = deadline - now()
    if (remainingMs <= 0) {
      throwServiceUnavailable()
    }

    try {
      const readiness = await getReadiness({
        fetchImpl: options.fetchImpl,
        signal: options.signal,
        timeoutMs: Math.min(READINESS_REQUEST_TIMEOUT_MS, remainingMs),
      })
      if (readiness === 'ready') {
        return
      }
      if (readiness === 'failed' || readiness === 'unavailable') {
        throwServiceUnavailable()
      }
    } catch (error) {
      if (isApiError(error) && error.kind === 'aborted') {
        throw error
      }
      if (isApiError(error) && !isTransientReadinessError(error)) {
        throw error
      }
    }

    const waitMs = deadline - now()
    if (waitMs <= 0) {
      throwServiceUnavailable()
    }
    await sleep(Math.min(pollIntervalMs, waitMs), options.signal)
  }
}

async function getReadinessResponse(options: GetReadinessOptions): Promise<Response> {
  const readinessUrl = `${getApiBaseUrl()}${READINESS_ENDPOINT_PATH}`
  try {
    return await fetchWithTimeout(
      readinessUrl,
      { method: 'GET' },
      { signal: options.signal, timeoutMs: options.timeoutMs ?? READINESS_REQUEST_TIMEOUT_MS },
      options.fetchImpl ?? fetch,
    )
  } catch (error) {
    if (isApiError(error)) {
      throw error
    }
    throw createApiError({
      kind: 'network',
      message: 'Could not contact the readiness service.',
    })
  }
}

async function readReadinessPayload(response: Response): Promise<{ status?: unknown }> {
  try {
    return (await response.json()) as { status?: unknown }
  } catch {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The readiness API returned a non-JSON response.',
    })
  }
}

function isReadinessStatus(value: unknown): value is ReadinessStatus {
  return value === 'pending' || value === 'ready' || value === 'failed' || value === 'unavailable'
}

function isTransientReadinessError(error: { kind: string }): boolean {
  return error.kind === 'network' || error.kind === 'timeout'
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortedError()
  }
}

function throwServiceUnavailable(): never {
  throw createApiError({
    kind: 'service-unavailable',
    message: 'The analysis service is not ready.',
  })
}

function sleepFor(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortedError())
      return
    }

    let timeoutId: number | undefined = undefined
    const cleanup = () => {
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId)
      }
      signal?.removeEventListener('abort', abort)
    }
    const abort = () => {
      cleanup()
      reject(createAbortedError())
    }
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) {
      abort()
      return
    }
    timeoutId = window.setTimeout(() => {
      cleanup()
      resolve()
    }, milliseconds)
  })
}

function createAbortedError() {
  return createApiError({ kind: 'aborted', message: 'The request was cancelled.' })
}
