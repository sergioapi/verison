import { createApiError } from './errors'

type FetchWithTimeoutOptions = {
  signal?: AbortSignal
  timeoutMs: number
}

export async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  options: FetchWithTimeoutOptions,
  fetchImpl: typeof fetch,
): Promise<Response> {
  const controller = new AbortController()
  let timedOut = false
  const abortFromParent = () => controller.abort()
  const timeoutId = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, options.timeoutMs)

  options.signal?.addEventListener('abort', abortFromParent, { once: true })
  if (options.signal?.aborted) {
    controller.abort()
  }

  try {
    return await fetchImpl(url, { ...init, signal: controller.signal })
  } catch (error) {
    if (options.signal?.aborted) {
      throw createApiError({ kind: 'aborted', message: 'The request was cancelled.' })
    }
    if (timedOut) {
      throw createApiError({ kind: 'timeout', message: 'The request timed out.' })
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
    options.signal?.removeEventListener('abort', abortFromParent)
  }
}
