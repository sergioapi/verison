import { describe, expect, it, vi } from 'vitest'
import {
  READINESS_POLL_INTERVAL_MS,
  READINESS_REQUEST_TIMEOUT_MS,
  READINESS_TIMEOUT_MS,
  waitUntilReady,
} from './readiness'

const API_URL = 'https://api.example.test'

describe('waitUntilReady', () => {
  it('returns immediately when readiness is ready', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      readinessResponse(200, 'ready'),
    )

    await waitUntilReady({ fetchImpl })

    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('polls pending readiness until ready', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(readinessResponse(503, 'pending'))
      .mockResolvedValueOnce(readinessResponse(200, 'ready'))

    await waitUntilReady({ fetchImpl, sleep: async () => {} })

    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(fetchImpl).toHaveBeenNthCalledWith(
      1,
      `${API_URL}/ready`,
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it.each(['failed', 'unavailable'] as const)(
    'stops without retrying when readiness is %s',
    async (status) => {
      vi.stubEnv('VITE_API_BASE_URL', API_URL)
      const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
        readinessResponse(503, status),
      )

      await expect(waitUntilReady({ fetchImpl })).rejects.toMatchObject({
        kind: 'service-unavailable',
      })
      expect(fetchImpl).toHaveBeenCalledTimes(1)
    },
  )

  it('recovers from a transient readiness network error within its budget', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>()
      .mockRejectedValueOnce(new TypeError('network interrupted'))
      .mockResolvedValueOnce(readinessResponse(200, 'ready'))

    await waitUntilReady({ fetchImpl, sleep: async () => {} })

    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('stops polling after the readiness budget is exhausted', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    vi.useFakeTimers()
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )
    const waiting = waitUntilReady({ fetchImpl })
    const unavailableExpectation = expect(waiting).rejects.toMatchObject({
      kind: 'service-unavailable',
    })

    await vi.advanceTimersByTimeAsync(READINESS_TIMEOUT_MS)

    await unavailableExpectation
    expect(fetchImpl).toHaveBeenCalledTimes(3)
    expect(vi.getTimerCount()).toBe(0)
    vi.useRealTimers()
  })

  it('retries a timed-out readiness request before succeeding', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const fetchImpl = vi.fn<typeof fetch>()
      .mockImplementationOnce((_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'))
          })
        }),
      )
      .mockResolvedValueOnce(readinessResponse(200, 'ready'))
    const waiting = waitUntilReady({ fetchImpl })

    await vi.advanceTimersByTimeAsync(
      READINESS_REQUEST_TIMEOUT_MS + READINESS_POLL_INTERVAL_MS,
    )
    await waiting

    expect(fetchImpl).toHaveBeenCalledTimes(2)
    vi.useRealTimers()
  })

  it('does not request readiness when the signal is already aborted', async () => {
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const fetchImpl = vi.fn<typeof fetch>()
    controller.abort()

    await expect(
      waitUntilReady({ fetchImpl, signal: controller.signal }),
    ).rejects.toMatchObject({ kind: 'aborted' })
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('stops polling when cancelled between readiness checks', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )
    const waiting = waitUntilReady({ fetchImpl, signal: controller.signal })
    const abortedExpectation = expect(waiting).rejects.toMatchObject({ kind: 'aborted' })

    await vi.advanceTimersByTimeAsync(0)
    expect(fetchImpl).toHaveBeenCalledTimes(1)

    controller.abort()
    await vi.advanceTimersByTimeAsync(READINESS_POLL_INTERVAL_MS)

    await abortedExpectation
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it('handles cancellation while registering the sleep listener', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', API_URL)
    const controller = new AbortController()
    const addEventListener = controller.signal.addEventListener.bind(controller.signal)
    let abortListenerRegistrations = 0
    const addEventListenerSpy = vi.spyOn(controller.signal, 'addEventListener')
      .mockImplementation((type, listener, options) => {
        addEventListener(type, listener, options)
        if (type === 'abort' && ++abortListenerRegistrations === 2) {
          controller.abort()
        }
      })
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation(async () =>
      readinessResponse(503, 'pending'),
    )
    const waiting = waitUntilReady({ fetchImpl, signal: controller.signal })
    const abortedExpectation = expect(waiting).rejects.toMatchObject({ kind: 'aborted' })

    await vi.advanceTimersByTimeAsync(0)
    await abortedExpectation
    await vi.runAllTimersAsync()

    expect(fetchImpl).toHaveBeenCalledTimes(1)
    addEventListenerSpy.mockRestore()
    vi.useRealTimers()
  })
})

function readinessResponse(statusCode: number, status: string): Response {
  return new Response(JSON.stringify({ status }), {
    status: statusCode,
    headers: { 'Content-Type': 'application/json' },
  })
}
