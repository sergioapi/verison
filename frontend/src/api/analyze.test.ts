import { describe, expect, it, vi } from 'vitest'
import { analyzeAudio } from './analyze'
import { buildAnalyzeResponse } from '../test/fixtures'

describe('analyzeAudio', () => {
  it('returns a valid analysis response', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const expected = buildAnalyzeResponse()
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(200, expected))

    const result = await analyzeAudio(
      new File(['audio'], 'song.wav', { type: 'audio/wav' }),
      { fetchImpl },
    )

    expect(result.predicted_class).toBe(expected.predicted_class)
    expect(result.fragments).toHaveLength(expected.fragments.length)
    expect(result.timings.total_seconds).toBe(expected.timings.total_seconds)
    expect(result.model.model_id).toBe(expected.model.model_id)
  })

  it('rejects a malformed successful response', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(200, { predicted_class: 'ai_generated' }),
    )

    await expect(
      analyzeAudio(new File(['audio'], 'song.wav', { type: 'audio/wav' }), { fetchImpl }),
    ).rejects.toMatchObject({ kind: 'unexpected-response', status: 200 })
  })

  it('preserves structured backend errors', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(422, {
        detail: { code: 'invalid_audio', message: 'Audio file could not be processed' },
      }),
    )

    await expect(
      analyzeAudio(new File(['audio'], 'song.wav', { type: 'audio/wav' }), { fetchImpl }),
    ).rejects.toMatchObject({ kind: 'http', status: 422, code: 'invalid_audio' })
  })

  it('aborts a timed-out POST without retrying it', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>((_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
      }),
    )

    const request = analyzeAudio(
      new File(['audio'], 'song.wav', { type: 'audio/wav' }),
      { fetchImpl, timeoutMs: 10 },
    )
    const timeoutExpectation = expect(request).rejects.toMatchObject({ kind: 'timeout' })
    await vi.advanceTimersByTimeAsync(10)

    await timeoutExpectation
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it('does not retry a failed POST', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test')
    const fetchImpl = vi.fn<typeof fetch>().mockRejectedValue(new TypeError('network'))

    await expect(
      analyzeAudio(new File(['audio'], 'song.wav', { type: 'audio/wav' }), { fetchImpl }),
    ).rejects.toMatchObject({ kind: 'network' })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })
})

function jsonResponse(status: number, payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
