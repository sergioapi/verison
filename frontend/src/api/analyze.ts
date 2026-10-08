import { getApiBaseUrl } from './config'
import { ANALYZE_ENDPOINT_PATH, ANALYZE_FILE_FIELD } from './endpoints'
import {
  createApiError,
  isApiError,
  isFastApiCustomErrorDetail,
  type ApiError,
} from './errors'
import type { AnalyzeResponse } from './types'
import { isAnalyzeResponse } from './validators'
import { fetchWithTimeout } from './request'

export const ANALYSIS_TIMEOUT_MS = 180_000

type AnalyzeAudioOptions = {
  fetchImpl?: typeof fetch
  signal?: AbortSignal
  timeoutMs?: number
}

export async function analyzeAudio(
  file: File,
  options: AnalyzeAudioOptions = {},
): Promise<AnalyzeResponse> {
  const formData = new FormData()
  formData.append(ANALYZE_FILE_FIELD, file)

  const response = await postAnalyzeRequest(formData, options)
  const payload = await readJsonResponse(response)

  if (!response.ok) {
    throw httpErrorFromPayload(response.status, payload)
  }

  if (!isAnalyzeResponse(payload)) {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The analysis API returned an unexpected response.',
    })
  }

  return payload
}

async function postAnalyzeRequest(
  formData: FormData,
  options: AnalyzeAudioOptions,
): Promise<Response> {
  const analyzeUrl = `${getApiBaseUrl()}${ANALYZE_ENDPOINT_PATH}`

  try {
    return await fetchWithTimeout(
      analyzeUrl,
      { method: 'POST', body: formData },
      {
        signal: options.signal,
        timeoutMs: options.timeoutMs ?? ANALYSIS_TIMEOUT_MS,
      },
      options.fetchImpl ?? fetch,
    )
  } catch (error) {
    if (isApiError(error)) {
      throw error
    }
    throw createApiError({
      kind: 'network',
      message: 'Could not contact the analysis service.',
    })
  }
}

async function readJsonResponse(response: Response): Promise<unknown> {
  const rawBody = await response.text()

  if (!rawBody.trim()) {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The analysis API returned an empty response.',
    })
  }

  try {
    return JSON.parse(rawBody) as unknown
  } catch {
    throw createApiError({
      kind: 'unexpected-response',
      status: response.status,
      message: 'The analysis API returned a non-JSON response.',
    })
  }
}

function httpErrorFromPayload(status: number, payload: unknown): ApiError {
  if (isRecord(payload) && isFastApiCustomErrorDetail(payload.detail)) {
    return createApiError({
      kind: 'http',
      status,
      code: payload.detail.code,
      message: payload.detail.message,
    })
  }

  if (isRecord(payload) && 'detail' in payload) {
    return createApiError({
      kind: 'validation',
      status,
      message: 'The analysis API rejected the request.',
    })
  }

  return createApiError({
    kind: 'unexpected-response',
    status,
    message: 'The analysis API returned an unexpected error response.',
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
