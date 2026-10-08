export type ApiErrorKind =
  | 'configuration'
  | 'network'
  | 'http'
  | 'validation'
  | 'unexpected-response'
  | 'timeout'
  | 'aborted'
  | 'service-unavailable'

export type ApiError = {
  kind: ApiErrorKind
  message: string
  status?: number
  code?: string
}

export type FastApiCustomErrorDetail = {
  code: string
  message: string
}

export function createApiError(error: ApiError): ApiError {
  return error
}

export function isApiError(value: unknown): value is ApiError {
  return (
    isRecord(value) &&
    typeof value.kind === 'string' &&
    typeof value.message === 'string'
  )
}

export function isFastApiCustomErrorDetail(
  value: unknown,
): value is FastApiCustomErrorDetail {
  return (
    isRecord(value) &&
    typeof value.code === 'string' &&
    typeof value.message === 'string'
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
