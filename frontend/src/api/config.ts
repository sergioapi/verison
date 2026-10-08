const API_BASE_URL_ENV_VAR = 'VITE_API_BASE_URL'

type ApiEnvironment = {
  readonly VITE_API_BASE_URL?: string
}

export function getApiBaseUrl(
  env: ApiEnvironment = import.meta.env as ApiEnvironment,
): string {
  const rawValue = env[API_BASE_URL_ENV_VAR]?.trim()

  if (!rawValue) {
    throw createApiConfigError(
      `${API_BASE_URL_ENV_VAR} must be configured before calling the API.`,
    )
  }

  let parsedUrl: URL
  try {
    parsedUrl = new URL(rawValue)
  } catch {
    throw createApiConfigError(
      `${API_BASE_URL_ENV_VAR} must be an absolute HTTP or HTTPS URL.`,
    )
  }

  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    throw createApiConfigError(
      `${API_BASE_URL_ENV_VAR} must use HTTP or HTTPS.`,
    )
  }

  return parsedUrl.href.replace(/\/+$/, '')
}

function createApiConfigError(message: string) {
  return {
    kind: 'configuration',
    message,
  } as const
}
