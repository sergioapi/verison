import { describe, expect, it } from 'vitest'
import { getApiErrorMessage } from './errorMessages'

describe('getApiErrorMessage', () => {
  it('maps network errors to a user-facing message', () => {
    expect(
      getApiErrorMessage({
        kind: 'network',
        message: 'Could not contact the analysis service.',
      }),
    ).toBe('No se ha podido contactar con el servicio de análisis.')
  })

  it('maps unsupported file errors to the supported formats message', () => {
    expect(
      getApiErrorMessage({
        kind: 'http',
        status: 415,
        code: 'unsupported_file_type',
        message: 'Unsupported audio file type',
      }),
    ).toBe('El archivo debe estar en formato WAV o MP3.')
  })

  it('uses a safe fallback for unknown error responses', () => {
    expect(
      getApiErrorMessage({
        kind: 'unexpected-response',
        message: 'Unexpected response',
      }),
    ).toBe('El servicio ha devuelto una respuesta inesperada.')
  })
})
