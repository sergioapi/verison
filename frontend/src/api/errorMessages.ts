import { getSupportedAudioFormatsLabel } from '../audio/formats'
import type { ApiError } from './errors'

const ERROR_MESSAGES_BY_CODE: Record<string, string> = {
  unsupported_file_type: `El archivo debe estar en formato ${getSupportedAudioFormatsLabel()}.`,
  unsupported_media_type: 'El tipo de audio no está admitido.',
  empty_file: 'El archivo seleccionado está vacío.',
  file_too_large: 'El archivo supera el tamaño máximo permitido.',
  audio_too_long: 'El audio supera la duración máxima permitida.',
  invalid_audio: 'No se ha podido procesar el audio.',
  model_unavailable: 'El servicio no está disponible en este momento.',
  prediction_failed: 'No se ha podido completar el análisis.',
  internal_error: 'Ha ocurrido un error durante el análisis.',
}

export function getApiErrorMessage(error: ApiError): string {
  if (error.code && ERROR_MESSAGES_BY_CODE[error.code]) {
    return ERROR_MESSAGES_BY_CODE[error.code]
  }

  switch (error.kind) {
    case 'configuration':
      return 'La aplicación no está configurada para realizar el análisis.'
    case 'network':
      return 'No se ha podido contactar con el servicio de análisis.'
    case 'validation':
      return 'La solicitud no es válida para el servicio de análisis.'
    case 'http':
      return 'El servicio ha rechazado la solicitud.'
    case 'unexpected-response':
      return 'El servicio ha devuelto una respuesta inesperada.'
    case 'timeout':
    case 'service-unavailable':
      return 'El análisis no está disponible en este momento. Vuelve a intentarlo en breve.'
    case 'aborted':
      return 'La solicitud de análisis se ha cancelado.'
  }
}
