type VisibleClassResult = {
  label: string
  description: string
}

const CLASS_RESULTS: Record<string, VisibleClassResult> = {
  ai_generated: {
    label: 'Posible generación con IA',
    description:
      'El modelo ha identificado patrones acústicos más compatibles con música generada mediante IA.',
  },
  human: {
    label: 'Posible origen humano',
    description:
      'El modelo ha identificado patrones acústicos más compatibles con música de origen humano.',
  },
}

const FALLBACK_RESULT: VisibleClassResult = {
  label: 'No se ha podido interpretar el resultado',
  description:
    'El servicio ha devuelto una clase que la interfaz no reconoce.',
}

export function getVisibleClassLabel(predictedClass: string): string {
  return getVisibleClassResult(predictedClass).label
}

export function getVisibleClassDescription(predictedClass: string): string {
  return getVisibleClassResult(predictedClass).description
}

function getVisibleClassResult(predictedClass: string): VisibleClassResult {
  return CLASS_RESULTS[predictedClass] ?? FALLBACK_RESULT
}
