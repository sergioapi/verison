import type { AnalyzeResponse } from './types'

export function isAnalyzeResponse(value: unknown): value is AnalyzeResponse {
  if (!isRecord(value)) {
    return false
  }

  return (
    isFiniteNumber(value.predicted_label) &&
    typeof value.predicted_class === 'string' &&
    isFiniteNumber(value.ai_score) &&
    isFiniteNumber(value.decision_threshold) &&
    isFiniteNumber(value.audio_duration_seconds) &&
    isFiniteNumber(value.n_fragments) &&
    Array.isArray(value.fragments) &&
    value.fragments.every(isFragmentLike) &&
    hasFiniteNumber(value.timings, 'total_seconds') &&
    isModelMetadataLike(value.model) &&
    typeof value.usage_warning === 'string'
  )
}

function isFragmentLike(value: unknown): boolean {
  return (
    isRecord(value) &&
    isFiniteNumber(value.ai_score) &&
    isFiniteNumber(value.start_seconds) &&
    isFiniteNumber(value.end_seconds) &&
    typeof value.predicted_class === 'string'
  )
}

function isModelMetadataLike(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.model_id === 'string' &&
    typeof value.score_type === 'string' &&
    typeof value.score_is_calibrated_probability === 'boolean' &&
    typeof value.aggregation_strategy === 'string'
  )
}

function hasFiniteNumber(
  value: unknown,
  propertyName: string,
): value is Record<string, number> {
  return isRecord(value) && isFiniteNumber(value[propertyName])
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
