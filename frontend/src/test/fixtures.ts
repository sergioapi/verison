import type {
  AnalyzeModelResponse,
  AnalyzeResponse,
  InferenceTimingsResponse,
} from '../api'

type AnalyzeResponseOverrides = Omit<
  Partial<AnalyzeResponse>,
  'model' | 'timings'
> & {
  model?: Partial<AnalyzeModelResponse>
  timings?: Partial<InferenceTimingsResponse>
}

export function buildAnalyzeResponse(
  overrides: AnalyzeResponseOverrides = {},
): AnalyzeResponse {
  const { model, timings, ...responseOverrides } = overrides

  const response: AnalyzeResponse = {
    predicted_label: 1,
    predicted_class: 'ai_generated',
    ai_score: 0.428,
    decision_threshold: 0,
    audio_duration_seconds: 92.4,
    original_sample_rate: 44100,
    n_fragments: 10,
    fragments: [
      {
        index: 0,
        start_seconds: 0,
        end_seconds: 10,
        duration_seconds: 10,
        ai_score: 0.428,
        predicted_label: 1,
        predicted_class: 'ai_generated',
        was_padded: false,
      },
    ],
    timings: {
      decode_seconds: 0.01,
      segmentation_seconds: 0.01,
      preprocessing_seconds: 0.01,
      mfcc_seconds: 0.01,
      prediction_seconds: 0.01,
      aggregation_seconds: 0.01,
      total_seconds: 0.047,
      ...timings,
    },
    model: {
      model_id: 'mfcc-svm-baseline',
      sha256: 'abc123',
      classes: [0, 1],
      positive_label: 1,
      score_type: 'decision_function',
      score_is_calibrated_probability: false,
      target_sample_rate: 16000,
      fragment_duration_seconds: 10,
      n_mfcc: 20,
      n_features: 40,
      aggregation_strategy: 'duration_weighted_mean_decision_score',
      ...model,
    },
    usage_warning:
      'La salida es una estimación; el score no es una probabilidad calibrada.',
    ...responseOverrides,
  }

  return response
}
