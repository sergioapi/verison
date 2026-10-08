export type AnalyzeModelResponse = {
  model_id: string
  sha256: string
  classes: number[]
  positive_label: number
  score_type: string
  score_is_calibrated_probability: boolean
  target_sample_rate: number
  fragment_duration_seconds: number
  n_mfcc: number
  n_features: number
  aggregation_strategy: string
}

export type FragmentPredictionResponse = {
  index: number
  start_seconds: number
  end_seconds: number
  duration_seconds: number
  ai_score: number
  predicted_label: number
  predicted_class: string
  was_padded: boolean
}

export type InferenceTimingsResponse = {
  decode_seconds: number
  segmentation_seconds: number
  preprocessing_seconds: number
  mfcc_seconds: number
  prediction_seconds: number
  aggregation_seconds: number
  total_seconds: number
}

export type AnalyzeResponse = {
  predicted_label: number
  predicted_class: string
  ai_score: number
  decision_threshold: number
  audio_duration_seconds: number
  original_sample_rate: number
  n_fragments: number
  fragments: FragmentPredictionResponse[]
  timings: InferenceTimingsResponse
  model: AnalyzeModelResponse
  usage_warning: string
}
