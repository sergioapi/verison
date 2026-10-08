import type { AnalyzeResponse } from '../api'
import {
  getVisibleClassDescription,
  getVisibleClassLabel,
} from '../results/classLabels'

type AnalysisResultProps = {
  result: AnalyzeResponse
  fileName: string
  onReset: () => void
}

export function AnalysisResult({
  result,
  fileName,
  onReset,
}: AnalysisResultProps) {
  const labelClassName = `result-label ${resultLabelModifier(result.predicted_class)}`

  return (
    <section className="result-card" aria-labelledby="result-title" role="status">
      <div className="result-header">
        <h2 id="result-title" className="visually-hidden">
          Resultado
        </h2>
        <button type="button" className="secondary-action" onClick={onReset}>
          Analizar otra canción
        </button>
      </div>
      <div className="result-main">
        <p className="result-file">
          <svg
            className="result-file-icon"
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
          <span className="result-file-name">{fileName}</span>
        </p>
        <p className={labelClassName}>
          {getVisibleClassLabel(result.predicted_class)}
        </p>
        <p className="result-description">
          {getVisibleClassDescription(result.predicted_class)}
        </p>
      </div>
    </section>
  )
}

function resultLabelModifier(predictedClass: string): string {
  if (predictedClass === 'ai_generated') {
    return 'result-label-ai'
  }
  if (predictedClass === 'human') {
    return 'result-label-human'
  }
  return 'result-label-unknown'
}
