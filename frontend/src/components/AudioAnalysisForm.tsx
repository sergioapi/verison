import {
  AUDIO_FILE_ACCEPT,
  getSupportedAudioFormatsLabel,
  isSupportedAudioFile,
} from '../audio/formats'

type Feedback = {
  kind: 'status' | 'error'
  message: string
}

type AudioAnalysisFormProps = {
  selectedFile: File | null
  isAnalyzing: boolean
  isPreparing: boolean
  feedback: Feedback | null
  onFileAccepted: (file: File) => void
  onFileRejected: () => void
  onFileCleared: () => void
  onAnalyze: () => void
}

export function AudioAnalysisForm({
  selectedFile,
  isAnalyzing,
  isPreparing,
  feedback,
  onFileAccepted,
  onFileRejected,
  onFileCleared,
  onAnalyze,
}: AudioAnalysisFormProps) {
  const canAnalyze = selectedFile !== null && !isAnalyzing
  const feedbackId = feedback ? 'analysis-feedback' : undefined

  return (
    <form
      className="analysis-form"
      aria-busy={isAnalyzing}
      onSubmit={(event) => {
        event.preventDefault()
        if (canAnalyze) {
          onAnalyze()
        }
      }}
    >
      <div className="field-group">
        <label htmlFor="audio-file">Archivo de audio</label>
        <input
          id="audio-file"
          name="audio-file"
          type="file"
          accept={AUDIO_FILE_ACCEPT}
          disabled={isAnalyzing}
          aria-describedby="audio-file-help"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0] ?? null

            if (!file) {
              onFileCleared()
              return
            }

            if (!isSupportedAudioFile(file)) {
              event.currentTarget.value = ''
              onFileRejected()
              return
            }

            onFileAccepted(file)
          }}
        />
        <p id="audio-file-help" className="help-text">
          Formatos admitidos: {getSupportedAudioFormatsLabel()}.
        </p>
      </div>

      <button
        type="submit"
        className="primary-action"
        disabled={!canAnalyze}
        aria-describedby={feedbackId}
      >
        {isPreparing ? 'Preparando el análisis...' : isAnalyzing ? 'Analizando...' : 'Analizar audio'}
      </button>

      {feedback?.kind === 'status' ? (
        <>
          <div className="analysis-progress" aria-hidden="true">
            <span className="analysis-progress-indicator" />
          </div>
          <p
            id="analysis-feedback"
            className="visually-hidden"
            role="status"
            aria-live="polite"
          >
            {feedback.message}
          </p>
        </>
      ) : null}

      {feedback?.kind === 'error' ? (
        <p
          id="analysis-feedback"
          className="feedback feedback-error"
          role="alert"
          aria-live="assertive"
        >
          {feedback.message}
        </p>
      ) : null}
    </form>
  )
}
