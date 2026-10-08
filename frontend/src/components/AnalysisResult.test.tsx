import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { buildAnalyzeResponse } from '../test/fixtures'
import { AnalysisResult } from './AnalysisResult'

const noop = () => undefined

describe('AnalysisResult', () => {
  it('shows the AI-generated label and explanation for ai_generated results', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse()}
        fileName="cumbia_pcf.mp3"
        onReset={noop}
      />,
    )

    expect(screen.getByRole('status', { name: 'Resultado' })).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: 'Analizar otra canción' }),
    ).toBeInTheDocument()
    expect(screen.getByText('cumbia_pcf.mp3')).toBeInTheDocument()
    expect(screen.getByText('Posible generación con IA')).toBeInTheDocument()
    expect(
      screen.getByText(
        'El modelo ha identificado patrones acústicos más compatibles con música generada mediante IA.',
      ),
    ).toBeInTheDocument()
  })

  it('shows the human-origin label and explanation for human results', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({
          predicted_class: 'human',
          predicted_label: 0,
        })}
        fileName="voz_humana.wav"
        onReset={noop}
      />,
    )

    expect(screen.getByText('voz_humana.wav')).toBeInTheDocument()
    expect(screen.getByText('Posible origen humano')).toBeInTheDocument()
    expect(
      screen.getByText(
        'El modelo ha identificado patrones acústicos más compatibles con música de origen humano.',
      ),
    ).toBeInTheDocument()
  })

  it('shows a safe fallback for unknown result classes', () => {
    render(
      <AnalysisResult
        result={buildAnalyzeResponse({ predicted_class: 'unexpected' })}
        fileName="resultado_desconocido.wav"
        onReset={noop}
      />,
    )

    expect(
      screen.getByText('No se ha podido interpretar el resultado'),
    ).toBeInTheDocument()
  })
})
