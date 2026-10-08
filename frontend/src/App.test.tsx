import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { analyzeAudio, waitUntilReady } from './api'
import { buildAnalyzeResponse } from './test/fixtures'

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')

  return {
    ...actual,
    analyzeAudio: vi.fn(),
    waitUntilReady: vi.fn(),
  }
})

const mockedAnalyzeAudio = vi.mocked(analyzeAudio)
const mockedWaitUntilReady = vi.mocked(waitUntilReady)
const scrollIntoViewMock = vi.fn()

describe('App', () => {
  beforeEach(() => {
    mockedAnalyzeAudio.mockReset()
    mockedWaitUntilReady.mockReset()
    scrollIntoViewMock.mockReset()
    Object.defineProperty(window.HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollIntoViewMock,
    })
  })

  it('runs the main analysis flow for a valid audio file', async () => {
    mockedWaitUntilReady.mockResolvedValue()
    mockedAnalyzeAudio.mockResolvedValue(buildAnalyzeResponse())
    render(<App />)

    const file = new File(['audio'], 'cumbia_pcf.wav', { type: 'audio/wav' })
    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: { files: [file] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    await waitFor(() => expect(mockedAnalyzeAudio).toHaveBeenCalledTimes(1))
    expect(mockedAnalyzeAudio).toHaveBeenCalledWith(file, expect.any(Object))
    expect(await screen.findByText('Posible generación con IA')).toBeInTheDocument()
    await waitFor(() => expect(scrollIntoViewMock).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Analizar otra canción' }))

    expect(screen.getByLabelText('Archivo de audio')).toBeInTheDocument()
    expect(screen.queryByText('Posible generación con IA')).not.toBeInTheDocument()
  })

  it('shows an error when analysis fails', async () => {
    mockedWaitUntilReady.mockResolvedValue()
    mockedAnalyzeAudio.mockRejectedValue({
      kind: 'network',
      message: 'Could not contact the analysis service.',
    })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: {
        files: [new File(['audio'], 'song.mp3', { type: 'audio/mpeg' })],
      },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
  })

  it('keeps the form processing until readiness resolves, then posts once', async () => {
    let resolveReadiness: (() => void) | undefined
    mockedWaitUntilReady.mockImplementation(
      () => new Promise<void>((resolve) => {
        resolveReadiness = resolve
      }),
    )
    mockedAnalyzeAudio.mockResolvedValue(buildAnalyzeResponse())
    render(<App />)

    const file = new File(['audio'], 'song.wav', { type: 'audio/wav' })
    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: { files: [file] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByRole('button')).toBeDisabled()
    expect(mockedAnalyzeAudio).not.toHaveBeenCalled()

    resolveReadiness?.()

    await waitFor(() => expect(mockedAnalyzeAudio).toHaveBeenCalledTimes(1))
  })

  it('allows another attempt after readiness fails without posting audio', async () => {
    mockedWaitUntilReady.mockRejectedValueOnce({
      kind: 'service-unavailable',
      message: 'The analysis service is not ready.',
    })
    mockedWaitUntilReady.mockResolvedValueOnce()
    mockedAnalyzeAudio.mockResolvedValue(buildAnalyzeResponse())
    render(<App />)

    const file = new File(['audio'], 'song.wav', { type: 'audio/wav' })
    fireEvent.change(screen.getByLabelText('Archivo de audio'), {
      target: { files: [file] },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(mockedAnalyzeAudio).not.toHaveBeenCalled()
    const fileInput = screen.getByLabelText('Archivo de audio') as HTMLInputElement
    expect(fileInput.files?.[0]).toBe(file)
    expect(screen.getByRole('button')).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: 'Analizar audio' }))

    await waitFor(() => expect(mockedAnalyzeAudio).toHaveBeenCalledTimes(1))
  })
})
