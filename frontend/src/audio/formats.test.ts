import { describe, expect, it } from 'vitest'
import { isSupportedAudioFile } from './formats'

function fileWithName(name: string): File {
  return new File(['audio'], name, { type: 'audio/wav' })
}

describe('isSupportedAudioFile', () => {
  it('accepts WAV and MP3 files', () => {
    expect(isSupportedAudioFile(fileWithName('song.wav'))).toBe(true)
    expect(isSupportedAudioFile(fileWithName('song.mp3'))).toBe(true)
  })

  it('accepts supported extensions case-insensitively', () => {
    expect(isSupportedAudioFile(fileWithName('song.WAV'))).toBe(true)
    expect(isSupportedAudioFile(fileWithName('song.Mp3'))).toBe(true)
  })

  it('rejects unsupported or missing extensions', () => {
    expect(isSupportedAudioFile(fileWithName('song.flac'))).toBe(false)
    expect(isSupportedAudioFile(fileWithName('song'))).toBe(false)
  })
})
