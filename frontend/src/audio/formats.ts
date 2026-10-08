export const SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3'] as const

export const AUDIO_FILE_ACCEPT = SUPPORTED_AUDIO_EXTENSIONS.join(',')

export function isSupportedAudioFile(file: File): boolean {
  const extension = getFileExtension(file.name)
  return SUPPORTED_AUDIO_EXTENSIONS.includes(
    extension as (typeof SUPPORTED_AUDIO_EXTENSIONS)[number],
  )
}

export function getSupportedAudioFormatsLabel(): string {
  return SUPPORTED_AUDIO_EXTENSIONS
    .map((extension) => extension.slice(1).toUpperCase())
    .join(' o ')
}

function getFileExtension(filename: string): string {
  const extensionStart = filename.lastIndexOf('.')

  if (extensionStart < 0) {
    return ''
  }

  return filename.slice(extensionStart).toLowerCase()
}
