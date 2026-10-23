import { useEffect, useState } from 'react'
import { getSourceLabels } from '@/api/lightrag'

const BATCH_SIZE = 500

// file_path -> "Authors · Title · Year · DOI", or null when the path has no Zotero entry
const cache = new Map<string, string | null>()
const inFlight = new Set<string>()
const listeners = new Set<() => void>()

/**
 * Resolve corpus file paths to their Zotero label. The returned function gives the label,
 * or undefined while loading or when the path has no entry (callers fall back to the filename).
 */
export const useSourceLabels = (filePaths: string[]): ((filePath: string) => string | undefined) => {
  const [, setVersion] = useState(0)

  useEffect(() => {
    const notify = () => setVersion((v) => v + 1)
    listeners.add(notify)
    return () => {
      listeners.delete(notify)
    }
  }, [])

  const wanted = Array.from(new Set(filePaths.filter(Boolean))).sort()
  const wantedKey = wanted.join('\n')

  useEffect(() => {
    const missing = wanted.filter((p) => !cache.has(p) && !inFlight.has(p))
    if (missing.length === 0) return
    missing.forEach((p) => inFlight.add(p))
    ;(async () => {
      for (let i = 0; i < missing.length; i += BATCH_SIZE) {
        const batch = missing.slice(i, i + BATCH_SIZE)
        try {
          const labels = await getSourceLabels(batch)
          labels.forEach((l) => cache.set(l.file_path, l.label))
        } catch {
          // display only: keep showing the filename, and do not retry in a loop
        }
        batch.forEach((p) => {
          if (!cache.has(p)) cache.set(p, null)
          inFlight.delete(p)
        })
        listeners.forEach((notify) => notify())
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantedKey])

  return (filePath: string) => cache.get(filePath) ?? undefined
}
