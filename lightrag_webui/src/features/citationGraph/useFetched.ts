import { useEffect, useState } from 'react'

export type Fetched<T> = {
  data: T | null
  error: unknown
  loading: boolean
}

// What one request answered. `load` is the request: a new function asks again, so callers wrap
// it in useCallback over what it reads, and null asks for nothing. An answer to an earlier
// request is dropped, and one that is not for the current `load` counts as still loading.
export const useFetched = <T>(load: (() => Promise<T>) | null): Fetched<T> => {
  const [settled, setSettled] = useState<{
    load: () => Promise<T>
    data: T | null
    error: unknown
  } | null>(null)

  useEffect(() => {
    if (!load) return
    let current = true
    load().then(
      (data) => current && setSettled({ load, data, error: null }),
      (error) => current && setSettled({ load, data: null, error })
    )
    return () => {
      current = false
    }
  }, [load])

  const done = load !== null && settled?.load === load ? settled : null
  return {
    data: done ? done.data : null,
    error: done ? done.error : null,
    loading: load !== null && !done
  }
}
