import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { isPaperMapUnavailable } from '@/lib/paperFilter'

// What a panel shows instead of its list while the request runs, after it fails, or when the
// server answers with a `reason` (an unknown paper, no embeddings, ...).
const PanelBody = ({
  loading,
  error,
  reason,
  children
}: {
  loading: boolean
  error: unknown
  reason?: string
  children: ReactNode
}) => {
  const { t } = useTranslation()
  let message: string | null = null
  if (loading) {
    message = t('citationGraph.loading')
  } else if (error) {
    message = isPaperMapUnavailable(error)
      ? t('citationGraph.unavailable')
      : t('citationGraph.error', { message: error instanceof Error ? error.message : String(error) })
  } else if (reason) {
    message = t(`citationGraph.reason.${reason}`, { defaultValue: reason })
  }
  if (message) {
    return (
      <p className="text-muted-foreground p-3 text-sm" role={error ? 'alert' : 'status'}>
        {message}
      </p>
    )
  }
  return <>{children}</>
}

export default PanelBody
