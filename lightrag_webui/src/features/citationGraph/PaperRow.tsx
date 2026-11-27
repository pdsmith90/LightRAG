import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import type { StructPaper } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import { paperByline, paperTitle } from '@/lib/citationGraph'

// One paper of a list: its title (a click makes it the tab's paper), who and where, how often
// the library cites it and, as `children`, what the list says about it.
const PaperRow = ({
  paper,
  onOpen,
  children
}: {
  paper: StructPaper
  onOpen: (id: string) => void
  children?: ReactNode
}) => {
  const { t } = useTranslation()
  return (
    <li className="border-border/60 flex flex-col gap-1 border-b px-3 py-2">
      <button
        type="button"
        className="text-left text-sm font-medium hover:underline"
        title={t('citationGraph.openPaper')}
        onClick={() => onOpen(paper.paper_id)}
      >
        {paperTitle(paper)}
      </button>
      <div className="text-muted-foreground flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <span>{paperByline(paper)}</span>
        {paper.cited_by_library > 0 && (
          <span>{t('citationGraph.citedByLibrary', { count: paper.cited_by_library })}</span>
        )}
        {!paper.in_kb && (
          <Badge variant="outline" className="px-1.5 py-0 font-normal">
            {t('citationGraph.notInKb')}
          </Badge>
        )}
      </div>
      {paper.tldr && (
        <p className="text-muted-foreground line-clamp-2 text-xs" title={paper.tldr}>
          {paper.tldr}
        </p>
      )}
      {children}
    </li>
  )
}

export default PaperRow
