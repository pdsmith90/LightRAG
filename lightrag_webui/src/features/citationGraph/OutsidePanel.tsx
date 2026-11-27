import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'

import { getOutsideWorks } from '@/api/lightrag'
import { fannedMissing, fannedReason, fanOut, mergeOutside, paperTitle } from '@/lib/citationGraph'
import PanelBody from './PanelBody'
import { useFetched } from './useFetched'

const LIMIT = 30

// The works one paper cites that the library does not hold and at least two library papers
// cite, most cited by the library first, with some of the library papers that cite them. Several
// papers: the works any of them cites.
const OutsidePanel = ({ papers, onOpen }: { papers: string[]; onOpen: (id: string) => void }) => {
  const { t } = useTranslation()
  const load = useCallback(() => fanOut(papers, (paper) => getOutsideWorks(paper, LIMIT)), [papers])
  const { data, error, loading } = useFetched(load)
  const works = data ? mergeOutside(data) : []
  const total = data?.answers.reduce((sum, { answer }) => sum + (answer.total ?? answer.works?.length ?? 0), 0) ?? 0

  return (
    <PanelBody
      loading={loading}
      error={error}
      reason={fannedReason(data)}
      missing={data ? { count: fannedMissing(data), total: data.total } : undefined}
    >
      {works.length ? (
        <>
          <p className="text-muted-foreground px-3 py-2 text-xs">
            {t('citationGraph.outside.count', { shown: works.length, total: Math.max(total, works.length) })}
          </p>
          <ul>
            {works.map((work) => (
              <li key={work.work_id} className="border-border/60 flex flex-col gap-1 border-b px-3 py-2">
                {work.doi ? (
                  <a
                    className="text-sm font-medium hover:underline"
                    href={`https://doi.org/${work.doi}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {work.title || work.doi}
                  </a>
                ) : (
                  <span className="text-sm font-medium">{work.title || work.work_id}</span>
                )}
                <div className="text-muted-foreground flex flex-wrap gap-x-2 text-xs">
                  <span>{[work.first_author, work.year, work.venue].filter(Boolean).join(' · ')}</span>
                  <span>{t('citationGraph.outside.libraryCiters', { count: work.n_library_citers })}</span>
                </div>
                {work.also_cited_by.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1 text-xs">
                    <span className="text-muted-foreground">{t('citationGraph.outside.alsoCitedBy')}</span>
                    {work.also_cited_by.map((citer) => (
                      <button
                        key={citer.paper_id}
                        type="button"
                        className="text-primary max-w-[200px] truncate hover:underline"
                        title={paperTitle(citer)}
                        onClick={() => onOpen(citer.paper_id)}
                      >
                        {citer.label}
                      </button>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="text-muted-foreground p-3 text-sm">{t('citationGraph.outside.empty')}</p>
      )}
    </PanelBody>
  )
}

export default OutsidePanel
