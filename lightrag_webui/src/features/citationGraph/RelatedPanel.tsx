import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { getRelatedPapers, type PaperRelation, type RelatedPaper } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { RELATION_FILTERS, type CitationPair } from '@/lib/citationGraph'
import PanelBody from './PanelBody'
import PaperRow from './PaperRow'
import { useFetched } from './useFetched'

const LIMIT = 30

// The papers one paper cites or is cited by, shares references or authors with, one relation
// at a time. A reference or a citing paper whose sentences are stored can open them.
const RelatedPanel = ({
  paper,
  onOpen,
  onPickCitation
}: {
  paper: string
  onOpen: (id: string) => void
  onPickCitation: (pair: CitationPair) => void
}) => {
  const { t } = useTranslation()
  const [relation, setRelation] = useState<PaperRelation>('cites')
  const load = useCallback(() => getRelatedPapers(paper, [relation], LIMIT), [paper, relation])
  const { data, error, loading } = useFetched(load)

  const detail = (item: RelatedPaper) => {
    const facts = item.citation
    const pair =
      relation === 'cites'
        ? { citing: paper, cited: item.paper_id }
        : relation === 'cited_by'
          ? { citing: item.paper_id, cited: paper }
          : null
    return (
      <div className="flex flex-wrap items-center gap-1 text-xs">
        {relation === 'coupled' && item.shared_refs != null && (
          <Badge variant="secondary" className="font-normal">
            {t('citationGraph.related.sharedRefs', { count: item.shared_refs })}
          </Badge>
        )}
        {relation === 'coauthor' && item.shared_authors != null && (
          <Badge variant="secondary" className="font-normal">
            {t('citationGraph.related.sharedAuthors', { count: item.shared_authors })}
          </Badge>
        )}
        {facts?.is_influential && (
          <Badge variant="secondary" className="font-normal">
            {t('citationGraph.influential')}
          </Badge>
        )}
        {facts?.intents.map((intent) => (
          <Badge key={intent} variant="outline" className="font-normal">
            {intent}
          </Badge>
        ))}
        {pair && !!facts?.n_contexts && (
          <Button variant="ghost" className="h-6 px-2 text-xs" onClick={() => onPickCitation(pair)}>
            {t('citationGraph.related.readSentences', { count: facts.n_contexts })}
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex flex-wrap gap-1 p-2" role="group" aria-label={t('citationGraph.related.filter')}>
        {RELATION_FILTERS.map((name) => (
          <Button
            key={name}
            size="sm"
            className="h-7 px-2 text-xs"
            variant={relation === name ? 'secondary' : 'ghost'}
            aria-pressed={relation === name}
            onClick={() => setRelation(name)}
          >
            {t(`citationGraph.relation.${name}`)}
          </Button>
        ))}
      </div>
      <PanelBody loading={loading} error={error} reason={data?.reason}>
        {data && (
          <>
            <p className="text-muted-foreground px-3 pb-1 text-xs">
              {t('citationGraph.related.count', {
                shown: data.related?.length ?? 0,
                total: data.totals?.[relation] ?? data.related?.length ?? 0
              })}
            </p>
            {data.related?.length ? (
              <ul>
                {data.related.map((item) => (
                  <PaperRow key={item.paper_id} paper={item} onOpen={onOpen}>
                    {detail(item)}
                  </PaperRow>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground p-3 text-sm">{t('citationGraph.related.empty')}</p>
            )}
          </>
        )}
      </PanelBody>
    </div>
  )
}

export default RelatedPanel
