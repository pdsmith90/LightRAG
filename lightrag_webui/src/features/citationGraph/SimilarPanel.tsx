import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'

import { getSimilarPapers } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import PanelBody from './PanelBody'
import PaperRow from './PaperRow'
import { useFetched } from './useFetched'

const LIMIT = 20

// The papers with the nearest SPECTER2 embeddings, cited or not, with how the library already
// links each to the paper (none: similar work the paper does not cite).
const SimilarPanel = ({ paper, onOpen }: { paper: string; onOpen: (id: string) => void }) => {
  const { t } = useTranslation()
  const load = useCallback(() => getSimilarPapers(paper, LIMIT), [paper])
  const { data, error, loading } = useFetched(load)

  return (
    <PanelBody loading={loading} error={error} reason={data?.reason}>
      {data?.similar?.length ? (
        <ul>
          {data.similar.map((item) => (
            <PaperRow key={item.paper_id} paper={item} onOpen={onOpen}>
              <div className="flex flex-wrap items-center gap-1 text-xs">
                <Badge variant="secondary" className="font-normal">
                  {t('citationGraph.similar.similarity', { value: Math.round(item.similarity * 100) })}
                </Badge>
                {item.links.length ? (
                  item.links.map((link) => (
                    <Badge key={link} variant="outline" className="font-normal">
                      {t(`citationGraph.relation.${link}`)}
                    </Badge>
                  ))
                ) : (
                  <span className="text-muted-foreground">{t('citationGraph.similar.unlinked')}</span>
                )}
              </div>
            </PaperRow>
          ))}
        </ul>
      ) : (
        <p className="text-muted-foreground p-3 text-sm">{t('citationGraph.similar.empty')}</p>
      )}
    </PanelBody>
  )
}

export default SimilarPanel
