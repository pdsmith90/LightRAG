import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowLeftRight } from 'lucide-react'

import { getPaperCitation } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { paperTitle, type CitationPair } from '@/lib/citationGraph'
import PanelBody from './PanelBody'
import { useFetched } from './useFetched'

// The sentences in which one paper cites another, with the citation's intents and whether it
// is influential. The pair comes from a link on the map or a reference in the related panel.
const ContextsPanel = ({
  pair,
  onSwap
}: {
  pair: CitationPair | null
  onSwap: () => void
}) => {
  const { t } = useTranslation()
  const citing = pair?.citing
  const cited = pair?.cited
  const load = useMemo(
    () => (citing && cited ? () => getPaperCitation(citing, cited) : null),
    [citing, cited]
  )
  const { data, error, loading } = useFetched(load)

  if (!pair) {
    return <p className="text-muted-foreground p-3 text-sm">{t('citationGraph.citation.pick')}</p>
  }

  const edge = data?.edge
  // These two are shown inside the panel: the paper titles, the facts or the swap button go with them.
  const inline = data?.reason === 'no_contexts_table' || data?.reason === 'no_citation'
  return (
    <PanelBody loading={loading} error={error} reason={inline ? undefined : data?.reason}>
      {data && (
        <div className="flex flex-col gap-2 p-3">
          {edge && data.citing && data.cited && (
            <p className="text-sm">
              {t('graphPanel.paperFilter.linkDirectionCites', {
                source: paperTitle(data.citing),
                target: paperTitle(data.cited)
              })}
            </p>
          )}
          {data.reason === 'no_citation' && (
            <p className="text-muted-foreground text-sm">{t('citationGraph.reason.no_citation')}</p>
          )}
          {data.reason === 'no_citation' && data.cited_cites_citing && (
            <Button variant="outline" size="sm" className="w-fit" onClick={onSwap}>
              <ArrowLeftRight />
              {t('citationGraph.citation.swap')}
            </Button>
          )}
          {data.reason === 'no_contexts_table' && (
            <p className="text-muted-foreground text-xs">{t('citationGraph.reason.no_contexts_table')}</p>
          )}
          {edge && (
            <div className="flex flex-wrap items-center gap-1 text-xs">
              {edge.is_influential && (
                <Badge variant="secondary" className="font-normal">
                  {t('citationGraph.influential')}
                </Badge>
              )}
              {edge.intents.map((intent) => (
                <Badge key={intent} variant="outline" className="font-normal">
                  {intent}
                </Badge>
              ))}
              {edge.source && (
                <span className="text-muted-foreground">{t('citationGraph.citation.source', { source: edge.source })}</span>
              )}
            </div>
          )}
          {edge && !data.contexts?.length && data.reason !== 'no_contexts_table' && (
            <p className="text-muted-foreground text-sm">{t('citationGraph.citation.noSentences')}</p>
          )}
          {data.contexts?.length ? (
            <ol className="flex flex-col gap-2">
              {data.contexts.map((context) => (
                <li key={context.ordinal} className="border-border/60 rounded-md border p-2 text-sm">
                  <p>{context.text}</p>
                  {context.intents.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {context.intents.map((intent) => (
                        <Badge key={intent} variant="outline" className="font-normal">
                          {intent}
                        </Badge>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          ) : null}
        </div>
      )}
    </PanelBody>
  )
}

export default ContextsPanel
