import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { searchPapers, type PaperSummary } from '@/api/lightrag'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { PAPER_PANELS, type ActivePaper, type CitationMode, type CitationPair, type PaperPanel } from '@/lib/citationGraph'
import { useBackendState } from '@/stores/state'
import ContextsPanel from '@/features/citationGraph/ContextsPanel'
import OutsidePanel from '@/features/citationGraph/OutsidePanel'
import PaperMap from '@/features/citationGraph/PaperMap'
import RelatedPanel from '@/features/citationGraph/RelatedPanel'
import Segmented from '@/features/citationGraph/Segmented'
import SimilarPanel from '@/features/citationGraph/SimilarPanel'
import TopicsPanel from '@/features/citationGraph/TopicsPanel'

/**
 * Fork: the citation structure of the library as a tab of its own. The Paper view searches a
 * paper and shows its citation and coupling map with the papers it is related to, similar to,
 * the sentences of a citation and the works it cites outside the library; the Topics view
 * browses the topics and opens one of their papers in the Paper view. Needs the server's
 * paper structure (`zotero_struct` in /health).
 */
export default function CitationGraphView() {
  const { t } = useTranslation()
  const status = useBackendState.use.status()
  const [mode, setMode] = useState<CitationMode>('paper')
  const [paper, setPaper] = useState<ActivePaper | null>(null)
  const [panel, setPanel] = useState<PaperPanel>('related')
  const [pair, setPair] = useState<CitationPair | null>(null)

  const openPaper = useCallback((id: string) => {
    setPaper({ id })
    setMode('paper')
    setPair(null)
  }, [])

  const pickCitation = useCallback((next: CitationPair) => {
    setPair(next)
    setPanel('citation')
  }, [])

  const swapCitation = useCallback(
    () => setPair((current) => current && { citing: current.cited, cited: current.citing }),
    []
  )

  const fetchPapers = useCallback(async (query?: string): Promise<PaperSummary[]> => {
    if (!query || !query.trim()) return []
    return searchPapers(query.trim())
  }, [])

  if (!status) {
    return <p className="text-muted-foreground p-4 text-sm">{t('citationGraph.loading')}</p>
  }
  if (status.configuration?.zotero_struct !== true) {
    return <p className="text-muted-foreground p-4 text-sm">{t('citationGraph.unavailable')}</p>
  }

  return (
    <div className="flex h-full flex-col overflow-auto">
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <Segmented<CitationMode>
          value={mode}
          label={t('citationGraph.modeLabel')}
          options={[
            { value: 'paper', label: t('citationGraph.modePaper') },
            { value: 'topics', label: t('citationGraph.modeTopics') }
          ]}
          onChange={setMode}
        />
        {mode === 'paper' && (
          <div className="w-full min-w-[200px] max-w-[420px]">
            <AsyncSelect<PaperSummary>
              className="min-w-[420px]"
              triggerClassName="max-h-8 w-full overflow-hidden"
              searchInputClassName="max-h-8"
              fetcher={fetchPapers}
              renderOption={(item) => (
                <div className="min-w-0 truncate" title={`${item.citation}\n${item.doi || item.paper}`}>
                  <div className="truncate">{item.citation || item.paper}</div>
                  <div className="text-muted-foreground truncate text-xs">{item.doi || item.paper}</div>
                  {item.tldr && (
                    <div className="text-muted-foreground line-clamp-2 w-0 min-w-full text-xs whitespace-normal">{item.tldr}</div>
                  )}
                </div>
              )}
              getOptionValue={(item) => item.paper}
              getDisplayValue={(item) => (
                <div className="min-w-0 flex-1 truncate text-left" title={item.citation}>
                  {item.citation || item.paper}
                </div>
              )}
              ariaLabel={t('citationGraph.search.label')}
              placeholder={t('citationGraph.search.placeholder')}
              searchPlaceholder={t('citationGraph.search.searchPlaceholder')}
              noResultsMessage={t('citationGraph.search.noPapers')}
              value=""
              onChange={(id) => id && openPaper(id)}
              clearable={false}
              debounceTime={400}
            />
          </div>
        )}
      </div>
      {mode === 'topics' ? (
        <div className="mx-auto min-h-0 w-full max-w-4xl flex-1 overflow-auto">
          <TopicsPanel onOpenPaper={openPaper} />
        </div>
      ) : paper ? (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <PaperMap
              key={paper.id}
              paper={paper.id}
              onOpenPaper={openPaper}
              onPickCitation={pickCitation}
            />
          </div>
          <aside className="flex min-h-[320px] w-full flex-col border-t lg:w-[420px] lg:border-t-0 lg:border-l">
            <div className="border-b p-2">
              <Segmented<PaperPanel>
                value={panel}
                label={t('citationGraph.panelLabel')}
                options={PAPER_PANELS.map((name) => ({ value: name, label: t(`citationGraph.panels.${name}`) }))}
                onChange={setPanel}
              />
            </div>
            <div className="min-h-0 flex-1 overflow-auto">
              {panel === 'related' && (
                <RelatedPanel
                  paper={paper.id}
                  onOpen={openPaper}
                  onPickCitation={pickCitation}
                />
              )}
              {panel === 'similar' && <SimilarPanel paper={paper.id} onOpen={openPaper} />}
              {panel === 'citation' && <ContextsPanel pair={pair} onSwap={swapCitation} />}
              {panel === 'outside' && <OutsidePanel paper={paper.id} onOpen={openPaper} />}
            </div>
          </aside>
        </div>
      ) : (
        <p className="text-muted-foreground p-4 text-sm">{t('citationGraph.emptyPaper')}</p>
      )}
    </div>
  )
}
