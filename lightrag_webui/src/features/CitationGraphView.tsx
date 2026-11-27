import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Check, X } from 'lucide-react'

import { searchPapers, type PaperSummary } from '@/api/lightrag'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import {
  bareId,
  findChosenPaper,
  PAPER_PANELS,
  toggleChosenPaper,
  type CitationMode,
  type CitationPair,
  type PaperPanel
} from '@/lib/citationGraph'
import { useBackendState } from '@/stores/state'
import ContextsPanel from '@/features/citationGraph/ContextsPanel'
import OutsidePanel from '@/features/citationGraph/OutsidePanel'
import PaperMap from '@/features/citationGraph/PaperMap'
import RelatedPanel from '@/features/citationGraph/RelatedPanel'
import Segmented from '@/features/citationGraph/Segmented'
import SimilarPanel from '@/features/citationGraph/SimilarPanel'
import TopicsPanel from '@/features/citationGraph/TopicsPanel'

// Papers seen in search results, so a chip can show a citation instead of a DOI.
const knownPapers = new Map<string, PaperSummary>()
const lookups = new Set<string>()

// eslint-disable-next-line react-refresh/only-export-components
export const __resetCitationGraphForTests = (): void => {
  knownPapers.clear()
  lookups.clear()
}

/**
 * Fork: the citation structure of the library as a tab of its own. The Paper view searches a
 * papers (up to MAX_PAPERS, like the Knowledge Graph's paper filter) and shows their citation and
 * coupling map with the papers they are related to, similar to and the works they cite outside the
 * library, merged over the chosen papers, and the sentences of a citation; the Topics view browses
 * the topics and adds one of their papers in the Paper view. Needs the server's paper structure
 * (`zotero_struct` in /health).
 */
export default function CitationGraphView() {
  const { t } = useTranslation()
  const status = useBackendState.use.status()
  const [mode, setMode] = useState<CitationMode>('paper')
  const [papers, setPapers] = useState<string[]>([])
  // Bumped when a lookup fills knownPapers, to redraw the chips.
  const [, setKnownVersion] = useState(0)
  const [panel, setPanel] = useState<PaperPanel>('related')
  const [pair, setPair] = useState<CitationPair | null>(null)

  // Ids that came from a map or a list are structure ids, which have no citation yet.
  useEffect(() => {
    for (const id of papers) {
      if (knownPapers.has(id) || lookups.has(id)) continue
      lookups.add(id)
      searchPapers(bareId(id), 1)
        .then((found) => {
          const hit = found[0]
          if (hit && findChosenPaper([id], hit.paper, hit.keys)) knownPapers.set(id, hit)
          setKnownVersion((v) => v + 1)
        })
        .catch(() => lookups.delete(id))
    }
  }, [papers])

  // Adds the paper to the chosen ones, or takes it out when it is chosen already (under any of
  // its ids: a DOI or a key).
  const togglePaper = useCallback(
    (id: string, keys: string[] = []) => {
      const next = toggleChosenPaper(papers, id, keys)
      if (!next) {
        toast.info(t('graphPanel.paperFilter.maxPapers'))
        return
      }
      setPapers(next)
      setMode('paper')
    },
    [papers, t]
  )

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
    const found = await searchPapers(query.trim())
    for (const item of found) knownPapers.set(item.paper, item)
    return found
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
                <div className="flex min-w-0 items-center gap-2">
                  <Check
                    className={`h-3 w-3 shrink-0 ${findChosenPaper(papers, item.paper, item.keys) ? 'opacity-100' : 'opacity-0'}`}
                  />
                  <div className="min-w-0 truncate" title={`${item.citation}\n${item.doi || item.paper}`}>
                    <div className="truncate">{item.citation || item.paper}</div>
                    <div className="text-muted-foreground truncate text-xs">{item.doi || item.paper}</div>
                    {item.tldr && (
                      <div className="text-muted-foreground line-clamp-2 w-0 min-w-full text-xs whitespace-normal">{item.tldr}</div>
                    )}
                  </div>
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
              onChange={(id) => id && togglePaper(id, knownPapers.get(id)?.keys)}
              clearable={false}
              closeOnSelect={false}
              debounceTime={400}
            />
          </div>
        )}
        {mode === 'paper' && papers.length > 0 && (
          <div className="flex min-w-0 flex-wrap items-center gap-1">
            {papers.map((id) => {
              const citation = knownPapers.get(id)?.citation || id
              return (
                <Badge key={id} variant="secondary" className="gap-1 font-normal" title={id}>
                  <span className="max-w-[160px] truncate">{citation}</span>
                  <button
                    type="button"
                    aria-label={t('graphPanel.paperFilter.removePaper', { citation })}
                    className="hover:text-destructive"
                    onClick={() => setPapers(papers.filter((p) => p !== id))}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              )
            })}
            {papers.length >= 2 && (
              <Button variant="ghost" className="h-5 px-2 text-xs" onClick={() => setPapers([])}>
                {t('graphPanel.paperFilter.clearPapers')}
              </Button>
            )}
          </div>
        )}
      </div>
      {mode === 'topics' ? (
        <div className="mx-auto min-h-0 w-full max-w-4xl flex-1 overflow-auto">
          <TopicsPanel onOpenPaper={togglePaper} />
        </div>
      ) : papers.length ? (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <PaperMap
              key={papers.join(' ')}
              papers={papers}
              onTogglePaper={togglePaper}
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
                  papers={papers}
                  onOpen={togglePaper}
                  onPickCitation={pickCitation}
                />
              )}
              {panel === 'similar' && <SimilarPanel papers={papers} onOpen={togglePaper} />}
              {panel === 'citation' && <ContextsPanel pair={pair} onSwap={swapCitation} />}
              {panel === 'outside' && <OutsidePanel papers={papers} onOpen={togglePaper} />}
            </div>
          </aside>
        </div>
      ) : (
        <p className="text-muted-foreground p-4 text-sm">{t('citationGraph.emptyPaper')}</p>
      )}
    </div>
  )
}
