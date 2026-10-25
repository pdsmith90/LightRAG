import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  Check,
  FileText,
  Network,
  Share2,
  Spline,
  SquaresIntersect,
  Waypoints,
  X
} from 'lucide-react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { controlButtonVariant } from '@/lib/constants'
import {
  MAX_PAPERS,
  nextContext,
  nextMode,
  paperLabel,
  parsePaperLabel,
  PaperContext,
  PaperGraphMode,
  readPaperContext,
  togglePaper,
  writePaperContext
} from '@/lib/paperFilter'
import { searchPapers, PaperSummary } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

// Papers seen in search results, so a chip can show a citation instead of a DOI.
const knownPapers = new Map<string, PaperSummary>()
const lookups = new Set<string>()
// The label the picker itself last set, so its next change can replace that
// history entry instead of adding one per pick.
let lastPickerLabel = ''

// eslint-disable-next-line react-refresh/only-export-components
export const __resetPaperFilterForTests = (): void => {
  knownPapers.clear()
  lookups.clear()
  lastPickerLabel = ''
}

const modeIcons = {
  paper: FileText,
  shared: Network,
  common: SquaresIntersect
}
const contextIcons = {
  own: Spline,
  kb: Share2,
  linked: Waypoints
}

// Fork: shows the part of the graph that one or more papers contributed, the part
// they share with the library papers they cite, or the part at least two of them
// contributed, by setting a paper label (see lib/paperFilter).
const PaperFilter = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const current = useMemo(() => parsePaperLabel(label), [label])
  const papers = useMemo(() => current?.papers ?? [], [current])
  // The mode for the next pick while no paper is shown; a shown paper's label
  // carries its own.
  const [pendingMode, setPendingMode] = useState<PaperGraphMode>('paper')
  const mode = current?.mode ?? pendingMode
  const [context, setContext] = useState<PaperContext>(readPaperContext)
  // Bumped when a lookup fills knownPapers, to redraw the chips.
  const [, setKnownVersion] = useState(0)

  // Ids that came from a reload or the search history have no citation yet.
  useEffect(() => {
    for (const id of papers) {
      if (knownPapers.has(id) || lookups.has(id)) continue
      lookups.add(id)
      searchPapers(id, 1)
        .then((found) => {
          // The best match is only this paper when it has this DOI or key.
          const hit = found[0]
          if (hit && (hit.paper.toLowerCase() === id.toLowerCase() || hit.keys?.includes(id.toUpperCase()))) {
            knownPapers.set(id, hit)
          }
          setKnownVersion((v) => v + 1)
        })
        .catch(() => lookups.delete(id))
    }
  }, [papers])

  const show = useCallback((nextPapers: string[], nextMode: PaperGraphMode) => {
    // No papers left: an empty label resets the graph, '*' would query all of it.
    const newLabel = nextPapers.length ? paperLabel(nextPapers, nextMode) : ''
    // An edit replaces the entry the picker made; clearing keeps it for later.
    const shown = useSettingsStore.getState().queryLabel
    if (newLabel && shown && shown === lastPickerLabel) SearchHistoryManager.removeLabel(shown)
    SearchHistoryManager.addToHistory(newLabel)
    lastPickerLabel = newLabel
    useGraphStore.getState().setGraphDataFetchAttempted(false)
    useSettingsStore.getState().setQueryLabel(newLabel)
    useGraphStore.getState().incrementGraphDataVersion()
  }, [])

  const apply = useCallback(
    (nextPapers: string[]) => {
      // Common needs two papers.
      const nextModeValue = nextPapers.length < 2 && mode === 'common' ? 'paper' : mode
      setPendingMode(nextModeValue)
      show(nextPapers, nextModeValue)
    },
    [mode, show]
  )

  const fetchPapers = useCallback(async (query?: string): Promise<PaperSummary[]> => {
    if (!query || !query.trim()) return []
    const found = await searchPapers(query.trim())
    for (const item of found) knownPapers.set(item.paper, item)
    return found
  }, [])

  const pick = useCallback(
    (paper: string) => {
      if (!paper) return
      // The same paper may already be selected under its other id (key or DOI).
      const keys = knownPapers.get(paper)?.keys ?? []
      const selected = papers.find((p) => p === paper || keys.includes(p))
      if (!selected && papers.length >= MAX_PAPERS) {
        toast.info(t('graphPanel.paperFilter.maxPapers'))
        return
      }
      apply(togglePaper(papers, selected ?? paper))
    },
    [papers, apply, t]
  )

  const toggleMode = useCallback(() => {
    const next = nextMode(mode, papers.length)
    setPendingMode(next)
    if (current) show(papers, next)
  }, [mode, papers, current, show])

  const toggleContext = useCallback(() => {
    const next = nextContext(context)
    writePaperContext(next)
    setContext(next)
    // The context is not part of the label, so a refetch has to be asked for.
    if (current) {
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      useGraphStore.getState().incrementGraphDataVersion()
    }
  }, [context, current])

  const ModeIcon = modeIcons[mode]
  const ContextIcon = contextIcons[context]
  const modeTooltip = t(
    mode === 'common'
      ? 'graphPanel.paperFilter.modeCommonTooltip'
      : mode === 'shared'
        ? 'graphPanel.paperFilter.modeSharedTooltip'
        : 'graphPanel.paperFilter.modePaperTooltip'
  )
  const contextTooltip = t(
    context === 'linked'
      ? 'graphPanel.paperFilter.contextLinkedTooltip'
      : context === 'kb'
        ? 'graphPanel.paperFilter.contextKbTooltip'
        : 'graphPanel.paperFilter.contextOwnTooltip'
  )

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center">
        <div className="w-full min-w-[200px] max-w-[360px]">
          <AsyncSelect<PaperSummary>
            className="min-w-[420px]"
            triggerClassName="max-h-8 w-full overflow-hidden"
            searchInputClassName="max-h-8"
            triggerTooltip={t('graphPanel.paperFilter.selectTooltip')}
            fetcher={fetchPapers}
            renderOption={(item) => (
              <div className="flex min-w-0 items-center gap-2">
                <Check
                  className={`h-3 w-3 shrink-0 ${papers.some((p) => p === item.paper || item.keys?.includes(p)) ? 'opacity-100' : 'opacity-0'}`}
                />
                <div className="min-w-0 truncate" title={`${item.citation}\n${item.doi || item.paper}`}>
                  <div className="truncate">{item.citation || item.paper}</div>
                  <div className="text-muted-foreground truncate text-xs">
                    {item.doi || item.paper}
                    {item.n_cited_in_kb != null &&
                      ` · ${t('graphPanel.paperFilter.referencesInKb', { count: item.n_cited_in_kb })}`}
                  </div>
                </div>
              </div>
            )}
            getOptionValue={(item) => item.paper}
            getDisplayValue={(item) => (
              <div className="min-w-0 flex-1 truncate text-left" title={item.citation}>
                {item.citation || item.paper}
              </div>
            )}
            ariaLabel={t('graphPanel.paperFilter.label')}
            placeholder={t('graphPanel.paperFilter.placeholder')}
            searchPlaceholder={t('graphPanel.paperFilter.searchPlaceholder')}
            noResultsMessage={t('graphPanel.paperFilter.noPapers')}
            value=""
            onChange={pick}
            clearable={false}
            closeOnSelect={false}
            debounceTime={400}
          />
        </div>
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={toggleMode}
          tooltip={modeTooltip}
          aria-label={modeTooltip}
          className="ml-1"
        >
          <ModeIcon className="h-4 w-4" />
        </Button>
        <Button
          size="icon"
          variant={controlButtonVariant}
          onClick={toggleContext}
          tooltip={contextTooltip}
          aria-label={contextTooltip}
        >
          <ContextIcon className="h-4 w-4" />
        </Button>
      </div>
      {papers.length > 0 && (
        <div className="flex max-w-[440px] flex-wrap items-center gap-1">
          {papers.map((id) => {
            const citation = knownPapers.get(id)?.citation || id
            return (
              <Badge key={id} variant="secondary" className="gap-1 font-normal" title={id}>
                <span className="max-w-[160px] truncate">{citation}</span>
                <button
                  type="button"
                  aria-label={t('graphPanel.paperFilter.removePaper', { citation })}
                  className="hover:text-destructive"
                  onClick={() => apply(papers.filter((p) => p !== id))}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            )
          })}
          {papers.length >= 2 && (
            <Button
              variant="ghost"
              className="h-5 px-2 text-xs"
              onClick={() => apply([])}
            >
              {t('graphPanel.paperFilter.clearPapers')}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

export default PaperFilter
