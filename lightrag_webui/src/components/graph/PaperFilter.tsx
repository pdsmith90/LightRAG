import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { FileText, Network } from 'lucide-react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import Button from '@/components/ui/Button'
import { controlButtonVariant } from '@/lib/constants'
import { paperLabel, parsePaperLabel, PaperGraphMode } from '@/lib/paperFilter'
import { searchPapers, PaperSummary } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'

// Fork: shows one paper's part of the graph, or the part it shares with the
// library papers it cites, by setting a paper label (see lib/paperFilter).
const PaperFilter = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const current = parsePaperLabel(label)
  // The mode for the next pick while no paper is shown; a shown paper's label
  // carries its own.
  const [pendingMode, setPendingMode] = useState<PaperGraphMode>('paper')
  const mode = current?.mode ?? pendingMode

  const show = useCallback((paper: string, nextMode: PaperGraphMode) => {
    const newLabel = paperLabel(paper, nextMode)
    SearchHistoryManager.addToHistory(newLabel)
    useGraphStore.getState().setGraphDataFetchAttempted(false)
    useSettingsStore.getState().setQueryLabel(newLabel)
    useGraphStore.getState().incrementGraphDataVersion()
  }, [])

  const fetchPapers = useCallback(
    async (query?: string): Promise<PaperSummary[]> =>
      query && query.trim() ? searchPapers(query.trim()) : [],
    []
  )

  const toggleMode = useCallback(() => {
    const nextMode: PaperGraphMode = mode === 'paper' ? 'shared' : 'paper'
    setPendingMode(nextMode)
    if (current) show(current.paper, nextMode)
  }, [mode, current, show])

  return (
    <div className="flex items-center">
      <div className="w-full min-w-[200px] max-w-[360px]">
        <AsyncSelect<PaperSummary>
          className="min-w-[420px]"
          triggerClassName="max-h-8 w-full overflow-hidden"
          searchInputClassName="max-h-8"
          triggerTooltip={t('graphPanel.paperFilter.selectTooltip')}
          fetcher={fetchPapers}
          renderOption={(item) => (
            <div className="min-w-0 truncate" title={`${item.citation}\n${item.doi || item.paper}`}>
              <div className="truncate">{item.citation || item.paper}</div>
              <div className="text-muted-foreground truncate text-xs">
                {item.doi || item.paper}
                {item.n_cited_in_kb != null &&
                  ` · ${t('graphPanel.paperFilter.referencesInKb', { count: item.n_cited_in_kb })}`}
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
          value={current?.paper ?? ''}
          onChange={(paper) => {
            if (paper) show(paper, mode)
          }}
          clearable={false}
          debounceTime={400}
        />
      </div>
      <Button
        size="icon"
        variant={controlButtonVariant}
        onClick={toggleMode}
        tooltip={
          mode === 'shared'
            ? t('graphPanel.paperFilter.modeSharedTooltip')
            : t('graphPanel.paperFilter.modePaperTooltip')
        }
        className="ml-1"
      >
        {mode === 'shared' ? <Network className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
      </Button>
    </div>
  )
}

export default PaperFilter
