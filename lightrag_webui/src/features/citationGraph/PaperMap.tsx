import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { SigmaContainer, useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { EdgeRectangleProgram, NodeCircleProgram } from 'sigma/rendering'
import type { Settings as SigmaSettings } from 'sigma/settings'
import type { AbstractGraph } from 'graphology-types'
import { Check, CircleDot, Orbit, Plus } from 'lucide-react'

import { queryPaperMap } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import useIsDarkMode from '@/hooks/useIsDarkMode'
import { edgeColorDarkTheme, labelColorDarkTheme, labelColorLightTheme } from '@/lib/constants'
import {
  buildPaperMapGraph,
  mapEdgeCitation,
  type CitationPair,
  type PaperMapGraph
} from '@/lib/citationGraph'
import { MapArrowProgram, MapCurvedArrowProgram } from '@/lib/mapArrowPrograms'
import { nearestLink } from '@/lib/mapLinkPick'
import {
  MAP_CITATION_COLOR,
  MAP_COUPLING_COLOR,
  MAP_DEFAULT_LIMIT,
  mapHasMorePapers,
  mapLinkKind,
  nextHops,
  nextMapLimit,
  paperSummaryMessage,
  readMapHops,
  writeMapHops,
  type MapHops
} from '@/lib/paperFilter'
import { useFetched } from './useFetched'

import '@react-sigma/core/lib/style.css'

type Selection = { graph: PaperMapGraph; kind: 'node' | 'edge'; id: string }

const createSigmaSettings = (isDarkTheme: boolean): Partial<SigmaSettings> => ({
  allowInvalidContainer: true,
  defaultNodeType: 'circle',
  defaultEdgeType: 'rect',
  renderEdgeLabels: false,
  edgeProgramClasses: {
    rect: EdgeRectangleProgram,
    arrow: MapArrowProgram,
    curvedArrow: MapCurvedArrowProgram
  },
  nodeProgramClasses: { circle: NodeCircleProgram },
  labelGridCellSize: 60,
  labelRenderedSizeThreshold: 8,
  labelSize: 12,
  defaultEdgeColor: isDarkTheme ? edgeColorDarkTheme : '#d3d3d3',
  labelColor: { color: isDarkTheme ? labelColorDarkTheme : labelColorLightTheme }
})

// Binds the graph to the canvas, marks the selected paper or link, and reports clicks. A click
// beside a link selects it (see nearestLink: sigma picks only the drawn line).
const MapController = ({
  built,
  selection,
  onSelect
}: {
  built: PaperMapGraph
  selection: Selection | null
  onSelect: (next: { kind: 'node' | 'edge'; id: string } | null) => void
}) => {
  const sigma = useSigma()
  const registerEvents = useRegisterEvents()
  const setSettings = useSetSettings()

  useEffect(() => {
    sigma.setGraph(built.graph as unknown as AbstractGraph)
    sigma.refresh()
  }, [sigma, built])

  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => onSelect({ kind: 'node', id: node }),
      clickStage: ({ event }) => {
        const edge = nearestLink(sigma as any, event)
        onSelect(edge === null ? null : { kind: 'edge', id: edge })
      }
    })
  }, [sigma, registerEvents, onSelect])

  useEffect(() => {
    const graph = built.graph
    setSettings({
      nodeReducer: (node, data) => {
        const picked =
          selection?.kind === 'node' &&
          (selection.id === node || graph.areNeighbors(selection.id, node))
        return picked ? { ...data, forceLabel: true, highlighted: selection.id === node } : data
      },
      edgeReducer: (edge, data) =>
        selection?.kind === 'edge' && selection.id === edge
          ? { ...data, size: data.size + 2, zIndex: 1 }
          : data
    })
  }, [setSettings, built, selection])

  return null
}

// The papers around the chosen papers over citations and coupling (the server's paper map), on a
// canvas of its own: a click on a paper or a link says what it is and offers to add that paper to
// the chosen ones (or take it out) or read the sentences of that citation.
const PaperMap = ({
  papers,
  onTogglePaper,
  onPickCitation
}: {
  papers: string[]
  onTogglePaper: (id: string, keys: string[]) => void
  onPickCitation: (pair: CitationPair) => void
}) => {
  const { t } = useTranslation()
  const isDarkMode = useIsDarkMode()
  const [hops, setHops] = useState<MapHops>(readMapHops)
  const [limit, setLimit] = useState(MAP_DEFAULT_LIMIT)
  const load = useCallback(() => queryPaperMap(papers, hops, limit), [papers, hops, limit])
  const { data, error, loading } = useFetched(load)
  const built = useMemo(() => (data?.nodes?.length ? buildPaperMapGraph(data) : null), [data])
  const [picked, setPicked] = useState<Selection | null>(null)
  const selection = picked && picked.graph === built ? picked : null
  const settings = useMemo(() => createSigmaSettings(isDarkMode), [isDarkMode])

  const select = useCallback(
    (next: { kind: 'node' | 'edge'; id: string } | null) =>
      setPicked(next && built ? { graph: built, ...next } : null),
    [built]
  )

  const toggleHops = () => {
    const next = nextHops(hops)
    writeMapHops(next)
    setHops(next)
  }

  const summary = data?.paper
  const HopsIcon = hops === 2 ? Orbit : CircleDot
  const hopsTooltip = t(
    hops === 2 ? 'graphPanel.paperFilter.mapTwoHopsTooltip' : 'graphPanel.paperFilter.mapOneHopTooltip',
    { limit }
  )

  const node = selection?.kind === 'node' ? data?.nodes.find((n) => n.id === selection.id) : undefined
  const edge = selection?.kind === 'edge' ? data?.edges.find((e) => e.id === selection.id) : undefined
  const nodeName = (id: string) => data?.nodes.find((n) => n.id === id)?.labels?.[0] ?? id
  const structId = (id: string) =>
    (data?.nodes.find((n) => n.id === id)?.properties?.paper_id as string | undefined) ?? id
  // A chosen paper may be named by its DOI or by a key: the node's other ids say which it is.
  const togglePaper = (id: string) => {
    const info = data?.nodes.find((n) => n.id === id)?.properties ?? {}
    const keys = [id, info.doi, info.attachment_key, ...(info.attachment_keys ?? [])]
    onTogglePaper(structId(id), keys.filter((key): key is string => typeof key === 'string' && !!key))
  }

  const message = error
    ? t('citationGraph.error', { message: error instanceof Error ? error.message : String(error) })
    : summary
      ? paperSummaryMessage(t, summary, data?.nodes.length ?? 0)
      : null

  const link = edge ? mapLinkKind(edge.properties) : null
  const citation = edge ? mapEdgeCitation(edge) : null
  const names = edge ? { source: nodeName(edge.source), target: nodeName(edge.target) } : null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          {summary?.papers && summary.papers.length > 1 ? (
            <div className="flex flex-col text-sm font-medium" title={summary?.title}>
              {summary.papers.map((p) => (
                <div key={p.paper} className="truncate">
                  {p.citation || p.paper}
                </div>
              ))}
            </div>
          ) : (
            <div className="truncate text-sm font-medium" title={summary?.title}>
              {summary?.citation || summary?.title || papers[0]}
            </div>
          )}
          <div className="text-muted-foreground truncate text-xs" role="status">
            {loading ? t('citationGraph.loading') : message}
          </div>
        </div>
        <Button
          size="icon"
          variant="outline"
          onClick={toggleHops}
          tooltip={hopsTooltip}
          aria-label={hopsTooltip}
        >
          <HopsIcon />
        </Button>
        {mapHasMorePapers(summary) && (
          <Button variant="outline" size="sm" onClick={() => setLimit((l) => nextMapLimit(l, summary?.max_nodes_cap ?? 0))}>
            {t('graphPanel.paperFilter.mapShowMore', {
              shown: summary?.shown_papers ?? 0,
              found: summary?.n_papers ?? 0,
              limit: nextMapLimit(limit, summary?.max_nodes_cap ?? 0)
            })}
          </Button>
        )}
      </div>
      <div className="relative min-h-[320px] flex-1">
        {built ? (
          <SigmaContainer settings={settings} className="!bg-background !size-full overflow-hidden">
            <MapController built={built} selection={selection} onSelect={select} />
          </SigmaContainer>
        ) : (
          <p className="text-muted-foreground p-3 text-sm">
            {loading ? t('citationGraph.loading') : message}
          </p>
        )}
        {built && (
          <div className="bg-background/70 absolute bottom-2 left-2 flex max-w-[60%] flex-col gap-1 rounded-md border p-2 text-xs backdrop-blur-lg">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {[...built.fields].map(([field, color]) => (
                <span key={field} className="flex items-center gap-1">
                  <span className="inline-block size-2.5 rounded-full" style={{ backgroundColor: color }} />
                  {field}
                </span>
              ))}
            </div>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4" style={{ backgroundColor: MAP_CITATION_COLOR }} />
              {t('graphPanel.paperFilter.legendCitation')}
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4" style={{ backgroundColor: MAP_COUPLING_COLOR }} />
              {t('graphPanel.paperFilter.legendCoupling')}
            </span>
          </div>
        )}
        {(node || edge) && (
          <div className="bg-background/80 absolute top-2 right-2 flex max-h-[70%] w-[320px] max-w-[60%] flex-col gap-1 overflow-auto rounded-md border p-3 text-sm backdrop-blur-lg">
            {node && (
              <>
                <div className="font-medium">{node.properties?.title ?? nodeName(node.id)}</div>
                <div className="text-muted-foreground text-xs">
                  {[nodeName(node.id), node.properties?.venue, node.properties?.topic].filter(Boolean).join(' · ')}
                </div>
                {node.properties?.tldr && (
                  <p className="text-muted-foreground text-xs">{node.properties.tldr}</p>
                )}
                <Button
                  variant={node.properties?.hop === 0 ? 'secondary' : 'outline'}
                  size="sm"
                  className="mt-1 h-6 w-fit gap-1 px-2 text-xs"
                  aria-pressed={node.properties?.hop === 0}
                  onClick={() => togglePaper(node.id)}
                >
                  {node.properties?.hop === 0 ? <Check className="size-3" /> : <Plus className="size-3" />}
                  {t(node.properties?.hop === 0 ? 'citationGraph.chosen' : 'citationGraph.notChosen')}
                </Button>
              </>
            )}
            {edge && link && names && (
              <>
                <div className="font-medium">
                  {link.cites === 'one'
                    ? t('graphPanel.paperFilter.linkDirectionCites', names)
                    : link.cites === 'both'
                      ? t('graphPanel.paperFilter.linkDirectionMutual', names)
                      : t('graphPanel.paperFilter.linkDirectionNone')}
                </div>
                {link.sharedRefs !== null && (
                  <div className="text-muted-foreground text-xs">
                    {t('graphPanel.paperFilter.linkSharedRefs')}: {link.sharedRefs}
                  </div>
                )}
                {citation && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-1 w-fit"
                    onClick={() =>
                      onPickCitation({ citing: structId(citation.citing), cited: structId(citation.cited) })
                    }
                  >
                    {t('citationGraph.map.readSentences')}
                  </Button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default PaperMap
