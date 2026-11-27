// Helpers of the Citation Graph tab (fork): the paper map drawn on its own sigma canvas, the
// citation a map link stands for, and the topic list. The routes are in api/lightrag.

import { UndirectedGraph } from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'

import type { PaperTopic, StructPaper } from '@/api/lightrag'
import { maxNodeSize, minNodeSize } from '@/lib/constants'
import { mapLinkKind, paperMapEdgeAttrs } from '@/lib/paperFilter'
import { resolveNodeColor } from '@/utils/graphColor'

export type CitationMode = 'paper' | 'topics'

// The paper the tab shows: the id every route takes (a search result's `paper` or a structure
// `paper_id`).
export type ActivePaper = { id: string }

// The panels beside the map.
export type PaperPanel = 'related' | 'similar' | 'citation' | 'outside'

export const PAPER_PANELS: PaperPanel[] = ['related', 'similar', 'citation', 'outside']

// The relation filters of the related panel, in the order they are offered.
export const RELATION_FILTERS = ['cites', 'cited_by', 'coupled', 'coauthor'] as const

// The pair of papers the citation panel is about: the citing paper first.
export type CitationPair = { citing: string; cited: string }

export const paperTitle = (paper: Pick<StructPaper, 'title' | 'label' | 'paper_id'>): string =>
  paper.title || paper.label || paper.paper_id

// "Kratzert 2019 · Journal" style one-line description of a paper.
export const paperByline = (paper: Pick<StructPaper, 'label' | 'venue'>): string =>
  [paper.label, paper.venue].filter(Boolean).join(' · ')

// The citation a link of the paper map stands for. The server sends the edge from the citing
// paper; a link that is only shared references (coupling) is no citation.
export const mapEdgeCitation = (rawEdge: {
  source: string
  target: string
  properties?: Record<string, any>
}): (CitationPair & { both: boolean }) | null => {
  const link = mapLinkKind(rawEdge.properties)
  if (link.cites === 'none') return null
  return { citing: rawEdge.source, cited: rawEdge.target, both: link.cites === 'both' }
}

export type MapRawNode = { id: string; labels?: string[]; properties?: Record<string, any> }
export type MapRawEdge = {
  id: string
  source: string
  target: string
  properties?: Record<string, any>
}

export type PaperMapGraph = {
  graph: UndirectedGraph
  // Field of study -> node colour, for the map's legend.
  fields: Map<string, string>
}

const LAYOUT_ITERATIONS = 150

// The sigma graph of a paper map: nodes coloured by field and sized by how many library papers
// cite them (the chosen paper is the largest), links as the Knowledge Graph's paper map draws
// them (see paperMapEdgeAttrs), laid out by ForceAtlas2 from rings around the chosen paper. A
// map has at most a few hundred papers, so the layout runs once, here, on the main thread.
export const buildPaperMapGraph = (data: {
  nodes: MapRawNode[]
  edges: MapRawEdge[]
}): PaperMapGraph => {
  const graph = new UndirectedGraph()
  const fields = new Map<string, string>()
  let colors = new Map<string, string>()
  const maxCited = Math.max(1, ...data.nodes.map((n) => Number(n.properties?.cited_by_library) || 0))
  const ringCount = new Map<number, number>()

  for (const node of data.nodes) {
    if (graph.hasNode(node.id)) continue
    const props = node.properties ?? {}
    const hop = typeof props.hop === 'number' ? props.hop : 1
    const resolved = resolveNodeColor(props.entity_type, colors)
    colors = resolved.map
    if (typeof props.entity_type === 'string' && props.entity_type) {
      fields.set(props.entity_type, resolved.color)
    }
    const index = ringCount.get(hop) ?? 0
    ringCount.set(hop, index + 1)
    const angle = index * 2.399963 // the golden angle keeps a ring's papers apart
    const radius = hop * 10
    const cited = Number(props.cited_by_library) || 0
    graph.addNode(node.id, {
      label: node.labels?.[0] ?? node.id,
      color: resolved.color,
      x: Math.cos(angle) * radius + index * 0.01,
      y: Math.sin(angle) * radius,
      size:
        hop === 0
          ? maxNodeSize + 4
          : Math.round(minNodeSize + (maxNodeSize - minNodeSize) * Math.sqrt(cited / maxCited)),
      forceLabel: hop === 0
    })
  }

  for (const edge of data.edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue
    if (graph.hasEdge(edge.source, edge.target)) continue
    const weight = Number(edge.properties?.weight)
    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      size: Math.min(3, 0.5 + (Number.isFinite(weight) ? weight : 1)),
      ...paperMapEdgeAttrs(edge, data.edges.length)
    })
  }

  if (graph.order > 1) {
    forceAtlas2.assign(graph, {
      iterations: LAYOUT_ITERATIONS,
      settings: forceAtlas2.inferSettings(graph)
    })
  }
  return { graph, fields }
}

// A topic list grouped by the path above each topic (a subfield's field, a topic's subfield),
// the groups and the topics in them by the papers they hold, most first.
export type TopicGroup = { path: string; topics: PaperTopic[] }

export const groupTopics = (topics: PaperTopic[]): TopicGroup[] => {
  const groups = new Map<string, PaperTopic[]>()
  for (const topic of topics) {
    const parts = topic.path.split(' > ')
    const parent = parts.length > 1 ? parts.slice(0, -1).join(' > ') : ''
    groups.set(parent, [...(groups.get(parent) ?? []), topic])
  }
  const bySize = (a: PaperTopic, b: PaperTopic) => b.n_papers - a.n_papers
  const total = (list: PaperTopic[]) => list.reduce((sum, t) => sum + t.n_papers, 0)
  return [...groups.entries()]
    .map(([path, list]) => ({ path, topics: [...list].sort(bySize) }))
    .sort((a, b) => total(b.topics) - total(a.topics))
}
