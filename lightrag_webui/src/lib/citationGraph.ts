// Helpers of the Citation Graph tab (fork): the paper map drawn on its own sigma canvas, the
// citation a map link stands for, and the topic list. The routes are in api/lightrag.

import { UndirectedGraph } from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'

import type {
  OutsideWork,
  OutsideWorks,
  PaperTopic,
  RelatedPaper,
  RelatedPapers,
  SimilarPapers,
  StructPaper
} from '@/api/lightrag'
import { maxNodeSize, minNodeSize } from '@/lib/constants'
import { MAX_PAPERS, mapLinkKind, paperMapEdgeAttrs, togglePaper } from '@/lib/paperFilter'
import { resolveNodeColor } from '@/utils/graphColor'

export type CitationMode = 'paper' | 'topics'

// A paper id as the routes take it, without the `doi:` or `zot:` of a structure id or the
// doi.org link.
export const bareId = (id: string): string =>
  id
    .trim()
    .replace(/^https?:\/\/(dx\.)?doi\.org\//, '')
    .replace(/^(doi|zot):/, '')

// The paper chosen under any of these ids: a paper is a DOI or a Zotero key in a search result,
// `doi:` or `zot:` in a structure id, and may have several keys.
export const findChosenPaper = (papers: string[], id: string, keys: string[] = []): string | undefined => {
  const same = [id, ...keys].map((alias) => bareId(alias).toLowerCase())
  return papers.find((p) => same.includes(bareId(p).toLowerCase()))
}

// The chosen papers with this one added, or taken out when it is chosen already (under any of
// its ids); null when it is new and MAX_PAPERS are chosen.
export const toggleChosenPaper = (papers: string[], id: string, keys: string[] = []): string[] | null => {
  const chosen = findChosenPaper(papers, id, keys)
  if (chosen !== undefined) return togglePaper(papers, chosen)
  return papers.length >= MAX_PAPERS ? null : [...papers, id]
}

// The panels beside the map.
export type PaperPanel = 'related' | 'similar' | 'citation' | 'outside'

export const PAPER_PANELS: PaperPanel[] = ['related', 'similar', 'citation', 'outside']

// What a panel got by asking the server once for each chosen paper (the routes take one paper):
// the answers of the requests that succeeded, each with its paper, and how many were asked.
export type Fanned<T> = { answers: { paper: string; answer: T }[]; total: number }

// Asks for every paper at once. Fails only when every request does (with the first error, so a
// status stays readable); otherwise the papers that failed are left out and `total` says so.
export const fanOut = async <T>(papers: string[], ask: (paper: string) => Promise<T>): Promise<Fanned<T>> => {
  const settled = await Promise.allSettled(papers.map(ask))
  const answers = papers.flatMap((paper, i) => {
    const result = settled[i]
    return result.status === 'fulfilled' ? [{ paper, answer: result.value }] : []
  })
  const failed = settled.find((result): result is PromiseRejectedResult => result.status === 'rejected')
  if (failed && !answers.length) throw failed.reason
  return { answers, total: papers.length }
}

// The reason every answer gave, when none has anything to show.
export const fannedReason = (fanned: Fanned<{ reason?: string }> | null): string | undefined =>
  fanned?.answers.length && fanned.answers.every(({ answer }) => answer.reason)
    ? fanned.answers[0].answer.reason
    : undefined

// How many of the chosen papers gave nothing: their request failed or the answer is a reason.
export const fannedMissing = (fanned: Fanned<{ reason?: string }>): number =>
  fanned.total - fanned.answers.filter(({ answer }) => !answer.reason).length

// A related paper with the chosen paper it was found from: the pair of a citation reads from it.
export type RelatedFrom = RelatedPaper & { source: string }

// The related papers of every answer, each once: where several chosen papers reach it, the
// one that has citing sentences is kept. Best score first once several papers answered.
export const mergeRelated = (fanned: Fanned<RelatedPapers>): RelatedFrom[] => {
  const merged = new Map<string, RelatedFrom>()
  for (const { paper, answer } of fanned.answers) {
    for (const item of answer.related ?? []) {
      const have = merged.get(item.paper_id)
      if (!have || (!have.citation?.n_contexts && item.citation?.n_contexts)) {
        merged.set(item.paper_id, { ...item, source: paper })
      }
    }
  }
  const list = [...merged.values()]
  return fanned.answers.length > 1 ? list.sort((a, b) => b.score - a.score) : list
}

// The similar papers of every answer, each once with its best similarity and every way the
// library links it to a chosen paper. Most similar first once several papers answered.
export const mergeSimilar = (fanned: Fanned<SimilarPapers>): NonNullable<SimilarPapers['similar']> => {
  const merged = new Map<string, NonNullable<SimilarPapers['similar']>[number]>()
  for (const { answer } of fanned.answers) {
    for (const item of answer.similar ?? []) {
      const have = merged.get(item.paper_id)
      if (!have) {
        merged.set(item.paper_id, item)
        continue
      }
      merged.set(item.paper_id, {
        ...(item.similarity > have.similarity ? item : have),
        links: [...new Set([...have.links, ...item.links])]
      })
    }
  }
  const list = [...merged.values()]
  return fanned.answers.length > 1 ? list.sort((a, b) => b.similarity - a.similarity) : list
}

// The outside works of every answer, each once with the library papers that cite it from any
// answer. Most cited by the library first once several papers answered.
export const mergeOutside = (fanned: Fanned<OutsideWorks>): OutsideWork[] => {
  const merged = new Map<string, OutsideWork>()
  for (const { answer } of fanned.answers) {
    for (const work of answer.works ?? []) {
      const have = merged.get(work.work_id)
      if (!have) {
        merged.set(work.work_id, work)
        continue
      }
      const seen = new Set(have.also_cited_by.map((citer) => citer.paper_id))
      merged.set(work.work_id, {
        ...have,
        also_cited_by: [...have.also_cited_by, ...work.also_cited_by.filter((c) => !seen.has(c.paper_id))]
      })
    }
  }
  const list = [...merged.values()]
  return fanned.answers.length > 1 ? list.sort((a, b) => b.n_library_citers - a.n_library_citers) : list
}

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
