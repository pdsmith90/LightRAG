// The paper filter (fork) rides on the graph query label, so the label's
// fetch, refresh, retry and history handling cover it unchanged: a label
// `paper:<ids>` shows what the papers contributed, `paper+refs:<ids>` only what
// they share with the library papers they cite, `paper+common:<ids>` what at
// least two of them contributed, `paper+map:<ids>` the papers around them
// over citations and coupling (a server with the paper structure only). <ids>
// are DOIs or Zotero keys separated by whitespace.

import type { PaperSummary } from '@/api/lightrag'

export type PaperGraphMode = 'paper' | 'shared' | 'common' | 'map'

// How far the paper map reaches from the chosen papers: a viewing preference
// like the density, so it is not part of the label either.
export type MapHops = 1 | 2

// How much of the wider knowledge base the graph shows around the papers:
// their own relations, plus every relation between the shown entities, plus
// a few linking entities. A viewing preference, so it is not part of the label.
export type PaperContext = 'own' | 'kb' | 'linked'

export const MAX_PAPERS = 10

const PREFIXES: Record<PaperGraphMode, string> = {
  paper: 'paper:',
  shared: 'paper+refs:',
  common: 'paper+common:',
  map: 'paper+map:'
}

const CONTEXT_KEY = 'lightrag-paper-context'
const HOPS_KEY = 'lightrag-paper-map-hops'

// Relations and linking entities the selected papers did not contribute.
export const CONTEXT_EDGE_COLOR = 'rgba(59,130,246,0.6)'

export const paperLabel = (papers: string[], mode: PaperGraphMode): string => {
  const ids = papers.map((p) => p.trim()).filter(Boolean)
  return `${PREFIXES[mode]}${[...new Set(ids)].join(' ')}`
}

export const parsePaperLabel = (
  label: string | null | undefined
): { papers: string[]; mode: PaperGraphMode } | null => {
  for (const mode of ['shared', 'common', 'map', 'paper'] as const) {
    const prefix = PREFIXES[mode]
    if (label?.startsWith(prefix)) {
      const papers = [...new Set(label.slice(prefix.length).split(/\s+/).filter(Boolean))]
      return papers.length ? { papers, mode } : null
    }
  }
  return null
}

export const togglePaper = (papers: string[], id: string): string[] =>
  papers.includes(id) ? papers.filter((p) => p !== id) : [...papers, id]

// How a chosen paper may be written: a DOI (bare, `doi:` or as a doi.org link), an
// attachment key or a structure id (`zot:<item key>`), compared without case.
const normalizePaperId = (id: string): string =>
  id
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(dx\.)?doi\.org\//, '')
    .replace(/^doi:/, '')

// A paper-map node (as the server sends it) is one of the chosen papers when its id, DOI,
// attachment key or structure id is one of them: a paper chosen by its key appears on the
// map under its DOI.
export const isChosenMapNode = (
  papers: string[],
  node: { id: string; properties?: Record<string, any> }
): boolean => {
  const props = node.properties ?? {}
  const ids = [node.id, props.doi, props.attachment_key, props.paper_id]
    .filter((id): id is string => typeof id === 'string' && id.trim() !== '')
    .map(normalizePaperId)
  return papers.some((p) => ids.includes(normalizePaperId(p)))
}

// Expanding a paper-map node adds that paper to the map's papers: the next
// label, or why not (the i18n key under graphPanel.paperFilter).
export const expandMapLabel = (
  papers: string[],
  node: { id: string; properties?: Record<string, any> }
): { label: string } | { reason: 'mapAlreadyChosen' | 'maxPapers' } => {
  if (isChosenMapNode(papers, node)) return { reason: 'mapAlreadyChosen' }
  if (papers.length >= MAX_PAPERS) return { reason: 'maxPapers' }
  return { label: paperLabel([...papers, node.id], 'map') }
}

// A server without the paper structure answers the map route with 501 (the setting is off or
// the schema is missing) or, when it predates the route, 404. Anything else is a failure that
// may pass.
export const isPaperMapUnavailable = (error: unknown): boolean => {
  const status = (error as { status?: number } | null)?.status
  return status === 501 || status === 404
}

// The label a map label falls back to when the server has no paper map: the same papers in
// the plain paper view (what the mode button shows after the map).
export const mapFallbackLabel = (papers: string[]): string => paperLabel(papers, 'paper')

// paper -> shared -> common -> map; common needs two papers, the map a server
// that reports the paper structure.
export const nextMode = (
  mode: PaperGraphMode,
  n: number,
  mapAvailable: boolean = false
): PaperGraphMode => {
  if (mode === 'paper') return 'shared'
  if (mode === 'shared' && n >= 2) return 'common'
  if ((mode === 'shared' || mode === 'common') && mapAvailable) return 'map'
  return 'paper'
}

const CONTEXTS: PaperContext[] = ['own', 'kb', 'linked']

export const nextContext = (context: PaperContext): PaperContext =>
  CONTEXTS[(CONTEXTS.indexOf(context) + 1) % CONTEXTS.length]

// The one value the density button and the fetch both read, so they agree
// even when storage is blocked. Loaded from storage on first use.
let currentContext: PaperContext | null = null

export const readPaperContext = (): PaperContext => {
  if (currentContext === null) {
    currentContext = 'kb'
    try {
      const stored = localStorage.getItem(CONTEXT_KEY)
      if (CONTEXTS.includes(stored as PaperContext)) currentContext = stored as PaperContext
    } catch {
      // storage unavailable: keep the default
    }
  }
  return currentContext
}

export const writePaperContext = (context: PaperContext): void => {
  currentContext = context
  try {
    localStorage.setItem(CONTEXT_KEY, context)
  } catch {
    // storage unavailable: the choice lasts until reload at most
  }
}

export const nextHops = (hops: MapHops): MapHops => (hops === 1 ? 2 : 1)

let currentHops: MapHops | null = null

export const readMapHops = (): MapHops => {
  if (currentHops === null) {
    currentHops = 1
    try {
      if (localStorage.getItem(HOPS_KEY) === '2') currentHops = 2
    } catch {
      // storage unavailable: keep the default
    }
  }
  return currentHops
}

export const writeMapHops = (hops: MapHops): void => {
  currentHops = hops
  try {
    localStorage.setItem(HOPS_KEY, String(hops))
  } catch {
    // storage unavailable: the choice lasts until reload at most
  }
}

export const __resetPaperContextForTests = (): void => {
  currentContext = null
  currentHops = null
}

// A paper result that arrives after the selection changed is out of date: the
// fetch effect runs again for the new label.
export const isStalePaperResult = (fetchedLabel: string, currentLabel: string): boolean =>
  !!parsePaperLabel(fetchedLabel) && fetchedLabel !== currentLabel

// An empty result clears the label, except a paper label (the selection stays
// editable) and a label the user changed while the fetch was running.
export const clearsLabelOnEmpty = (fetchedLabel: string, currentLabel: string): boolean =>
  !!fetchedLabel && !parsePaperLabel(fetchedLabel) && currentLabel === fetchedLabel

// #rrggbb -> #rrggbb73 (about 45% opacity); any other colour format is unchanged.
export const fadeColor = (color: string): string =>
  /^#[0-9a-fA-F]{6}$/.test(color) ? `${color}73` : color

// A node that is not from the selected papers is drawn faded. It keeps its
// size and label: a linking entity is usually a hub the view is about.
export const paperNodeAttrs = (rawNode: {
  color?: string
  properties?: Record<string, any>
}): { color?: string } =>
  rawNode.properties?.paper_context && rawNode.color
    ? { color: fadeColor(rawNode.color) }
    : {}

// A relation that is not from the selected papers is drawn in a translucent blue.
export const paperEdgeAttrs = (rawEdge: {
  properties?: Record<string, any>
}): { color?: string } =>
  rawEdge.properties?.paper_context ? { color: CONTEXT_EDGE_COLOR } : {}

// What a paper-filter load showed, or why it is empty.
export const paperSummaryMessage = (
  t: (key: string, options?: Record<string, unknown>) => string,
  summary: PaperSummary,
  nodes: number
): string => {
  const k = (key: string) => `graphPanel.paperFilter.${key}`
  const papers = summary.papers ?? []
  if (summary.reason) {
    const many =
      papers.length >= 2 &&
      ['not_in_kb', 'no_reference_list', 'no_references_in_kb', 'not_in_structure'].includes(
        summary.reason
      )
    return t(k(`reason.${summary.reason}${many ? '_many' : ''}`))
  }

  if (summary.mode === 'map') {
    let map = t(k('mapSummary'), {
      count: papers.length,
      nodes,
      hop1: summary.shown_hop1 ?? 0,
      hop2: summary.shown_hop2 ?? 0
    })
    const found = summary.n_links ?? 0
    const shown = summary.shown_links ?? found
    if (shown < found) map += `; ${t(k('mapLinksThinned'), { shown, found })}`
    const unknown = papers.filter((p) => p.reason === 'not_in_structure').length
    if (unknown) map += `; ${t(k('someNotInStructure'), { count: unknown })}`
    return map
  }

  // Linking entities are in the node count but are not what the papers contributed.
  const linking = summary.n_linking ?? 0
  nodes -= linking
  let message: string
  if (papers.length >= 2) {
    const counts = { count: papers.length, nodes }
    message =
      summary.mode === 'shared'
        ? t(k('sharedSummaryMany'), { ...counts, sharing: summary.cited_sharing ?? 0 })
        : summary.mode === 'common'
          ? t(k('commonSummary'), counts)
          : t(k('papersSummary'), counts)
  } else {
    const counts = { citation: summary.citation, nodes, inKb: summary.n_cited_in_kb ?? 0 }
    message =
      summary.mode === 'shared'
        ? t(k('sharedSummary'), { ...counts, sharing: summary.cited_sharing ?? 0 })
        : t(k('paperSummary'), counts)
  }

  const edges = summary.n_context_edges ?? 0
  if (linking > 0) message += t(k('contextLinkingSuffix'), { edges, linking })
  else if (edges > 0) message += t(k('contextSuffix'), { edges })
  const missing = papers.filter((p) => p.reason === 'not_in_kb').length
  if (missing) message += `; ${t(k('someNotInKb'), { count: missing })}`
  return message
}
