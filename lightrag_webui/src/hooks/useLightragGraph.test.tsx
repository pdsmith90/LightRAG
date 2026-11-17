/// <reference types="bun" />
import { afterAll, beforeEach, describe, expect, test } from 'bun:test'
import { screen, waitFor, act } from '@testing-library/react'
import { UndirectedGraph } from 'graphology'
import { Toaster, toast } from 'sonner'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import { RawGraph, useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import useLightragGraph from '@/hooks/useLightragGraph'
import {
  __resetPaperContextForTests,
  MAP_CITATION_COLOR,
  MAP_COUPLING_COLOR,
  raiseMapLimit
} from '@/lib/paperFilter'

/**
 * The graph hook, driven through the real stores and the real request layer (the HTTP
 * adapter is stubbed). It covers what the paper map changes: the map is marked on the graph
 * it loads, expanding a map node adds a paper instead of expanding an entity, what is on
 * screen (not the label) decides that, and a server without the map is told once.
 */
const strings = en.graphPanel.paperFilter

const Harness = () => {
  useLightragGraph()
  return <Toaster />
}

type Call = { path: string; query: URLSearchParams }
let calls: Call[] = []
let mapStatus = 200

const mapNode = (id: string, doi: string, key: string) => ({
  id,
  labels: [doi],
  properties: { entity_type: 'Field', doi, attachment_key: key, paper_id: `doi:${doi}` }
})
const mapEdge = (source: string, target: string, properties: Record<string, unknown>) => ({
  id: `${source}-${target}`,
  type: 'DIRECTED',
  source,
  target,
  properties
})
const MAP: Record<string, any> = {
  nodes: [mapNode('10.1/a', '10.1/a', 'AAAA0001'), mapNode('10.1/b', '10.1/b', 'BBBB0001')],
  edges: [],
  is_truncated: false,
  paper: { mode: 'map', papers: [] }
}
let mapBody: Record<string, any> = MAP

const entityRequests = () => calls.filter((c) => c.path === '/graphs')
const mapRequests = () => calls.filter((c) => c.path === '/graphs/paper/map')
const label = () => useSettingsStore.getState().queryLabel
const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const respond = (config: any, data: unknown, status = 200) => {
  if (status >= 400) {
    const error: any = new Error(`Request failed with status code ${status}`)
    error.config = config
    error.response = { status, statusText: 'x', data: { detail: 'x' }, headers: {}, config }
    return Promise.reject(error)
  }
  return Promise.resolve({ data, status, statusText: 'OK', headers: {}, config })
}

beforeEach(() => {
  toast.dismiss() // toasts outlive a test's render
  calls = []
  mapStatus = 200
  mapBody = MAP
  __resetPaperContextForTests()
  useGraphStore.getState().reset()
  useGraphStore.getState().triggerNodeExpand(null)
  useGraphStore.getState().setGraphDataFetchAttempted(false)
  useSettingsStore.getState().setQueryLabel('')
  __setAxiosAdapterForTests(async (config: any) => {
    const url = new URL(config.url, 'http://x')
    calls.push({ path: url.pathname, query: url.searchParams })
    if (url.pathname === '/graphs/paper/map') return respond(config, mapBody, mapStatus)
    return respond(config, { nodes: [], edges: [], is_truncated: false })
  })
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

const loadedMap = () => waitFor(() => expect(useGraphStore.getState().rawGraph?.isPaperMap).toBe(true))

// An entity graph on screen, as if a map fetch had failed after it.
const showEntityGraph = () => {
  const raw = new RawGraph()
  raw.nodes = [
    { id: 'GRACE', labels: ['GRACE'], properties: { entity_id: 'GRACE' }, size: 10, x: 0, y: 0, color: '#fff', degree: 0 }
  ] as any
  raw.nodeIdMap = { GRACE: 0 }
  const sigma = new UndirectedGraph()
  sigma.addNode('GRACE', { label: 'GRACE', x: 0, y: 0, size: 10, color: '#fff' })
  useGraphStore.getState().setSigmaGraph(sigma)
  useGraphStore.getState().setRawGraph(raw)
  useGraphStore.getState().setGraphDataFetchAttempted(true)
}

describe('paper map in the graph hook', () => {
  test('a loaded map is marked as one, and an entity graph is not', async () => {
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()
    expect(mapRequests().length).toBe(1)
    expect(mapRequests()[0].query.getAll('paper')).toEqual(['AAAA0001'])
  })

  test('expanding a map node adds a paper and never expands an entity', async () => {
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()

    // the paper chosen by its key is on the map under its DOI: already chosen
    await act(async () => useGraphStore.getState().triggerNodeExpand('10.1/a'))
    await wait(50)
    expect(label()).toBe('paper+map:AAAA0001')

    await act(async () => useGraphStore.getState().triggerNodeExpand('10.1/b'))
    await waitFor(() => expect(label()).toBe('paper+map:AAAA0001 10.1/b'))
    // the new map is fetched with both papers; no entity graph was asked for
    await waitFor(() => expect(mapRequests().length).toBe(2))
    expect(mapRequests()[1].query.getAll('paper')).toEqual(['AAAA0001', '10.1/b'])
    expect(entityRequests()).toEqual([])
  })

  test('an entity graph still on screen expands as an entity graph under a map label', async () => {
    // the map fetch failed (and was not retried yet); the label names a map
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    showEntityGraph()
    renderWithProviders(<Harness />)

    await act(async () => useGraphStore.getState().triggerNodeExpand('GRACE'))
    await waitFor(() => expect(entityRequests().length).toBe(1))
    expect(entityRequests()[0].query.get('label')).toBe('GRACE')
    expect(label()).toBe('paper+map:AAAA0001')
  })

  test('a map stays a map while its label is changing', async () => {
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()

    // the label moves to an entity; the map is still on screen until that loads
    useGraphStore.getState().setGraphDataFetchAttempted(true)
    act(() => useSettingsStore.getState().setQueryLabel('GRACE'))
    await act(async () => useGraphStore.getState().triggerNodeExpand('10.1/b'))
    await wait(50)
    expect(entityRequests()).toEqual([])
    expect(label()).toBe('GRACE')
  })

  test.each([501, 404])(
    'a server answering %d to a map label says so once, does not retry and shows the papers',
    async (status) => {
      mapStatus = status
      useSettingsStore.getState().setQueryLabel('paper+map:10.1/a AAAA0001')
      renderWithProviders(<Harness />)

      await waitFor(() => expect(label()).toBe('paper:10.1/a AAAA0001'))
      expect(await screen.findByText(strings.mapUnavailable)).toBeTruthy()
      // the plain paper view is requested; the map is not asked again
      await waitFor(() => expect(calls.some((c) => c.path === '/graphs/paper')).toBe(true))
      await wait(1500) // past the first retry delay
      expect(mapRequests().length).toBe(1)
      expect(screen.getAllByText(strings.mapUnavailable).length).toBe(1)
    }
  )

  test('a map asks for its own paper limit, not the graph viewer\'s node limit', async () => {
    useSettingsStore.getState().setGraphMaxNodes(1000)
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()
    expect(mapRequests()[0].query.get('max_nodes')).toBe('150')
  })

  test('the limit raised for this map is the one asked for, and only for this map', async () => {
    raiseMapLimit('paper+map:AAAA0001', 1000)
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()
    expect(mapRequests()[0].query.get('max_nodes')).toBe('300')
    // the shown map keeps the server's summary for the control that asks for more
    expect(useGraphStore.getState().rawGraph?.paperSummary?.mode).toBe('map')
    act(() => {
      useGraphStore.getState().setGraphDataFetchAttempted(false)
      useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001 10.1/b')
      useGraphStore.getState().incrementGraphDataVersion()
    })
    await waitFor(() => expect(mapRequests().length).toBe(2))
    expect(mapRequests()[1].query.get('max_nodes')).toBe('150')
  })

  test('citations are arrows from the citing paper and coupling links plain, in two colours', async () => {
    mapBody = {
      ...MAP,
      nodes: [...MAP.nodes, mapNode('10.1/c', '10.1/c', 'CCCC0001'), mapNode('10.1/d', '10.1/d', 'DDDD0001')],
      edges: [
        mapEdge('10.1/a', '10.1/b', { keywords: 'cites', direction: 'source cites target' }),
        mapEdge('10.1/a', '10.1/c', { keywords: 'coupled', shared_refs: 7 }),
        mapEdge('10.1/b', '10.1/c', { keywords: 'cite each other', direction: 'both' })
      ]
    }
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    renderWithProviders(<Harness />)
    await loadedMap()
    await waitFor(() => expect(useGraphStore.getState().sigmaGraph?.size).toBe(3))
    const graph = useGraphStore.getState().sigmaGraph!
    const edge = (a: string, b: string) => graph.edge(a, b) as string
    const cites = graph.getEdgeAttributes(edge('10.1/a', '10.1/b'))
    expect(cites.color).toBe(MAP_CITATION_COLOR)
    expect(cites.type).toBe('curvedArrow')
    // the arrow points at the cited paper: the edge runs from the citing one
    expect([graph.source(edge('10.1/a', '10.1/b')), graph.target(edge('10.1/a', '10.1/b'))]).toEqual([
      '10.1/a',
      '10.1/b'
    ])
    const coupled = graph.getEdgeAttributes(edge('10.1/a', '10.1/c'))
    expect(coupled.color).toBe(MAP_COUPLING_COLOR)
    expect(coupled.type).toBeUndefined()
    const mutual = graph.getEdgeAttributes(edge('10.1/b', '10.1/c'))
    expect(mutual.color).toBe(MAP_CITATION_COLOR)
    expect(mutual.type).toBeUndefined()
  })

  test('an entity graph keeps its edges as they were', async () => {
    useSettingsStore.getState().setQueryLabel('GRACE')
    __setAxiosAdapterForTests(async (config: any) =>
      respond(config, {
        nodes: [
          { id: 'A', labels: ['A'], properties: { entity_type: 'x' } },
          { id: 'B', labels: ['B'], properties: { entity_type: 'x' } }
        ],
        edges: [{ id: 'A-B', type: 'DIRECTED', source: 'A', target: 'B', properties: { keywords: 'cites', direction: 'source cites target' } }],
        is_truncated: false
      })
    )
    renderWithProviders(<Harness />)
    await waitFor(() => expect(useGraphStore.getState().sigmaGraph?.size).toBe(1))
    const graph = useGraphStore.getState().sigmaGraph!
    const attrs = graph.getEdgeAttributes(graph.edge('A', 'B') as string)
    expect(attrs.type).toBeUndefined()
    expect(attrs.color).toBeUndefined()
  })

  test('the truncation toast means papers were left out; thinned links only reach the summary', async () => {
    const summary = {
      mode: 'map',
      papers: [],
      n_papers: 300,
      shown_papers: 300,
      max_nodes: 300,
      max_nodes_cap: 1000,
      n_links: 900,
      shown_links: 800,
      links_thinned: true
    }
    mapBody = { ...MAP, is_truncated: false, paper: summary }
    useSettingsStore.getState().setQueryLabel('paper+map:AAAA0001')
    const { unmount } = renderWithProviders(<Harness />)
    expect(await screen.findByText(/800 of the 900 links between them shown/)).toBeTruthy()
    expect(screen.queryByText('Graph data is truncated to Max Nodes')).toBeNull()
    unmount()

    toast.dismiss()
    mapBody = { ...MAP, is_truncated: true, paper: { ...summary, n_papers: 612, shown_papers: 300 } }
    useGraphStore.getState().reset()
    useGraphStore.getState().setGraphDataFetchAttempted(false)
    useSettingsStore.getState().setQueryLabel('paper+map:10.1/b')
    renderWithProviders(<Harness />)
    // a map says what it left out in its own words: how many papers, and how to see more
    expect(await screen.findByText(/612 papers found, 2 shown/)).toBeTruthy()
    expect(await screen.findByText(/"Show more papers" raises the limit/)).toBeTruthy()
    expect(screen.queryByText('Graph data is truncated to Max Nodes')).toBeNull()
  })

  test('an entity graph that was cut short keeps the generic truncation toast', async () => {
    __setAxiosAdapterForTests(async (config: any) =>
      respond(config, {
        nodes: [{ id: 'A', labels: ['A'], properties: { entity_type: 'x' } }],
        edges: [],
        is_truncated: true
      })
    )
    useSettingsStore.getState().setQueryLabel('GRACE')
    renderWithProviders(<Harness />)
    expect(await screen.findByText('Graph data is truncated to Max Nodes')).toBeTruthy()
  })

  test('another failure of a map fetch is still retried', async () => {
    mapStatus = 500
    useSettingsStore.getState().setQueryLabel('paper+map:10.1/a')
    renderWithProviders(<Harness />)
    await waitFor(() => expect(mapRequests().length).toBe(2), { timeout: 3000 })
    expect(label()).toBe('paper+map:10.1/a')
  })
})
