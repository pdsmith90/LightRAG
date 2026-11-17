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
const MAP = {
  nodes: [mapNode('10.1/a', '10.1/a', 'AAAA0001'), mapNode('10.1/b', '10.1/b', 'BBBB0001')],
  edges: [],
  is_truncated: false,
  paper: { mode: 'map', papers: [] }
}

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
  useGraphStore.getState().reset()
  useGraphStore.getState().triggerNodeExpand(null)
  useGraphStore.getState().setGraphDataFetchAttempted(false)
  useSettingsStore.getState().setQueryLabel('')
  __setAxiosAdapterForTests(async (config: any) => {
    const url = new URL(config.url, 'http://x')
    calls.push({ path: url.pathname, query: url.searchParams })
    if (url.pathname === '/graphs/paper/map') return respond(config, MAP, mapStatus)
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

  test('another failure of a map fetch is still retried', async () => {
    mapStatus = 500
    useSettingsStore.getState().setQueryLabel('paper+map:10.1/a')
    renderWithProviders(<Harness />)
    await waitFor(() => expect(mapRequests().length).toBe(2), { timeout: 3000 })
    expect(label()).toBe('paper+map:10.1/a')
  })
})
