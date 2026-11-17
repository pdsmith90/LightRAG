/// <reference types="bun" />
import { beforeEach, describe, expect, test } from 'bun:test'
import { UndirectedGraph } from 'graphology'

import { renderWithProviders } from '@/test/render'
import { RawGraph, useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import PropertiesView from './PropertiesView'

/**
 * Whether the properties panel offers editing follows the graph on screen: a paper map is
 * read-only, an entity graph is editable, whatever the query label says.
 */
const show = (isPaperMap: boolean) => {
  const raw = new RawGraph()
  raw.isPaperMap = isPaperMap
  raw.nodes = [
    {
      id: 'N1',
      labels: ['N1'],
      properties: { entity_type: 'Field', title: 'A title' },
      size: 10,
      x: 0,
      y: 0,
      color: '#fff',
      degree: 0
    }
  ] as any
  raw.nodeIdMap = { N1: 0 }
  const sigma = new UndirectedGraph()
  sigma.addNode('N1', { label: 'N1', x: 0, y: 0, size: 10, color: '#fff' })
  const state = useGraphStore.getState()
  state.setSigmaGraph(sigma)
  state.setRawGraph(raw)
  state.setSelectedNode('N1')
}

const pencils = (container: HTMLElement) => container.querySelectorAll('svg.lucide-pencil').length

beforeEach(() => {
  // editing is hidden while the pipeline writes; these tests are about the graph
  useBackendState.setState({ pipelineBusy: false })
  useGraphStore.getState().reset()
  // no fetch: the graph below is the one on screen
  useGraphStore.getState().setGraphDataFetchAttempted(true)
})

// A selected link between two papers of a map (or of an entity graph).
const showLink = (isPaperMap: boolean, properties: Record<string, unknown>) => {
  const raw = new RawGraph()
  raw.isPaperMap = isPaperMap
  raw.nodes = ['A', 'B'].map((id) => ({
    id,
    labels: [id === 'A' ? 'Okafor 2010' : 'Varga 2014'],
    properties: { entity_type: 'Field' },
    size: 10,
    x: 0,
    y: 0,
    color: '#fff',
    degree: 1
  })) as any
  raw.nodeIdMap = { A: 0, B: 1 }
  const sigma = new UndirectedGraph()
  sigma.addNode('A', { label: 'A', x: 0, y: 0, size: 10, color: '#fff' })
  sigma.addNode('B', { label: 'B', x: 0, y: 0, size: 10, color: '#fff' })
  const dynamicId = sigma.addEdge('A', 'B', {})
  raw.edges = [
    { id: 'A-B', dynamicId, type: 'DIRECTED', source: 'A', target: 'B', properties }
  ] as any
  raw.edgeIdMap = { 'A-B': 0 }
  raw.edgeDynamicIdMap = { [dynamicId]: 0 }
  const state = useGraphStore.getState()
  state.setSigmaGraph(sigma)
  state.setRawGraph(raw)
  state.setSelectedEdge(dynamicId)
  useSettingsStore.getState().setQueryLabel(isPaperMap ? 'paper+map:10.1/a' : 'GRACE')
}

describe('PropertiesView on a selected paper-map link', () => {
  test('a citation says its kind and its direction in words', () => {
    showLink(true, { keywords: 'cites', direction: 'source cites target', weight: 1 })
    const { getByText, queryByText } = renderWithProviders(<PropertiesView />)
    expect(getByText('Citation')).toBeTruthy()
    expect(getByText('Okafor 2010 cites Varga 2014')).toBeTruthy()
    expect(queryByText('Common references')).toBeNull()
  })

  test('citations in both directions are named as such', () => {
    showLink(true, { keywords: 'cite each other', direction: 'both' })
    const { getByText } = renderWithProviders(<PropertiesView />)
    expect(getByText('Citation both ways')).toBeTruthy()
    expect(getByText('Okafor 2010 and Varga 2014 cite each other')).toBeTruthy()
  })

  test('shared references have no direction and say how many are shared', () => {
    showLink(true, { keywords: 'coupled', shared_refs: 7, coupling_score: 0.4 })
    const { getByText, getAllByText } = renderWithProviders(<PropertiesView />)
    expect(getAllByText('Shared references').length).toBeGreaterThan(0)
    expect(getByText(/^None: the two papers cite some of the same references/)).toBeTruthy()
    expect(getByText('Common references')).toBeTruthy()
    expect(getByText('Common references').parentElement?.textContent).toContain('7')
  })

  test('a citation that is also coupling shows both kinds', () => {
    showLink(true, { keywords: 'cites, coupled', direction: 'source cites target', shared_refs: 3 })
    const { getByText } = renderWithProviders(<PropertiesView />)
    expect(getByText('Citation · Shared references')).toBeTruthy()
    expect(getByText('Okafor 2010 cites Varga 2014')).toBeTruthy()
  })

  test('an entity graph relationship gets no link rows', () => {
    showLink(false, { keywords: 'cites', direction: 'source cites target' })
    const { queryByText } = renderWithProviders(<PropertiesView />)
    expect(queryByText('Okafor 2010 cites Varga 2014')).toBeNull()
    expect(queryByText('Direction')).toBeNull()
  })
})

describe('PropertiesView on a paper map', () => {
  test('a paper map is read-only', () => {
    show(true)
    useSettingsStore.getState().setQueryLabel('paper+map:10.1/a')
    const { container, getByText } = renderWithProviders(<PropertiesView />)
    expect(getByText('A title')).toBeTruthy()
    expect(pencils(container)).toBe(0)
  })

  test('an entity graph stays editable under a map label', () => {
    // the map fetch failed: the label names a map, an entity graph is on screen
    show(false)
    useSettingsStore.getState().setQueryLabel('paper+map:10.1/a')
    const { container } = renderWithProviders(<PropertiesView />)
    expect(pencils(container)).toBe(1) // entity_type
  })

  test('a map stays read-only while its label changes to an entity', () => {
    show(true)
    useSettingsStore.getState().setQueryLabel('GRACE')
    const { container } = renderWithProviders(<PropertiesView />)
    expect(pencils(container)).toBe(0)
  })
})
