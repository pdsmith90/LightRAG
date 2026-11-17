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
