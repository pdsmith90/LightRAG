/// <reference types="bun" />
import { beforeEach, describe, expect, test } from 'bun:test'
import { act, render } from '@testing-library/react'
import { UndirectedGraph } from 'graphology'
import '@/test/webglStub'

import { RawGraph, useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

const { SigmaProvider } = await import('@react-sigma/core')
const { default: GraphControl } = await import('./GraphControl')

/**
 * Which pointer events a graph registers on its sigma instance, and what they do to the store.
 * The instance is a fake that records its listeners (no WebGL here); the real picking of edges
 * by sigma is not exercised, only what GraphControl wires up around it.
 */
type Listener = (...args: any[]) => void

const makeSigma = () => {
  const listeners: Record<string, Listener> = {}
  const record = (event: string, fn: Listener) => {
    listeners[event] = fn
  }
  const forget = (event: string) => {
    delete listeners[event]
  }
  const graph = new UndirectedGraph()
  graph.addNode('a', { x: 0, y: 0 })
  graph.addNode('b', { x: 100, y: 0 })
  graph.addEdgeWithKey('ab', 'a', 'b', { type: 'curvedArrow' })
  const captor = { on: record, off: forget, removeListener: forget, isMoving: false, draggedEvents: 0, currentWheelDirection: 0 }
  const camera = { on: record, off: forget, removeListener: forget, isAnimated: () => false }
  const sigma = {
    on: record,
    off: forget,
    getGraph: () => graph,
    getCamera: () => camera,
    getMouseCaptor: () => captor,
    getTouchCaptor: () => captor,
    setSetting: () => {},
    setGraph: () => {},
    setCustomBBox: () => {},
    refresh: () => {},
    graphToViewport: (p: { x: number; y: number }) => p
  }
  return { sigma, listeners }
}

const mount = (isPaperMap: boolean, edgeEventsSetting: boolean) => {
  const raw = new RawGraph()
  raw.isPaperMap = isPaperMap
  act(() => {
    useSettingsStore.setState({ enableEdgeEvents: edgeEventsSetting })
    useGraphStore.getState().setRawGraph(raw)
    useGraphStore.getState().setGraphCounts(2, 1)
  })
  const { sigma, listeners } = makeSigma()
  render(
    <SigmaProvider value={{ sigma: sigma as never, container: document.createElement('div') }}>
      <GraphControl />
    </SigmaProvider>
  )
  return listeners
}

beforeEach(() => useGraphStore.getState().reset())

describe('GraphControl edge events', () => {
  test('a paper map picks links near the pointer, whatever the Edge Events setting says', () => {
    const listeners = mount(true, false)
    expect(listeners.clickEdge).toBeDefined()
    expect(listeners.mousemove).toBeDefined()
    expect(listeners.enterEdge).toBeUndefined()

    // a click on the stage beside the link selects it
    act(() => listeners.clickStage({ event: { x: 50, y: -9 } }))
    expect(useGraphStore.getState().selectedEdge).toBe('ab')
    // and one far from any link clears the selection, as on any graph
    act(() => listeners.clickStage({ event: { x: 400, y: 400 } }))
    expect(useGraphStore.getState().selectedEdge).toBeNull()
  })

  test('an entity graph keeps sigma\'s own picking, and nothing when the setting is off', () => {
    const on = mount(false, true)
    expect(on.enterEdge).toBeDefined()
    expect(on.clickEdge).toBeDefined()
    expect(on.mousemove).toBeUndefined()
    act(() => on.clickStage({ event: { x: 50, y: -9 } }))
    expect(useGraphStore.getState().selectedEdge).toBeNull()
  })

  test('an entity graph with the setting off registers no edge events', () => {
    const off = mount(false, false)
    expect(off.clickEdge).toBeUndefined()
    expect(off.enterEdge).toBeUndefined()
    expect(off.mousemove).toBeUndefined()
  })
})
