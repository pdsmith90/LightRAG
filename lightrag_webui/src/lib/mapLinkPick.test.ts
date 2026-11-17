/// <reference types="bun" />
import { beforeEach, describe, expect, test } from 'bun:test'
import { UndirectedGraph } from 'graphology'
import '@/test/webglStub'

import { useGraphStore } from '@/stores/graph'

// The module under test reads @sigma/edge-curve, which needs the stub first.
const { DEFAULT_EDGE_CURVATURE } = await import('@sigma/edge-curve')
const { MAP_LINK_PICK_RADIUS, distanceToLink, mapLinkEvents, nearestLink } = await import(
  './mapLinkPick'
)

/**
 * What sigma gives a map: a link is picked only on its own one or two pixels. These tests cover
 * the proximity picking the map adds, on a fake sigma whose viewport is the graph's own
 * coordinates. They do not cover that the curve built here is the one the GPU draws; that was
 * checked against sigma's picking buffer in a browser.
 */
const makeSigma = (hidden: string[] = []) => {
  const graph = new UndirectedGraph()
  graph.addNode('a', { x: 0, y: 0 })
  graph.addNode('b', { x: 100, y: 0 })
  graph.addNode('c', { x: 0, y: 200 })
  graph.addNode('d', { x: 100, y: 200 })
  graph.addEdgeWithKey('ab', 'a', 'b', { type: 'curvedArrow', hidden: hidden.includes('ab') })
  graph.addEdgeWithKey('cd', 'c', 'd', { type: 'arrow' })
  return {
    getGraph: () => graph,
    graphToViewport: (p: { x: number; y: number }) => ({ x: p.x, y: p.y })
  }
}

describe('distanceToLink', () => {
  test('a straight link is measured to its segment, ends included', () => {
    expect(distanceToLink({ x: 50, y: 3 }, { x: 0, y: 0 }, { x: 100, y: 0 }, 0)).toBe(3)
    expect(distanceToLink({ x: -4, y: 3 }, { x: 0, y: 0 }, { x: 100, y: 0 }, 0)).toBe(5)
  })

  test('a curved link is measured to the curve, which bulges to the left of source to target', () => {
    // y points down on the screen: the curve of (0,0) to (100,0) rises to y = -12.5
    expect(DEFAULT_EDGE_CURVATURE).toBe(0.25)
    expect(distanceToLink({ x: 50, y: -12.5 }, { x: 0, y: 0 }, { x: 100, y: 0 })).toBeLessThan(0.01)
    expect(distanceToLink({ x: 50, y: 0 }, { x: 0, y: 0 }, { x: 100, y: 0 })).toBeGreaterThan(10)
    // the other direction bulges the other way
    expect(distanceToLink({ x: 50, y: 12.5 }, { x: 100, y: 0 }, { x: 0, y: 0 })).toBeLessThan(0.01)
  })
})

describe('nearestLink', () => {
  test('a point within reach of a curve picks it, one beside the chord does not', () => {
    const sigma = makeSigma()
    expect(nearestLink(sigma, { x: 50, y: -10 })).toBe('ab')
    expect(nearestLink(sigma, { x: 50, y: -12.5 - MAP_LINK_PICK_RADIUS + 1 })).toBe('ab')
    expect(nearestLink(sigma, { x: 50, y: -12.5 - MAP_LINK_PICK_RADIUS - 2 })).toBeNull()
  })

  test('an arrow that sigma draws straight is measured as a line', () => {
    const sigma = makeSigma()
    expect(nearestLink(sigma, { x: 50, y: 200 + MAP_LINK_PICK_RADIUS - 1 })).toBe('cd')
    expect(nearestLink(sigma, { x: 50, y: 200 - 20 })).toBeNull()
  })

  test('a link the graph hides is not picked, and the nearest of two wins', () => {
    expect(nearestLink(makeSigma(['ab']), { x: 50, y: -12 })).toBeNull()
    const sigma = makeSigma()
    sigma.getGraph().addNode('e', { x: 100, y: -8 })
    sigma.getGraph().addEdgeWithKey('ae', 'a', 'e', { type: 'arrow' })
    // the straight link runs from (0,0) to (100,-8), the curve of 'ab' rises above it
    expect(nearestLink(sigma, { x: 50, y: -3 })).toBe('ae')
    expect(nearestLink(sigma, { x: 50, y: -11 })).toBe('ab')
  })
})

describe('mapLinkEvents', () => {
  beforeEach(() => useGraphStore.getState().reset())
  const events = () => mapLinkEvents(makeSigma(), () => useGraphStore.getState())

  test('a click near a link selects it and clears a selected node; far from any it clears', () => {
    const { clickStage } = events()
    useGraphStore.getState().setSelectedNode('a')
    clickStage({ event: { x: 50, y: -9 } })
    expect(useGraphStore.getState().selectedEdge).toBe('ab')
    expect(useGraphStore.getState().selectedNode).toBeNull()

    clickStage({ event: { x: 500, y: 500 } })
    expect(useGraphStore.getState().selectedEdge).toBeNull()
  })

  test('the pointer near a link highlights it, away from it lets go', () => {
    const { mousemove } = events()
    mousemove({ x: 50, y: -9 })
    expect(useGraphStore.getState().focusedEdge).toBe('ab')
    mousemove({ x: 500, y: 500 })
    expect(useGraphStore.getState().focusedEdge).toBeNull()
  })

  test('a hovered node or a drag keeps the highlight off', () => {
    const { mousemove } = events()
    useGraphStore.getState().setFocusedNode('a')
    mousemove({ x: 50, y: -9 })
    expect(useGraphStore.getState().focusedEdge).toBeNull()

    useGraphStore.getState().setFocusedNode(null)
    mousemove({ x: 50, y: -9, original: new MouseEvent('mousemove', { buttons: 1 }) })
    expect(useGraphStore.getState().focusedEdge).toBeNull()
    mousemove({ x: 50, y: -9, original: new MouseEvent('mousemove', { buttons: 0 }) })
    expect(useGraphStore.getState().focusedEdge).toBe('ab')
  })
})
