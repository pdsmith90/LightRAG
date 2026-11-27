/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'

import type { PaperTopic } from '@/api/lightrag'
import { maxNodeSize } from '@/lib/constants'
import { MAP_CITATION_COLOR, MAP_COUPLING_COLOR } from '@/lib/paperFilter'
import {
  buildPaperMapGraph,
  groupTopics,
  mapEdgeCitation,
  paperByline,
  paperTitle
} from './citationGraph'

const node = (id: string, hop: number, field: string, cited: number) => ({
  id,
  labels: [`Label ${id}`],
  properties: { entity_type: field, hop, cited_by_library: cited }
})

const edge = (source: string, target: string, properties: Record<string, any>) => ({
  id: `${source}-${target}`,
  source,
  target,
  properties
})

describe('paperTitle and paperByline', () => {
  test('the title falls back to the label, then to the id', () => {
    expect(paperTitle({ title: 'A title', label: 'Li 2020', paper_id: 'doi:10.1/a' })).toBe('A title')
    expect(paperTitle({ title: null, label: 'Li 2020', paper_id: 'doi:10.1/a' })).toBe('Li 2020')
    expect(paperTitle({ title: null, label: '', paper_id: 'doi:10.1/a' })).toBe('doi:10.1/a')
  })

  test('the byline joins the label and the venue and skips an empty one', () => {
    expect(paperByline({ label: 'Li 2020', venue: 'Nature' })).toBe('Li 2020 · Nature')
    expect(paperByline({ label: 'Li 2020', venue: null })).toBe('Li 2020')
  })
})

describe('mapEdgeCitation', () => {
  test('a one-way citation runs from the source to the target', () => {
    const link = { source: 'a', target: 'b', properties: { direction: 'source cites target' } }
    expect(mapEdgeCitation(link)).toEqual({ citing: 'a', cited: 'b', both: false })
  })

  test('papers that cite each other are flagged, and shared references alone are no citation', () => {
    expect(mapEdgeCitation({ source: 'a', target: 'b', properties: { direction: 'both' } })?.both).toBe(true)
    expect(mapEdgeCitation({ source: 'a', target: 'b', properties: { shared_refs: 4 } })).toBeNull()
  })
})

describe('buildPaperMapGraph', () => {
  const data = {
    nodes: [node('a', 0, 'Physics', 0), node('b', 1, 'Physics', 40), node('c', 1, 'Chemistry', 10)],
    edges: [
      edge('a', 'b', { direction: 'source cites target', weight: 1.5 }),
      edge('b', 'c', { shared_refs: 3, weight: 0.4 }),
      edge('c', 'b', { shared_refs: 3 }),
      edge('a', 'missing', { direction: 'source cites target' })
    ]
  }

  test('keeps the papers, drops dangling and repeated links', () => {
    const { graph } = buildPaperMapGraph(data)
    expect(graph.order).toBe(3)
    expect(graph.size).toBe(2)
  })

  test('the chosen paper is the largest and always labelled; the rest are sized by citers', () => {
    const { graph } = buildPaperMapGraph(data)
    expect(graph.getNodeAttribute('a', 'size')).toBe(maxNodeSize + 4)
    expect(graph.getNodeAttribute('a', 'forceLabel')).toBe(true)
    expect(graph.getNodeAttribute('b', 'size')).toBeGreaterThan(graph.getNodeAttribute('c', 'size'))
    expect(graph.getNodeAttribute('b', 'label')).toBe('Label b')
  })

  test('draws citations as arrows and shared references plain, in the Knowledge Graph colours', () => {
    const { graph } = buildPaperMapGraph(data)
    expect(graph.getEdgeAttributes('a-b')).toMatchObject({ color: MAP_CITATION_COLOR, type: 'curvedArrow' })
    const coupling = graph.getEdgeAttributes('b-c')
    expect(coupling.color).toBe(MAP_COUPLING_COLOR)
    expect(coupling.type).toBeUndefined()
  })

  test('colours by field, one legend entry per field, and lays every paper out', () => {
    const { graph, fields } = buildPaperMapGraph(data)
    expect([...fields.keys()].sort()).toEqual(['Chemistry', 'Physics'])
    expect(graph.getNodeAttribute('a', 'color')).toBe(graph.getNodeAttribute('b', 'color'))
    expect(graph.getNodeAttribute('a', 'color')).not.toBe(graph.getNodeAttribute('c', 'color'))
    graph.forEachNode((_, attributes) => {
      expect(Number.isFinite(attributes.x)).toBe(true)
      expect(Number.isFinite(attributes.y)).toBe(true)
    })
  })

  test('a single paper with no links builds without a layout run', () => {
    const { graph } = buildPaperMapGraph({ nodes: [node('a', 0, 'Physics', 0)], edges: [] })
    expect(graph.order).toBe(1)
    expect(graph.size).toBe(0)
  })
})

describe('groupTopics', () => {
  const topic = (id: string, path: string, papers: number): PaperTopic => ({
    topic_id: id,
    name: path.split(' > ').pop() as string,
    level: 'subfield',
    path,
    n_papers: papers,
    n_primary: papers
  })

  test('groups by the path above, largest group and largest topic first', () => {
    const groups = groupTopics([
      topic('s:1', 'Physical > Earth > Geophysics', 10),
      topic('s:2', 'Physical > Earth > Space', 30),
      topic('s:3', 'Life > Biology > Ecology', 25),
      topic('s:4', 'Life > Biology > Genetics', 5)
    ])
    expect(groups.map((g) => g.path)).toEqual(['Physical > Earth', 'Life > Biology'])
    expect(groups[0].topics.map((t) => t.name)).toEqual(['Space', 'Geophysics'])
  })

  test('a top-level entry (a domain) has an empty group path', () => {
    const groups = groupTopics([topic('d:1', 'Physical', 100)])
    expect(groups).toEqual([{ path: '', topics: [expect.objectContaining({ topic_id: 'd:1' })] }])
  })
})
