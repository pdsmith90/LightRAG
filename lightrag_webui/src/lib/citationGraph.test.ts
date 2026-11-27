/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'

import type { OutsideWorks, PaperTopic, RelatedPapers, SimilarPapers } from '@/api/lightrag'
import { maxNodeSize } from '@/lib/constants'
import { MAP_CITATION_COLOR, MAP_COUPLING_COLOR, MAX_PAPERS } from '@/lib/paperFilter'
import {
  bareId,
  buildPaperMapGraph,
  fannedMissing,
  fannedReason,
  fanOut,
  findChosenPaper,
  groupTopics,
  mapEdgeCitation,
  mergeOutside,
  mergeRelated,
  mergeSimilar,
  paperByline,
  paperTitle,
  toggleChosenPaper,
  type Fanned
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

describe('the chosen papers', () => {
  test('a structure id or a doi.org link is the same paper as the bare DOI or key', () => {
    expect(bareId('doi:10.1/a')).toBe('10.1/a')
    expect(bareId('zot:AAAA1111')).toBe('AAAA1111')
    expect(bareId('https://doi.org/10.1/a')).toBe('10.1/a')
    expect(findChosenPaper(['10.1/a', 'AAAA1111'], 'doi:10.1/A')).toBe('10.1/a')
    expect(findChosenPaper(['10.1/a', 'AAAA1111'], 'zot:AAAA1111')).toBe('AAAA1111')
    expect(findChosenPaper(['10.1/a'], '10.1/b')).toBeUndefined()
  })

  test('a paper chosen by its key is found by its DOI when the keys are given', () => {
    expect(findChosenPaper(['AAAA1111'], '10.1/a')).toBeUndefined()
    expect(findChosenPaper(['AAAA1111'], '10.1/a', ['AAAA1111', 'BBBB2222'])).toBe('AAAA1111')
  })

  test('toggling adds at the end, and removes a paper chosen under another of its ids', () => {
    expect(toggleChosenPaper([], '10.1/a')).toEqual(['10.1/a'])
    expect(toggleChosenPaper(['10.1/a'], 'doi:10.1/b')).toEqual(['10.1/a', 'doi:10.1/b'])
    expect(toggleChosenPaper(['10.1/a', '10.1/b'], 'doi:10.1/a')).toEqual(['10.1/b'])
    expect(toggleChosenPaper(['AAAA1111', '10.1/b'], '10.1/a', ['AAAA1111'])).toEqual(['10.1/b'])
    expect(toggleChosenPaper(['10.1/a'], 'doi:10.1/a')).toEqual([])
  })

  test('a new paper beyond MAX_PAPERS is refused, a chosen one can still be removed', () => {
    const full = Array.from({ length: MAX_PAPERS }, (_, i) => `10.1/p${i}`)
    expect(toggleChosenPaper(full, '10.1/new')).toBeNull()
    expect(toggleChosenPaper(full, '10.1/p3')).toHaveLength(MAX_PAPERS - 1)
  })
})

describe('fanOut', () => {
  test('asks once for each paper and keeps the answers with their paper', async () => {
    const asked: string[] = []
    const fanned = await fanOut(['a', 'b'], async (paper) => {
      asked.push(paper)
      return { reason: undefined, id: paper }
    })
    expect(asked).toEqual(['a', 'b'])
    expect(fanned.total).toBe(2)
    expect(fanned.answers.map((a) => a.paper)).toEqual(['a', 'b'])
  })

  test('a failed request is left out while another answered', async () => {
    const fanned = await fanOut(['a', 'b'], async (paper) => {
      if (paper === 'a') throw new Error('boom')
      return { reason: undefined }
    })
    expect(fanned.answers.map((a) => a.paper)).toEqual(['b'])
    expect(fannedMissing(fanned)).toBe(1)
  })

  test('it fails with the first error, status and all, when every request fails', async () => {
    const error = Object.assign(new Error('nope'), { status: 501 })
    await expect(fanOut(['a', 'b'], () => Promise.reject(error))).rejects.toBe(error)
  })

  test('a reason is shown only when every answer has one, and a reason counts as missing', () => {
    const fanned: Fanned<{ reason?: string }> = {
      total: 3,
      answers: [
        { paper: 'a', answer: { reason: 'not_in_structure' } },
        { paper: 'b', answer: {} }
      ]
    }
    expect(fannedReason(fanned)).toBeUndefined()
    expect(fannedMissing(fanned)).toBe(2)
    fanned.answers[1].answer.reason = 'no_embedding'
    expect(fannedReason(fanned)).toBe('not_in_structure')
    expect(fannedReason(null)).toBeUndefined()
  })
})

describe('merging the answers of several papers', () => {
  const struct = (id: string) => ({ paper_id: `doi:${id}`, doi: id, label: id }) as any
  const fan = <T>(answers: Record<string, T>): Fanned<T> => ({
    total: Object.keys(answers).length,
    answers: Object.entries(answers).map(([paper, answer]) => ({ paper, answer }))
  })

  test('related papers appear once, from the paper whose citation has sentences, best score first', () => {
    const item = (id: string, score: number, n_contexts: number) =>
      ({ ...struct(id), score, reasons: ['cites'], citation: { n_contexts } }) as any
    const merged = mergeRelated(
      fan<RelatedPapers>({
        p: { paper: null, related: [item('10.1/shared', 1, 0), item('10.1/only-p', 3, 0)] },
        q: { paper: null, related: [item('10.1/shared', 2, 4)] }
      })
    )
    expect(merged.map((r) => [r.paper_id, r.source])).toEqual([
      ['doi:10.1/only-p', 'p'],
      ['doi:10.1/shared', 'q']
    ])
  })

  test('related papers of one paper keep the server order', () => {
    const item = (id: string, score: number) => ({ ...struct(id), score, reasons: ['cites'] }) as any
    const merged = mergeRelated(fan<RelatedPapers>({ p: { paper: null, related: [item('a', 1), item('b', 2)] } }))
    expect(merged.map((r) => r.paper_id)).toEqual(['doi:a', 'doi:b'])
  })

  test('similar papers appear once with the best similarity and every link', () => {
    const item = (id: string, similarity: number, links: string[]) => ({ ...struct(id), similarity, links }) as any
    const merged = mergeSimilar(
      fan<SimilarPapers>({
        p: { paper: null, similar: [item('a', 0.9, ['cites']), item('b', 0.8, [])] },
        q: { paper: null, similar: [item('a', 0.95, ['coupled']), item('c', 0.7, [])] }
      })
    )
    expect(merged.map((m) => [m.paper_id, m.similarity])).toEqual([
      ['doi:a', 0.95],
      ['doi:b', 0.8],
      ['doi:c', 0.7]
    ])
    expect(merged[0].links).toEqual(['cites', 'coupled'])
  })

  test('outside works appear once with the library papers that cite them, most cited first', () => {
    const work = (id: string, citers: number, also: string[]) =>
      ({ work_id: id, n_library_citers: citers, also_cited_by: also.map(struct) }) as any
    const merged = mergeOutside(
      fan<OutsideWorks>({
        p: { paper: null, works: [work('W2', 3, ['c']), work('W1', 5, ['c'])] },
        q: { paper: null, works: [work('W1', 5, ['c', 'd'])] }
      })
    )
    expect(merged.map((w) => w.work_id)).toEqual(['W1', 'W2'])
    expect(merged[0].also_cited_by.map((c) => c.paper_id)).toEqual(['doi:c', 'doi:d'])
  })
})
