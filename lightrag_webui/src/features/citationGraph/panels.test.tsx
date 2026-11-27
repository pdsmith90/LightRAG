/// <reference types="bun" />
import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { fireEvent, screen } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import ContextsPanel from './ContextsPanel'
import OutsidePanel from './OutsidePanel'
import RelatedPanel from './RelatedPanel'
import SimilarPanel from './SimilarPanel'
import TopicsPanel from './TopicsPanel'

/**
 * The panels beside the paper map and the topic browser, driven against the real axios
 * instance with its HTTP adapter stubbed. Each answers a route by its path; `requests`
 * records the URLs asked for.
 */
const strings = en.citationGraph

const struct = (id: string, title: string, extra: Record<string, unknown> = {}) => ({
  paper_id: `doi:${id}`,
  doi: id,
  label: `Author ${id.split('/')[1]}`,
  title,
  year: 2020,
  venue: 'Journal',
  authors: [],
  field: 'Physics',
  topic: 'Optics',
  in_kb: true,
  keys: [],
  cited_by_library: 3,
  cites_library: 1,
  cited_by_count: 10,
  tldr: null,
  ...extra
})

let requests: string[] = []
let answers: Record<string, unknown> = {}

beforeEach(() => {
  requests = []
  answers = {}
  __setAxiosAdapterForTests(async (config: any) => {
    requests.push(config.url)
    const url = new URL(config.url, 'http://x')
    // A function answers by the request (the paper asked about), and may throw.
    const answer = answers[url.pathname]
    const data = typeof answer === 'function' ? (answer as (url: URL) => unknown)(url) : answer
    return { data, status: 200, statusText: 'OK', headers: {}, config }
  })
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

describe('RelatedPanel', () => {
  const related = (relation: string, extra: Record<string, unknown> = {}) => ({
    ...struct('10.1/ref', 'A reference'),
    score: 1,
    reasons: [relation],
    citation: { source: 'both', is_influential: true, intents: ['background'], n_contexts: 2 },
    ...extra
  })

  test('lists the references first and opens a paper by its structure id', async () => {
    answers['/graph/papers/related'] = { paper: struct('10.1/p', 'P'), totals: { cites: 5 }, related: [related('cites')] }
    const onOpen = mock(() => {})
    renderWithProviders(<RelatedPanel papers={['10.1/p']} onOpen={onOpen} onPickCitation={() => {}} />)

    fireEvent.click(await screen.findByRole('button', { name: 'A reference' }))
    expect(onOpen).toHaveBeenCalledWith('doi:10.1/ref')
    expect(requests[0]).toContain('relations=cites')
    expect(requests[0]).toContain('paper=10.1%2Fp')
    expect(screen.getByText('Showing 1 of 5')).toBeTruthy()
  })

  test('a reference with stored sentences opens them as the paper citing the reference', async () => {
    answers['/graph/papers/related'] = { paper: struct('10.1/p', 'P'), related: [related('cites')] }
    const onPickCitation = mock(() => {})
    renderWithProviders(<RelatedPanel papers={['10.1/p']} onOpen={() => {}} onPickCitation={onPickCitation} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Read citing sentences (2)' }))
    expect(onPickCitation).toHaveBeenCalledWith({ citing: '10.1/p', cited: 'doi:10.1/ref' })
    expect(screen.getByText('Influential')).toBeTruthy()
  })

  test('another relation asks again for that relation', async () => {
    answers['/graph/papers/related'] = { paper: struct('10.1/p', 'P'), related: [related('coupled', { shared_refs: 8 })] }
    renderWithProviders(<RelatedPanel papers={['10.1/p']} onOpen={() => {}} onPickCitation={() => {}} />)
    await screen.findByText('A reference')

    fireEvent.click(screen.getByRole('button', { name: strings.relation.coupled }))
    expect(await screen.findByText('Shared references: 8')).toBeTruthy()
    expect(requests[requests.length - 1]).toContain('relations=coupled')
  })

  test('says why when the paper is not in the structure', async () => {
    answers['/graph/papers/related'] = { paper: null, asked: 'x', reason: 'not_in_structure' }
    renderWithProviders(<RelatedPanel papers={['x']} onOpen={() => {}} onPickCitation={() => {}} />)
    expect(await screen.findByText(strings.reason.not_in_structure)).toBeTruthy()
  })
})

describe('SimilarPanel', () => {
  test('shows the similarity and how the library links each paper, or that it does not', async () => {
    answers['/graph/papers/similar'] = {
      paper: struct('10.1/p', 'P'),
      similar: [
        { ...struct('10.1/a', 'Linked one'), similarity: 0.9569, links: ['cites', 'coupled'] },
        { ...struct('10.1/b', 'Unlinked one'), similarity: 0.8, links: [] }
      ]
    }
    const onOpen = mock(() => {})
    renderWithProviders(<SimilarPanel papers={['10.1/p']} onOpen={onOpen} />)

    expect(await screen.findByText('Similarity 96%')).toBeTruthy()
    expect(screen.getByText(strings.relation.cites)).toBeTruthy()
    expect(screen.getByText(strings.similar.unlinked)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Unlinked one' }))
    expect(onOpen).toHaveBeenCalledWith('doi:10.1/b')
  })

  test('a paper without an embedding gets the reason, not a crash', async () => {
    answers['/graph/papers/similar'] = { paper: struct('10.1/p', 'P'), in_kb: false, similar: [], reason: 'no_embedding' }
    renderWithProviders(<SimilarPanel papers={['10.1/p']} onOpen={() => {}} />)
    expect(await screen.findByText(strings.reason.no_embedding)).toBeTruthy()
  })
})

describe('ContextsPanel', () => {
  const pair = { citing: 'doi:10.1/a', cited: 'doi:10.1/b' }

  test('asks to pick a citation while there is none, and asks for nothing', () => {
    renderWithProviders(<ContextsPanel pair={null} onSwap={() => {}} />)
    expect(screen.getByText(strings.citation.pick)).toBeTruthy()
    expect(requests).toEqual([])
  })

  test('shows the citing sentences with their intents', async () => {
    answers['/graph/papers/citation'] = {
      citing: struct('10.1/a', 'Citing paper'),
      cited: struct('10.1/b', 'Cited paper'),
      edge: { source: 'both', is_influential: false, intents: ['methodology'], n_contexts: 1 },
      contexts: [{ ordinal: 1, text: 'Prior work (Cited, 2020) did this.', intents: ['background'] }]
    }
    renderWithProviders(<ContextsPanel pair={pair} onSwap={() => {}} />)

    expect(await screen.findByText('Prior work (Cited, 2020) did this.')).toBeTruthy()
    expect(screen.getByText('Citing paper cites Cited paper')).toBeTruthy()
    expect(screen.getByText('background')).toBeTruthy()
    expect(requests[0]).toContain('citing=doi%3A10.1%2Fa')
    expect(requests[0]).toContain('cited=doi%3A10.1%2Fb')
  })

  test('a citation that runs the other way offers to swap', async () => {
    answers['/graph/papers/citation'] = {
      citing: struct('10.1/a', 'Citing paper'),
      cited: struct('10.1/b', 'Cited paper'),
      edge: null,
      contexts: [],
      reason: 'no_citation',
      cited_cites_citing: true
    }
    const onSwap = mock(() => {})
    renderWithProviders(<ContextsPanel pair={pair} onSwap={onSwap} />)

    fireEvent.click(await screen.findByRole('button', { name: strings.citation.swap }))
    expect(onSwap).toHaveBeenCalledTimes(1)
  })

  test('with no citation either way it says so, without a headline or a swap', async () => {
    answers['/graph/papers/citation'] = {
      citing: struct('10.1/a', 'Citing paper'),
      cited: struct('10.1/b', 'Cited paper'),
      edge: null,
      contexts: [],
      reason: 'no_citation',
      cited_cites_citing: false
    }
    renderWithProviders(<ContextsPanel pair={pair} onSwap={() => {}} />)

    expect(await screen.findByText(strings.reason.no_citation)).toBeTruthy()
    expect(screen.queryByRole('button', { name: strings.citation.swap })).toBeNull()
    expect(screen.queryByText('Citing paper cites Cited paper')).toBeNull()
  })

  test('without a sentences table the edge facts still show', async () => {
    answers['/graph/papers/citation'] = {
      citing: struct('10.1/a', 'Citing paper'),
      cited: struct('10.1/b', 'Cited paper'),
      edge: { source: 'openalex', is_influential: true, intents: [], n_contexts: null },
      contexts: [],
      reason: 'no_contexts_table'
    }
    renderWithProviders(<ContextsPanel pair={pair} onSwap={() => {}} />)
    expect(await screen.findByText(strings.reason.no_contexts_table)).toBeTruthy()
    expect(screen.getByText('Influential')).toBeTruthy()
  })
})

describe('OutsidePanel', () => {
  const work = {
    work_id: 'oa:W1',
    doi: '10.9/outside',
    title: 'An outside work',
    year: 1995,
    first_author: 'Alan Burns',
    venue: 'JGR',
    cited_by_count: 220,
    n_library_citers: 21,
    also_cited_by: [struct('10.1/c', 'Citer paper')]
  }

  test('lists the works with a DOI link and the library papers that also cite them', async () => {
    answers['/graph/papers/outside'] = { paper: struct('10.1/p', 'P'), total: 13, works: [work], truncated: true }
    const onOpen = mock(() => {})
    renderWithProviders(<OutsidePanel papers={['10.1/p']} onOpen={onOpen} />)

    const link = await screen.findByRole('link', { name: 'An outside work' })
    expect(link.getAttribute('href')).toBe('https://doi.org/10.9/outside')
    expect(screen.getByText('Cited by library papers: 21')).toBeTruthy()
    // One paper: its total is exact, so the count says how many it left out.
    expect(
      screen.getByText('Showing 1 of 13 works cited by this paper and by other library papers')
    ).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Author c' }))
    expect(onOpen).toHaveBeenCalledWith('doi:10.1/c')
  })

  test('no outside works gives the reason', async () => {
    answers['/graph/papers/outside'] = { paper: struct('10.1/p', 'P'), total: 0, works: [], reason: 'no_external_works' }
    renderWithProviders(<OutsidePanel papers={['10.1/p']} onOpen={() => {}} />)
    expect(await screen.findByText(strings.reason.no_external_works)).toBeTruthy()
  })
})

describe('TopicsPanel', () => {
  const topic = (id: string, path: string, papers: number, level = 'subfield') => ({
    topic_id: id,
    name: path.split(' > ').pop(),
    level,
    path,
    n_papers: papers,
    n_primary: papers
  })

  test('groups the subfields by their field, biggest topic first, and opens a topic\'s papers', async () => {
    answers['/graph/papers/topics'] = {
      total: 3,
      topics: [
        topic('subfield:1', 'Physical > Earth > Geophysics', 10),
        topic('subfield:2', 'Physical > Earth > Space', 30)
      ]
    }
    answers['/graph/papers/by_topic'] = {
      topic: topic('subfield:2', 'Physical > Earth > Space', 30),
      alternatives: [],
      total: 30,
      in_kb: 25,
      papers: [{ ...struct('10.1/s', 'A space paper'), primary: true }]
    }
    const onOpenPaper = mock(() => {})
    renderWithProviders(<TopicsPanel onOpenPaper={onOpenPaper} />)

    expect(await screen.findByText('Physical > Earth')).toBeTruthy()
    expect(requests[0]).toContain('level=subfield')
    const names = screen.getAllByRole('button').map((b) => b.textContent)
    expect(names.findIndex((n) => n?.startsWith('Space'))).toBeLessThan(names.findIndex((n) => n?.startsWith('Geophysics')))

    fireEvent.click(screen.getByRole('button', { name: /^Space/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'A space paper' }))
    expect(onOpenPaper).toHaveBeenCalledWith('doi:10.1/s')
    expect(requests[requests.length - 1]).toContain('topic=subfield%3A2')
  })

  test('a level button asks for that level, and a topic with no match says so', async () => {
    answers['/graph/papers/topics'] = { total: 1, topics: [topic('field:1', 'Physical > Earth', 7, 'field')] }
    renderWithProviders(<TopicsPanel onOpenPaper={() => {}} />)
    await screen.findByText('Physical')

    fireEvent.click(screen.getByRole('button', { name: strings.topics.levels.field }))
    await screen.findByText('Earth')
    expect(requests[requests.length - 1]).toContain('level=field')

    answers['/graph/papers/by_topic'] = { topic: null, reason: 'no_topic', papers: [], total: 0 }
    fireEvent.click(screen.getByRole('button', { name: /^Earth/ }))
    expect(await screen.findByText(strings.reason.no_topic)).toBeTruthy()
  })
})

// The routes take one paper: with several chosen, each panel asks once per paper (never a
// repeated `paper=`, of which a route would read only the last) and merges the answers.
describe('several chosen papers', () => {
  const PAPERS = ['10.1/p', '10.1/q']
  const asked = (url: string) => new URL(url, 'http://x').searchParams.getAll('paper')
  const byPaper = (answer: Record<string, unknown>) => (url: URL) => {
    const body = answer[url.searchParams.get('paper') ?? '']
    if (body instanceof Error) throw body
    return body
  }
  const related = (id: string, title: string, n_contexts: number, extra: Record<string, unknown> = {}) => ({
    ...struct(id, title),
    score: 1,
    reasons: ['cites'],
    citation: { source: 'both', is_influential: false, intents: [], n_contexts },
    ...extra
  })

  test('the related panel asks once per paper and lists a shared paper once', async () => {
    answers['/graph/papers/related'] = byPaper({
      '10.1/p': {
        paper: struct('10.1/p', 'P'),
        totals: { cites: 5 },
        related: [related('10.1/ref', 'A reference', 0), related('10.1/only', 'Only P', 0, { score: 0.5 })]
      },
      '10.1/q': { paper: struct('10.1/q', 'Q'), totals: { cites: 4 }, related: [related('10.1/ref', 'A reference', 3)] }
    })
    const onPickCitation = mock(() => {})
    renderWithProviders(<RelatedPanel papers={PAPERS} onOpen={() => {}} onPickCitation={onPickCitation} />)

    await screen.findByText('Only P')
    expect(requests.map(asked).sort()).toEqual([['10.1/p'], ['10.1/q']])
    expect(requests.every((url) => url.includes('relations=cites'))).toBe(true)
    expect(screen.getAllByRole('button', { name: 'A reference' })).toHaveLength(1)
    // The two totals add up to 9, but the shared reference is one paper: only what is shown is counted.
    expect(screen.getByText('Showing 2 papers')).toBeTruthy()
    // The copy of the shared reference that has sentences is the one from Q.
    fireEvent.click(screen.getByRole('button', { name: 'Read citing sentences (3)' }))
    expect(onPickCitation).toHaveBeenCalledWith({ citing: '10.1/q', cited: 'doi:10.1/ref' })
  })

  test('a paper that fails leaves the others\' papers and a note', async () => {
    answers['/graph/papers/related'] = byPaper({
      '10.1/p': { paper: struct('10.1/p', 'P'), related: [related('10.1/ref', 'A reference', 0)] },
      '10.1/q': new Error('boom')
    })
    renderWithProviders(<RelatedPanel papers={PAPERS} onOpen={() => {}} onPickCitation={() => {}} />)

    expect(await screen.findByRole('button', { name: 'A reference' })).toBeTruthy()
    expect(screen.getByText('No answer for 1 of the 2 chosen papers')).toBeTruthy()
  })

  test('every paper failing is an error, not an empty list', async () => {
    answers['/graph/papers/related'] = byPaper({ '10.1/p': new Error('boom'), '10.1/q': new Error('boom') })
    renderWithProviders(<RelatedPanel papers={PAPERS} onOpen={() => {}} onPickCitation={() => {}} />)

    expect(await screen.findByText('Could not load this data: boom')).toBeTruthy()
    expect(screen.queryByText(strings.related.empty)).toBeNull()
  })

  test('a reason from one paper does not hide what the other found', async () => {
    answers['/graph/papers/related'] = byPaper({
      '10.1/p': { paper: null, reason: 'not_in_structure' },
      '10.1/q': { paper: struct('10.1/q', 'Q'), related: [related('10.1/ref', 'A reference', 0)] }
    })
    renderWithProviders(<RelatedPanel papers={PAPERS} onOpen={() => {}} onPickCitation={() => {}} />)

    expect(await screen.findByRole('button', { name: 'A reference' })).toBeTruthy()
    expect(screen.getByText('No answer for 1 of the 2 chosen papers')).toBeTruthy()
    expect(screen.queryByText(strings.reason.not_in_structure)).toBeNull()
  })

  test('the same reason from every paper is the panel\'s message', async () => {
    answers['/graph/papers/related'] = byPaper({
      '10.1/p': { paper: null, reason: 'not_in_structure' },
      '10.1/q': { paper: null, reason: 'not_in_structure' }
    })
    renderWithProviders(<RelatedPanel papers={PAPERS} onOpen={() => {}} onPickCitation={() => {}} />)
    expect(await screen.findByText(strings.reason.not_in_structure)).toBeTruthy()
  })

  test('the similar panel asks once per paper and keeps the best similarity of a shared paper', async () => {
    const similar = (id: string, title: string, similarity: number, links: string[]) => ({
      ...struct(id, title),
      similarity,
      links
    })
    answers['/graph/papers/similar'] = byPaper({
      '10.1/p': { paper: struct('10.1/p', 'P'), similar: [similar('10.1/a', 'Shared', 0.9, ['cites']), similar('10.1/b', 'Only P', 0.8, [])] },
      '10.1/q': { paper: struct('10.1/q', 'Q'), similar: [similar('10.1/a', 'Shared', 0.95, ['coupled'])] }
    })
    renderWithProviders(<SimilarPanel papers={PAPERS} onOpen={() => {}} />)

    expect(await screen.findByText('Similarity 95%')).toBeTruthy()
    expect(requests.map(asked).sort()).toEqual([['10.1/p'], ['10.1/q']])
    expect(screen.queryByText('Similarity 90%')).toBeNull()
    expect(screen.getAllByRole('button', { name: 'Shared' })).toHaveLength(1)
    expect(screen.getByText(strings.relation.cites)).toBeTruthy()
    expect(screen.getByText(strings.relation.coupled)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Only P' })).toBeTruthy()
  })

  test('the outside panel asks once per paper and joins the citers of a shared work', async () => {
    const work = (id: string, title: string, citers: string[]) => ({
      work_id: `oa:${id}`,
      doi: `10.9/${id}`,
      title,
      year: 1995,
      first_author: 'Alan Burns',
      venue: 'JGR',
      cited_by_count: 220,
      n_library_citers: 21,
      also_cited_by: citers.map((c) => struct(`10.1/${c}`, `Citer ${c}`))
    })
    answers['/graph/papers/outside'] = byPaper({
      '10.1/p': { paper: struct('10.1/p', 'P'), total: 13, works: [work('w1', 'Shared work', ['c'])] },
      '10.1/q': { paper: struct('10.1/q', 'Q'), total: 2, works: [work('w1', 'Shared work', ['c', 'd']), work('w2', 'Other work', [])] }
    })
    renderWithProviders(<OutsidePanel papers={PAPERS} onOpen={() => {}} />)

    await screen.findByRole('link', { name: 'Other work' })
    expect(requests.map(asked).sort()).toEqual([['10.1/p'], ['10.1/q']])
    expect(screen.getAllByRole('link', { name: 'Shared work' })).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: 'Author c' })).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Author d' })).toBeTruthy()
    // The two totals add up to 15, but the shared work is one work: only what is shown is counted.
    expect(
      screen.getByText('Showing 2 works cited by these papers and by other library papers')
    ).toBeTruthy()
  })
})
