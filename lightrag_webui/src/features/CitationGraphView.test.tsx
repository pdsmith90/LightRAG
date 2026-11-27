/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { fireEvent, screen, within } from '@testing-library/react'
import { Toaster, toast } from 'sonner'
import '@/test/webglStub'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'

const { default: CitationGraphView, __resetCitationGraphForTests } = await import('./CitationGraphView')

/**
 * The tab itself: it needs the server's paper structure, and offers the Paper and Topics views.
 * The map is not drawn here (no WebGL); the panels have their own tests.
 */
const strings = en.citationGraph
const filterStrings = en.graphPanel.paperFilter

const setStatus = (zoteroStruct: boolean | undefined) =>
  useBackendState.setState({
    status: { status: 'healthy', configuration: { zotero_struct: zoteroStruct } } as any
  })

beforeEach(() => {
  __resetCitationGraphForTests()
  __setAxiosAdapterForTests(async (config: any) => ({
    data: { total: 0, topics: [] },
    status: 200,
    statusText: 'OK',
    headers: {},
    config
  }))
})

afterEach(() => {
  toast.dismiss() // toasts outlive a test's render
  __setAxiosAdapterForTests(undefined)
  useBackendState.setState({ status: null })
})

describe('CitationGraphView', () => {
  test('without the paper structure it shows a notice and no views', () => {
    setStatus(false)
    renderWithProviders(<CitationGraphView />)
    expect(screen.getByText(strings.unavailable)).toBeTruthy()
    expect(screen.queryByRole('tab', { name: strings.modePaper })).toBeNull()
  })

  test('a server that does not report the setting is treated as without it', () => {
    setStatus(undefined)
    renderWithProviders(<CitationGraphView />)
    expect(screen.getByText(strings.unavailable)).toBeTruthy()
  })

  test('before the first health answer it only says it is loading', () => {
    useBackendState.setState({ status: null })
    renderWithProviders(<CitationGraphView />)
    expect(screen.getByText(strings.loading)).toBeTruthy()
    expect(screen.queryByText(strings.unavailable)).toBeNull()
  })

  test('search results show the TLDR under the title when the paper has one', async () => {
    const paper = (id: string, citation: string, tldr: string | null) => ({
      paper: id,
      doi: id,
      keys: [],
      title: citation,
      citation,
      year: '2020',
      n_refs: null,
      n_cited: null,
      n_cited_in_kb: null,
      tldr
    })
    __setAxiosAdapterForTests(async (config: any) => {
      const found = new URL(config.url, 'http://x').pathname === '/graph/papers/search'
      const data = found
        ? [paper('10.1/a', 'Alpha 2020', 'Alpha measures gravity leakage.'), paper('10.1/b', 'Beta 2020', null)]
        : { total: 0, topics: [] }
      return { data, status: 200, statusText: 'OK', headers: {}, config }
    })
    setStatus(true)
    renderWithProviders(<CitationGraphView />)

    fireEvent.click(screen.getByRole('combobox', { name: strings.search.label }))
    fireEvent.change(await screen.findByPlaceholderText(strings.search.searchPlaceholder), {
      target: { value: 'a' }
    })
    expect(await screen.findByText('Alpha measures gravity leakage.', {}, { timeout: 3000 })).toBeTruthy()
    expect(screen.getByText('Beta 2020')).toBeTruthy()
  })

  test('opens on the Paper view with a search and switches to the Topics view', async () => {
    setStatus(true)
    renderWithProviders(<CitationGraphView />)
    expect(screen.getByRole('tab', { name: strings.modePaper }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('combobox', { name: strings.search.label })).toBeTruthy()
    expect(screen.getByText(strings.emptyPaper)).toBeTruthy()

    fireEvent.click(screen.getByRole('tab', { name: strings.modeTopics }))
    expect(await screen.findByText(strings.topics.empty)).toBeTruthy()
    expect(screen.queryByRole('combobox', { name: strings.search.label })).toBeNull()
  })
})

/**
 * Choosing papers: the search is a repeat-pick like the Knowledge Graph's paper filter, the
 * chosen papers are chips, and the map and the panels ask about all of them. The map's canvas is
 * not drawn here (no WebGL), so its request is what shows it was asked.
 */
describe('CitationGraphView paper selection', () => {
  const paper = (id: string, citation: string, keys: string[] = []) => ({
    paper: id,
    doi: id,
    keys,
    title: citation,
    citation,
    year: '2020',
    n_refs: null,
    n_cited: null,
    n_cited_in_kb: null
  })
  // A paper chosen by its key (a library paper without a DOI) is the same as the one by its DOI.
  const LIBRARY = [
    paper('10.1/a', 'Alpha 2020', ['AAAA1111']),
    paper('10.1/b', 'Beta 2020'),
    paper('AAAA1111', 'Alpha by key'),
    paper('10.1/ref', 'Reference 2019'),
    ...Array.from({ length: 10 }, (_, i) => paper(`10.2/n${i}`, `Extra ${i}`))
  ]
  const REFERENCE = {
    paper_id: 'doi:10.1/ref',
    doi: '10.1/ref',
    label: 'Ref',
    title: 'A reference',
    in_kb: true,
    cited_by_library: 1,
    score: 1,
    reasons: ['cites']
  }
  let requests: string[] = []

  const mapPapers = () =>
    requests
      .filter((url) => url.startsWith('/graphs/paper/map'))
      .map((url) => new URL(url, 'http://x').searchParams.getAll('paper'))
  const chips = () => screen.queryAllByRole('button', { name: /^Remove / }).map((b) => b.getAttribute('aria-label'))

  beforeEach(() => {
    requests = []
    __setAxiosAdapterForTests(async (config: any) => {
      requests.push(config.url)
      const url = new URL(config.url, 'http://x')
      const q = url.searchParams.get('q') ?? ''
      let data: unknown = { total: 0, topics: [] }
      if (url.pathname === '/graph/papers/search') {
        data = LIBRARY.filter((p) => p.paper === q || p.keys.includes(q) || p.citation.includes(q) || q === 'all')
      } else if (url.pathname === '/graphs/paper/map') {
        data = { nodes: [], edges: [] }
      } else if (url.pathname === '/graph/papers/related') {
        // Only Alpha has a reference.
        data = { paper: null, related: url.searchParams.get('paper') === '10.1/a' ? [REFERENCE] : [] }
      }
      return { data, status: 200, statusText: 'OK', headers: {}, config }
    })
    setStatus(true)
  })

  const combobox = () => screen.getByRole('combobox', { name: strings.search.label })
  const openList = async () => {
    fireEvent.click(combobox())
    fireEvent.change(await screen.findByPlaceholderText(strings.search.searchPlaceholder), {
      target: { value: 'all' }
    })
  }
  // Picks from the list as it stands; the list is not reopened.
  const pick = async (citation: string) => {
    const list = await screen.findByRole('listbox')
    fireEvent.click(await within(list).findByText(citation, {}, { timeout: 3000 }))
  }

  test('picking two papers one after the other chooses both, and the list stays open', async () => {
    renderWithProviders(<CitationGraphView />)

    await openList()
    await pick('Alpha 2020')
    expect(combobox().getAttribute('aria-expanded')).toBe('true')
    expect(chips()).toEqual(['Remove Alpha 2020'])
    await pick('Beta 2020')
    expect(chips()).toEqual(['Remove Alpha 2020', 'Remove Beta 2020'])
    expect(screen.queryByText(strings.emptyPaper)).toBeNull()
  })

  test('the map asks about every chosen paper, and each panel once per paper', async () => {
    renderWithProviders(<CitationGraphView />)
    await openList()
    await pick('Alpha 2020')
    await pick('Beta 2020')

    await screen.findByRole('button', { name: 'A reference' })
    expect(mapPapers().pop()).toEqual(['10.1/a', '10.1/b'])
    const related = requests.filter((url) => url.startsWith('/graph/papers/related'))
    expect(related.map((url) => new URL(url, 'http://x').searchParams.getAll('paper')).slice(-2)).toEqual([
      ['10.1/a'],
      ['10.1/b']
    ])
  })

  test('picking a chosen paper again takes it out, also when it was chosen by its key', async () => {
    renderWithProviders(<CitationGraphView />)
    await openList()
    await pick('Alpha by key')
    await pick('Beta 2020')
    expect(chips()).toEqual(['Remove Alpha by key', 'Remove Beta 2020'])

    // Alpha's DOI result lists the key as one of its own.
    await pick('Alpha 2020')
    expect(chips()).toEqual(['Remove Beta 2020'])
    await pick('Beta 2020')
    expect(chips()).toEqual([])
    expect(screen.getByText(strings.emptyPaper)).toBeTruthy()
  })

  test('a chip removes its paper, and the last one empties the view', async () => {
    renderWithProviders(<CitationGraphView />)
    await openList()
    await pick('Alpha 2020')
    await pick('Beta 2020')

    fireEvent.click(screen.getByRole('button', { name: 'Remove Alpha 2020' }))
    expect(chips()).toEqual(['Remove Beta 2020'])
    expect(screen.queryByRole('button', { name: filterStrings.clearPapers })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Remove Beta 2020' }))
    expect(screen.getByText(strings.emptyPaper)).toBeTruthy()
  })

  test('Clear appears with two papers and drops them all', async () => {
    renderWithProviders(<CitationGraphView />)
    await openList()
    await pick('Alpha 2020')
    expect(screen.queryByRole('button', { name: filterStrings.clearPapers })).toBeNull()
    await pick('Beta 2020')

    fireEvent.click(screen.getByRole('button', { name: filterStrings.clearPapers }))
    expect(chips()).toEqual([])
    expect(screen.getByText(strings.emptyPaper)).toBeTruthy()
  })

  test('an eleventh paper is refused with a toast', async () => {
    renderWithProviders(
      <>
        <CitationGraphView />
        <Toaster />
      </>
    )
    await openList()
    await pick('Alpha 2020')
    for (let i = 0; i < 9; i++) await pick(`Extra ${i}`)
    expect(chips()).toHaveLength(10)

    await pick('Extra 9')
    expect(await screen.findByText(filterStrings.maxPapers)).toBeTruthy()
    expect(chips()).toHaveLength(10)
    expect(screen.queryByRole('button', { name: 'Remove Extra 9' })).toBeNull()
  })

  test('a paper picked in a panel is chosen too, and shows its citation once it is looked up', async () => {
    renderWithProviders(<CitationGraphView />)
    await openList()
    await pick('Alpha 2020')

    fireEvent.click(await screen.findByRole('button', { name: 'A reference' }))
    expect(await screen.findByRole('button', { name: 'Remove Reference 2019' })).toBeTruthy()
    expect(chips()).toEqual(['Remove Alpha 2020', 'Remove Reference 2019'])

    // The same row again takes it out.
    fireEvent.click(await screen.findByRole('button', { name: 'A reference' }))
    expect(chips()).toEqual(['Remove Alpha 2020'])
  })
})
