/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { fireEvent, screen } from '@testing-library/react'
import '@/test/webglStub'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'

const { default: CitationGraphView } = await import('./CitationGraphView')

/**
 * The tab itself: it needs the server's paper structure, and offers the Paper and Topics views.
 * The map is not drawn here (no WebGL); the panels have their own tests.
 */
const strings = en.citationGraph

const setStatus = (zoteroStruct: boolean | undefined) =>
  useBackendState.setState({
    status: { status: 'healthy', configuration: { zotero_struct: zoteroStruct } } as any
  })

beforeEach(() => {
  __setAxiosAdapterForTests(async (config: any) => ({
    data: { total: 0, topics: [] },
    status: 200,
    statusText: 'OK',
    headers: {},
    config
  }))
})

afterEach(() => {
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
