/// <reference types="bun" />
import { afterAll, beforeEach, describe, expect, test } from 'bun:test'
import { fireEvent, screen, within } from '@testing-library/react'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { __setAxiosAdapterForTests } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { __resetPaperContextForTests } from '@/lib/paperFilter'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'
import PaperFilter, { __resetPaperFilterForTests } from './PaperFilter'

/**
 * The paper filter has no state of its own beyond the mode and density
 * buttons: the selection IS the query label, and every change goes through the
 * settings and graph stores. The tests drive the real component and read those
 * stores. Papers come from the search endpoint, stubbed at the HTTP adapter.
 */
const strings = en.graphPanel.paperFilter

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
const LIBRARY = [paper('10.1/a', 'Alpha 2020', ['AAAA1111']), paper('10.1/b', 'Beta 2020')]

const label = () => useSettingsStore.getState().queryLabel
const version = () => useGraphStore.getState().graphDataVersion

beforeEach(() => {
  localStorage.clear()
  __resetPaperFilterForTests()
  __resetPaperContextForTests()
  useSettingsStore.getState().setQueryLabel('')
  __setAxiosAdapterForTests(async (config: any) => {
    const q = new URL(config.url, 'http://x').searchParams.get('q') ?? ''
    const data = LIBRARY.filter((p) => p.paper === q || p.keys.includes(q) || p.citation.includes(q) || q === 'all')
    return { data, status: 200, statusText: 'OK', headers: {}, config }
  })
})

afterAll(() => {
  __setAxiosAdapterForTests(undefined)
})

const combobox = () => screen.getByRole('combobox', { name: strings.label })

const openList = async () => {
  fireEvent.click(combobox())
  fireEvent.change(await screen.findByPlaceholderText(strings.searchPlaceholder), {
    target: { value: 'all' }
  })
}

// Picks from the list as it stands; the list is not reopened.
const pick = async (citation: string) => {
  const list = await screen.findByRole('listbox')
  fireEvent.click(await within(list).findByText(citation, {}, { timeout: 3000 }))
}

describe('PaperFilter', () => {
  test('picking two papers one after the other gives a two-id label', async () => {
    renderWithProviders(<PaperFilter />)

    await openList()
    await pick('Alpha 2020')
    expect(label()).toBe('paper:10.1/a')
    // The list stays open for the next pick.
    expect(combobox().getAttribute('aria-expanded')).toBe('true')
    await pick('Beta 2020')
    expect(label()).toBe('paper:10.1/a 10.1/b')
  })

  test('a paper selected by its key is removed by picking it from the list by DOI', async () => {
    useSettingsStore.getState().setQueryLabel('paper:AAAA1111 10.1/b')
    renderWithProviders(<PaperFilter />)

    await openList()
    await pick('Alpha 2020')
    expect(label()).toBe('paper:10.1/b')
  })

  test('a burst of picks leaves one history entry, and a restored label is kept', async () => {
    SearchHistoryManager.addToHistory('paper:10.1/old')
    useSettingsStore.getState().setQueryLabel('paper:10.1/old')
    renderWithProviders(<PaperFilter />)

    await openList()
    await pick('Alpha 2020')
    await pick('Beta 2020')
    fireEvent.click(screen.getByRole('button', { name: strings.modePaperTooltip }))
    expect(label()).toBe('paper+refs:10.1/old 10.1/a 10.1/b')
    expect(SearchHistoryManager.getHistoryLabels()).toEqual([
      'paper+refs:10.1/old 10.1/a 10.1/b',
      'paper:10.1/old'
    ])
  })

  test('clearing the papers keeps the last selection in the history', async () => {
    renderWithProviders(<PaperFilter />)

    await openList()
    await pick('Alpha 2020')
    await pick('Beta 2020')
    fireEvent.click(screen.getByRole('button', { name: strings.clearPapers }))
    expect(label()).toBe('')
    expect(SearchHistoryManager.getHistoryLabels()).toEqual(['paper:10.1/a 10.1/b'])
  })

  test('picking a selected paper again removes it', async () => {
    useSettingsStore.getState().setQueryLabel('paper:10.1/a 10.1/b')
    renderWithProviders(<PaperFilter />)
    await screen.findByRole('button', { name: 'Remove Alpha 2020' })

    await openList()
    await pick('Alpha 2020')
    expect(label()).toBe('paper:10.1/b')
  })

  test('removing the last paper empties the label instead of querying everything', async () => {
    useSettingsStore.getState().setQueryLabel('paper:10.1/a')
    renderWithProviders(<PaperFilter />)

    const remove = await screen.findByRole('button', { name: 'Remove Alpha 2020' })
    fireEvent.click(remove)
    expect(label()).toBe('')
  })

  test('Clear drops every paper', () => {
    useSettingsStore.getState().setQueryLabel('paper+refs:10.1/a 10.1/b')
    renderWithProviders(<PaperFilter />)

    fireEvent.click(screen.getByRole('button', { name: strings.clearPapers }))
    expect(label()).toBe('')
  })

  test('the mode button skips common with one paper and offers it with two', () => {
    useSettingsStore.getState().setQueryLabel('paper:10.1/a')
    const { unmount } = renderWithProviders(<PaperFilter />)

    fireEvent.click(screen.getByRole('button', { name: strings.modePaperTooltip }))
    expect(label()).toBe('paper+refs:10.1/a')
    fireEvent.click(screen.getByRole('button', { name: strings.modeSharedTooltip }))
    expect(label()).toBe('paper:10.1/a')
    unmount()

    useSettingsStore.getState().setQueryLabel('paper+refs:10.1/a 10.1/b')
    renderWithProviders(<PaperFilter />)
    fireEvent.click(screen.getByRole('button', { name: strings.modeSharedTooltip }))
    expect(label()).toBe('paper+common:10.1/a 10.1/b')
  })

  test('a common label left with one paper becomes a paper label', async () => {
    useSettingsStore.getState().setQueryLabel('paper+common:10.1/a 10.1/b')
    renderWithProviders(<PaperFilter />)

    fireEvent.click(await screen.findByRole('button', { name: 'Remove Alpha 2020' }))
    expect(label()).toBe('paper:10.1/b')
  })

  test('the density button does not refetch while no paper label is shown', () => {
    renderWithProviders(<PaperFilter />)
    const before = version()

    fireEvent.click(screen.getByRole('button', { name: strings.contextKbTooltip }))
    expect(version()).toBe(before)
    expect(localStorage.getItem('lightrag-paper-context')).toBe('linked')
  })

  test('the density button refetches and stores the choice while a paper label is shown', () => {
    useSettingsStore.getState().setQueryLabel('paper:10.1/a')
    renderWithProviders(<PaperFilter />)
    const before = version()

    fireEvent.click(screen.getByRole('button', { name: strings.contextKbTooltip }))
    expect(version()).toBe(before + 1)
    expect(localStorage.getItem('lightrag-paper-context')).toBe('linked')
    expect(useGraphStore.getState().graphDataFetchAttempted).toBe(false)
    // The selection is untouched: the density is not part of the label.
    expect(label()).toBe('paper:10.1/a')
    expect(screen.getByRole('button', { name: strings.contextLinkedTooltip })).toBeTruthy()
  })
})

describe('AsyncSelect', () => {
  test('closes after a pick unless closeOnSelect is false', async () => {
    renderWithProviders(
      <AsyncSelect<{ id: string }>
        fetcher={async () => [{ id: 'one' }]}
        renderOption={(item) => <span>{item.id}</span>}
        getOptionValue={(item) => item.id}
        getDisplayValue={(item) => <span>{item.id}</span>}
        ariaLabel="pick"
        value=""
        onChange={() => {}}
      />
    )

    fireEvent.click(screen.getByRole('combobox', { name: 'pick' }))
    fireEvent.click(await screen.findByText('one'))
    expect(screen.getByRole('combobox', { name: 'pick' }).getAttribute('aria-expanded')).toBe('false')
  })
})
