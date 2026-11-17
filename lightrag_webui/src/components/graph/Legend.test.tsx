/// <reference types="bun" />
import { beforeEach, describe, expect, test } from 'bun:test'

import { renderWithProviders } from '@/test/render'
import en from '@/locales/en.json'
import { RawGraph, useGraphStore } from '@/stores/graph'
import Legend from './Legend'

const strings = en.graphPanel.paperFilter

const show = (isPaperMap: boolean) => {
  const raw = new RawGraph()
  raw.isPaperMap = isPaperMap
  useGraphStore.getState().setRawGraph(raw)
  useGraphStore.getState().setTypeColorMap(new Map([['Geophysics', '#ff0000']]))
}

beforeEach(() => useGraphStore.getState().reset())

describe('Legend', () => {
  test('a paper map adds its two link kinds to the node types', () => {
    show(true)
    const { getByText } = renderWithProviders(<Legend />)
    expect(getByText('Geophysics')).toBeTruthy()
    expect(getByText(strings.legendCitation)).toBeTruthy()
    expect(getByText(strings.legendCoupling)).toBeTruthy()
  })

  test('an entity graph shows node types only', () => {
    show(false)
    const { getByText, queryByText } = renderWithProviders(<Legend />)
    expect(getByText('Geophysics')).toBeTruthy()
    expect(queryByText(strings.legendCitation)).toBeNull()
  })
})
