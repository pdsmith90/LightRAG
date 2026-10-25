import { afterEach, describe, expect, test } from 'bun:test'

import en from '@/locales/en.json'
import type { PaperSummary } from '@/api/lightrag'
import {
  __resetPaperContextForTests,
  CONTEXT_EDGE_COLOR,
  clearsLabelOnEmpty,
  isStalePaperResult,
  fadeColor,
  nextContext,
  nextMode,
  paperEdgeAttrs,
  paperLabel,
  paperNodeAttrs,
  paperSummaryMessage,
  parsePaperLabel,
  readPaperContext,
  togglePaper,
  writePaperContext
} from './paperFilter'

describe('paper filter labels', () => {
  test('a paper label round-trips its paper and mode', () => {
    expect(parsePaperLabel(paperLabel(['10.1002/2013jb010900'], 'paper'))).toEqual({
      papers: ['10.1002/2013jb010900'],
      mode: 'paper'
    })
    expect(parsePaperLabel(paperLabel([' ABCD1234 '], 'shared'))).toEqual({
      papers: ['ABCD1234'],
      mode: 'shared'
    })
  })

  test('labels written before several papers could be picked still parse', () => {
    expect(parsePaperLabel('paper:10.1/a')).toEqual({ papers: ['10.1/a'], mode: 'paper' })
    expect(parsePaperLabel('paper+refs:ABCD1234')).toEqual({ papers: ['ABCD1234'], mode: 'shared' })
  })

  test('several papers round-trip, trimmed and de-duplicated in pick order', () => {
    const label = paperLabel([' 10.1/b', 'ABCD1234 ', '', '10.1/b', '10.1/a'], 'paper')
    expect(label).toBe('paper:10.1/b ABCD1234 10.1/a')
    expect(parsePaperLabel(label)).toEqual({
      papers: ['10.1/b', 'ABCD1234', '10.1/a'],
      mode: 'paper'
    })
    // Hand-typed whitespace of any kind separates ids too.
    expect(parsePaperLabel('paper:  10.1/a\t10.1/b \n10.1/a')?.papers).toEqual(['10.1/a', '10.1/b'])
  })

  test('each mode has its own prefix', () => {
    expect(paperLabel(['a', 'b'], 'paper')).toBe('paper:a b')
    expect(paperLabel(['a', 'b'], 'shared')).toBe('paper+refs:a b')
    expect(paperLabel(['a', 'b'], 'common')).toBe('paper+common:a b')
    expect(parsePaperLabel('paper+common:a b')).toEqual({ papers: ['a', 'b'], mode: 'common' })
    expect(parsePaperLabel('paper+refs:a b')?.mode).toBe('shared')
    expect(parsePaperLabel('paper:a b')?.mode).toBe('paper')
  })

  test('entity labels and empty papers are not paper labels', () => {
    expect(parsePaperLabel('GRACE')).toBeNull()
    expect(parsePaperLabel('*')).toBeNull()
    expect(parsePaperLabel('')).toBeNull()
    expect(parsePaperLabel(null)).toBeNull()
    expect(parsePaperLabel('paper:')).toBeNull()
    expect(parsePaperLabel('paper+refs:  ')).toBeNull()
    expect(parsePaperLabel('paper+common: ')).toBeNull()
  })
})

describe('picking papers and modes', () => {
  test('togglePaper adds at the end and removes a paper picked again', () => {
    expect(togglePaper([], 'a')).toEqual(['a'])
    expect(togglePaper(['a'], 'b')).toEqual(['a', 'b'])
    expect(togglePaper(['a', 'b', 'c'], 'b')).toEqual(['a', 'c'])
  })

  test('nextMode cycles paper, shared, common and skips common below two papers', () => {
    expect(nextMode('paper', 3)).toBe('shared')
    expect(nextMode('shared', 2)).toBe('common')
    expect(nextMode('common', 2)).toBe('paper')
    expect(nextMode('shared', 1)).toBe('paper')
    expect(nextMode('shared', 0)).toBe('paper')
    // A common label left with one paper falls back to paper.
    expect(nextMode('common', 1)).toBe('paper')
  })

  test('nextContext cycles own, kb, linked', () => {
    expect(nextContext('own')).toBe('kb')
    expect(nextContext('kb')).toBe('linked')
    expect(nextContext('linked')).toBe('own')
  })
})

describe('stale and empty paper results', () => {
  test('only a paper result for a changed selection is stale', () => {
    expect(isStalePaperResult('paper:a', 'paper:a b')).toBe(true)
    expect(isStalePaperResult('paper:a', '')).toBe(true)
    expect(isStalePaperResult('paper:a', 'paper:a')).toBe(false)
    expect(isStalePaperResult('GRACE', 'Other')).toBe(false)
    expect(isStalePaperResult('*', '')).toBe(false)
  })

  test('an empty result clears an entity label but keeps a paper label', () => {
    expect(clearsLabelOnEmpty('GRACE', 'GRACE')).toBe(true)
    expect(clearsLabelOnEmpty('paper:a', 'paper:a')).toBe(false)
    expect(clearsLabelOnEmpty('paper+common:a b', 'paper+common:a b')).toBe(false)
    // The label changed while the fetch ran: leave the new one alone.
    expect(clearsLabelOnEmpty('GRACE', 'paper:a')).toBe(false)
    expect(clearsLabelOnEmpty('', '')).toBe(false)
  })
})

describe('paper context setting', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  const setStorage = (value: unknown) =>
    Object.defineProperty(globalThis, 'localStorage', { value, configurable: true })
  const throwingStorage = {
    getItem: () => {
      throw new Error('blocked')
    },
    setItem: () => {
      throw new Error('blocked')
    }
  }

  afterEach(() => {
    __resetPaperContextForTests()
    if (original) Object.defineProperty(globalThis, 'localStorage', original)
  })

  test('is stored and read back, defaulting to kb', () => {
    __resetPaperContextForTests()
    const data = new Map<string, string>()
    setStorage({
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v)
    })
    expect(readPaperContext()).toBe('kb')
    writePaperContext('linked')
    expect(data.get('lightrag-paper-context')).toBe('linked')
    expect(readPaperContext()).toBe('linked')
    // A fresh page load reads what was stored; nonsense falls back to the default.
    __resetPaperContextForTests()
    expect(readPaperContext()).toBe('linked')
    __resetPaperContextForTests()
    data.set('lightrag-paper-context', 'nonsense')
    expect(readPaperContext()).toBe('kb')
  })

  test('a storage that throws gives the default, and a write still sticks for reads', () => {
    __resetPaperContextForTests()
    setStorage(throwingStorage)
    expect(readPaperContext()).toBe('kb')
    expect(() => writePaperContext('own')).not.toThrow()
    expect(readPaperContext()).toBe('own')
  })
})

describe('context styling', () => {
  test('fadeColor adds alpha to a six-digit hex colour and leaves other formats alone', () => {
    expect(fadeColor('#aabbcc')).toBe('#aabbcc73')
    expect(fadeColor('#ABC')).toBe('#ABC')
    expect(fadeColor('#aabbccdd')).toBe('#aabbccdd')
    expect(fadeColor('rgb(1,2,3)')).toBe('rgb(1,2,3)')
    expect(fadeColor('red')).toBe('red')
  })

  test('a linking node is faded; other nodes are untouched', () => {
    expect(
      paperNodeAttrs({ color: '#112233', properties: { paper_context: 'Not from the selected papers' } })
    ).toEqual({ color: '#11223373' })
    expect(paperNodeAttrs({ color: '#112233', properties: { entity_type: 'x' } })).toEqual({})
    expect(paperNodeAttrs({ color: '#112233' })).toEqual({})
  })

  test('only a marked edge gets the context colour', () => {
    expect(paperEdgeAttrs({ properties: { paper_context: 'Relation from other documents' } })).toEqual({
      color: CONTEXT_EDGE_COLOR
    })
    expect(paperEdgeAttrs({ properties: { keywords: 'k' } })).toEqual({})
    expect(paperEdgeAttrs({})).toEqual({})
  })
})

describe('paperSummaryMessage', () => {
  // The English strings with their {{placeholders}} filled, as the app shows them.
  const t = (key: string, options: Record<string, unknown> = {}): string => {
    const text = key.split('.').reduce<any>((node, part) => node?.[part], en)
    expect(typeof text).toBe('string')
    return (text as string).replace(/{{(\w+)}}/g, (_, name) => String(options[name]))
  }
  const base: PaperSummary = {
    paper: '10.1/a',
    doi: '10.1/a',
    keys: ['ABCD1234'],
    title: 'A',
    citation: 'Smith 2020',
    year: '2020',
    n_refs: 10,
    n_cited: 8,
    n_cited_in_kb: 5
  }

  test('one paper gives the strings the single-paper filter always had', () => {
    expect(paperSummaryMessage(t, { ...base, mode: 'paper', papers: [base] }, 12)).toBe(
      'Smith 2020: 12 nodes; 5 of its references are in the knowledge base'
    )
    expect(
      paperSummaryMessage(t, { ...base, mode: 'shared', cited_sharing: 3, papers: [base] }, 7)
    ).toBe('Smith 2020: 7 nodes shared with 3 of its 5 references in the knowledge base')
    // Without the new fields at all (an older server).
    expect(paperSummaryMessage(t, { ...base, mode: 'paper' }, 12)).toBe(
      'Smith 2020: 12 nodes; 5 of its references are in the knowledge base'
    )
  })

  test('several papers get a count-based message per mode', () => {
    const two = [base, { ...base, paper: '10.1/b' }]
    expect(paperSummaryMessage(t, { ...base, mode: 'paper', papers: two }, 30)).toBe(
      '2 papers: 30 nodes'
    )
    expect(paperSummaryMessage(t, { ...base, mode: 'common', papers: two }, 4)).toBe(
      '2 papers: 4 nodes at least two of them contributed'
    )
    expect(
      paperSummaryMessage(t, { ...base, mode: 'shared', cited_sharing: 6, papers: two }, 9)
    ).toBe('2 papers: 9 nodes shared with 6 cited works in the knowledge base')
  })

  test('context counts are appended only when something was added', () => {
    const summary = { ...base, mode: 'paper' as const, papers: [base] }
    const plain = paperSummaryMessage(t, summary, 12)
    expect(paperSummaryMessage(t, { ...summary, n_context_edges: 0, n_linking: 0 }, 12)).toBe(plain)
    expect(paperSummaryMessage(t, { ...summary, n_context_edges: 5, n_linking: 0 }, 12)).toBe(
      `${plain} (+5 relations from the wider graph)`
    )
  })

  test('linking entities are not counted among the nodes the papers contributed', () => {
    const summary = { ...base, mode: 'paper' as const, papers: [base], n_context_edges: 5, n_linking: 2 }
    expect(paperSummaryMessage(t, summary, 14)).toBe(
      'Smith 2020: 12 nodes; 5 of its references are in the knowledge base' +
        ' (+5 relations and 2 linking entities from the wider graph)'
    )
  })

  test('papers missing from the knowledge base are mentioned', () => {
    const papers = [base, { ...base, paper: '10.1/b' }, { paper: '10.1/c', reason: 'not_in_kb' }]
    const message = paperSummaryMessage(
      t,
      { ...base, mode: 'paper', papers: papers as PaperSummary[] },
      30
    )
    expect(message).toContain('3 papers: 30 nodes')
    expect(message).toContain('1 of the selected papers are not in the knowledge base')
  })

  test('a view reason for several papers uses the plural wording', () => {
    const two = [base, { ...base, paper: '10.1/b' }]
    const reasons = en.graphPanel.paperFilter.reason as Record<string, string>
    for (const code of ['not_in_kb', 'no_reference_list', 'no_references_in_kb']) {
      expect(paperSummaryMessage(t, { ...base, reason: code, papers: two }, 0)).toBe(
        reasons[`${code}_many`]
      )
      expect(paperSummaryMessage(t, { ...base, reason: code, papers: [base] }, 0)).toBe(reasons[code])
    }
    expect(paperSummaryMessage(t, { ...base, reason: 'nothing_in_common', papers: two }, 0)).toBe(
      reasons.nothing_in_common
    )
  })

  test('a view reason replaces the summary', () => {
    for (const code of Object.keys(en.graphPanel.paperFilter.reason)) {
      expect(
        paperSummaryMessage(t, { ...base, reason: code, n_context_edges: 5, papers: [base] }, 0)
      ).toBe((en.graphPanel.paperFilter.reason as Record<string, string>)[code])
    }
  })
})
