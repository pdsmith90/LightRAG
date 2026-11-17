import { afterEach, describe, expect, test } from 'bun:test'

import en from '@/locales/en.json'
import ar from '@/locales/ar.json'
import de from '@/locales/de.json'
import fr from '@/locales/fr.json'
import id from '@/locales/id.json'
import ja from '@/locales/ja.json'
import ko from '@/locales/ko.json'
import ru from '@/locales/ru.json'
import uk from '@/locales/uk.json'
import vi from '@/locales/vi.json'
import zh from '@/locales/zh.json'
import zhTW from '@/locales/zh_TW.json'
import type { PaperSummary } from '@/api/lightrag'
import {
  __resetPaperContextForTests,
  CONTEXT_EDGE_COLOR,
  clearsLabelOnEmpty,
  edgeEventsActive,
  expandMapLabel,
  isChosenMapNode,
  isPaperMapUnavailable,
  isStalePaperResult,
  mapFallbackLabel,
  mapHasMorePapers,
  mapLinkKind,
  MAP_CITATION_COLOR,
  MAP_COUPLING_COLOR,
  MAP_DEFAULT_LIMIT,
  fadeColor,
  nextContext,
  nextHops,
  nextMapLimit,
  nextMode,
  paperEdgeAttrs,
  paperLabel,
  paperMapEdgeAttrs,
  paperNodeAttrs,
  paperSummaryMessage,
  parsePaperLabel,
  raiseMapLimit,
  readMapHops,
  readMapLimit,
  readPaperContext,
  togglePaper,
  writeMapHops,
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

describe('paper map', () => {
  test('a map label has its own prefix and round-trips', () => {
    expect(paperLabel(['10.1/a', 'ABCD1234'], 'map')).toBe('paper+map:10.1/a ABCD1234')
    expect(parsePaperLabel('paper+map:10.1/a ABCD1234')).toEqual({
      papers: ['10.1/a', 'ABCD1234'],
      mode: 'map'
    })
    // not read as a paper label whose first id is "+map:..."
    expect(parsePaperLabel('paper+map:10.1/a')?.mode).toBe('map')
  })

  test('the mode cycle offers the map only when the server has it', () => {
    expect(nextMode('shared', 1, true)).toBe('map')
    expect(nextMode('shared', 2, true)).toBe('common')
    expect(nextMode('common', 2, true)).toBe('map')
    expect(nextMode('map', 2, true)).toBe('paper')
    expect(nextMode('map', 1, false)).toBe('paper')
    // unavailable: the cycle is the one it always was
    expect(nextMode('shared', 1, false)).toBe('paper')
    expect(nextMode('common', 2)).toBe('paper')
  })

  test('expanding a map node adds it to the papers, at most ten', () => {
    expect(expandMapLabel(['10.1/a'], { id: 'ABCD1234' })).toEqual({
      label: 'paper+map:10.1/a ABCD1234'
    })
    expect(expandMapLabel(['10.1/a'], { id: '10.1/a' })).toEqual({ reason: 'mapAlreadyChosen' })
    const ten = Array.from({ length: 10 }, (_, i) => `10.1/${i}`)
    expect(expandMapLabel(ten, { id: '10.1/new' })).toEqual({ reason: 'maxPapers' })
    expect(en.graphPanel.paperFilter.mapAlreadyChosen).toBeTruthy()
  })

  test('a node is a chosen paper by its DOI, attachment key or structure id', () => {
    // chosen by key, shown under its DOI (the server's node id)
    const node = {
      id: '10.1/a',
      properties: { doi: '10.1/a', attachment_key: 'AAAA0001', paper_id: 'doi:10.1/a' }
    }
    expect(isChosenMapNode(['AAAA0001'], node)).toBe(true)
    expect(isChosenMapNode(['aaaa0001'], node)).toBe(true)
    expect(isChosenMapNode(['10.1/A'], node)).toBe(true)
    expect(isChosenMapNode(['https://doi.org/10.1/a'], node)).toBe(true)
    expect(isChosenMapNode(['doi:10.1/a'], node)).toBe(true)
    expect(isChosenMapNode(['10.1/b', 'BBBB0001'], node)).toBe(false)
    // a paper without a DOI: its structure id or the key it is shown under
    const report = {
      id: 'KKKK0001',
      properties: { paper_id: 'zot:ITEM0001', attachment_key: 'KKKK0001' }
    }
    expect(isChosenMapNode(['zot:ITEM0001'], report)).toBe(true)
    expect(isChosenMapNode(['KKKK0001'], report)).toBe(true)
    expect(isChosenMapNode(['10.1/a'], report)).toBe(false)
    expect(isChosenMapNode([], node)).toBe(false)
    // expanding such a node says it is already chosen instead of adding it
    expect(expandMapLabel(['AAAA0001'], node)).toEqual({ reason: 'mapAlreadyChosen' })
    expect(expandMapLabel(['AAAA0001'], { id: '10.1/b', properties: { doi: '10.1/b' } })).toEqual({
      label: 'paper+map:AAAA0001 10.1/b'
    })
  })

  test('only a 501 or 404 means the server has no paper map', () => {
    expect(isPaperMapUnavailable(Object.assign(new Error('x'), { status: 501 }))).toBe(true)
    expect(isPaperMapUnavailable(Object.assign(new Error('x'), { status: 404 }))).toBe(true)
    expect(isPaperMapUnavailable(Object.assign(new Error('x'), { status: 500 }))).toBe(false)
    expect(isPaperMapUnavailable(Object.assign(new Error('x'), { status: 422 }))).toBe(false)
    expect(isPaperMapUnavailable(new Error('Network Error'))).toBe(false)
    expect(isPaperMapUnavailable(null)).toBe(false)
    expect(mapFallbackLabel(['10.1/a', 'AAAA0001'])).toBe('paper:10.1/a AAAA0001')
    expect(en.graphPanel.paperFilter.mapUnavailable).toBeTruthy()
  })

  test('the reach is stored and read back, defaulting to one hop', () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
    const data = new Map<string, string>()
    Object.defineProperty(globalThis, 'localStorage', {
      value: {
        getItem: (k: string) => data.get(k) ?? null,
        setItem: (k: string, v: string) => void data.set(k, v)
      },
      configurable: true
    })
    try {
      __resetPaperContextForTests()
      expect(readMapHops()).toBe(1)
      writeMapHops(nextHops(readMapHops()))
      expect(data.get('lightrag-paper-map-hops')).toBe('2')
      __resetPaperContextForTests()
      expect(readMapHops()).toBe(2)
      expect(nextHops(2)).toBe(1)
    } finally {
      __resetPaperContextForTests()
      if (original) Object.defineProperty(globalThis, 'localStorage', original)
    }
  })

  test('a map summary counts the papers at each reach and the unknown ones', () => {
    const t = (key: string, options: Record<string, unknown> = {}): string => {
      const text = key.split('.').reduce<any>((node, part) => node?.[part], en)
      expect(typeof text).toBe('string')
      return (text as string).replace(/{{(\w+)}}/g, (_, name) => String(options[name]))
    }
    const seed = { paper: '10.1/a', doi: '10.1/a', keys: [], title: 'A', citation: 'A', year: '2020', n_refs: 1, n_cited: 1, n_cited_in_kb: 1 } as PaperSummary
    const map = { ...seed, mode: 'map', hops: 2, shown_hop1: 20, shown_hop2: 9, papers: [seed] } as PaperSummary
    expect(paperSummaryMessage(t, map, 30)).toBe(
      'Paper map: 30 papers found and shown (20 one step away, 9 two steps away)'
    )
    const unknown = { ...map, papers: [seed, { paper: 'ZZZZ9999', reason: 'not_in_structure' } as PaperSummary] }
    expect(paperSummaryMessage(t, unknown, 30)).toContain(
      '; 1 of the selected papers are not in the paper structure'
    )
    // links the server thinned are said so; all links shown say nothing
    const thinned = { ...map, n_links: 25209, shown_links: 4000, links_thinned: true }
    expect(paperSummaryMessage(t, thinned, 30)).toBe(
      'Paper map: 30 papers found and shown (20 one step away, 9 two steps away); ' +
        '4000 of the 25209 links between them shown, the strongest for each paper'
    )
    const whole = { ...map, n_links: 12, shown_links: 12 }
    expect(paperSummaryMessage(t, whole, 30)).toBe(paperSummaryMessage(t, map, 30))
    const none = { ...map, reason: 'not_in_structure', papers: [seed, seed] }
    expect(paperSummaryMessage(t, none, 0)).toBe(
      en.graphPanel.paperFilter.reason.not_in_structure_many
    )
    expect(paperSummaryMessage(t, { ...map, reason: 'no_links' }, 1)).toBe(
      en.graphPanel.paperFilter.reason.no_links
    )
  })
})

describe('paper map size', () => {
  const t = (key: string, options: Record<string, unknown> = {}): string => {
    const text = key.split('.').reduce<any>((node, part) => node?.[part], en)
    expect(typeof text).toBe('string')
    return (text as string).replace(/{{(\w+)}}/g, (_, name) => String(options[name]))
  }
  const seed = { paper: '10.1/a', doi: '10.1/a', keys: [], title: 'A', citation: 'A', year: '2020', n_refs: 1, n_cited: 1, n_cited_in_kb: 1 } as PaperSummary
  const map = {
    ...seed,
    mode: 'map',
    hops: 2,
    n_papers: 612,
    shown_papers: 150,
    max_nodes: 150,
    max_nodes_cap: 1000,
    shown_hop1: 40,
    shown_hop2: 109,
    papers: [seed]
  } as PaperSummary

  afterEach(() => __resetPaperContextForTests())

  test('the summary line says how many papers were found and how many are shown', () => {
    expect(paperSummaryMessage(t, map, 150)).toBe(
      'Paper map: 612 papers found, 150 shown (limit 150; 40 one step away, 109 two steps away). ' +
        '"Show more papers" raises the limit.'
    )
    // at the server's ceiling there is no more to show, and the line does not say there is
    expect(paperSummaryMessage(t, { ...map, max_nodes: 1000 }, 150)).toBe(
      'Paper map: 612 papers found, 150 shown (limit 1000; 40 one step away, 109 two steps away)'
    )
    // nothing left out: said once
    const all = { ...map, n_papers: 150 }
    expect(paperSummaryMessage(t, all, 150)).toBe(
      'Paper map: 150 papers found and shown (40 one step away, 109 two steps away)'
    )
    // a server that does not count them: every paper is taken as shown
    const old = { ...map, n_papers: undefined, shown_papers: undefined, max_nodes: undefined }
    expect(paperSummaryMessage(t, old, 150)).toContain('150 papers found and shown')
  })

  test('thinned links are said only when the server says so, apart from the papers', () => {
    const thinned = { ...map, n_links: 900, shown_links: 800, links_thinned: true }
    expect(paperSummaryMessage(t, thinned, 150)).toBe(
      'Paper map: 612 papers found, 150 shown (limit 150; 40 one step away, 109 two steps away); ' +
        '800 of the 900 links between them shown, the strongest for each paper. ' +
        '"Show more papers" raises the limit.'
    )
    const whole = { ...map, n_links: 900, shown_links: 900, links_thinned: false }
    expect(paperSummaryMessage(t, whole, 150)).not.toContain('links between them')
  })

  test('the limit belongs to one map, starts at 150 and doubles up to the server ceiling', () => {
    expect(MAP_DEFAULT_LIMIT).toBe(150)
    expect(readMapLimit('paper+map:10.1/a')).toBe(150)
    expect(nextMapLimit(150, 1000)).toBe(300)
    expect(nextMapLimit(600, 1000)).toBe(1000)
    expect(nextMapLimit(1000, 1000)).toBe(1000)
    // a server whose ceiling is below the default never asks for more than the default
    expect(nextMapLimit(150, 100)).toBe(150)
    expect(raiseMapLimit('paper+map:10.1/a', 1000)).toBe(300)
    expect(raiseMapLimit('paper+map:10.1/a', 1000)).toBe(600)
    expect(readMapLimit('paper+map:10.1/a')).toBe(600)
    // another selection starts again
    expect(readMapLimit('paper+map:10.1/a 10.1/b')).toBe(150)
    expect(readMapLimit('paper+map:10.1/b')).toBe(150)
  })

  test('more papers are offered only for a map that left papers out under the ceiling', () => {
    expect(mapHasMorePapers(map)).toBe(true)
    expect(mapHasMorePapers({ ...map, shown_papers: 612 })).toBe(false)
    expect(mapHasMorePapers({ ...map, max_nodes: 1000 })).toBe(false)
    expect(mapHasMorePapers({ ...map, reason: 'no_links' })).toBe(false)
    expect(mapHasMorePapers({ ...map, mode: 'paper' })).toBe(false)
    expect(mapHasMorePapers(undefined)).toBe(false)
    expect(mapHasMorePapers({ ...map, n_papers: undefined })).toBe(false)
  })

  test('the reach tooltips name the limit and the new strings exist in every locale', () => {
    expect(en.graphPanel.paperFilter.mapOneHopTooltip).toContain('{{limit}}')
    expect(en.graphPanel.paperFilter.mapTwoHopsTooltip).toContain('{{limit}}')
    const locales = { ar, de, fr, id, ja, ko, ru, uk, vi, zh, zhTW }
    for (const key of ['mapSummaryLimited', 'mapShowMore', 'mapMoreSuffix', 'legendCitation', 'legendCoupling', 'linkKind', 'linkDirectionNone']) {
      expect(key in de.graphPanel.paperFilter, key).toBe(true)
    }
    const keys = (o: Record<string, any>): string[] => Object.keys(o).sort()
    // the locales carry the paper map's strings (English alone has the older filter ones)
    const mapKeys = keys(en.graphPanel.paperFilter).filter((k) => k in de.graphPanel.paperFilter)
    for (const [name, locale] of Object.entries(locales)) {
      const have = locale.graphPanel.paperFilter as Record<string, any>
      expect(keys(have), name).toEqual(mapKeys)
      expect(keys(have.reason), name).toEqual(keys(de.graphPanel.paperFilter.reason))
      for (const key of ['mapOneHopTooltip', 'mapTwoHopsTooltip', 'mapSummaryLimited', 'mapShowMore']) {
        expect(have[key], `${name}.${key}`).toContain('{{limit}}')
      }
      for (const key of ['mapSummary', 'mapSummaryLimited']) {
        expect(have[key], `${name}.${key}`).toContain('{{nodes}}')
      }
      expect(have.linkDirectionCites).toContain('{{source}}')
      expect(have.linkDirectionCites).toContain('{{target}}')
    }
  })
})

describe('paper map links', () => {
  test('a link is a citation one way or both, shared references, or both kinds', () => {
    expect(mapLinkKind({ keywords: 'cites', direction: 'source cites target' })).toEqual({
      cites: 'one',
      coupled: false,
      sharedRefs: null
    })
    expect(mapLinkKind({ keywords: 'cite each other', direction: 'both' }).cites).toBe('both')
    expect(mapLinkKind({ keywords: 'coupled', shared_refs: 7 })).toEqual({
      cites: 'none',
      coupled: true,
      sharedRefs: 7
    })
    expect(
      mapLinkKind({ keywords: 'cites, coupled', direction: 'source cites target', shared_refs: 3 })
    ).toEqual({ cites: 'one', coupled: true, sharedRefs: 3 })
    expect(mapLinkKind(undefined)).toEqual({ cites: 'none', coupled: false, sharedRefs: null })
  })

  test('citations are arrows in one colour, coupling is a plain line in another', () => {
    const cites = paperMapEdgeAttrs({ properties: { direction: 'source cites target' } }, 100)
    expect(cites).toEqual({ color: MAP_CITATION_COLOR, type: 'curvedArrow' })
    // straight arrows when the graph is too large for curves
    expect(paperMapEdgeAttrs({ properties: { direction: 'source cites target' } }, 6000).type).toBe(
      'arrow'
    )
    const coupled = paperMapEdgeAttrs({ properties: { keywords: 'coupled', shared_refs: 4 } }, 100)
    expect(coupled).toEqual({ color: MAP_COUPLING_COLOR })
    expect('type' in coupled).toBe(false)
    // each cites the other: the citation colour, no arrowhead that points one way only
    expect(paperMapEdgeAttrs({ properties: { direction: 'both' } }, 100)).toEqual({
      color: MAP_CITATION_COLOR
    })
    // a citation that is also coupling is a citation
    expect(
      paperMapEdgeAttrs(
        { properties: { direction: 'source cites target', shared_refs: 2, keywords: 'cites, coupled' } },
        100
      ).type
    ).toBe('curvedArrow')
    expect(MAP_CITATION_COLOR).not.toBe(MAP_COUPLING_COLOR)
  })

  test('a paper map picks its edges whatever the setting says, within the edge limit', () => {
    expect(edgeEventsActive(false, false, 10)).toBe(false)
    expect(edgeEventsActive(true, false, 10)).toBe(true)
    expect(edgeEventsActive(false, true, 10)).toBe(true)
    expect(edgeEventsActive(true, true, 5000)).toBe(true)
    expect(edgeEventsActive(false, true, 5001)).toBe(false)
    expect(edgeEventsActive(true, false, 5001)).toBe(false)
  })

  test('a paper chosen by its second attachment key is recognised and not added again', () => {
    const node = {
      id: '10.1/a',
      properties: {
        doi: '10.1/a',
        attachment_key: 'AAAA0002',
        attachment_keys: ['AAAA0001', 'AAAA0002'],
        paper_id: 'doi:10.1/a'
      }
    }
    expect(isChosenMapNode(['AAAA0001'], node)).toBe(true)
    expect(isChosenMapNode(['aaaa0002'], node)).toBe(true)
    expect(isChosenMapNode(['AAAA0003'], node)).toBe(false)
    expect(expandMapLabel(['AAAA0001'], node)).toEqual({ reason: 'mapAlreadyChosen' })
    // a server that sends no list: the one key it sends still counts
    expect(isChosenMapNode(['AAAA0002'], { ...node, properties: { attachment_key: 'AAAA0002' } })).toBe(true)
  })
})
