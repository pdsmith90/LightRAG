import { describe, expect, test } from 'bun:test'

import { paperLabel, parsePaperLabel } from './paperFilter'

describe('paper filter labels', () => {
  test('a paper label round-trips its paper and mode', () => {
    expect(parsePaperLabel(paperLabel('10.1002/2013jb010900', 'paper'))).toEqual({
      paper: '10.1002/2013jb010900',
      mode: 'paper'
    })
    expect(parsePaperLabel(paperLabel(' ABCD1234 ', 'shared'))).toEqual({
      paper: 'ABCD1234',
      mode: 'shared'
    })
  })

  test('entity labels and empty papers are not paper labels', () => {
    expect(parsePaperLabel('GRACE')).toBeNull()
    expect(parsePaperLabel('*')).toBeNull()
    expect(parsePaperLabel('')).toBeNull()
    expect(parsePaperLabel(null)).toBeNull()
    expect(parsePaperLabel('paper:')).toBeNull()
    expect(parsePaperLabel('paper+refs:  ')).toBeNull()
  })
})
