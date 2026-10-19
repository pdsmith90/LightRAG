// The paper filter (fork) rides on the graph query label, so the label's
// fetch, refresh, retry and history handling cover it unchanged: a label
// `paper:<id>` shows what paper <id> contributed, `paper+refs:<id>` only what
// it shares with the library papers it cites. <id> is a DOI or a Zotero key.

export type PaperGraphMode = 'paper' | 'shared'

const PREFIXES: Record<PaperGraphMode, string> = {
  paper: 'paper:',
  shared: 'paper+refs:'
}

export const paperLabel = (paper: string, mode: PaperGraphMode): string =>
  `${PREFIXES[mode]}${paper.trim()}`

export const parsePaperLabel = (
  label: string | null | undefined
): { paper: string; mode: PaperGraphMode } | null => {
  for (const mode of ['shared', 'paper'] as const) {
    const prefix = PREFIXES[mode]
    if (label?.startsWith(prefix)) {
      const paper = label.slice(prefix.length).trim()
      return paper ? { paper, mode } : null
    }
  }
  return null
}
