import { useCallback, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowLeft } from 'lucide-react'

import { getPapersByTopic, getPaperTopics, type PaperTopic } from '@/api/lightrag'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { useDebounce } from '@/hooks/useDebounce'
import { groupTopics } from '@/lib/citationGraph'
import PanelBody from './PanelBody'
import PaperRow from './PaperRow'
import { useFetched } from './useFetched'

const TOPIC_LIMIT = 200
const PAPER_LIMIT = 50
const LEVELS = ['domain', 'field', 'subfield', 'topic'] as const

// The papers under one topic, best first; a click on a paper opens it in the paper view.
const TopicPapers = ({
  topic,
  onSelectTopic,
  onBack,
  onOpenPaper
}: {
  topic: PaperTopic
  onSelectTopic: (topic: PaperTopic) => void
  onBack: () => void
  onOpenPaper: (id: string) => void
}) => {
  const { t } = useTranslation()
  const topicId = topic.topic_id
  const load = useCallback(() => getPapersByTopic(topicId, PAPER_LIMIT), [topicId])
  const { data, error, loading } = useFetched(load)

  return (
    <div className="flex flex-col">
      <div className="flex flex-col gap-1 border-b px-3 py-2">
        <Button variant="ghost" size="sm" className="w-fit" onClick={onBack}>
          <ArrowLeft />
          {t('citationGraph.topics.back')}
        </Button>
        <div className="text-sm font-medium">{topic.name}</div>
        <div className="text-muted-foreground text-xs">{topic.path}</div>
        {data && !data.reason && (
          <div className="text-muted-foreground text-xs">
            {t('citationGraph.topics.paperCount', {
              shown: data.papers.length,
              total: data.total,
              inKb: data.in_kb ?? 0
            })}
          </div>
        )}
        {!!data?.alternatives?.length && (
          <div className="flex flex-wrap items-center gap-1 text-xs">
            <span className="text-muted-foreground">{t('citationGraph.topics.alternatives')}</span>
            {data.alternatives.map((alternative) => (
              <button
                key={alternative.topic_id}
                type="button"
                className="text-primary hover:underline"
                onClick={() => onSelectTopic(alternative)}
              >
                {alternative.name}
              </button>
            ))}
          </div>
        )}
      </div>
      <PanelBody loading={loading} error={error} reason={data?.reason}>
        <ul>
          {data?.papers.map((paper) => (
            <PaperRow key={paper.paper_id} paper={paper} onOpen={onOpenPaper}>
              {paper.primary && (
                <Badge variant="secondary" className="w-fit font-normal">
                  {t('citationGraph.topics.primary')}
                </Badge>
              )}
            </PaperRow>
          ))}
        </ul>
      </PanelBody>
    </div>
  )
}

// The topic browser: the library's topics, subfields, fields and domains grouped by the path
// above them and sorted by their papers, and the papers of the one picked.
const TopicsPanel = ({ onOpenPaper }: { onOpenPaper: (id: string) => void }) => {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [level, setLevel] = useState<PaperTopic['level'] | null>('subfield')
  const [topic, setTopic] = useState<PaperTopic | null>(null)
  const debounced = useDebounce(query.trim(), 400)
  const load = useCallback(() => getPaperTopics(debounced, level, TOPIC_LIMIT), [debounced, level])
  const { data, error, loading } = useFetched(load)
  const groups = useMemo(() => groupTopics(data?.topics ?? []), [data])

  if (topic) {
    return (
      <TopicPapers
        topic={topic}
        onSelectTopic={setTopic}
        onBack={() => setTopic(null)}
        onOpenPaper={onOpenPaper}
      />
    )
  }

  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b p-2">
        <Input
          className="h-8 max-w-[260px]"
          value={query}
          aria-label={t('citationGraph.topics.search')}
          placeholder={t('citationGraph.topics.search')}
          onChange={(event) => setQuery(event.target.value)}
        />
        <div className="flex gap-1" role="group" aria-label={t('citationGraph.topics.level')}>
          {[null, ...LEVELS].map((name) => (
            <Button
              key={name ?? 'all'}
              size="sm"
              className="h-7 px-2 text-xs"
              variant={level === name ? 'secondary' : 'ghost'}
              aria-pressed={level === name}
              onClick={() => setLevel(name)}
            >
              {t(`citationGraph.topics.levels.${name ?? 'all'}`)}
            </Button>
          ))}
        </div>
      </div>
      <PanelBody loading={loading} error={error}>
        {data && (
          <>
            <p className="text-muted-foreground px-3 py-2 text-xs">
              {t('citationGraph.topics.count', { shown: data.topics.length, total: data.total })}
            </p>
            {groups.length === 0 && (
              <p className="text-muted-foreground px-3 pb-3 text-sm">{t('citationGraph.topics.empty')}</p>
            )}
            {groups.map((group) => (
              <section key={group.path} className="pb-2">
                {group.path && (
                  <h3 className="text-muted-foreground bg-muted/40 px-3 py-1 text-xs font-medium">{group.path}</h3>
                )}
                <ul>
                  {group.topics.map((item) => (
                    <li key={item.topic_id}>
                      <button
                        type="button"
                        className="hover:bg-accent flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-sm"
                        onClick={() => setTopic(item)}
                      >
                        <span>{item.name}</span>
                        <span className="text-muted-foreground shrink-0 text-xs">
                          {t('citationGraph.topics.papers', { count: item.n_papers })}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ))}
          </>
        )}
      </PanelBody>
    </div>
  )
}

export default TopicsPanel
