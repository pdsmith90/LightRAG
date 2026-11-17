import React from 'react'
import { useTranslation } from 'react-i18next'
import { useGraphStore } from '@/stores/graph'
import { Card } from '@/components/ui/Card'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { MAP_CITATION_COLOR, MAP_COUPLING_COLOR } from '@/lib/paperFilter'

interface LegendProps {
  className?: string
}

// A sample of a paper-map link: a line, with an arrowhead for a citation.
const LinkSample = ({ color, arrow }: { color: string; arrow?: boolean }) => (
  <svg width="24" height="12" viewBox="0 0 24 12" className="shrink-0" aria-hidden="true">
    <line x1="1" y1="6" x2={arrow ? 17 : 23} y2="6" stroke={color} strokeWidth="2" />
    {arrow && <polygon points="23,6 15,2 15,10" fill={color} />}
  </svg>
)

const Legend: React.FC<LegendProps> = ({ className }) => {
  const { t } = useTranslation()
  const typeColorMap = useGraphStore.use.typeColorMap()
  const paperMap = useGraphStore.use.rawGraph()?.isPaperMap === true

  if (!typeColorMap || typeColorMap.size === 0) {
    return null
  }

  return (
    <Card className={`p-2 max-w-xs ${className}`}>
      <h3 className="text-sm font-medium mb-2">{t('graphPanel.legend')}</h3>
      <ScrollArea className="max-h-80">
        <div className="flex flex-col gap-1">
          {Array.from(typeColorMap.entries()).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs truncate" title={type}>
                {t(`graphPanel.nodeTypes.${String(type).toLowerCase().replace(/\s+/g, '')}`, type)}
              </span>
            </div>
          ))}
          {paperMap && (
            <>
              <div className="flex items-center gap-2 pt-1">
                <LinkSample color={MAP_CITATION_COLOR} arrow />
                <span className="text-xs">{t('graphPanel.paperFilter.legendCitation')}</span>
              </div>
              <div className="flex items-center gap-2">
                <LinkSample color={MAP_COUPLING_COLOR} />
                <span className="text-xs">{t('graphPanel.paperFilter.legendCoupling')}</span>
              </div>
            </>
          )}
        </div>
      </ScrollArea>
    </Card>
  )
}

export default Legend
