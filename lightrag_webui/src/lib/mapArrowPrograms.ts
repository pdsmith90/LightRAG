// The arrowheads of the paper map's citation links. sigma sizes an arrowhead as a multiple of
// the link's thickness (length 2.5, width 2 by default): at the default link width of one pixel
// that is a head two or three pixels long, lost among 150 papers. A larger multiple makes the
// head clear without thickening the line.

import { createEdgeArrowProgram } from 'sigma/rendering'
import { createEdgeCurveProgram } from '@sigma/edge-curve'

export const MAP_ARROW_HEAD = { lengthToThicknessRatio: 8, widenessToThicknessRatio: 6 }

// Straight arrow (a map too large for curves) and curved arrow (the usual one).
export const MapArrowProgram = createEdgeArrowProgram(MAP_ARROW_HEAD)
export const MapCurvedArrowProgram = createEdgeCurveProgram({
  arrowHead: { extremity: 'target', ...MAP_ARROW_HEAD }
})
