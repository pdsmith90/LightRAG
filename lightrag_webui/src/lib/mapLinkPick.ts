// Picking the links of the paper map. sigma picks an edge only where its drawn line is, which is
// one or two pixels wide at the default link width: a click beside it reaches the stage instead.
// The library has no wider picking area, so the map looks for the link nearest to the pointer
// itself, within a few pixels: a hover highlights that link and a click on the stage selects it.

import { DEFAULT_EDGE_CURVATURE } from '@sigma/edge-curve'

export type Point = { x: number; y: number }

// How far (CSS pixels) from a link the pointer still reaches it.
export const MAP_LINK_PICK_RADIUS = 6

// Edge types sigma draws straight; every other edge of the viewer is a curve (see GraphControl).
const STRAIGHT_TYPES = new Set(['arrow', 'rect', 'line'])

const CURVE_STEPS = 16

const segmentDistance = (p: Point, a: Point, b: Point): number => {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const length2 = dx * dx + dy * dy
  const t = length2 === 0 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length2))
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy))
}

// The control point of the quadratic curve from `source` to `target` as @sigma/edge-curve draws
// it (in viewport pixels, y pointing down): the middle of the chord, pushed sideways by
// `curvature` times the chord's length.
const controlPoint = (source: Point, target: Point, curvature: number): Point => ({
  x: (source.x + target.x) / 2 + curvature * (target.y - source.y),
  y: (source.y + target.y) / 2 - curvature * (target.x - source.x)
})

// The distance from `p` to the link between two viewport points: the line (curvature 0) or the
// curve sigma draws.
export const distanceToLink = (
  p: Point,
  source: Point,
  target: Point,
  curvature: number = DEFAULT_EDGE_CURVATURE
): number => {
  if (curvature === 0) return segmentDistance(p, source, target)
  const c = controlPoint(source, target, curvature)
  let previous = source
  let best = Infinity
  for (let i = 1; i <= CURVE_STEPS; i++) {
    const t = i / CURVE_STEPS
    const u = 1 - t
    const next = {
      x: u * u * source.x + 2 * u * t * c.x + t * t * target.x,
      y: u * u * source.y + 2 * u * t * c.y + t * t * target.y
    }
    best = Math.min(best, segmentDistance(p, previous, next))
    previous = next
  }
  return best
}

// The part of sigma that picking reads.
export type LinkPickSigma = {
  getGraph(): {
    forEachEdge(
      callback: (
        edge: string,
        attributes: Record<string, any>,
        source: string,
        target: string,
        sourceAttributes: Record<string, any>,
        targetAttributes: Record<string, any>
      ) => void
    ): void
  }
  graphToViewport(coordinates: Point): Point
}

// The link nearest to `point` (viewport pixels) within `radius`, or null. A link the graph itself
// hides is skipped; one that the viewer hides while another is highlighted (the reducers) is
// not, or the highlighted link would be the only one to pick.
export const nearestLink = (
  sigma: LinkPickSigma,
  point: Point,
  radius: number = MAP_LINK_PICK_RADIUS
): string | null => {
  const places = new Map<string, Point>()
  const place = (node: string, attributes: Record<string, any>): Point => {
    let at = places.get(node)
    if (!at) {
      at = sigma.graphToViewport({ x: attributes.x, y: attributes.y })
      places.set(node, at)
    }
    return at
  }

  let best: string | null = null
  let bestDistance = radius
  sigma.getGraph().forEachEdge((edge, attributes, source, target, sourceAttributes, targetAttributes) => {
    if (attributes.hidden) return
    const a = place(source, sourceAttributes)
    const b = place(target, targetAttributes)
    const curvature = STRAIGHT_TYPES.has(attributes.type) ? 0 : DEFAULT_EDGE_CURVATURE
    // The curve stays inside the triangle of its end points and control point.
    const c = curvature === 0 ? a : controlPoint(a, b, curvature)
    if (
      point.x < Math.min(a.x, b.x, c.x) - radius ||
      point.x > Math.max(a.x, b.x, c.x) + radius ||
      point.y < Math.min(a.y, b.y, c.y) - radius ||
      point.y > Math.max(a.y, b.y, c.y) + radius
    ) {
      return
    }
    const distance = distanceToLink(point, a, b, curvature)
    if (distance > bestDistance) return
    best = edge
    bestDistance = distance
  })
  return best
}

// The part of the graph store that the map's pointer handling reads and writes.
export type LinkPickStore = {
  focusedNode: string | null
  focusedEdge: string | null
  setFocusedEdge: (edge: string | null) => void
  setSelectedEdge: (edge: string | null) => void
  setSelectedNode: (node: string | null) => void
  clearSelection: () => void
}

type PointerEvent = { x: number; y: number; original?: MouseEvent | TouchEvent }
type StageEvent = { event: PointerEvent }

// The sigma events that give a map its tolerant link picking. `getStore` reads the live store.
export const mapLinkEvents = (sigma: LinkPickSigma, getStore: () => LinkPickStore) => ({
  // Hover: the nearest link is highlighted, unless the pointer is on a node or drags the map.
  mousemove: (event: PointerEvent) => {
    const store = getStore()
    if (store.focusedNode) return
    const original = event.original
    if (original && original.type.startsWith('mouse') && (original as MouseEvent).buttons !== 0) return
    const edge = nearestLink(sigma, event)
    if (edge !== store.focusedEdge) store.setFocusedEdge(edge)
  },
  // A click on the stage within reach of a link selects the link; elsewhere it clears the
  // selection as before.
  clickStage: ({ event }: StageEvent) => {
    const store = getStore()
    const edge = nearestLink(sigma, event)
    if (edge === null) {
      store.clearSelection()
      return
    }
    store.setSelectedEdge(edge)
    store.setSelectedNode(null)
  }
})
