/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import '@/test/webglStub'

const { DEFAULT_EDGE_ARROW_HEAD_PROGRAM_OPTIONS, EdgeArrowProgram } = await import('sigma/rendering')
const { EdgeCurvedArrowProgram } = await import('@sigma/edge-curve')
const { MAP_ARROW_HEAD, MapArrowProgram, MapCurvedArrowProgram } = await import(
  './mapArrowPrograms'
)

/**
 * What this can show: the map registers its own arrow programs, with a head that is a clearly
 * larger multiple of the link's thickness than sigma's default (2.5 long, 2 wide: two or three
 * pixels at the default link width). How large that looks on screen needs WebGL and a browser.
 */
describe('the arrowheads of a map', () => {
  test('are larger than sigma\'s own, for the straight and the curved arrow', () => {
    expect(MAP_ARROW_HEAD.lengthToThicknessRatio).toBeGreaterThanOrEqual(
      3 * DEFAULT_EDGE_ARROW_HEAD_PROGRAM_OPTIONS.lengthToThicknessRatio
    )
    expect(MAP_ARROW_HEAD.widenessToThicknessRatio).toBeGreaterThanOrEqual(
      3 * DEFAULT_EDGE_ARROW_HEAD_PROGRAM_OPTIONS.widenessToThicknessRatio
    )
    expect(MapArrowProgram).not.toBe(EdgeArrowProgram)
    expect(MapCurvedArrowProgram).not.toBe(EdgeCurvedArrowProgram)
  })
})
