/// <reference types="bun" />
import { afterEach, describe, expect, spyOn, test } from 'bun:test'
import '@/test/webglStub'
import { render } from '@testing-library/react'

import { useGraphStore } from '@/stores/graph'

// sigma is loaded after the stub (a static import of a CommonJS package runs before it).
const { SigmaProvider } = await import('@react-sigma/core')
const { Camera } = await import('sigma')
const { default: FocusOnNode } = await import('./FocusOnNode')

/**
 * The viewer puts the camera back to the default view whenever a graph has been loaded. The
 * SigmaContainer copies the camera of the instance it replaces into the new one, which is what
 * happens to a paper map once its graph is in (it turns edge events on, and those cannot be
 * switched on in a live instance). A camera that is NaN at that moment is copied as NaN and
 * nothing is drawn.
 *
 * What this proves: a real sigma Camera, driven by FocusOnNode, holds the default view at once.
 * What it cannot prove: sigma's rendering, which needs WebGL; the rebuild itself is the
 * container's own code and is not run here.
 */
const render_ = (move: boolean) => {
  const camera = new Camera()
  const sigma = {
    getGraph: () => ({ hasNode: () => false }),
    setCustomBBox: () => {},
    getCamera: () => camera
  }
  render(
    <SigmaProvider value={{ sigma: sigma as never, container: document.createElement('div') }}>
      <FocusOnNode node={null} move={move} />
    </SigmaProvider>
  )
  return camera
}

afterEach(() => {
  useGraphStore.getState().setMoveToSelectedNode(false)
})

describe('FocusOnNode', () => {
  test('the default view is a finite camera state at once, even within one millisecond', () => {
    // sigma divides the elapsed time of an animation by its duration: 0 / 0 within one
    // millisecond. Fixing the clock makes that the case every time.
    const now = spyOn(Date, 'now').mockReturnValue(1_000_000)
    try {
      const camera = render_(true)
      const state = camera.getState()
      expect(state).toEqual({ x: 0.5, y: 0.5, angle: 0, ratio: 1 })
      expect(Object.values(state).every(Number.isFinite)).toBe(true)
      // nothing left animating that would write a state later
      expect(camera.isAnimated()).toBe(false)
    } finally {
      now.mockRestore()
    }
  })

  test('without a move request the camera is left alone', () => {
    const camera = render_(false)
    camera.setState({ x: 0.2, y: 0.9, ratio: 3 })
    expect(camera.getState().ratio).toBe(3)
  })
})
