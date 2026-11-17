import { useCamera, useSigma } from '@react-sigma/core'
import { useEffect } from 'react'
import { useGraphStore } from '@/stores/graph'

/**
 * Component that highlights a node and centers the camera on it.
 */
const FocusOnNode = ({ node, move }: { node: string | null; move?: boolean }) => {
  const sigma = useSigma()
  const { gotoNode } = useCamera()

  /**
   * When the selected item changes, highlighted the node and center the camera on it.
   */
  useEffect(() => {
    const graph = sigma.getGraph();

    if (move) {
      if (node && graph.hasNode(node)) {
        try {
          graph.setNodeAttribute(node, 'highlighted', true);
          gotoNode(node);
        } catch (error) {
          console.error('Error focusing on node:', error);
        }
      } else {
        // If no node is selected but move is true, reset to default view
        sigma.setCustomBBox(null);
        // Set the state, do not animate with duration 0: sigma divides the elapsed time by the
        // duration, so within one millisecond the camera becomes NaN until the next frame. The
        // container copies a camera's state into the instance it rebuilds (a paper map turns
        // edge events on for itself once its graph is loaded), and would keep the NaN.
        sigma.getCamera().setState({ x: 0.5, y: 0.5, ratio: 1 });
      }
      useGraphStore.getState().setMoveToSelectedNode(false);
    } else if (node && graph.hasNode(node)) {
      try {
        graph.setNodeAttribute(node, 'highlighted', true);
      } catch (error) {
        console.error('Error highlighting node:', error);
      }
    }

    return () => {
      if (node && graph.hasNode(node)) {
        try {
          graph.setNodeAttribute(node, 'highlighted', false);
        } catch (error) {
          console.error('Error cleaning up node highlight:', error);
        }
      }
    }
  }, [node, move, sigma, gotoNode])

  return null
}

export default FocusOnNode
