// A static 3-d tree over the lit nodes, for "what is under the pointer".
//
// Picking a point cloud cannot be done by raycasting geometry: the nodes are
// GPU point sprites with no faces to hit, and `THREE.Points` raycasting walks
// every vertex per event anyway. So the pointer ray is intersected with the
// cloud analytically — nearest node to the ray, within a radius — and that
// needs a spatial index rather than a scan of the whole set on every mousemove.
//
// Rebuilt when the lit set changes and not otherwise: the tree is over the
// *active* nodes (~1,000 at cell scope, ~6,000 across a whole trace), never
// over the 20,000-node idle sample, which is not interactive.
//
// Free of runtime imports, like lens.ts and lit.ts, so it can be exercised
// under node without a WebGL context.

export interface Point3 {
  x: number
  y: number
  z: number
}

interface Node<T> {
  point: T
  axis: 0 | 1 | 2
  left: Node<T> | null
  right: Node<T> | null
}

const AXES = ['x', 'y', 'z'] as const

function coord(point: Point3, axis: 0 | 1 | 2): number {
  return point[AXES[axis]]
}

/**
 * A balanced 3-d tree, built once per lit set.
 *
 * Balanced by construction — the median along the widest-variance axis is
 * chosen at each level — because the node cloud is anything but uniform: UMAP
 * output is clumped, and a naive axis-cycling split on it degenerates toward a
 * list.
 */
export class KdTree<T extends Point3> {
  private readonly root: Node<T> | null

  constructor(points: readonly T[]) {
    this.root = build(points.slice(), 0)
  }

  /**
   * The nearest point to `query` within `maxDistance`, or null.
   *
   * The radius is not a convenience: without it the nearest node to a pointer
   * anywhere on screen is *some* node, and the interface would report a
   * feature for a click on empty space.
   */
  nearest(query: Point3, maxDistance: number): T | null {
    let best: T | null = null
    let bestSquared = maxDistance * maxDistance

    const visit = (node: Node<T> | null): void => {
      if (node === null) return

      const dx = query.x - node.point.x
      const dy = query.y - node.point.y
      const dz = query.z - node.point.z
      const squared = dx * dx + dy * dy + dz * dz
      if (squared < bestSquared) {
        bestSquared = squared
        best = node.point
      }

      const delta = coord(query, node.axis) - coord(node.point, node.axis)
      const near = delta < 0 ? node.left : node.right
      const far = delta < 0 ? node.right : node.left

      visit(near)
      // The far side can only hold something closer if the splitting plane
      // itself is within the current best radius.
      if (delta * delta < bestSquared) visit(far)
    }

    visit(this.root)
    return best
  }
}

function build<T extends Point3>(points: T[], depth: number): Node<T> | null {
  if (points.length === 0) return null

  const axis = widestAxis(points)
  points.sort((a, b) => coord(a, axis) - coord(b, axis))
  const middle = points.length >> 1

  return {
    point: points[middle],
    axis,
    left: build(points.slice(0, middle), depth + 1),
    right: build(points.slice(middle + 1), depth + 1),
  }
}

/** The axis with the largest spread, so the split actually divides the set. */
function widestAxis(points: readonly Point3[]): 0 | 1 | 2 {
  let best: 0 | 1 | 2 = 0
  let bestSpread = -1
  for (const axis of [0, 1, 2] as const) {
    let low = Infinity
    let high = -Infinity
    for (const point of points) {
      const value = coord(point, axis)
      if (value < low) low = value
      if (value > high) high = value
    }
    const spread = high - low
    if (spread > bestSpread) {
      bestSpread = spread
      best = axis
    }
  }
  return best
}

/**
 * The point on a ray closest to each node, for pointer picking.
 *
 * Returns the query point to hand to `nearest`: the ray is walked at a fixed
 * step and the closest sample is used, which is enough for a pointer test and
 * avoids projecting every node onto the ray.
 */
export function rayPoints(
  origin: Point3,
  direction: Point3,
  near: number,
  far: number,
  steps: number,
): Point3[] {
  const out: Point3[] = []
  for (let i = 0; i <= steps; i++) {
    const t = near + ((far - near) * i) / steps
    out.push({
      x: origin.x + direction.x * t,
      y: origin.y + direction.y * t,
      z: origin.z + direction.z * t,
    })
  }
  return out
}
