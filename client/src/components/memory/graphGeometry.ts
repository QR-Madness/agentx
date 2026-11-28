// ─── Graph geometry ────────────────────────────────────────────────────────
// Pure helpers behind Memory → Explore (MemoryGraphView): where the neighbours
// of the centre node sit, and where a straight edge between two nodes starts
// and ends. They know nothing of React Flow, so the routing is unit-tested
// (graphGeometry.test.ts). Units are React Flow's flow units, with y pointing
// down.

export interface Point {
  x: number;
  y: number;
}

/** A node's box: its top-left corner and its size. */
export interface Box extends Point {
  width: number;
  height: number;
}

/**
 * The widest a node gets. `.graph-node` in MemoryPanel.css caps its width so
 * that long names wrap, and the ring below is sized against that cap
 * (graphGeometry.test.ts keeps the two in step).
 */
export const NODE_MAX_WIDTH = 200;

/**
 * Radii of the ring of neighbours, measured between node centres. The ring is
 * wider than it is tall because the nodes are: a neighbour level with the
 * centre needs room for two half-widths plus the edge label between them.
 */
export const RING_RADIUS_X = 340;
export const RING_RADIUS_Y = 220;

/**
 * How many neighbours the ring holds before it grows to keep them apart. Past
 * it, the radii grow with the count, so the gap between neighbours stays what
 * it is at capacity (graphGeometry.test.ts checks up to 50 neighbours, the
 * most the entity graph endpoint returns).
 */
export const RING_CAPACITY = 10;

const centreOf = (box: Box): Point => ({ x: box.x + box.width / 2, y: box.y + box.height / 2 });

/**
 * Centres for `count` neighbours on the ring around the origin, where the
 * centre node sits. The first is on the right and the rest follow clockwise.
 * Past RING_CAPACITY the ring grows in proportion to the count.
 */
export function ringPositions(count: number): Point[] {
  const scale = Math.max(1, count / RING_CAPACITY);
  return Array.from({ length: count }, (_, i) => {
    const angle = (i * 2 * Math.PI) / count;
    return { x: Math.cos(angle) * RING_RADIUS_X * scale, y: Math.sin(angle) * RING_RADIUS_Y * scale };
  });
}

/**
 * Where the line from the centre of `box` towards `toward` leaves the box, so
 * that an edge starts on whichever side faces the other node. Fixed left and
 * right handles made every edge leave the centre's right side, and the edges
 * to neighbours on the left, above or below looped round behind the nodes.
 */
export function borderPoint(box: Box, toward: Point): Point {
  const centre = centreOf(box);
  const dx = toward.x - centre.x;
  const dy = toward.y - centre.y;
  if (dx === 0 && dy === 0) return centre;
  // Stretch the direction until it meets the nearer pair of sides.
  const t = Math.min(
    dx === 0 ? Infinity : box.width / 2 / Math.abs(dx),
    dy === 0 ? Infinity : box.height / 2 / Math.abs(dy),
  );
  return { x: centre.x + dx * t, y: centre.y + dy * t };
}

/**
 * The two ends of a straight edge between two nodes: where the line between
 * their centres crosses each node's border. The label sits halfway between
 * the ends, in the gap between the two nodes.
 */
export function edgeEnds(source: Box, target: Box): { source: Point; target: Point } {
  return {
    source: borderPoint(source, centreOf(target)),
    target: borderPoint(target, centreOf(source)),
  };
}
