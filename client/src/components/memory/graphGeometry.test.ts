/**
 * graphGeometry — how Memory → Explore routes its edges and lays out its ring.
 *
 * Until v0.21.306 every edge left the centre node's right side and entered the
 * neighbour's left side. Edges to neighbours on the left, above or below looped
 * round behind the nodes, and their labels could end up under a node. Edges now
 * run straight between the borders that face each other. These tests check
 * that at every angle, and check that the ring leaves each label clear of every
 * node.
 *
 * If a ring test fails after a change to the node CSS or the ring constants,
 * resize the ring (RING_RADIUS_X/Y, RING_CAPACITY) rather than loosening the
 * test. The sizes below were measured in the browser.
 */

import { describe, it, expect } from 'vitest';
import memoryPanelCss from '../../styles/MemoryPanel.css?raw';
import {
  NODE_MAX_WIDTH, RING_CAPACITY, RING_RADIUS_X, RING_RADIUS_Y,
  borderPoint, edgeEnds, ringPositions,
  type Box, type Point,
} from './graphGeometry';

// Flow-unit sizes measured in the browser: a node whose name fits on one line
// is 64 tall, and a name wrapped to three lines makes it about 104. A typical
// name gives a node about 150 wide. The edge labels (CONTRASTS_WITH and the
// like) measure up to about 96 by 20.
const NODE_HEIGHT = 64;
const TALL_NODE_HEIGHT = 104;
const TYPICAL_NODE_WIDTH = 150;
const LABEL = { width: 100, height: 20 };
/** The most relationships the entity graph endpoint returns. */
const MAX_NEIGHBOURS = 50;

const EPS = 1e-6;
const ORIGIN: Point = { x: 0, y: 0 };

/** A box of the given size centred on `p`. */
const boxAt = (p: Point, width: number, height: number): Box =>
  ({ x: p.x - width / 2, y: p.y - height / 2, width, height });

/** The point `distance` away from the origin at `deg` degrees, clockwise from the right. */
const towards = (deg: number, distance = 1000): Point => ({
  x: Math.cos((deg * Math.PI) / 180) * distance,
  y: Math.sin((deg * Math.PI) / 180) * distance,
});

const everyAngle = (step: number) => Array.from({ length: 360 / step }, (_, i) => i * step);

const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

/** Strictly inside: a point on the border does not count. */
const inside = (p: Point, b: Box) =>
  p.x > b.x + EPS && p.x < b.x + b.width - EPS && p.y > b.y + EPS && p.y < b.y + b.height - EPS;

const overlaps = (a: Box, b: Box) =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

const onBorder = (p: Point, b: Box) => {
  const withinX = p.x >= b.x - EPS && p.x <= b.x + b.width + EPS;
  const withinY = p.y >= b.y - EPS && p.y <= b.y + b.height + EPS;
  const onLeftOrRight = Math.abs(p.x - b.x) < EPS || Math.abs(p.x - (b.x + b.width)) < EPS;
  const onTopOrBottom = Math.abs(p.y - b.y) < EPS || Math.abs(p.y - (b.y + b.height)) < EPS;
  return withinX && withinY && (onLeftOrRight || onTopOrBottom);
};

const expectPoint = (p: Point, x: number, y: number) => {
  expect(p.x).toBeCloseTo(x, 6);
  expect(p.y).toBeCloseTo(y, 6);
};

describe('borderPoint', () => {
  const box = boxAt(ORIGIN, 100, 40);

  it('leaves through the side that faces the other node', () => {
    expectPoint(borderPoint(box, { x: 500, y: 0 }), 50, 0);
    expectPoint(borderPoint(box, { x: -500, y: 0 }), -50, 0);
    expectPoint(borderPoint(box, { x: 0, y: -500 }), 0, -20);
    expectPoint(borderPoint(box, { x: 0, y: 500 }), 0, 20);
  });

  it('meets the corner on the diagonal', () => {
    expectPoint(borderPoint(box, { x: 100, y: 40 }), 50, 20);
    expectPoint(borderPoint(box, { x: -100, y: -40 }), -50, -20);
  });

  it('lands on the border at every angle', () => {
    for (const deg of everyAngle(5)) {
      expect(onBorder(borderPoint(box, towards(deg)), box), `${deg}°`).toBe(true);
    }
  });

  it('gives the centre when the two centres coincide', () => {
    expectPoint(borderPoint(box, ORIGIN), 0, 0);
  });
});

describe('edgeEnds', () => {
  it('runs the short way between two nodes at every angle', () => {
    const centre = boxAt(ORIGIN, NODE_MAX_WIDTH, TALL_NODE_HEIGHT);
    for (const deg of everyAngle(15)) {
      const neighbour = boxAt(towards(deg, 300), TYPICAL_NODE_WIDTH, NODE_HEIGHT);
      const { source, target } = edgeEnds(centre, neighbour);
      const where = `${deg}°`;

      expect(onBorder(source, centre), where).toBe(true);
      expect(onBorder(target, neighbour), where).toBe(true);

      // Nothing between the ends lies inside either node, so the edge never
      // passes behind one (the old right-to-left edges crossed the centre node).
      for (let step = 1; step < 20; step++) {
        const p = {
          x: source.x + ((target.x - source.x) * step) / 20,
          y: source.y + ((target.y - source.y) * step) / 20,
        };
        expect(inside(p, centre) || inside(p, neighbour), `${where}, step ${step}`).toBe(false);
      }

      // And it heads from the centre towards the neighbour.
      const direction = towards(deg);
      const along = (target.x - source.x) * direction.x + (target.y - source.y) * direction.y;
      expect(along, where).toBeGreaterThan(0);
    }
  });
});

describe('ringPositions', () => {
  it('starts on the right and runs clockwise round the centre node', () => {
    const [right, below, left, above] = ringPositions(4);
    expectPoint(right, RING_RADIUS_X, 0);
    expectPoint(below, 0, RING_RADIUS_Y);
    expectPoint(left, -RING_RADIUS_X, 0);
    expectPoint(above, 0, -RING_RADIUS_Y);
  });

  it('places nothing when there are no neighbours', () => {
    expect(ringPositions(0)).toEqual([]);
  });

  it('grows once it is past its capacity', () => {
    expectPoint(ringPositions(RING_CAPACITY)[0], RING_RADIUS_X, 0);
    expectPoint(ringPositions(RING_CAPACITY * 2)[0], RING_RADIUS_X * 2, 0);
  });

  it('keeps every label clear of every node when the nodes are as large as they get', () => {
    // The centre's name wraps to three lines, and every node is at the width cap.
    for (let count = 1; count <= RING_CAPACITY; count++) {
      const centre = boxAt(ORIGIN, NODE_MAX_WIDTH, TALL_NODE_HEIGHT);
      const neighbours = ringPositions(count).map(p => boxAt(p, NODE_MAX_WIDTH, NODE_HEIGHT));
      for (const [i, neighbour] of neighbours.entries()) {
        const { source, target } = edgeEnds(centre, neighbour);
        const label = boxAt(midpoint(source, target), LABEL.width, LABEL.height);
        for (const [j, node] of [centre, ...neighbours].entries()) {
          expect(overlaps(label, node), `${count} neighbours: label ${i} over node ${j}`).toBe(false);
        }
      }
    }
  });

  it('keeps typical nodes apart and their labels clear, up to the most the API returns', () => {
    for (let count = 1; count <= MAX_NEIGHBOURS; count++) {
      const centre = boxAt(ORIGIN, NODE_MAX_WIDTH, TALL_NODE_HEIGHT);
      const neighbours = ringPositions(count).map(p => boxAt(p, TYPICAL_NODE_WIDTH, NODE_HEIGHT));
      const nodes = [centre, ...neighbours];
      for (const [i, a] of nodes.entries()) {
        for (const [j, b] of nodes.entries()) {
          if (j > i) expect(overlaps(a, b), `${count} neighbours: node ${i} over node ${j}`).toBe(false);
        }
      }
      for (const [i, neighbour] of neighbours.entries()) {
        const { source, target } = edgeEnds(centre, neighbour);
        const label = boxAt(midpoint(source, target), LABEL.width, LABEL.height);
        for (const [j, node] of nodes.entries()) {
          expect(overlaps(label, node), `${count} neighbours: label ${i} over node ${j}`).toBe(false);
        }
      }
    }
  });
});

describe('NODE_MAX_WIDTH', () => {
  it('matches the width cap on `.graph-node` in MemoryPanel.css', () => {
    const rule = memoryPanelCss.match(/\.graph-node\s*\{[^}]*\}/)?.[0] ?? '';
    expect(rule, '.graph-node rule').toMatch(new RegExp(`max-width:\\s*${NODE_MAX_WIDTH}px`));
  });
});
