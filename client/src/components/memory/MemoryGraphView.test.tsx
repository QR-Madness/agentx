/**
 * MemoryGraphView — what React Flow actually draws for Memory → Explore.
 *
 * graphGeometry.test.ts checks the pure layout and routing helpers. These tests
 * check that the component wires them into React Flow, so that a change to the
 * component cannot bring the old bugs back while the helper tests stay green:
 *
 * - **Spoke edges.** Every edge is a straight line between the borders the two
 *   nodes turn to each other, and the ring sits around the centre node's
 *   middle. Dropping `type: 'spoke'`, `edgeTypes` or `nodeOrigin` from the
 *   component fails the first test (v0.21.306 fixed edges that looped round
 *   behind the nodes).
 * - **Re-fit.** Each new subject is fitted to the canvas. A procedure chosen
 *   after a 50-neighbour entity used to keep that entity's zoom (about 0.25),
 *   because the canvas stays mounted. Removing the re-fit fails the second and
 *   third tests.
 *
 * jsdom lays nothing out, so the setup below gives React Flow fixed sizes: the
 * canvas is 1000 × 700 and every node 150 × 64, and the ResizeObserver reports
 * everything observed in one commit together. The memory hooks are stubbed
 * (vi.mock is hoisted above the imports).
 */

import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { EntityGraph, MemoryEntity, MemoryProcedure } from '../../lib/api';
import { edgeEnds, ringPositions, type Box, type Point } from './graphGeometry';
import { MemoryGraphView } from './MemoryGraphView';

// ── Fixtures ────────────────────────────────────────────────────────────────

const entity = (id: string, name: string): MemoryEntity => ({
  id, name, type: 'concept', channel: '_global', salience: 0.5,
  last_accessed: '2026-11-28T00:00:00Z', access_count: 1,
});

const neighbours = Array.from({ length: 50 }, (_, i) => entity(`n${i}`, `Neighbour ${i}`));
const HUB = entity('hub', 'Hub entity');
const SMALL = entity('small', 'Small entity');

const graphs: Record<string, EntityGraph> = {
  hub: { entity: HUB, facts: [], relationships: neighbours.map(target => ({ type: 'RELATED_TO', target })) },
  small: { entity: SMALL, facts: [], relationships: neighbours.slice(0, 3).map(target => ({ type: 'RELATED_TO', target })) },
};

const procedure = (id: string, trigger: string, agent: string | null, conversations: number): MemoryProcedure => ({
  id, trigger, body: 'Weigh both readings.', rationale: '', scope: 'global', agent_id: agent, strength: 3,
  signal_kinds: [], evidence_refs: Array.from({ length: conversations }, (_, i) => `conv:${i}0000000-conv`),
  channel: '_global',
});

// Eight neighbours: the scope, the agent and six conversations.
const WIDE = procedure('p-wide', 'When the question is broad', 'agent-a', 6);
// One neighbour: the scope.
const LONE = procedure('p-lone', 'When only a scope is known', null, 0);

// ── Stubbed data hooks ──────────────────────────────────────────────────────

// Stable references, as the real hooks give: the component lists them as effect deps.
const refresh = vi.fn();
const entitiesResult = { entities: [HUB, SMALL], loading: false, refresh };
const proceduresResult = { procedures: [WIDE, LONE], loading: false, refresh };
const graphResults = Object.fromEntries(
  Object.entries(graphs).map(([id, graph]) => [id, { graph, loading: false, refresh }]),
);
const noGraph = { graph: null, loading: false, refresh };

vi.mock('../../lib/hooks', () => ({
  useMemoryEntities: () => entitiesResult,
  useMemoryProcedures: () => proceduresResult,
  useEntityGraph: (id: string | null) => (id && graphResults[id]) || noGraph,
}));

// ── Layout stand-ins for jsdom ──────────────────────────────────────────────

const CANVAS = { width: 1000, height: 700 };
const NODE = { width: 150, height: 64 };

const sizeOf = (el: HTMLElement) =>
  el.classList.contains('react-flow') ? CANVAS
    : el.classList.contains('react-flow__node') ? NODE
      : { width: 0, height: 0 };

const restore: Array<() => void> = [];

beforeAll(() => {
  const proto = HTMLElement.prototype;
  for (const key of ['offsetWidth', 'offsetHeight'] as const) {
    const original = Object.getOwnPropertyDescriptor(proto, key);
    Object.defineProperty(proto, key, {
      configurable: true,
      get(this: HTMLElement) { return sizeOf(this)[key === 'offsetWidth' ? 'width' : 'height']; },
    });
    restore.push(() => { if (original) Object.defineProperty(proto, key, original); });
  }

  const OriginalResizeObserver = globalThis.ResizeObserver;
  // Like a browser's, it reports everything observed in one commit together.
  // Reporting each node on its own would let React Flow's mount-time fit run
  // after the first node, which a browser never does.
  globalThis.ResizeObserver = class {
    private pending: Element[] = [];
    constructor(private readonly callback: ResizeObserverCallback) {}
    observe(target: Element) {
      if (this.pending.push(target) > 1) return;
      queueMicrotask(() => {
        const entries = this.pending.map(t => ({ target: t }) as ResizeObserverEntry);
        this.pending = [];
        this.callback(entries, this as unknown as ResizeObserver);
      });
    }
    unobserve() {}
    disconnect() { this.pending = []; }
  };
  restore.push(() => { globalThis.ResizeObserver = OriginalResizeObserver; });

  // React Flow reads the zoom off the viewport's transform (`m22`).
  const win = window as unknown as { DOMMatrixReadOnly?: unknown };
  const OriginalMatrix = win.DOMMatrixReadOnly;
  win.DOMMatrixReadOnly = class {
    readonly m22: number;
    constructor(transform?: string) {
      const scale = transform?.match(/scale\(([\d.]+)\)/)?.[1];
      this.m22 = scale === undefined ? 1 : Number(scale);
    }
  };
  restore.push(() => { win.DOMMatrixReadOnly = OriginalMatrix; });

  // Edge labels measure their text.
  const svgProto = SVGElement.prototype as unknown as { getBBox?: () => DOMRect };
  const originalBBox = svgProto.getBBox;
  svgProto.getBBox = () => ({ x: 0, y: 0, width: 80, height: 16 }) as DOMRect;
  restore.push(() => { svgProto.getBBox = originalBBox; });
});

afterAll(() => restore.forEach(undo => undo()));

// ── Helpers ─────────────────────────────────────────────────────────────────

const select = (name: string | RegExp) => fireEvent.click(screen.getByRole('button', { name }));

const viewport = () => document.querySelector<HTMLElement>('.react-flow__viewport');

/** The viewport's transform, as `{x, y, zoom}`. */
function transform() {
  const match = viewport()?.style.transform.match(/translate\((-?[\d.]+)px, (-?[\d.]+)px\) scale\(([\d.]+)\)/);
  if (!match) return null;
  return { x: Number(match[1]), y: Number(match[2]), zoom: Number(match[3]) };
}

const centreName = () => document.querySelector('.graph-node.center .graph-node-name')?.textContent;

/**
 * Waits until `subject` is the centre node and the view has stopped moving:
 * the fit lands a frame or two after the nodes are measured.
 */
async function settledOn(subject: string) {
  await waitFor(() => expect(centreName()).toBe(subject));
  let last = '';
  let steady = 0;
  await waitFor(() => {
    const now = JSON.stringify(transform());
    steady = now === last ? steady + 1 : 0;
    last = now;
    expect(steady).toBeGreaterThanOrEqual(5);
  }, { interval: 30, timeout: 3000 });
  return transform()!;
}

/** The transform a fresh canvas fits `subject` to: select it on a new mount. */
async function freshFit(button: string | RegExp, subject: string) {
  const { unmount } = render(<MemoryGraphView />);
  select(button);
  const fitted = await settledOn(subject);
  unmount();
  return fitted;
}

const boxAt = (p: Point): Box =>
  ({ x: p.x - NODE.width / 2, y: p.y - NODE.height / 2, width: NODE.width, height: NODE.height });

const near = (a: number, b: number) => Math.abs(a - b) < 1e-6;

// ── Tests ───────────────────────────────────────────────────────────────────

describe('MemoryGraphView', () => {
  it('draws each edge straight between the borders the centre and its neighbour face', async () => {
    render(<MemoryGraphView />);
    select(/Small entity/);
    await settledOn('Small entity');

    // nodeOrigin [0.5, 0.5]: the centre node's middle sits on the origin, so
    // its top-left corner is half a node up and to the left.
    const centre = document.querySelector<HTMLElement>('.react-flow__node[data-id="small"]');
    expect(centre?.style.transform).toBe(`translate(${-NODE.width / 2}px, ${-NODE.height / 2}px)`);

    const ring = ringPositions(3);
    for (let i = 0; i < 3; i++) {
      const path = document.querySelector(`.react-flow__edge[data-id="e-${i}"] .react-flow__edge-path`);
      const d = path?.getAttribute('d') ?? '';
      // A spoke edge is one straight segment (a bezier would carry a `C`).
      const match = d.match(/^M (-?[\d.e-]+),(-?[\d.e-]+)\s*L (-?[\d.e-]+),(-?[\d.e-]+)$/);
      expect(match, `edge e-${i} is a straight line: ${d}`).not.toBeNull();
      const [sx, sy, tx, ty] = match!.slice(1).map(Number);
      const ends = edgeEnds(boxAt({ x: 0, y: 0 }), boxAt(ring[i]));
      expect(
        near(sx, ends.source.x) && near(sy, ends.source.y) && near(tx, ends.target.x) && near(ty, ends.target.y),
        `edge e-${i} runs ${d}, expected M ${ends.source.x},${ends.source.y} L ${ends.target.x},${ends.target.y}`,
      ).toBe(true);
    }
  });

  it('fits a procedure chosen after a large entity, as a fresh canvas would', async () => {
    const expected = await freshFit(/When the question is broad/, WIDE.trigger);

    render(<MemoryGraphView />);
    select(/Hub entity/);
    const hub = await settledOn('Hub entity');
    // The 50-neighbour ring needs a far smaller zoom; otherwise this proves nothing.
    expect(hub.zoom).toBeLessThan(expected.zoom / 2);

    select(/When the question is broad/);
    await waitFor(() => expect(transform()).toEqual(expected));
  });

  it('fits each procedure in turn while the canvas stays mounted', async () => {
    const wide = await freshFit(/When the question is broad/, WIDE.trigger);
    const lone = await freshFit(/When only a scope is known/, LONE.trigger);
    expect(lone).not.toEqual(wide);

    render(<MemoryGraphView />);
    select(/When only a scope is known/);
    await waitFor(() => expect(transform()).toEqual(lone));
    select(/When the question is broad/);
    await waitFor(() => expect(transform()).toEqual(wide));
    select(/When only a scope is known/);
    await waitFor(() => expect(transform()).toEqual(lone));
  });
});
