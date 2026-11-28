/**
 * flow — every React Flow canvas is drawn with the design tokens.
 *
 * React Flow 13 follows the OS colour scheme unless a canvas forces one, and its
 * `[data-theme='light'|'dark']` rules match only our `light` theme id. A canvas
 * left alone therefore shows xyflow's own palette, which on the dark themes means
 * white Controls and white edge-label boxes (Memory → Explore did until v0.21.304).
 * `.ax-flow` (styles/flow.css) maps xyflow's `--xy-*` variables to the tokens and
 * inherits the app's colour scheme, so no canvas needs `forceColorMode`.
 *
 * This test loads every component module under src/ as raw text (via Vite's glob)
 * and fails when a module renders <ReactFlow> without `className="ax-flow"` and
 * the flow.css import, or when any module passes `forceColorMode`.
 *
 * If this fails: put `className="ax-flow"` on the <ReactFlow> element, import
 * `styles/flow.css` beside `@xyflow/react/dist/style.css`, and map any further
 * `--xy-*` variable the canvas needs in flow.css instead of forcing a mode.
 */

import { describe, it, expect } from 'vitest';

// Raw source of every component module under src/ (root-absolute keys like `/src/components/x.tsx`).
const modules = Object.entries(
  import.meta.glob('/src/**/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<string, string>,
).filter(([path]) => !path.endsWith('.test.tsx'));

/** A <ReactFlow> element (not <ReactFlowProvider>). */
const REACT_FLOW_ELEMENT = /<ReactFlow[\s>]/g;
const AX_FLOW_CLASS = /className="ax-flow"/g;
const FLOW_CSS_IMPORT = /import\s+['"](?:\.\.?\/)+styles\/flow\.css['"]/;
const FORCED_MODE = /\bforceColorMode\s*=/;

const count = (source: string, pattern: RegExp) => source.match(pattern)?.length ?? 0;

describe('React Flow canvases', () => {
  const canvases = modules.filter(([, source]) => count(source, REACT_FLOW_ELEMENT) > 0);

  it('finds the canvases it guards', () => {
    // The org chart and Memory → Explore; a rename that hid them would pass vacuously.
    expect(canvases.length).toBeGreaterThanOrEqual(2);
  });

  it('every <ReactFlow> takes the token theme', () => {
    const offenders = canvases
      .filter(([, source]) =>
        count(source, AX_FLOW_CLASS) < count(source, REACT_FLOW_ELEMENT) || !FLOW_CSS_IMPORT.test(source))
      .map(([path]) => path.replace(/^\//, ''))
      .sort();

    expect(
      offenders,
      `<ReactFlow> without className="ax-flow" or the styles/flow.css import:\n` + offenders.join('\n'),
    ).toEqual([]);
  });

  it('no module forces a colour mode', () => {
    const offenders = modules
      .filter(([, source]) => FORCED_MODE.test(source))
      .map(([path]) => path.replace(/^\//, ''))
      .sort();

    expect(
      offenders,
      `forceColorMode pins xyflow's own palette; map the --xy-* variable in styles/flow.css instead:\n` +
        offenders.join('\n'),
    ).toEqual([]);
  });
});
