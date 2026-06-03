import { describe, it, expect } from 'vitest';
import { exhibitFromWire } from './exhibits';

describe('exhibitFromWire', () => {
  it('coerces table cells to strings', () => {
    const ex = exhibitFromWire({
      id: 'e1',
      elements: [{ type: 'table', columns: ['M', 'Cost'], rows: [['opus', 0.4 as unknown as string]] }],
    });
    const el = ex.elements[0];
    expect(el.type).toBe('table');
    expect(el.type === 'table' && el.rows).toEqual([['opus', '0.4']]);
  });

  it('defaults citation source kind to passive and keeps active', () => {
    const ex = exhibitFromWire({
      id: 'e2',
      elements: [{ type: 'citation', sources: [{ label: 'A', kind: 'active' }, { label: 'B' }] }],
    });
    const el = ex.elements[0];
    expect(el.type === 'citation' && el.sources.map((s) => s.kind)).toEqual(['active', 'passive']);
  });

  it('drops an unrecognized source_type', () => {
    const ex = exhibitFromWire({
      id: 'e3',
      elements: [{ type: 'citation', sources: [{ label: 'A', source_type: 'bogus' }] }],
    });
    expect(ex.elements[0].type === 'citation' && ex.elements[0].sources[0].source_type).toBe(undefined);
  });
});
