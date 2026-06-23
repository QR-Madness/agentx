import { describe, it, expect } from 'vitest';
import { exhibitFromWire, citationExhibitFromWebSearch } from './exhibits';

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

  it('maps an image element (url/alt) for a generated image', () => {
    const ex = exhibitFromWire({
      id: 'e4',
      elements: [{ type: 'image', url: '/api/workspaces/ws_home/documents/doc_1/raw', alt: 'a sunset' }],
    });
    const el = ex.elements[0];
    expect(el.type).toBe('image');
    expect(el.type === 'image' && el.url).toBe('/api/workspaces/ws_home/documents/doc_1/raw');
    expect(el.type === 'image' && el.alt).toBe('a sunset');
  });
});

describe('citationExhibitFromWebSearch', () => {
  it('maps results to passive web sources, deduped by URL', () => {
    const ex = citationExhibitFromWebSearch(
      [
        { title: 'A', url: 'https://a' },
        { title: 'B', url: 'https://b' },
        { title: 'A again', url: 'https://a' }, // dup url
        { title: '', url: '' }, // blank → skipped
      ],
      'exh_src_1',
    );
    expect(ex).not.toBeNull();
    expect(ex!.id).toBe('exh_src_1');
    const el = ex!.elements[0];
    expect(el.type).toBe('citation');
    if (el.type === 'citation') {
      expect(el.sources).toHaveLength(2);
      expect(el.sources.every((s) => s.kind === 'passive' && s.source_type === 'web')).toBe(true);
      expect(el.sources.map((s) => s.url)).toEqual(['https://a', 'https://b']);
    }
  });

  it('returns null for empty / non-array input', () => {
    expect(citationExhibitFromWebSearch([], 'x')).toBeNull();
    expect(citationExhibitFromWebSearch(undefined, 'x')).toBeNull();
    expect(citationExhibitFromWebSearch([{ title: '', url: '' }], 'x')).toBeNull();
  });
});
