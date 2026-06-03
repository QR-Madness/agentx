import { describe, it, expect } from 'vitest';
import { buildBibliography } from './bibliography';
import type { ConversationMessage } from './messages';

function exhibit(id: string, sources: Array<{ label: string; url?: string; kind?: 'active' | 'passive' }>): ConversationMessage {
  return {
    id,
    type: 'exhibit',
    timestamp: '2026-06-03T00:00:00Z',
    exhibit: {
      schemaVersion: 1,
      id: `ex_${id}`,
      layout: 'stack',
      elements: [
        {
          type: 'citation',
          sources: sources.map((s) => ({ label: s.label, url: s.url, kind: s.kind ?? 'passive' })),
        },
      ],
    },
  } as ConversationMessage;
}

describe('buildBibliography', () => {
  it('dedupes by URL and numbers in first-appearance order', () => {
    const msgs: ConversationMessage[] = [
      { id: 'u', type: 'user', timestamp: 't', content: 'hi' } as ConversationMessage,
      exhibit('a', [
        { label: 'NLLB paper', url: 'https://a' },
        { label: 'Django', url: 'https://b', kind: 'active' },
      ]),
      exhibit('b', [
        { label: 'NLLB again', url: 'https://a' }, // dup URL → skipped
        { label: 'HF NLLB', url: 'https://c' },
      ]),
    ];
    const bib = buildBibliography(msgs);
    expect(bib.map((e) => e.n)).toEqual([1, 2, 3]);
    expect(bib.map((e) => e.url)).toEqual(['https://a', 'https://b', 'https://c']);
    expect(bib[1].kind).toBe('active');
  });

  it('returns empty for a conversation with no citations', () => {
    const msgs: ConversationMessage[] = [
      { id: 'u', type: 'user', timestamp: 't', content: 'hi' } as ConversationMessage,
    ];
    expect(buildBibliography(msgs)).toEqual([]);
  });

  it('dedupes label-only sources (no URL) by label', () => {
    const bib = buildBibliography([
      exhibit('a', [{ label: 'A memory fact' }, { label: 'A memory fact' }]),
    ]);
    expect(bib).toHaveLength(1);
  });
});
