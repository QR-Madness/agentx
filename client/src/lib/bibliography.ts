/**
 * Conversation Bibliography — a basic, static, deduped list of every source the
 * agent has cited in a conversation (from `citation` exhibits, incl. auto-captured
 * web_search/web_research sources). Numbered in first-appearance order; the number
 * is stable because messages restore in order. Pure + client-only: it recomputes
 * from the conversation's existing messages, so it can never drift from the
 * transcript and needs no storage.
 */

import type { ConversationMessage } from './messages';
import type { CitationSource } from './exhibits';

export interface BibliographyEntry {
  /** Stable 1-based number, first-appearance order. */
  n: number;
  label: string;
  url?: string;
  kind: 'active' | 'passive';
  sourceType?: CitationSource['source_type'];
}

/** Flatten a conversation's cited sources into a deduped, numbered bibliography. */
export function buildBibliography(messages: ConversationMessage[]): BibliographyEntry[] {
  const entries: BibliographyEntry[] = [];
  const seen = new Set<string>();
  for (const m of messages) {
    if (m.type !== 'exhibit') continue;
    for (const el of m.exhibit.elements) {
      if (el.type !== 'citation') continue;
      for (const s of el.sources) {
        const key = (s.url || s.label || '').trim().toLowerCase();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        entries.push({
          n: entries.length + 1,
          label: s.label,
          url: s.url,
          kind: s.kind,
          sourceType: s.source_type,
        });
      }
    }
  }
  return entries;
}
