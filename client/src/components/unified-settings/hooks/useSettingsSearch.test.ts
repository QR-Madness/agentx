/**
 * Settings search finds settings, not just sections.
 *
 * The defect this replaces: search filtered 21 section labels against hand-typed
 * keyword arrays, so typing the name of an actual setting found nothing.
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSettingsSearch, type SearchableSection } from './useSettingsSearch';
import type { SettingsManifestEntry } from '../../../lib/api';

const SECTIONS: SearchableSection[] = [
  { id: 'memory-recall', label: 'Recall', keywords: ['retrieval', 'hybrid'] },
  { id: 'context', label: 'Conversation Context', keywords: ['compaction'] },
  { id: 'search', label: 'Web Search', keywords: ['tavily'] },
];

const entry = (over: Partial<SettingsManifestEntry>): SettingsManifestEntry => ({
  key: 'x',
  store: 'memory',
  type: 'int',
  default: 1,
  value: 1,
  secret: false,
  writable_via: '/api/memory/recall-settings',
  ...over,
});

const ENTRIES: SettingsManifestEntry[] = [
  entry({ key: 'recall_candidate_pool', ui_section: 'memory-recall', default: 50, value: 50,
          help: { summary: 'How many candidates the reranker scores.' } }),
  entry({ key: 'recall_hybrid_bm25_weight', ui_section: 'memory-recall', default: 0.3, value: 0.3 }),
  entry({ key: 'context.verbatim_budget_ratio', store: 'config', ui_section: 'context',
          writable_via: '/api/config/update', default: 0.9, value: 0.75 }),
  entry({ key: 'neo4j_password', writable_via: null, secret: true }),
  entry({ key: 'search.max_results', store: 'config', ui_section: 'search',
          writable_via: '/api/config/update', default: 5, value: 5 }),
];

function search(query: string) {
  const { result } = renderHook(() => useSettingsSearch(SECTIONS, ENTRIES));
  act(() => result.current.setQuery(query));
  return result;
}

describe('useSettingsSearch', () => {
  it('finds a setting by its own name, not just its section', () => {
    const r = search('candidate pool');
    expect(r.current.settingHits.map(h => h.key)).toContain('recall_candidate_pool');
  });

  it('finds a setting whose words are not adjacent in the key', () => {
    // The motivating example: "verbatim budget" → context.verbatim_budget_ratio.
    const r = search('verbatim budget');
    expect(r.current.settingHits.map(h => h.key)).toContain('context.verbatim_budget_ratio');
  });

  it('matches authored help, so you can search by what a setting does', () => {
    const r = search('reranker');
    expect(r.current.settingHits.map(h => h.key)).toEqual(['recall_candidate_pool']);
  });

  it('ranks an exact key above an incidental mention', () => {
    const r = search('recall_candidate_pool');
    expect(r.current.settingHits[0].key).toBe('recall_candidate_pool');
  });

  it('reports which section a hit lives in, so it can be navigated to', () => {
    const r = search('bm25');
    expect(r.current.settingHits[0].sectionId).toBe('memory-recall');
  });

  it('flags a hit that differs from its shipped default', () => {
    const r = search('verbatim');
    expect(r.current.settingHits[0].isModified).toBe(true);
  });

  it('omits secrets and read-only plumbing — nothing to act on here', () => {
    const r = search('neo4j');
    expect(r.current.settingHits).toHaveLength(0);
  });

  it('still matches sections, including by keyword', () => {
    const r = search('tavily');
    expect(r.current.filtered.map(s => s.id)).toEqual(['search']);
  });

  it('returns every section when the query is empty', () => {
    const { result } = renderHook(() => useSettingsSearch(SECTIONS, ENTRIES));
    expect(result.current.filtered).toHaveLength(3);
    expect(result.current.settingHits).toHaveLength(0);
    expect(result.current.isSearching).toBe(false);
  });

  it('reports no results only when neither a section nor a setting matches', () => {
    const r = search('zzzznothing');
    expect(r.current.hasResults).toBe(false);
  });

  it('works with no manifest at all — sections still search', () => {
    const { result } = renderHook(() => useSettingsSearch(SECTIONS));
    act(() => result.current.setQuery('recall'));
    expect(result.current.filtered.map(s => s.id)).toEqual(['memory-recall']);
    expect(result.current.settingHits).toHaveLength(0);
  });
});
