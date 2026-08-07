/**
 * useSettingsSearch — search that finds *settings*, not just sections.
 *
 * This used to filter 21 section labels against hand-typed keyword arrays. Two
 * problems: typing "temperature" or "verbatim budget" found nothing, because no
 * section is called that; and the arrays rotted silently, since adding a control
 * never forced anyone to update them.
 *
 * The manifest already knows every setting's key, section, and authored help, so
 * settings are indexed from it — nothing to maintain by hand, and a new setting
 * is searchable the moment it's declared. Section matching stays (it's how you
 * navigate when you *do* know where you're going), keyword arrays included, so
 * sections without manifest coverage are no worse off than before.
 */

import { useState, useMemo } from 'react';
import type { SettingsManifestEntry } from '../../../lib/api';

export interface SearchableSection {
  id: string;
  label: string;
  keywords?: string[];
}

export interface SettingHit {
  /** `store:key` — the anchor the content area scrolls to. */
  id: string;
  key: string;
  label: string;
  /** Section id, or undefined when the setting isn't surfaced in the UI. */
  sectionId?: string;
  summary?: string;
  isModified: boolean;
  /** Lower sorts first. */
  rank: number;
}

/** `recall_candidate_pool` → "Recall candidate pool". */
function humanize(key: string): string {
  return key
    .split('.')
    .map(part => part.replace(/_/g, ' '))
    .join(' · ')
    .replace(/^\w/, c => c.toUpperCase());
}

/**
 * Rank a setting against a query. Lower is better; null means no match.
 *
 * The ordering is about what the user most likely meant: an exact key beats a
 * key prefix, which beats a word in the name, which beats a mention buried in
 * the help text.
 */
function scoreSetting(entry: SettingsManifestEntry, q: string): number | null {
  const key = entry.key.toLowerCase();
  const human = humanize(entry.key).toLowerCase();
  const summary = (entry.help?.summary || '').toLowerCase();

  if (key === q) return 0;
  if (key.startsWith(q) || human.startsWith(q)) return 1;
  if (key.includes(q) || human.includes(q)) return 2;
  if (summary.includes(q)) return 3;

  // Every word present somewhere — "verbatim budget" should find
  // `context.verbatim_budget_ratio` even though the words aren't adjacent.
  const words = q.split(/\s+/).filter(Boolean);
  if (words.length > 1) {
    const haystack = `${key} ${human} ${summary}`;
    if (words.every(w => haystack.includes(w))) return 4;
  }
  return null;
}

export function useSettingsSearch(
  sections: SearchableSection[],
  entries?: Iterable<SettingsManifestEntry>,
) {
  const [query, setQuery] = useState('');
  const trimmed = query.trim().toLowerCase();

  const filtered = useMemo(() => {
    if (!trimmed) return sections;
    return sections.filter(section => {
      const matchesLabel = section.label.toLowerCase().includes(trimmed);
      const matchesKeywords = section.keywords?.some(kw =>
        kw.toLowerCase().includes(trimmed)
      );
      return matchesLabel || matchesKeywords;
    });
  }, [sections, trimmed]);

  const settingHits = useMemo<SettingHit[]>(() => {
    if (!trimmed || !entries) return [];
    const hits: SettingHit[] = [];
    for (const entry of entries) {
      // Read-only plumbing (connection strings, credentials) isn't something the
      // user can act on here, so it stays out of the results.
      if (!entry.writable_via || entry.secret) continue;
      const rank = scoreSetting(entry, trimmed);
      if (rank === null) continue;
      hits.push({
        id: `${entry.store}:${entry.key}`,
        key: entry.key,
        label: humanize(entry.key),
        sectionId: entry.ui_section,
        summary: entry.help?.summary,
        isModified: entry.value !== entry.default,
        rank,
      });
    }
    hits.sort((a, b) => a.rank - b.rank || a.label.localeCompare(b.label));
    return hits.slice(0, 40);
  }, [entries, trimmed]);

  return {
    query,
    setQuery,
    filtered,
    settingHits,
    isSearching: Boolean(trimmed),
    hasResults: filtered.length > 0 || settingHits.length > 0,
  };
}
