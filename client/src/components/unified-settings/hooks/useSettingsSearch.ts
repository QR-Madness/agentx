/**
 * useSettingsSearch — Global search functionality for settings sections
 */

import { useState, useMemo } from 'react';

export interface SearchableSection {
  id: string;
  label: string;
  keywords?: string[];
}

export function useSettingsSearch(sections: SearchableSection[]) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    if (!query.trim()) return sections;

    const lowerQuery = query.toLowerCase();
    return sections.filter(section => {
      const matchesLabel = section.label.toLowerCase().includes(lowerQuery);
      const matchesKeywords = section.keywords?.some(kw =>
        kw.toLowerCase().includes(lowerQuery)
      );
      return matchesLabel || matchesKeywords;
    });
  }, [sections, query]);

  return {
    query,
    setQuery,
    filtered,
    hasResults: filtered.length > 0
  };
}
