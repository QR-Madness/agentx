/**
 * SettingsSearchContext — one search, two places to type it.
 *
 * Search state used to live inside `SettingsNav`. On desktop that was fine —
 * the nav is always on screen. On mobile the nav is a full-screen takeover, so
 * from the Overview the only way to search was to open the nav first, which is
 * exactly the "where do I even start" problem the Overview exists to answer.
 *
 * Lifting the state here lets the Overview host its own search box driving the
 * *same* query: type in one and the other reflects it. Two independent search
 * boxes over the same corpus would be worse than one — you'd never know which
 * of them was showing you the truth.
 */

import { createContext, useContext, type ReactNode } from 'react';
import { getAllSections } from './sections';
import { useSettingsSearch } from './hooks/useSettingsSearch';
import { useSettingsManifest } from './SettingsManifestContext';

export type SettingsSearchValue = ReturnType<typeof useSettingsSearch>;

const SettingsSearchContext = createContext<SettingsSearchValue | null>(null);

export function SettingsSearchProvider({ children }: { children: ReactNode }) {
  const manifest = useSettingsManifest();
  const value = useSettingsSearch(getAllSections(), manifest?.entries.values());

  return (
    <SettingsSearchContext.Provider value={value}>
      {children}
    </SettingsSearchContext.Provider>
  );
}

/**
 * The shared search, or null outside the provider. Null-safe like the manifest:
 * a consumer rendered without it simply shows no search box.
 */
export function useSharedSettingsSearch(): SettingsSearchValue | null {
  return useContext(SettingsSearchContext);
}
