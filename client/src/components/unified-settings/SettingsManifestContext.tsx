/**
 * SettingsManifestContext — server truth about the settings themselves.
 *
 * Every section still loads and saves its own values through its own endpoint;
 * this carries the *metadata* those values are missing — the shipped default,
 * declared bounds, tier, and authored help — from `GET /api/settings/manifest`.
 *
 * Two rules hold everything else together:
 *
 *  1. **Nothing requires it.** Every consumer is null-safe. If the fetch fails,
 *     or a panel renders outside the provider, the manifest chrome (default
 *     chips, reset, help) simply isn't there and the controls work exactly as
 *     they did before. The settings surface must never be gated on metadata.
 *  2. **Secrets are never compared or shown.** The server redacts them, so a
 *     "modified" check on a secret would compare `'***'` to `'***'`. They're
 *     reported as configured-or-not and nothing more.
 */

import {
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from 'react';
import { api } from '../../lib/api';
import { useApi } from '../../lib/hooks';
import type { ApiError } from '../../lib/api';
import type {
  SettingsManifest,
  SettingsManifestEntry,
  SettingsManifestSection,
  SettingsStore,
} from '../../lib/api';

/** Manifest keys are only unique per store (`config` and `memory` overlap). */
export function entryId(store: SettingsStore, key: string): string {
  return `${store}:${key}`;
}

export interface SettingsManifestValue {
  entries: Map<string, SettingsManifestEntry>;
  /** Screen metadata by SECTION_HIERARCHY id — blurb and writable count. */
  sections: Map<string, SettingsManifestSection>;
  /** Writable entries whose current value differs from the shipped default. */
  modified: SettingsManifestEntry[];
  /** How many changed settings each screen owns, for the Overview tiles. */
  modifiedBySection: Map<string, number>;
  loading: boolean;
  error: ApiError | null;
  refresh: () => Promise<void>;
}

const SettingsManifestContext = createContext<SettingsManifestValue | null>(null);

/** Structural equality — defaults can be lists (`entity_types`) or dicts. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (a === null || b === null || typeof a !== typeof b) return false;
  if (typeof a !== 'object') return false;
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}

/**
 * True when the user has moved this setting off its shipped default. Secrets
 * are excluded: their values arrive redacted, so any comparison is meaningless.
 */
export function isModifiedEntry(entry: SettingsManifestEntry): boolean {
  if (!entry.writable_via || entry.secret) return false;
  return !sameValue(entry.value, entry.default);
}

export function SettingsManifestProvider({ children }: { children: ReactNode }) {
  const { data, loading, error, refresh } = useApi<SettingsManifest>(
    () => api.getSettingsManifest(),
    [],
  );

  const entries = useMemo(() => {
    const map = new Map<string, SettingsManifestEntry>();
    for (const entry of data?.entries ?? []) {
      map.set(entryId(entry.store, entry.key), entry);
    }
    return map;
  }, [data]);

  const sections = useMemo(() => {
    const map = new Map<string, SettingsManifestSection>();
    for (const section of data?.sections ?? []) map.set(section.id, section);
    return map;
  }, [data]);

  const modified = useMemo(
    () => (data?.entries ?? []).filter(isModifiedEntry),
    [data],
  );

  const modifiedBySection = useMemo(() => {
    const counts = new Map<string, number>();
    for (const entry of modified) {
      if (!entry.ui_section) continue;
      counts.set(entry.ui_section, (counts.get(entry.ui_section) ?? 0) + 1);
    }
    return counts;
  }, [modified]);

  const value = useMemo<SettingsManifestValue>(
    () => ({ entries, sections, modified, modifiedBySection, loading, error, refresh }),
    [entries, sections, modified, modifiedBySection, loading, error, refresh],
  );

  return (
    <SettingsManifestContext.Provider value={value}>
      {children}
    </SettingsManifestContext.Provider>
  );
}

/** The whole manifest, or null outside the provider. Callers must handle null. */
export function useSettingsManifest(): SettingsManifestValue | null {
  return useContext(SettingsManifestContext);
}

/** What a field needs to render its chrome. Every field is happy with null. */
export interface SettingBinding {
  entry: SettingsManifestEntry;
  defaultValue: unknown;
  /** Differs from the shipped default *right now*, per the live draft value. */
  isModified: boolean;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  help?: SettingsManifestEntry['help'];
  tier?: SettingsManifestEntry['tier'];
}

/**
 * Build a binding from an already-fetched manifest. Plain function, not a hook
 * — panels bind controls inside conditional branches (a knob that only renders
 * when its technique is on), and a hook there would change the hook count
 * between renders. Take the manifest once with `useSettingsManifest()`, then
 * call this as often as you like.
 *
 * Pass `currentValue` (the live draft) so "modified" tracks what's on screen
 * rather than what the server last returned — otherwise the dot lags a debounce
 * behind the user. Returns null when the key isn't in the manifest, which is
 * the normal case for a control that hasn't been declared yet.
 */
export function bindSetting(
  manifest: SettingsManifestValue | null,
  store: SettingsStore,
  key: string,
  currentValue?: unknown,
): SettingBinding | null {
  const entry = manifest?.entries.get(entryId(store, key));
  if (!entry) return null;
  const live = currentValue === undefined ? entry.value : currentValue;
  return {
    entry,
    defaultValue: entry.default,
    isModified: !entry.secret && !sameValue(live, entry.default),
    min: entry.constraints?.min,
    max: entry.constraints?.max,
    step: entry.constraints?.step,
    unit: entry.constraints?.unit,
    help: entry.help,
    tier: entry.tier,
  };
}

/**
 * Hook form, for a control bound unconditionally at the top level of a
 * component. Inside conditional markup, use `bindSetting` instead.
 */
export function useSettingEntry(
  store: SettingsStore,
  key: string,
  currentValue?: unknown,
): SettingBinding | null {
  const manifest = useSettingsManifest();
  return useMemo(
    () => bindSetting(manifest, store, key, currentValue),
    [manifest, store, key, currentValue],
  );
}

/** What a field spreads to get its chrome: `<SliderField {...bind('recent_floor')} …>`. */
export interface FieldBinding {
  binding: SettingBinding | null;
  onReset?: () => void;
}

/**
 * Bind a whole section in one call.
 *
 * Sections keep flat local field names that don't match the config paths behind
 * them — `trajectory_enabled` is `trajectory_compression.enabled`, and one
 * section can span five config roots — so callers supply a map from the local
 * name to `store:key`. Anything absent from the map (or from the manifest) just
 * yields an empty binding, and the field renders as it always did.
 *
 * Reset writes the shipped default through the section's own `update`, so it
 * rides the normal autosave path: same debounce, same save chip, no special
 * case in the save handler.
 */
export function sectionBinder<T extends Record<string, unknown>>(
  manifest: SettingsManifestValue | null,
  keyMap: Partial<Record<keyof T & string, string>>,
  values: T | null,
  update: (patch: Partial<T>) => void,
): (localKey: keyof T & string) => FieldBinding {
  return (localKey) => {
    const target = keyMap[localKey];
    if (!target) return { binding: null };
    const [store, ...rest] = target.split(':');
    const binding = bindSetting(
      manifest,
      store as SettingsStore,
      rest.join(':'),
      values?.[localKey],
    );
    return {
      binding,
      onReset: binding
        ? () => update({ [localKey]: binding.defaultValue } as Partial<T>)
        : undefined,
    };
  };
}
