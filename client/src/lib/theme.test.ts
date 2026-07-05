/**
 * Theme registry invariants.
 *
 * `applyTheme` writes variables onto `document.documentElement` and never
 * clears them — so every theme MUST define the exact same key set, or switching
 * themes leaves stale values from the previous theme behind. Adding a token to
 * one theme and forgetting the others is the classic drift this locks out.
 */

import { describe, expect, it } from 'vitest';
import { THEMES, isThemePreference, applyTheme, COSMIC_THEME } from './theme';

describe('THEMES registry', () => {
  const names = Object.keys(THEMES);
  const reference = COSMIC_THEME;
  const referenceKeys = Object.keys(reference.variables).sort();

  it('includes the six design-system themes', () => {
    expect(names).toEqual(
      expect.arrayContaining(['cosmic', 'light', 'professional', 'ugentx', 'tango', 'blackhawk'])
    );
  });

  it.each(names)('theme %s defines the exact reference key set', (name) => {
    const keys = Object.keys(THEMES[name as keyof typeof THEMES].variables).sort();
    expect(keys).toEqual(referenceKeys);
  });

  it.each(names)('theme %s has picker metadata', (name) => {
    const t = THEMES[name as keyof typeof THEMES];
    expect(t.name).toBe(name);
    expect(t.displayName.length).toBeGreaterThan(0);
    expect(t.description.length).toBeGreaterThan(0);
    expect(t.icon.length).toBeGreaterThan(0);
  });

  it.each(names)('theme %s never uses bare "none" in shadow-list tokens', (name) => {
    // `none` inside a box-shadow LIST invalidates the whole declaration —
    // focus rings silently vanished in flat themes before this was fixed.
    const vars = THEMES[name as keyof typeof THEMES].variables;
    for (const key of Object.keys(vars)) {
      if (/^--(glow|border-glow)-/.test(key)) {
        expect(vars[key], `${name} ${key}`).not.toBe('none');
      }
    }
  });

  it('isThemePreference accepts every registry name + system, rejects junk', () => {
    for (const name of names) expect(isThemePreference(name)).toBe(true);
    expect(isThemePreference('system')).toBe(true);
    expect(isThemePreference('midnight')).toBe(false);
    expect(isThemePreference(undefined)).toBe(false);
  });

  it('applyTheme stamps data-theme on the root element', () => {
    applyTheme(COSMIC_THEME);
    expect(document.documentElement.dataset.theme).toBe('cosmic');
    applyTheme(THEMES.blackhawk);
    expect(document.documentElement.dataset.theme).toBe('blackhawk');
  });
});
