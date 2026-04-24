/**
 * Theme Context — Manages CSS variable-based theming with system preference support
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  ThemeDefinition,
  ThemePreference,
  THEMES,
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  applyTheme,
  getSystemTheme,
} from '../lib/theme';

interface ThemeContextValue {
  /** User's theme preference (cosmic, light, or system) */
  preference: ThemePreference;
  /** The resolved/applied theme name */
  currentTheme: 'cosmic' | 'light';
  /** Full theme definition with all variables */
  themeDefinition: ThemeDefinition;
  /** Whether the current theme is dark */
  isDark: boolean;
  /** Set theme preference */
  setTheme: (preference: ThemePreference) => void;
  /** Available theme names */
  availableThemes: string[];
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Load preference from storage
  const [preference, setPreference] = useState<ThemePreference>(() => {
    try {
      const stored = localStorage.getItem(THEME_STORAGE_KEY);
      if (stored === 'cosmic' || stored === 'light' || stored === 'system') {
        return stored;
      }
      return DEFAULT_THEME as ThemePreference;
    } catch {
      return DEFAULT_THEME as ThemePreference;
    }
  });

  // Resolve system preference to actual theme
  const [systemTheme, setSystemTheme] = useState<'cosmic' | 'light'>(() => getSystemTheme());

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');

    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'light' : 'cosmic');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Compute the actual applied theme
  const currentTheme = preference === 'system' ? systemTheme : preference;
  const themeDefinition = THEMES[currentTheme] || THEMES[DEFAULT_THEME];

  // Apply theme to DOM
  useEffect(() => {
    applyTheme(themeDefinition);
  }, [themeDefinition]);

  const setTheme = useCallback((newPreference: ThemePreference) => {
    setPreference(newPreference);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, newPreference);
    } catch {
      // localStorage unavailable
    }
  }, []);

  const availableThemes = Object.keys(THEMES);

  return (
    <ThemeContext.Provider
      value={{
        preference,
        currentTheme,
        themeDefinition,
        isDark: themeDefinition.isDark,
        setTheme,
        availableThemes,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
