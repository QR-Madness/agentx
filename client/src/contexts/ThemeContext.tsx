/**
 * Theme Context — Manages CSS variable-based theming with swappable themes
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  ThemeDefinition,
  THEMES,
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  applyTheme,
} from '../lib/theme';

interface ThemeContextValue {
  currentTheme: string;
  themeDefinition: ThemeDefinition;
  setTheme: (name: string) => void;
  availableThemes: string[];
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [currentTheme, setCurrentTheme] = useState<string>(() => {
    try {
      return localStorage.getItem(THEME_STORAGE_KEY) || DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  const themeDefinition = THEMES[currentTheme] || THEMES[DEFAULT_THEME];

  useEffect(() => {
    applyTheme(themeDefinition);
  }, [themeDefinition]);

  const setTheme = useCallback((name: string) => {
    if (THEMES[name]) {
      setCurrentTheme(name);
      try {
        localStorage.setItem(THEME_STORAGE_KEY, name);
      } catch {
        // localStorage unavailable
      }
    }
  }, []);

  const availableThemes = Object.keys(THEMES);

  return (
    <ThemeContext.Provider value={{ currentTheme, themeDefinition, setTheme, availableThemes }}>
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
