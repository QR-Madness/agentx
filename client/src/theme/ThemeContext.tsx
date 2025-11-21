import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { Theme, ThemeName } from '@/types/theme';
import { themes } from './themes';
import { GlobalStyles } from './GlobalStyles';

interface ThemeContextType {
  theme: Theme;
  themeName: ThemeName;
  setTheme: (name: ThemeName) => void;
  cycleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const themeOrder: ThemeName[] = ['dark', 'light', 'ocean', 'forest'];

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [themeName, setThemeName] = useState<ThemeName>(() => {
    const saved = localStorage.getItem('agentx-theme') as ThemeName;
    return saved && themes[saved] ? saved : 'dark';
  });

  const theme = themes[themeName];

  useEffect(() => {
    localStorage.setItem('agentx-theme', themeName);
  }, [themeName]);

  const setTheme = (name: ThemeName) => {
    if (themes[name]) {
      setThemeName(name);
    }
  };

  const cycleTheme = () => {
    const currentIndex = themeOrder.indexOf(themeName);
    const nextIndex = (currentIndex + 1) % themeOrder.length;
    setThemeName(themeOrder[nextIndex]);
  };

  return (
    <ThemeContext.Provider value={{ theme, themeName, setTheme, cycleTheme }}>
      <StyledThemeProvider theme={theme}>
        <GlobalStyles theme={theme} />
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
