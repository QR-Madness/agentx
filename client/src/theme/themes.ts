import { Theme } from '../types/theme';

export const darkTheme: Theme = {
  name: 'dark',
  colors: {
    bgPrimary: '#0f0f0f',
    bgSecondary: '#1a1a1a',
    bgTertiary: '#242424',
    bgHover: '#2a2a2a',
    textPrimary: '#e8e8e8',
    textSecondary: '#a8a8a8',
    textMuted: '#6a6a6a',
    accentPrimary: '#3b82f6',
    accentHover: '#2563eb',
    borderColor: '#333333',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  sidebar: {
    width: '240px',
  },
};

export const lightTheme: Theme = {
  name: 'light',
  colors: {
    bgPrimary: '#ffffff',
    bgSecondary: '#f5f5f5',
    bgTertiary: '#e8e8e8',
    bgHover: '#e0e0e0',
    textPrimary: '#1a1a1a',
    textSecondary: '#4a4a4a',
    textMuted: '#8a8a8a',
    accentPrimary: '#3b82f6',
    accentHover: '#2563eb',
    borderColor: '#d0d0d0',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  sidebar: {
    width: '240px',
  },
};

export const oceanTheme: Theme = {
  name: 'ocean',
  colors: {
    bgPrimary: '#0a1929',
    bgSecondary: '#132f4c',
    bgTertiary: '#1a4565',
    bgHover: '#21547a',
    textPrimary: '#e8e8e8',
    textSecondary: '#a8a8a8',
    textMuted: '#6a6a6a',
    accentPrimary: '#26a0e8',
    accentHover: '#1e8bd3',
    borderColor: '#1e3a52',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  sidebar: {
    width: '240px',
  },
};

export const forestTheme: Theme = {
  name: 'forest',
  colors: {
    bgPrimary: '#0f1e13',
    bgSecondary: '#1a2f1f',
    bgTertiary: '#243d2a',
    bgHover: '#2d4a36',
    textPrimary: '#e8e8e8',
    textSecondary: '#a8a8a8',
    textMuted: '#6a6a6a',
    accentPrimary: '#4ade80',
    accentHover: '#22c55e',
    borderColor: '#1e3824',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  sidebar: {
    width: '240px',
  },
};

export const themes = {
  dark: darkTheme,
  light: lightTheme,
  ocean: oceanTheme,
  forest: forestTheme,
};
