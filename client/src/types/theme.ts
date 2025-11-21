export interface Theme {
  name: string;
  colors: {
    bgPrimary: string;
    bgSecondary: string;
    bgTertiary: string;
    bgHover: string;
    textPrimary: string;
    textSecondary: string;
    textMuted: string;
    accentPrimary: string;
    accentHover: string;
    borderColor: string;
    success: string;
    warning: string;
    error: string;
  };
  sidebar: {
    width: string;
  };
}

export type ThemeName = 'dark' | 'light' | 'ocean' | 'forest';
