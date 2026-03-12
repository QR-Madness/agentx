/**
 * Theme System — CSS variable-based theming with swappable themes
 */

export interface ThemeDefinition {
  name: string;
  displayName: string;
  variables: Record<string, string>;
}

export const COSMIC_THEME: ThemeDefinition = {
  name: 'cosmic',
  displayName: 'Cosmic',
  variables: {
    // Background Palette
    '--bg-space': '#05070f',
    '--bg-nebula': '#0a0d17',
    '--bg-void': '#0d1120',
    '--bg-card': '#111827',
    '--bg-elevated': '#1a2035',
    '--bg-hover': 'rgba(139, 92, 246, 0.08)',
    '--border-color': 'rgba(139, 92, 246, 0.15)',
    '--border-glow': 'rgba(139, 92, 246, 0.3)',

    // Text Colors
    '--text-primary': '#f0f4ff',
    '--text-secondary': '#a5b4d4',
    '--text-muted': '#5a6b8a',

    // Accent Colors
    '--cosmic-purple': '#a855f7',
    '--cosmic-violet': '#8b5cf6',
    '--cosmic-cyan': '#06b6d4',
    '--cosmic-pink': '#ec4899',
    '--cosmic-blue': '#3b82f6',
    '--cosmic-indigo': '#6366f1',

    // Gradients
    '--gradient-nebula': 'linear-gradient(135deg, #a855f7 0%, #06b6d4 50%, #ec4899 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(139, 92, 246, 0.12) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',

    // Status Colors
    '--status-online': '#22c55e',
    '--status-warning': '#f59e0b',
    '--status-error': '#ef4444',
    '--status-inactive': '#475569',

    // Shadows & Glows
    '--shadow-sm': '0 2px 8px rgba(0, 0, 0, 0.4)',
    '--shadow-md': '0 4px 16px rgba(0, 0, 0, 0.5)',
    '--shadow-lg': '0 8px 32px rgba(0, 0, 0, 0.6)',
    '--glow-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
    '--glow-cyan': '0 0 20px rgba(6, 182, 212, 0.3)',
    '--glow-pink': '0 0 20px rgba(236, 72, 153, 0.3)',
  },
};

/** Registry of all available themes */
export const THEMES: Record<string, ThemeDefinition> = {
  cosmic: COSMIC_THEME,
};

/** Apply a theme's CSS variables to the document root */
export function applyTheme(theme: ThemeDefinition): void {
  const root = document.documentElement;
  for (const [property, value] of Object.entries(theme.variables)) {
    root.style.setProperty(property, value);
  }
}

export const DEFAULT_THEME = 'cosmic';
export const THEME_STORAGE_KEY = 'agentx:theme';
