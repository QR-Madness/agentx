/**
 * Theme System — CSS variable-based theming with semantic design tokens
 *
 * Token Hierarchy:
 * 1. Primitive tokens - Raw color values (defined inline)
 * 2. Semantic tokens - Purpose-based (surface, text, accent, feedback)
 * 3. Component tokens - Component-specific overrides (optional)
 */

export interface ThemeDefinition {
  name: string;
  displayName: string;
  isDark: boolean;
  variables: Record<string, string>;
}

/** Primitive color palette - Cosmic theme */
const COSMIC_PRIMITIVES = {
  // Blues/Purples
  space: '#05070f',
  nebula: '#0a0d17',
  void: '#0d1120',
  card: '#111827',
  elevated: '#1a2035',

  // Accent palette
  purple500: '#a855f7',
  violet500: '#8b5cf6',
  cyan500: '#06b6d4',
  pink500: '#ec4899',
  blue500: '#3b82f6',
  indigo500: '#6366f1',

  // Text
  textLight: '#f0f4ff',
  textMid: '#a5b4d4',
  textDim: '#5a6b8a',

  // Status
  green500: '#22c55e',
  amber500: '#f59e0b',
  red500: '#ef4444',
  slate500: '#475569',
};

/** Primitive color palette - Light theme */
const LIGHT_PRIMITIVES = {
  // Surfaces (light to dark)
  white: '#ffffff',
  gray50: '#f8fafc',
  gray100: '#f1f5f9',
  gray200: '#e2e8f0',
  gray300: '#cbd5e1',
  gray400: '#94a3b8',

  // Accent palette (slightly adjusted for light bg)
  purple600: '#9333ea',
  violet600: '#7c3aed',
  cyan600: '#0891b2',
  pink600: '#db2777',
  blue600: '#2563eb',
  indigo600: '#4f46e5',

  // Text
  textDark: '#0f172a',
  textMid: '#475569',
  textLight: '#64748b',

  // Status (same as dark, good contrast)
  green600: '#16a34a',
  amber600: '#d97706',
  red600: '#dc2626',
  slate400: '#94a3b8',
};

export const COSMIC_THEME: ThemeDefinition = {
  name: 'cosmic',
  displayName: 'Cosmic Dark',
  isDark: true,
  variables: {
    // === Semantic Surface Tokens ===
    '--surface-base': COSMIC_PRIMITIVES.space,
    '--surface-raised': COSMIC_PRIMITIVES.card,
    '--surface-overlay': COSMIC_PRIMITIVES.elevated,
    '--surface-sunken': COSMIC_PRIMITIVES.void,
    '--surface-hover': 'rgba(139, 92, 246, 0.08)',

    // === Semantic Text Tokens ===
    '--text-primary': COSMIC_PRIMITIVES.textLight,
    '--text-secondary': COSMIC_PRIMITIVES.textMid,
    '--text-muted': COSMIC_PRIMITIVES.textDim,
    '--text-inverse': COSMIC_PRIMITIVES.space,

    // === Semantic Border Tokens ===
    '--border-default': 'rgba(139, 92, 246, 0.15)',
    '--border-emphasis': 'rgba(139, 92, 246, 0.3)',
    '--border-interactive': COSMIC_PRIMITIVES.violet500,

    // === Semantic Accent Tokens ===
    '--accent-primary': COSMIC_PRIMITIVES.violet500,
    '--accent-secondary': COSMIC_PRIMITIVES.cyan500,
    '--accent-tertiary': COSMIC_PRIMITIVES.pink500,

    // === Semantic Feedback Tokens ===
    '--feedback-success': COSMIC_PRIMITIVES.green500,
    '--feedback-warning': COSMIC_PRIMITIVES.amber500,
    '--feedback-error': COSMIC_PRIMITIVES.red500,
    '--feedback-info': COSMIC_PRIMITIVES.blue500,
    '--feedback-inactive': COSMIC_PRIMITIVES.slate500,

    // === Interactive State Tokens ===
    '--interactive-default': COSMIC_PRIMITIVES.elevated,
    '--interactive-hover': 'rgba(139, 92, 246, 0.1)',
    '--interactive-active': 'rgba(139, 92, 246, 0.2)',
    '--interactive-disabled': 'rgba(139, 92, 246, 0.05)',

    // === Shadow & Glow Tokens ===
    '--shadow-sm': '0 2px 8px rgba(0, 0, 0, 0.4)',
    '--shadow-md': '0 4px 16px rgba(0, 0, 0, 0.5)',
    '--shadow-lg': '0 8px 32px rgba(0, 0, 0, 0.6)',
    '--glow-primary': '0 0 20px rgba(139, 92, 246, 0.3)',
    '--glow-secondary': '0 0 20px rgba(6, 182, 212, 0.3)',
    '--glow-accent': '0 0 20px rgba(236, 72, 153, 0.3)',

    // === Gradient Tokens ===
    '--gradient-primary': 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
    '--gradient-accent': 'linear-gradient(135deg, #a855f7 0%, #06b6d4 50%, #ec4899 100%)',
    '--gradient-surface': 'linear-gradient(180deg, rgba(139, 92, 246, 0.12) 0%, transparent 50%)',
    '--gradient-cosmic': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',

    // === Glass Effect Token ===
    '--glass-bg': 'rgba(17, 24, 39, 0.7)',
    '--glass-bg-hover': 'rgba(17, 24, 39, 0.85)',
    '--glass-blur': '12px',

    // === Backwards Compatibility (legacy variable names) ===
    // Background Palette
    '--bg-space': COSMIC_PRIMITIVES.space,
    '--bg-nebula': COSMIC_PRIMITIVES.nebula,
    '--bg-void': COSMIC_PRIMITIVES.void,
    '--bg-card': COSMIC_PRIMITIVES.card,
    '--bg-elevated': COSMIC_PRIMITIVES.elevated,
    '--bg-hover': 'rgba(139, 92, 246, 0.08)',
    '--border-color': 'rgba(139, 92, 246, 0.15)',
    '--border-glow': 'rgba(139, 92, 246, 0.3)',

    // Accent Colors (legacy)
    '--cosmic-purple': COSMIC_PRIMITIVES.purple500,
    '--cosmic-violet': COSMIC_PRIMITIVES.violet500,
    '--cosmic-cyan': COSMIC_PRIMITIVES.cyan500,
    '--cosmic-pink': COSMIC_PRIMITIVES.pink500,
    '--cosmic-blue': COSMIC_PRIMITIVES.blue500,
    '--cosmic-indigo': COSMIC_PRIMITIVES.indigo500,

    // Gradients (legacy)
    '--gradient-nebula': 'linear-gradient(135deg, #a855f7 0%, #06b6d4 50%, #ec4899 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(139, 92, 246, 0.12) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',

    // Status Colors (legacy)
    '--status-online': COSMIC_PRIMITIVES.green500,
    '--status-warning': COSMIC_PRIMITIVES.amber500,
    '--status-error': COSMIC_PRIMITIVES.red500,
    '--status-inactive': COSMIC_PRIMITIVES.slate500,

    // Glow Effects (legacy)
    '--glow-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
    '--glow-cyan': '0 0 20px rgba(6, 182, 212, 0.3)',
    '--glow-pink': '0 0 20px rgba(236, 72, 153, 0.3)',
  },
};

export const LIGHT_THEME: ThemeDefinition = {
  name: 'light',
  displayName: 'Light',
  isDark: false,
  variables: {
    // === Semantic Surface Tokens ===
    '--surface-base': LIGHT_PRIMITIVES.white,
    '--surface-raised': LIGHT_PRIMITIVES.gray50,
    '--surface-overlay': LIGHT_PRIMITIVES.gray100,
    '--surface-sunken': LIGHT_PRIMITIVES.gray200,
    '--surface-hover': 'rgba(124, 58, 237, 0.06)',

    // === Semantic Text Tokens ===
    '--text-primary': LIGHT_PRIMITIVES.textDark,
    '--text-secondary': LIGHT_PRIMITIVES.textMid,
    '--text-muted': LIGHT_PRIMITIVES.textLight,
    '--text-inverse': LIGHT_PRIMITIVES.white,

    // === Semantic Border Tokens ===
    '--border-default': LIGHT_PRIMITIVES.gray300,
    '--border-emphasis': LIGHT_PRIMITIVES.gray400,
    '--border-interactive': LIGHT_PRIMITIVES.violet600,

    // === Semantic Accent Tokens ===
    '--accent-primary': LIGHT_PRIMITIVES.violet600,
    '--accent-secondary': LIGHT_PRIMITIVES.cyan600,
    '--accent-tertiary': LIGHT_PRIMITIVES.pink600,

    // === Semantic Feedback Tokens ===
    '--feedback-success': LIGHT_PRIMITIVES.green600,
    '--feedback-warning': LIGHT_PRIMITIVES.amber600,
    '--feedback-error': LIGHT_PRIMITIVES.red600,
    '--feedback-info': LIGHT_PRIMITIVES.blue600,
    '--feedback-inactive': LIGHT_PRIMITIVES.slate400,

    // === Interactive State Tokens ===
    '--interactive-default': LIGHT_PRIMITIVES.gray100,
    '--interactive-hover': 'rgba(124, 58, 237, 0.08)',
    '--interactive-active': 'rgba(124, 58, 237, 0.15)',
    '--interactive-disabled': LIGHT_PRIMITIVES.gray200,

    // === Shadow & Glow Tokens ===
    '--shadow-sm': '0 1px 3px rgba(0, 0, 0, 0.1)',
    '--shadow-md': '0 4px 12px rgba(0, 0, 0, 0.1)',
    '--shadow-lg': '0 8px 24px rgba(0, 0, 0, 0.12)',
    '--glow-primary': '0 0 12px rgba(124, 58, 237, 0.2)',
    '--glow-secondary': '0 0 12px rgba(8, 145, 178, 0.2)',
    '--glow-accent': '0 0 12px rgba(219, 39, 119, 0.2)',

    // === Gradient Tokens ===
    '--gradient-primary': 'linear-gradient(135deg, #7c3aed 0%, #0891b2 100%)',
    '--gradient-accent': 'linear-gradient(135deg, #9333ea 0%, #0891b2 50%, #db2777 100%)',
    '--gradient-surface': 'linear-gradient(180deg, rgba(124, 58, 237, 0.05) 0%, transparent 50%)',
    '--gradient-cosmic': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%)',

    // === Glass Effect Token ===
    '--glass-bg': 'rgba(255, 255, 255, 0.8)',
    '--glass-bg-hover': 'rgba(255, 255, 255, 0.9)',
    '--glass-blur': '12px',

    // === Backwards Compatibility (legacy variable names) ===
    // Background Palette - mapped to light equivalents
    '--bg-space': LIGHT_PRIMITIVES.white,
    '--bg-nebula': LIGHT_PRIMITIVES.gray50,
    '--bg-void': LIGHT_PRIMITIVES.gray100,
    '--bg-card': LIGHT_PRIMITIVES.white,
    '--bg-elevated': LIGHT_PRIMITIVES.gray50,
    '--bg-hover': 'rgba(124, 58, 237, 0.06)',
    '--border-color': LIGHT_PRIMITIVES.gray300,
    '--border-glow': LIGHT_PRIMITIVES.gray400,

    // Accent Colors (legacy) - slightly darker for light bg
    '--cosmic-purple': LIGHT_PRIMITIVES.purple600,
    '--cosmic-violet': LIGHT_PRIMITIVES.violet600,
    '--cosmic-cyan': LIGHT_PRIMITIVES.cyan600,
    '--cosmic-pink': LIGHT_PRIMITIVES.pink600,
    '--cosmic-blue': LIGHT_PRIMITIVES.blue600,
    '--cosmic-indigo': LIGHT_PRIMITIVES.indigo600,

    // Gradients (legacy)
    '--gradient-nebula': 'linear-gradient(135deg, #9333ea 0%, #0891b2 50%, #db2777 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #7c3aed 0%, #0891b2 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(124, 58, 237, 0.06) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%)',

    // Status Colors (legacy)
    '--status-online': LIGHT_PRIMITIVES.green600,
    '--status-warning': LIGHT_PRIMITIVES.amber600,
    '--status-error': LIGHT_PRIMITIVES.red600,
    '--status-inactive': LIGHT_PRIMITIVES.slate400,

    // Glow Effects (legacy) - subtler for light theme
    '--glow-purple': '0 0 12px rgba(124, 58, 237, 0.2)',
    '--glow-cyan': '0 0 12px rgba(8, 145, 178, 0.2)',
    '--glow-pink': '0 0 12px rgba(219, 39, 119, 0.2)',
  },
};

/** Registry of all available themes */
export const THEMES: Record<string, ThemeDefinition> = {
  cosmic: COSMIC_THEME,
  light: LIGHT_THEME,
};

/** Theme preference options */
export type ThemePreference = 'cosmic' | 'light' | 'system';

/** Apply a theme's CSS variables to the document root */
export function applyTheme(theme: ThemeDefinition): void {
  const root = document.documentElement;

  // Set color-scheme for browser UI integration
  root.style.colorScheme = theme.isDark ? 'dark' : 'light';

  for (const [property, value] of Object.entries(theme.variables)) {
    root.style.setProperty(property, value);
  }
}

/** Get system preferred color scheme */
export function getSystemTheme(): 'cosmic' | 'light' {
  if (typeof window === 'undefined') return 'cosmic';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'cosmic';
}

/** Resolve theme preference to actual theme name */
export function resolveTheme(preference: ThemePreference): 'cosmic' | 'light' {
  if (preference === 'system') {
    return getSystemTheme();
  }
  return preference;
}

export const DEFAULT_THEME = 'cosmic';
export const THEME_STORAGE_KEY = 'agentx:theme';
