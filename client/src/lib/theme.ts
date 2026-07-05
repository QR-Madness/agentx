/**
 * Theme System — CSS variable-based theming with semantic design tokens
 *
 * Token Hierarchy:
 * 1. Primitive tokens - Raw color values (defined inline)
 * 2. Semantic tokens - Purpose-based (surface, text, accent, feedback)
 * 3. Component tokens - Component-specific overrides (optional)
 */

/** Icon key for theme pickers — mapped to lucide components in
 *  `components/common/themeIcons.tsx` (keeps this module React-free). */
export type ThemeIcon = 'moon' | 'sun' | 'contrast' | 'terminal' | 'palette' | 'crosshair';

export interface ThemeDefinition {
  name: string;
  displayName: string;
  /** One-line picker description (Settings → Appearance). */
  description: string;
  /** Picker icon key (see ThemeIcon). */
  icon: ThemeIcon;
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

/** Primitive color palette - Light theme (warm white / neutral, restrained accent) */
const LIGHT_PRIMITIVES = {
  // Surfaces — warm white → warm gray (slightly warm neutral, not cool slate)
  white: '#fdfdfc',
  gray50: '#f6f6f3',
  gray100: '#eeeeea',
  gray200: '#e4e4de',
  gray300: '#d7d7cf',
  gray400: '#a6a69b',

  // Accent — a single restrained slate-blue (deeper for light bg). The violet/
  // cyan/pink names are kept as keys (referenced below) but now all resolve to
  // the slate-blue family, so the chrome stays monochrome.
  purple600: '#4f6695',
  violet600: '#4f6695',
  cyan600: '#3e527d',
  pink600: '#6981ab',
  blue600: '#0969da', // feedback "info" — stays a real blue
  indigo600: '#4f6695',

  // Text — warm neutral
  textDark: '#23231f',
  textMid: '#56564d',
  textLight: '#87877b',

  // Status — kept colored (semantic), tuned for a light bg
  green600: '#1a7f37',
  amber600: '#9a6700',
  red600: '#cf222e',
  slate400: '#6e7781',
};

/** Primitive color palette - Professional theme (deep graphite, restrained accent) */
const PROFESSIONAL_PRIMITIVES = {
  // Surfaces — a deep, neutral graphite ramp with clear elevation steps
  base: '#0f0f12',
  nebula: '#131316',
  sunken: '#0a0a0c',
  card: '#18181c',
  elevated: '#212126',

  // Accent — single restrained slate-blue
  accent: '#7e93c9',
  accentDeep: '#6c80b8',
  accentSoft: '#93a6d6',

  // Text — calm off-white (never a stark/blinding pure white)
  textLight: '#d6d6db',
  textMid: '#96969f',
  textDim: '#63636c',

  // Status — kept colored (semantic)
  green: '#3fb950',
  amber: '#d29922',
  red: '#f85149',
  blue: '#58a6ff',
  slate: '#6e7681',
};

/**
 * Spacing scale — theme-independent, spread into every theme's variables so the
 * tokens exist regardless of the active theme. Replaces the magic-number
 * paddings/gaps scattered across component CSS (use `var(--space-lg)` etc.).
 */
const SPACING_TOKENS = {
  '--space-xs': '4px',
  '--space-sm': '8px',
  '--space-md': '12px',
  '--space-lg': '16px',
  '--space-xl': '24px',
  '--space-2xl': '32px',
  '--space-3xl': '40px',
} as const;

/**
 * Theme-independent scale tokens with no Tailwind namespace — runtime vars
 * only (the Tailwind-namespaced scale — radii, type, tracking — lives as
 * build-time literals in App.css `@theme static`; see the bridge notes there).
 */
const SCALE_TOKENS = {
  '--hit-target': '44px',
  '--reading-width': '768px',
  '--weight-regular': '400',
  '--weight-medium': '500',
  '--weight-semibold': '600',
  '--weight-bold': '700',
} as const;

/** A box-shadow-safe "no glow": `none` inside a shadow LIST invalidates the
 *  whole declaration (focus rings silently vanished in flat themes), while a
 *  zero-size transparent shadow composes fine. */
const NO_GLOW = '0 0 rgba(0, 0, 0, 0)';

export const COSMIC_THEME: ThemeDefinition = {
  name: 'cosmic',
  displayName: 'Cosmic Dark',
  description: 'Deep space, violet aurora accents, glass + glow',
  icon: 'moon',
  isDark: true,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
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
    '--border-subtle': 'rgba(255, 255, 255, 0.08)',

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
    '--accent-gradient': 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',

    // === Glass Effect Token ===
    '--glass-bg': 'rgba(17, 24, 39, 0.7)',
    '--glass-bg-hover': 'rgba(17, 24, 39, 0.85)',
    '--glass-blur': '12px',

    // === Settings / Panel Glass Layers (depth 0 → 3) ===
    '--glass-backdrop': 'rgba(5, 7, 15, 0.6)',
    '--glass-layer-0': 'rgba(17, 24, 39, 0.5)',
    '--glass-layer-1': 'rgba(17, 24, 39, 0.7)',
    '--glass-layer-2': 'rgba(26, 32, 53, 0.85)',
    '--glass-layer-3': 'rgba(31, 41, 70, 0.95)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': '0 0 12px rgba(139, 92, 246, 0.15)',
    '--border-glow-medium': '0 0 24px rgba(139, 92, 246, 0.3)',
    '--border-glow-strong': '0 0 36px rgba(139, 92, 246, 0.5)',
    '--accent-tint-soft': 'rgba(139, 92, 246, 0.15)',
    '--accent-tint-medium': 'rgba(139, 92, 246, 0.3)',
    '--accent-tint-strong': 'rgba(139, 92, 246, 0.5)',
    '--accent-tint-faint': 'rgba(139, 92, 246, 0.08)',
    // === Feedback Tint Tokens (subtle bg + border, in each theme's own hue) ===
    '--feedback-success-tint': 'rgba(34, 197, 94, 0.13)',
    '--feedback-success-border': 'rgba(34, 197, 94, 0.32)',
    '--feedback-warning-tint': 'rgba(245, 158, 11, 0.13)',
    '--feedback-warning-border': 'rgba(245, 158, 11, 0.32)',
    '--feedback-error-tint': 'rgba(239, 68, 68, 0.13)',
    '--feedback-error-border': 'rgba(239, 68, 68, 0.32)',
    '--feedback-info-tint': 'rgba(59, 130, 246, 0.13)',
    '--feedback-info-border': 'rgba(59, 130, 246, 0.32)',
    // === Focus ring + code-syntax palette (One Dark on dark surfaces) ===
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#c678dd',
    '--syntax-string': '#98c379',
    '--syntax-number': '#d19a66',
    '--syntax-comment': '#5c6370',
    '--syntax-function': '#61afef',
    '--syntax-variable': '#e06c75',
    '--syntax-class': '#e5c07b',
    '--syntax-operator': '#56b6c2',
    '--text-error': COSMIC_PRIMITIVES.red500,
    '--text-tertiary': COSMIC_PRIMITIVES.textDim,

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

    // Immersive chat surfaces (flat-row canvas + floating composer)
    '--chat-user-tint': 'rgba(139, 92, 246, 0.10)',
    '--composer-bg': 'rgba(17, 24, 39, 0.72)',

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
  description: 'Flat warm neutrals, restrained slate-blue accent',
  icon: 'sun',
  isDark: false,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
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
    '--border-subtle': 'rgba(0, 0, 0, 0.08)',

    // === Semantic Accent Tokens (single restrained slate-blue) ===
    '--accent-primary': LIGHT_PRIMITIVES.violet600,
    '--accent-secondary': LIGHT_PRIMITIVES.cyan600,
    '--accent-tertiary': LIGHT_PRIMITIVES.pink600,

    // === Semantic Feedback Tokens (kept colored) ===
    '--feedback-success': LIGHT_PRIMITIVES.green600,
    '--feedback-warning': LIGHT_PRIMITIVES.amber600,
    '--feedback-error': LIGHT_PRIMITIVES.red600,
    '--feedback-info': LIGHT_PRIMITIVES.blue600,
    '--feedback-inactive': LIGHT_PRIMITIVES.slate400,

    // === Interactive State Tokens ===
    '--interactive-default': LIGHT_PRIMITIVES.gray100,
    '--interactive-hover': 'rgba(79, 102, 149, 0.08)',
    '--interactive-active': 'rgba(79, 102, 149, 0.15)',
    '--interactive-disabled': LIGHT_PRIMITIVES.gray200,

    // === Shadow Tokens (glows OFF for the flat, professional look) ===
    '--shadow-sm': '0 1px 3px rgba(0, 0, 0, 0.08)',
    '--shadow-md': '0 4px 12px rgba(0, 0, 0, 0.08)',
    '--shadow-lg': '0 8px 24px rgba(0, 0, 0, 0.1)',
    '--glow-primary': NO_GLOW,
    '--glow-secondary': NO_GLOW,
    '--glow-accent': NO_GLOW,

    // === Gradient Tokens (flattened to solids/none) ===
    '--gradient-primary': LIGHT_PRIMITIVES.violet600,
    '--gradient-accent': LIGHT_PRIMITIVES.violet600,
    '--gradient-surface': 'none',
    '--gradient-cosmic': 'none',
    '--accent-gradient': LIGHT_PRIMITIVES.violet600,

    // === Glass Effect Token ===
    '--glass-bg': 'rgba(255, 255, 255, 0.8)',
    '--glass-bg-hover': 'rgba(255, 255, 255, 0.9)',
    '--glass-blur': '12px',

    // === Settings / Panel Glass Layers (depth 0 → 3) ===
    '--glass-backdrop': 'rgba(15, 23, 42, 0.4)',
    '--glass-layer-0': 'rgba(255, 255, 255, 0.65)',
    '--glass-layer-1': 'rgba(255, 255, 255, 0.8)',
    '--glass-layer-2': 'rgba(248, 250, 252, 0.9)',
    '--glass-layer-3': 'rgba(241, 245, 249, 0.96)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': NO_GLOW,
    '--border-glow-medium': NO_GLOW,
    '--border-glow-strong': NO_GLOW,
    '--accent-tint-soft': 'rgba(79, 102, 149, 0.1)',
    '--accent-tint-medium': 'rgba(79, 102, 149, 0.2)',
    '--accent-tint-strong': 'rgba(79, 102, 149, 0.35)',
    '--accent-tint-faint': 'rgba(79, 102, 149, 0.05)',
    // === Feedback Tint Tokens (subtle bg + border, in each theme's own hue) ===
    '--feedback-success-tint': 'rgba(26, 127, 55, 0.1)',
    '--feedback-success-border': 'rgba(26, 127, 55, 0.28)',
    '--feedback-warning-tint': 'rgba(154, 103, 0, 0.1)',
    '--feedback-warning-border': 'rgba(154, 103, 0, 0.28)',
    '--feedback-error-tint': 'rgba(207, 34, 46, 0.1)',
    '--feedback-error-border': 'rgba(207, 34, 46, 0.28)',
    '--feedback-info-tint': 'rgba(9, 105, 218, 0.1)',
    '--feedback-info-border': 'rgba(9, 105, 218, 0.28)',
    // === Focus ring + code-syntax palette (One Light on light surfaces) ===
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#a626a4',
    '--syntax-string': '#50a14f',
    '--syntax-number': '#986801',
    '--syntax-comment': '#a0a1a7',
    '--syntax-function': '#4078f2',
    '--syntax-variable': '#e45649',
    '--syntax-class': '#c18401',
    '--syntax-operator': '#0184bc',
    '--text-error': LIGHT_PRIMITIVES.red600,
    '--text-tertiary': LIGHT_PRIMITIVES.textLight,

    // === Backwards Compatibility (legacy variable names) ===
    // Background Palette - mapped to warm-neutral equivalents
    '--bg-space': LIGHT_PRIMITIVES.white,
    '--bg-nebula': LIGHT_PRIMITIVES.gray50,
    '--bg-void': LIGHT_PRIMITIVES.gray100,
    '--bg-card': LIGHT_PRIMITIVES.white,
    '--bg-elevated': LIGHT_PRIMITIVES.gray50,
    '--bg-hover': 'rgba(0, 0, 0, 0.04)',
    '--border-color': LIGHT_PRIMITIVES.gray300,
    '--border-glow': LIGHT_PRIMITIVES.gray400,

    // Immersive chat surfaces (flat-row canvas + floating composer)
    '--chat-user-tint': 'rgba(79, 102, 149, 0.07)',
    '--composer-bg': 'rgba(253, 253, 252, 0.8)',

    // Accent Colors (legacy) — all resolve to the slate-blue family (monochrome)
    '--cosmic-purple': LIGHT_PRIMITIVES.violet600,
    '--cosmic-violet': LIGHT_PRIMITIVES.violet600,
    '--cosmic-cyan': LIGHT_PRIMITIVES.cyan600,
    '--cosmic-pink': LIGHT_PRIMITIVES.pink600,
    '--cosmic-blue': LIGHT_PRIMITIVES.cyan600,
    '--cosmic-indigo': LIGHT_PRIMITIVES.violet600,

    // Gradients (legacy) — flattened
    '--gradient-nebula': LIGHT_PRIMITIVES.violet600,
    '--gradient-aurora': LIGHT_PRIMITIVES.violet600,
    '--gradient-stardust': 'none',
    '--gradient-cosmic-bg': 'none',

    // Status Colors (legacy)
    '--status-online': LIGHT_PRIMITIVES.green600,
    '--status-warning': LIGHT_PRIMITIVES.amber600,
    '--status-error': LIGHT_PRIMITIVES.red600,
    '--status-inactive': LIGHT_PRIMITIVES.slate400,

    // Glow Effects (legacy) — off
    '--glow-purple': NO_GLOW,
    '--glow-cyan': NO_GLOW,
    '--glow-pink': NO_GLOW,
  },
};

export const PROFESSIONAL_THEME: ThemeDefinition = {
  name: 'professional',
  displayName: 'Professional',
  description: 'Flat graphite monochrome, quiet slate-blue accent',
  icon: 'contrast',
  isDark: true,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
    // === Semantic Surface Tokens ===
    '--surface-base': PROFESSIONAL_PRIMITIVES.base,
    '--surface-raised': PROFESSIONAL_PRIMITIVES.card,
    '--surface-overlay': PROFESSIONAL_PRIMITIVES.elevated,
    '--surface-sunken': PROFESSIONAL_PRIMITIVES.sunken,
    '--surface-hover': 'rgba(255, 255, 255, 0.05)',

    // === Semantic Text Tokens ===
    '--text-primary': PROFESSIONAL_PRIMITIVES.textLight,
    '--text-secondary': PROFESSIONAL_PRIMITIVES.textMid,
    '--text-muted': PROFESSIONAL_PRIMITIVES.textDim,
    '--text-inverse': PROFESSIONAL_PRIMITIVES.base,

    // === Semantic Border Tokens ===
    '--border-default': 'rgba(255, 255, 255, 0.1)',
    '--border-emphasis': 'rgba(255, 255, 255, 0.18)',
    '--border-interactive': PROFESSIONAL_PRIMITIVES.accent,
    '--border-subtle': 'rgba(255, 255, 255, 0.08)',

    // === Semantic Accent Tokens (single restrained slate-blue) ===
    '--accent-primary': PROFESSIONAL_PRIMITIVES.accent,
    '--accent-secondary': PROFESSIONAL_PRIMITIVES.accentDeep,
    '--accent-tertiary': PROFESSIONAL_PRIMITIVES.accentSoft,

    // === Semantic Feedback Tokens (kept colored) ===
    '--feedback-success': PROFESSIONAL_PRIMITIVES.green,
    '--feedback-warning': PROFESSIONAL_PRIMITIVES.amber,
    '--feedback-error': PROFESSIONAL_PRIMITIVES.red,
    '--feedback-info': PROFESSIONAL_PRIMITIVES.blue,
    '--feedback-inactive': PROFESSIONAL_PRIMITIVES.slate,

    // === Interactive State Tokens ===
    '--interactive-default': PROFESSIONAL_PRIMITIVES.elevated,
    '--interactive-hover': 'rgba(255, 255, 255, 0.07)',
    '--interactive-active': 'rgba(255, 255, 255, 0.12)',
    '--interactive-disabled': 'rgba(255, 255, 255, 0.04)',

    // === Shadow Tokens (glows OFF — flat, professional) ===
    '--shadow-sm': '0 1px 3px rgba(0, 0, 0, 0.4)',
    '--shadow-md': '0 4px 12px rgba(0, 0, 0, 0.45)',
    '--shadow-lg': '0 8px 28px rgba(0, 0, 0, 0.5)',
    '--glow-primary': NO_GLOW,
    '--glow-secondary': NO_GLOW,
    '--glow-accent': NO_GLOW,

    // === Gradient Tokens (flattened to solids/none) ===
    '--gradient-primary': PROFESSIONAL_PRIMITIVES.accent,
    '--gradient-accent': PROFESSIONAL_PRIMITIVES.accent,
    '--gradient-surface': 'none',
    '--gradient-cosmic': 'none',
    '--accent-gradient': PROFESSIONAL_PRIMITIVES.accent,

    // === Glass Effect Token ===
    '--glass-bg': 'rgba(18, 18, 22, 0.7)',
    '--glass-bg-hover': 'rgba(18, 18, 22, 0.85)',
    '--glass-blur': '12px',

    // === Settings / Panel Glass Layers (depth 0 → 3) ===
    '--glass-backdrop': 'rgba(0, 0, 0, 0.55)',
    '--glass-layer-0': 'rgba(18, 18, 22, 0.5)',
    '--glass-layer-1': 'rgba(18, 18, 22, 0.72)',
    '--glass-layer-2': 'rgba(28, 28, 33, 0.86)',
    '--glass-layer-3': 'rgba(34, 34, 40, 0.95)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': NO_GLOW,
    '--border-glow-medium': NO_GLOW,
    '--border-glow-strong': NO_GLOW,
    '--accent-tint-soft': 'rgba(126, 147, 201, 0.15)',
    '--accent-tint-medium': 'rgba(126, 147, 201, 0.28)',
    '--accent-tint-strong': 'rgba(126, 147, 201, 0.45)',
    '--accent-tint-faint': 'rgba(126, 147, 201, 0.08)',
    // === Feedback Tint Tokens (subtle bg + border, in each theme's own hue) ===
    '--feedback-success-tint': 'rgba(63, 185, 80, 0.14)',
    '--feedback-success-border': 'rgba(63, 185, 80, 0.34)',
    '--feedback-warning-tint': 'rgba(210, 153, 34, 0.14)',
    '--feedback-warning-border': 'rgba(210, 153, 34, 0.34)',
    '--feedback-error-tint': 'rgba(248, 81, 73, 0.14)',
    '--feedback-error-border': 'rgba(248, 81, 73, 0.34)',
    '--feedback-info-tint': 'rgba(88, 166, 255, 0.14)',
    '--feedback-info-border': 'rgba(88, 166, 255, 0.34)',
    // === Focus ring + code-syntax palette (One Dark on the graphite surfaces) ===
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#c678dd',
    '--syntax-string': '#98c379',
    '--syntax-number': '#d19a66',
    '--syntax-comment': '#5c6370',
    '--syntax-function': '#61afef',
    '--syntax-variable': '#e06c75',
    '--syntax-class': '#e5c07b',
    '--syntax-operator': '#56b6c2',
    '--text-error': PROFESSIONAL_PRIMITIVES.red,
    '--text-tertiary': PROFESSIONAL_PRIMITIVES.textDim,

    // === Backwards Compatibility (legacy variable names) ===
    '--bg-space': PROFESSIONAL_PRIMITIVES.base,
    '--bg-nebula': PROFESSIONAL_PRIMITIVES.nebula,
    '--bg-void': PROFESSIONAL_PRIMITIVES.sunken,
    '--bg-card': PROFESSIONAL_PRIMITIVES.card,
    '--bg-elevated': PROFESSIONAL_PRIMITIVES.elevated,
    '--bg-hover': 'rgba(255, 255, 255, 0.05)',
    '--border-color': 'rgba(255, 255, 255, 0.1)',
    '--border-glow': 'rgba(255, 255, 255, 0.18)',

    '--chat-user-tint': 'rgba(126, 147, 201, 0.1)',
    '--composer-bg': 'rgba(18, 18, 22, 0.78)',

    // Accent Colors (legacy) — all resolve to the slate-blue family (monochrome)
    '--cosmic-purple': PROFESSIONAL_PRIMITIVES.accent,
    '--cosmic-violet': PROFESSIONAL_PRIMITIVES.accent,
    '--cosmic-cyan': PROFESSIONAL_PRIMITIVES.accentDeep,
    '--cosmic-pink': PROFESSIONAL_PRIMITIVES.accentSoft,
    '--cosmic-blue': PROFESSIONAL_PRIMITIVES.accentDeep,
    '--cosmic-indigo': PROFESSIONAL_PRIMITIVES.accent,

    // Gradients (legacy) — flattened
    '--gradient-nebula': PROFESSIONAL_PRIMITIVES.accent,
    '--gradient-aurora': PROFESSIONAL_PRIMITIVES.accent,
    '--gradient-stardust': 'none',
    '--gradient-cosmic-bg': 'none',

    // Status Colors (legacy)
    '--status-online': PROFESSIONAL_PRIMITIVES.green,
    '--status-warning': PROFESSIONAL_PRIMITIVES.amber,
    '--status-error': PROFESSIONAL_PRIMITIVES.red,
    '--status-inactive': PROFESSIONAL_PRIMITIVES.slate,

    // Glow Effects (legacy) — off
    '--glow-purple': NO_GLOW,
    '--glow-cyan': NO_GLOW,
    '--glow-pink': NO_GLOW,
  },
};

// === Design-kit themes (AgentX Design System `tokens/{colors,effects}.css`,
// values verbatim; repo-only legacy keys mapped per theme; glow "none" →
// NO_GLOW so shadow lists stay valid). ===

export const UGENTX_THEME: ThemeDefinition = {
  name: 'ugentx',
  displayName: 'Ugentx',
  description: 'Unix × AgentX — phosphor-green terminal, CRT glow',
  icon: 'terminal',
  isDark: true,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
    '--surface-base': '#060a06',
    '--surface-raised': '#0c130c',
    '--surface-overlay': '#121b12',
    '--surface-sunken': '#040704',
    '--surface-hover': 'rgba(43, 220, 107, 0.07)',
    '--text-primary': '#d6ffe0',
    '--text-secondary': '#6cc585',
    '--text-muted': '#3d7a52',
    '--text-inverse': '#060a06',
    '--border-default': 'rgba(43, 220, 107, 0.16)',
    '--border-emphasis': 'rgba(43, 220, 107, 0.32)',
    '--border-interactive': '#2bdc6b',
    '--border-subtle': 'rgba(43, 220, 107, 0.08)',
    '--accent-primary': '#2bdc6b',
    '--accent-secondary': '#19c59a',
    '--accent-tertiary': '#5effa0',
    '--feedback-success': '#2bdc6b',
    '--feedback-warning': '#ffd24a',
    '--feedback-error': '#ff5f56',
    '--feedback-info': '#4af0d0',
    '--feedback-inactive': '#3d5a45',
    '--interactive-default': '#121b12',
    '--interactive-hover': 'rgba(43, 220, 107, 0.1)',
    '--interactive-active': 'rgba(43, 220, 107, 0.2)',
    '--interactive-disabled': 'rgba(43, 220, 107, 0.05)',
    '--shadow-sm': '0 2px 8px rgba(0, 0, 0, 0.5)',
    '--shadow-md': '0 4px 16px rgba(0, 0, 0, 0.6)',
    '--shadow-lg': '0 8px 32px rgba(0, 0, 0, 0.7)',
    '--glow-primary': '0 0 18px rgba(43, 220, 107, 0.4)',
    '--glow-secondary': '0 0 18px rgba(25, 197, 154, 0.35)',
    '--glow-accent': '0 0 18px rgba(94, 255, 160, 0.35)',
    '--gradient-primary': 'linear-gradient(135deg, #2bdc6b 0%, #19c59a 100%)',
    '--gradient-accent': 'linear-gradient(135deg, #2bdc6b 0%, #19c59a 50%, #5effa0 100%)',
    '--gradient-surface': 'linear-gradient(180deg, rgba(43, 220, 107, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(43, 220, 107, 0.12) 0%, transparent 50%)',
    '--accent-gradient': 'linear-gradient(135deg, #2bdc6b 0%, #19c59a 100%)',
    '--glass-bg': 'rgba(12, 19, 12, 0.7)',
    '--glass-bg-hover': 'rgba(12, 19, 12, 0.85)',
    '--glass-blur': '12px',
    '--glass-backdrop': 'rgba(6, 10, 6, 0.6)',
    '--glass-layer-0': 'rgba(12, 19, 12, 0.5)',
    '--glass-layer-1': 'rgba(12, 19, 12, 0.7)',
    '--glass-layer-2': 'rgba(18, 27, 18, 0.86)',
    '--glass-layer-3': 'rgba(22, 33, 22, 0.95)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': '0 0 12px rgba(43, 220, 107, 0.18)',
    '--border-glow-medium': '0 0 24px rgba(43, 220, 107, 0.34)',
    '--border-glow-strong': '0 0 36px rgba(43, 220, 107, 0.5)',
    '--accent-tint-soft': 'rgba(43, 220, 107, 0.15)',
    '--accent-tint-medium': 'rgba(43, 220, 107, 0.3)',
    '--accent-tint-strong': 'rgba(43, 220, 107, 0.5)',
    '--accent-tint-faint': 'rgba(43, 220, 107, 0.08)',
    '--feedback-success-tint': 'rgba(43, 220, 107, 0.14)',
    '--feedback-success-border': 'rgba(43, 220, 107, 0.34)',
    '--feedback-warning-tint': 'rgba(255, 210, 74, 0.14)',
    '--feedback-warning-border': 'rgba(255, 210, 74, 0.34)',
    '--feedback-error-tint': 'rgba(255, 95, 86, 0.14)',
    '--feedback-error-border': 'rgba(255, 95, 86, 0.34)',
    '--feedback-info-tint': 'rgba(74, 240, 208, 0.14)',
    '--feedback-info-border': 'rgba(74, 240, 208, 0.34)',
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#5effa0',
    '--syntax-string': '#b6f7c2',
    '--syntax-number': '#ffd24a',
    '--syntax-comment': '#3d7a52',
    '--syntax-function': '#4af0d0',
    '--syntax-variable': '#ff5f56',
    '--syntax-class': '#ffd24a',
    '--syntax-operator': '#2bdc6b',
    '--text-error': '#ff5f56',
    '--text-tertiary': '#3d7a52',
    // Legacy aliases
    '--bg-space': '#060a06',
    '--bg-nebula': '#0a100a',
    '--bg-void': '#040704',
    '--bg-card': '#0c130c',
    '--bg-elevated': '#121b12',
    '--bg-hover': 'rgba(43, 220, 107, 0.07)',
    '--border-color': 'rgba(43, 220, 107, 0.16)',
    '--border-glow': 'rgba(43, 220, 107, 0.32)',
    '--chat-user-tint': 'rgba(43, 220, 107, 0.1)',
    '--composer-bg': 'rgba(12, 19, 12, 0.78)',
    '--cosmic-purple': '#2bdc6b',
    '--cosmic-violet': '#2bdc6b',
    '--cosmic-cyan': '#19c59a',
    '--cosmic-pink': '#5effa0',
    '--cosmic-blue': '#19c59a',
    '--cosmic-indigo': '#2bdc6b',
    '--gradient-nebula': 'linear-gradient(135deg, #2bdc6b 0%, #19c59a 50%, #5effa0 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #2bdc6b 0%, #19c59a 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(43, 220, 107, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(43, 220, 107, 0.12) 0%, transparent 50%)',
    '--status-online': '#2bdc6b',
    '--status-warning': '#ffd24a',
    '--status-error': '#ff5f56',
    '--status-inactive': '#3d5a45',
    '--glow-purple': '0 0 18px rgba(43, 220, 107, 0.4)',
    '--glow-cyan': '0 0 18px rgba(25, 197, 154, 0.35)',
    '--glow-pink': '0 0 18px rgba(94, 255, 160, 0.35)',
  },
};

export const TANGO_THEME: ThemeDefinition = {
  name: 'tango',
  displayName: 'Tango',
  description: 'Clean graphite with the colorful Tango palette',
  icon: 'palette',
  isDark: true,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
    '--surface-base': '#0f0f12',
    '--surface-raised': '#18181c',
    '--surface-overlay': '#212126',
    '--surface-sunken': '#0a0a0c',
    '--surface-hover': 'rgba(114, 159, 207, 0.08)',
    '--text-primary': '#dcdce2',
    '--text-secondary': '#9a9aa3',
    '--text-muted': '#66666f',
    '--text-inverse': '#0f0f12',
    '--border-default': 'rgba(255, 255, 255, 0.1)',
    '--border-emphasis': 'rgba(255, 255, 255, 0.18)',
    '--border-interactive': '#729fcf',
    '--border-subtle': 'rgba(255, 255, 255, 0.08)',
    '--accent-primary': '#729fcf',
    '--accent-secondary': '#ad7fa8',
    '--accent-tertiary': '#fcaf3e',
    '--feedback-success': '#8ae234',
    '--feedback-warning': '#f0c020',
    '--feedback-error': '#ef2929',
    '--feedback-info': '#729fcf',
    '--feedback-inactive': '#888a85',
    '--interactive-default': '#212126',
    '--interactive-hover': 'rgba(114, 159, 207, 0.1)',
    '--interactive-active': 'rgba(114, 159, 207, 0.18)',
    '--interactive-disabled': 'rgba(255, 255, 255, 0.04)',
    '--shadow-sm': '0 1px 3px rgba(0, 0, 0, 0.4)',
    '--shadow-md': '0 4px 12px rgba(0, 0, 0, 0.45)',
    '--shadow-lg': '0 8px 28px rgba(0, 0, 0, 0.5)',
    '--glow-primary': NO_GLOW,
    '--glow-secondary': NO_GLOW,
    '--glow-accent': NO_GLOW,
    '--gradient-primary': 'linear-gradient(135deg, #729fcf 0%, #ad7fa8 100%)',
    '--gradient-accent': 'linear-gradient(135deg, #729fcf 0%, #ad7fa8 50%, #fcaf3e 100%)',
    '--gradient-surface': 'linear-gradient(180deg, rgba(114, 159, 207, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(114, 159, 207, 0.1) 0%, transparent 50%)',
    '--accent-gradient': 'linear-gradient(135deg, #729fcf 0%, #ad7fa8 100%)',
    '--glass-bg': 'rgba(18, 18, 22, 0.7)',
    '--glass-bg-hover': 'rgba(18, 18, 22, 0.85)',
    '--glass-blur': '12px',
    '--glass-backdrop': 'rgba(0, 0, 0, 0.55)',
    '--glass-layer-0': 'rgba(18, 18, 22, 0.5)',
    '--glass-layer-1': 'rgba(18, 18, 22, 0.72)',
    '--glass-layer-2': 'rgba(28, 28, 33, 0.86)',
    '--glass-layer-3': 'rgba(34, 34, 40, 0.95)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': NO_GLOW,
    '--border-glow-medium': NO_GLOW,
    '--border-glow-strong': NO_GLOW,
    '--accent-tint-soft': 'rgba(114, 159, 207, 0.15)',
    '--accent-tint-medium': 'rgba(114, 159, 207, 0.28)',
    '--accent-tint-strong': 'rgba(114, 159, 207, 0.45)',
    '--accent-tint-faint': 'rgba(114, 159, 207, 0.08)',
    '--feedback-success-tint': 'rgba(138, 226, 52, 0.14)',
    '--feedback-success-border': 'rgba(138, 226, 52, 0.34)',
    '--feedback-warning-tint': 'rgba(240, 192, 32, 0.14)',
    '--feedback-warning-border': 'rgba(240, 192, 32, 0.34)',
    '--feedback-error-tint': 'rgba(239, 41, 41, 0.14)',
    '--feedback-error-border': 'rgba(239, 41, 41, 0.34)',
    '--feedback-info-tint': 'rgba(114, 159, 207, 0.14)',
    '--feedback-info-border': 'rgba(114, 159, 207, 0.34)',
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#ad7fa8',
    '--syntax-string': '#8ae234',
    '--syntax-number': '#fcaf3e',
    '--syntax-comment': '#66666f',
    '--syntax-function': '#729fcf',
    '--syntax-variable': '#ef2929',
    '--syntax-class': '#f0c020',
    '--syntax-operator': '#34e2e2',
    '--text-error': '#ef2929',
    '--text-tertiary': '#66666f',
    // Legacy aliases
    '--bg-space': '#0f0f12',
    '--bg-nebula': '#131316',
    '--bg-void': '#0a0a0c',
    '--bg-card': '#18181c',
    '--bg-elevated': '#212126',
    '--bg-hover': 'rgba(114, 159, 207, 0.08)',
    '--border-color': 'rgba(255, 255, 255, 0.1)',
    '--border-glow': 'rgba(114, 159, 207, 0.3)',
    '--chat-user-tint': 'rgba(114, 159, 207, 0.1)',
    '--composer-bg': 'rgba(18, 18, 22, 0.78)',
    '--cosmic-purple': '#ad7fa8',
    '--cosmic-violet': '#729fcf',
    '--cosmic-cyan': '#34e2e2',
    '--cosmic-pink': '#fcaf3e',
    '--cosmic-blue': '#729fcf',
    '--cosmic-indigo': '#ad7fa8',
    '--gradient-nebula': 'linear-gradient(135deg, #729fcf 0%, #ad7fa8 50%, #fcaf3e 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #729fcf 0%, #ad7fa8 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(114, 159, 207, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(114, 159, 207, 0.1) 0%, transparent 50%)',
    '--status-online': '#8ae234',
    '--status-warning': '#f0c020',
    '--status-error': '#ef2929',
    '--status-inactive': '#888a85',
    '--glow-purple': NO_GLOW,
    '--glow-cyan': NO_GLOW,
    '--glow-pink': NO_GLOW,
  },
};

export const BLACKHAWK_THEME: ThemeDefinition = {
  name: 'blackhawk',
  displayName: 'Blackhawk',
  description: 'Tactical night-ops — amber HUD on pure black',
  icon: 'crosshair',
  isDark: true,
  variables: {
    ...SPACING_TOKENS,
    ...SCALE_TOKENS,
    '--surface-base': '#070708',
    '--surface-raised': '#101012',
    '--surface-overlay': '#18181b',
    '--surface-sunken': '#030304',
    '--surface-hover': 'rgba(255, 138, 26, 0.07)',
    '--text-primary': '#ece9e3',
    '--text-secondary': '#9a958c',
    '--text-muted': '#615d56',
    '--text-inverse': '#070708',
    '--border-default': 'rgba(255, 138, 26, 0.16)',
    '--border-emphasis': 'rgba(255, 138, 26, 0.32)',
    '--border-interactive': '#ff8a1a',
    '--border-subtle': 'rgba(255, 255, 255, 0.07)',
    '--accent-primary': '#ff8a1a',
    '--accent-secondary': '#ffb000',
    '--accent-tertiary': '#ff6a3d',
    '--feedback-success': '#5ad17a',
    '--feedback-warning': '#ffb000',
    '--feedback-error': '#ff4d4d',
    '--feedback-info': '#4ab8ff',
    '--feedback-inactive': '#5a564f',
    '--interactive-default': '#18181b',
    '--interactive-hover': 'rgba(255, 138, 26, 0.1)',
    '--interactive-active': 'rgba(255, 138, 26, 0.2)',
    '--interactive-disabled': 'rgba(255, 255, 255, 0.04)',
    '--shadow-sm': '0 2px 8px rgba(0, 0, 0, 0.6)',
    '--shadow-md': '0 4px 16px rgba(0, 0, 0, 0.7)',
    '--shadow-lg': '0 8px 32px rgba(0, 0, 0, 0.8)',
    '--glow-primary': '0 0 18px rgba(255, 138, 26, 0.35)',
    '--glow-secondary': '0 0 18px rgba(255, 176, 0, 0.3)',
    '--glow-accent': '0 0 18px rgba(255, 106, 61, 0.3)',
    '--gradient-primary': 'linear-gradient(135deg, #ff8a1a 0%, #ff5a2d 100%)',
    '--gradient-accent': 'linear-gradient(135deg, #ffb000 0%, #ff8a1a 50%, #ff5a2d 100%)',
    '--gradient-surface': 'linear-gradient(180deg, rgba(255, 138, 26, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255, 138, 26, 0.1) 0%, transparent 50%)',
    '--accent-gradient': 'linear-gradient(135deg, #ff8a1a 0%, #ff5a2d 100%)',
    '--glass-bg': 'rgba(16, 16, 18, 0.7)',
    '--glass-bg-hover': 'rgba(16, 16, 18, 0.85)',
    '--glass-blur': '12px',
    '--glass-backdrop': 'rgba(0, 0, 0, 0.62)',
    '--glass-layer-0': 'rgba(16, 16, 18, 0.5)',
    '--glass-layer-1': 'rgba(16, 16, 18, 0.72)',
    '--glass-layer-2': 'rgba(24, 24, 27, 0.86)',
    '--glass-layer-3': 'rgba(30, 30, 33, 0.95)',
    '--blur-subtle': '8px',
    '--blur-medium': '16px',
    '--blur-strong': '24px',
    '--border-glow-subtle': '0 0 12px rgba(255, 138, 26, 0.18)',
    '--border-glow-medium': '0 0 24px rgba(255, 138, 26, 0.34)',
    '--border-glow-strong': '0 0 36px rgba(255, 138, 26, 0.5)',
    '--accent-tint-soft': 'rgba(255, 138, 26, 0.15)',
    '--accent-tint-medium': 'rgba(255, 138, 26, 0.3)',
    '--accent-tint-strong': 'rgba(255, 138, 26, 0.5)',
    '--accent-tint-faint': 'rgba(255, 138, 26, 0.08)',
    '--feedback-success-tint': 'rgba(90, 209, 122, 0.14)',
    '--feedback-success-border': 'rgba(90, 209, 122, 0.34)',
    '--feedback-warning-tint': 'rgba(255, 176, 0, 0.14)',
    '--feedback-warning-border': 'rgba(255, 176, 0, 0.34)',
    '--feedback-error-tint': 'rgba(255, 77, 77, 0.14)',
    '--feedback-error-border': 'rgba(255, 77, 77, 0.34)',
    '--feedback-info-tint': 'rgba(74, 184, 255, 0.14)',
    '--feedback-info-border': 'rgba(74, 184, 255, 0.34)',
    '--focus-ring': 'var(--accent-primary)',
    '--syntax-keyword': '#ff8a1a',
    '--syntax-string': '#5ad17a',
    '--syntax-number': '#ffb000',
    '--syntax-comment': '#615d56',
    '--syntax-function': '#4ab8ff',
    '--syntax-variable': '#ff6a3d',
    '--syntax-class': '#ffd24a',
    '--syntax-operator': '#ff8a1a',
    '--text-error': '#ff4d4d',
    '--text-tertiary': '#615d56',
    // Legacy aliases
    '--bg-space': '#070708',
    '--bg-nebula': '#0b0b0d',
    '--bg-void': '#030304',
    '--bg-card': '#101012',
    '--bg-elevated': '#18181b',
    '--bg-hover': 'rgba(255, 138, 26, 0.07)',
    '--border-color': 'rgba(255, 138, 26, 0.16)',
    '--border-glow': 'rgba(255, 138, 26, 0.32)',
    '--chat-user-tint': 'rgba(255, 138, 26, 0.1)',
    '--composer-bg': 'rgba(16, 16, 18, 0.78)',
    '--cosmic-purple': '#ff8a1a',
    '--cosmic-violet': '#ff8a1a',
    '--cosmic-cyan': '#ffb000',
    '--cosmic-pink': '#ff6a3d',
    '--cosmic-blue': '#4ab8ff',
    '--cosmic-indigo': '#ff8a1a',
    '--gradient-nebula': 'linear-gradient(135deg, #ffb000 0%, #ff8a1a 50%, #ff5a2d 100%)',
    '--gradient-aurora': 'linear-gradient(135deg, #ff8a1a 0%, #ff5a2d 100%)',
    '--gradient-stardust': 'linear-gradient(180deg, rgba(255, 138, 26, 0.1) 0%, transparent 50%)',
    '--gradient-cosmic-bg': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255, 138, 26, 0.1) 0%, transparent 50%)',
    '--status-online': '#5ad17a',
    '--status-warning': '#ffb000',
    '--status-error': '#ff4d4d',
    '--status-inactive': '#5a564f',
    '--glow-purple': '0 0 18px rgba(255, 138, 26, 0.35)',
    '--glow-cyan': '0 0 18px rgba(255, 176, 0, 0.3)',
    '--glow-pink': '0 0 18px rgba(255, 106, 61, 0.3)',
  },
};

/** Registry of all available themes. Adding a theme here is the ONLY required
 *  registration — pickers (command palette, Settings → Appearance) iterate this. */
export const THEMES = {
  cosmic: COSMIC_THEME,
  light: LIGHT_THEME,
  professional: PROFESSIONAL_THEME,
  ugentx: UGENTX_THEME,
  tango: TANGO_THEME,
  blackhawk: BLACKHAWK_THEME,
} as const satisfies Record<string, ThemeDefinition>;

/** A concrete (non-`system`) theme name. */
export type ThemeName = keyof typeof THEMES;

/** Theme preference options (a concrete theme, or follow the OS). */
export type ThemePreference = ThemeName | 'system';

/** Whether a stored/string value is a valid theme preference. */
export function isThemePreference(value: unknown): value is ThemePreference {
  return value === 'system' || (typeof value === 'string' && value in THEMES);
}

/** Apply a theme's CSS variables to the document root */
export function applyTheme(theme: ThemeDefinition): void {
  const root = document.documentElement;

  // Set color-scheme for browser UI integration
  root.style.colorScheme = theme.isDark ? 'dark' : 'light';

  // Expose the active theme to CSS ([data-theme='…'] — the expression layer
  // and any design-kit CSS key off this attribute).
  root.dataset.theme = theme.name;

  for (const [property, value] of Object.entries(theme.variables)) {
    root.style.setProperty(property, value);
  }
}

/** Get system preferred color scheme (maps to a concrete dark/light theme). */
export function getSystemTheme(): ThemeName {
  if (typeof window === 'undefined') return 'cosmic';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'cosmic';
}

/** Resolve theme preference to actual theme name */
export function resolveTheme(preference: ThemePreference): ThemeName {
  if (preference === 'system') {
    return getSystemTheme();
  }
  return preference;
}

export const DEFAULT_THEME = 'cosmic';
export const THEME_STORAGE_KEY = 'agentx:theme';
