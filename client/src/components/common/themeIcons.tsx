/**
 * themeIcons — maps a ThemeDefinition's string `icon` key to a lucide component,
 * so `lib/theme.ts` stays React-free while pickers (command palette, Settings →
 * Appearance) render registry-driven entries. Add a theme = add its icon key
 * here only if it introduces a NEW key; existing keys are reusable.
 */

import { Contrast, Crosshair, Moon, Palette, Sun, Terminal, type LucideIcon } from 'lucide-react';
import type { ThemeIcon } from '../../lib/theme';

export const THEME_ICONS: Record<ThemeIcon, LucideIcon> = {
  moon: Moon,
  sun: Sun,
  contrast: Contrast,
  terminal: Terminal,
  palette: Palette,
  crosshair: Crosshair,
};
