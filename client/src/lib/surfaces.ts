/**
 * Surfaces — the single source of truth for "open workspace surface X".
 *
 * Both the TopBar (its live status icons) and the command palette open these
 * same drawers/modals. Defining the `openModal(...)` descriptors once here means
 * the two surfaces can't drift (the historical bug: a surface added to the strip
 * but never the palette). Adding a surface is one entry.
 */

import type { ModalConfig } from '../contexts/ModalContext';

export type SurfaceKey =
  | 'settings'
  | 'tools'
  | 'memory'
  | 'sources'
  | 'ambassador'
  | 'plans'
  | 'translation'
  | 'logs'
  | 'toolOutputBrowser'
  | 'profileEditor'
  | 'conversations'
  | 'changePassword';

export const SURFACES: Record<SurfaceKey, ModalConfig> = {
  settings: { id: 'unified-settings', type: 'modal', component: 'unifiedSettings', size: 'full' },
  tools: { id: 'toolkit', type: 'modal', component: 'tools', size: 'full' },
  memory: { id: 'memory-drawer', type: 'drawer', component: 'memory', position: 'right', size: 'xxl' },
  sources: { id: 'sources-drawer', type: 'drawer', component: 'sources', position: 'right', size: 'xxl' },
  ambassador: { id: 'ambassador-drawer', type: 'drawer', component: 'ambassador', position: 'right', size: 'xxl' },
  plans: { id: 'plans-drawer', type: 'drawer', component: 'plans', position: 'right', size: 'xxl' },
  translation: { id: 'translation-modal', type: 'modal', component: 'translation', size: 'lg' },
  logs: { id: 'logs-drawer', type: 'drawer', component: 'logs', position: 'right', size: 'xxl' },
  toolOutputBrowser: { id: 'tool-output-browser', type: 'drawer', component: 'toolOutputBrowser', position: 'right', size: 'xxl' },
  profileEditor: { id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full' },
  conversations: { id: 'conversations-drawer', type: 'drawer', component: 'conversations', position: 'right', size: 'md' },
  changePassword: { id: 'change-password', type: 'modal', component: 'changePassword', size: 'sm' },
};
