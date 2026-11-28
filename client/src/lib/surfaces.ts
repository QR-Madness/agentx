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
  | 'consolidation'
  | 'ambassador'
  | 'ambassadorDeck'
  | 'plans'
  | 'translation'
  | 'logs'
  | 'toolOutputBrowser'
  | 'profileEditor'
  | 'conversations'
  | 'conversationState'
  | 'workspaces'
  | 'teams'
  | 'roster'
  | 'changePassword';

export const SURFACES: Record<SurfaceKey, ModalConfig> = {
  settings: { id: 'unified-settings', type: 'modal', component: 'unifiedSettings', size: 'full' },
  tools: { id: 'toolkit', type: 'modal', component: 'tools', size: 'full' },
  // Full-screen immersive workbench (renders bare via FULLSCREEN_SURFACES; type/size cosmetic).
  memory: { id: 'memory', type: 'modal', component: 'memory', position: 'center', size: 'full' },
  sources: { id: 'sources-drawer', type: 'drawer', component: 'sources', position: 'right', size: 'xxl' },
  consolidation: { id: 'consolidation-drawer', type: 'drawer', component: 'consolidation', position: 'right', size: 'lg' },
  ambassador: { id: 'ambassador-drawer', type: 'drawer', component: 'ambassador', position: 'right', size: 'xxl' },
  // The standalone, conversation-less command deck — full-screen, app-wide.
  ambassadorDeck: { id: 'ambassador-deck', type: 'modal', component: 'ambassadorDeck', size: 'full' },
  plans: { id: 'plans-drawer', type: 'drawer', component: 'plans', position: 'right', size: 'xxl' },
  translation: { id: 'translation-modal', type: 'modal', component: 'translation', size: 'lg' },
  logs: { id: 'logs-drawer', type: 'drawer', component: 'logs', position: 'right', size: 'xxl' },
  toolOutputBrowser: { id: 'tool-output-browser', type: 'drawer', component: 'toolOutputBrowser', position: 'right', size: 'xxl' },
  profileEditor: { id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full' },
  conversations: { id: 'conversations-drawer', type: 'drawer', component: 'conversations', position: 'right', size: 'md' },
  // Opened with a `conversationId` prop (the active tab's) — the editable
  // structured working-memory surface for that conversation.
  conversationState: { id: 'conversation-state-drawer', type: 'drawer', component: 'conversationState', position: 'right', size: 'lg' },
  workspaces: { id: 'workspaces-drawer', type: 'drawer', component: 'workspaces', position: 'right', size: 'xxl' },
  // The Organization surface — the whole org as a chart, with team editing in
  // its dock. Teams are internally Alloy workflows; the key stays `teams`.
  teams: { id: 'org-manager', type: 'modal', component: 'orgManager', size: 'full' },
  // The full two-pane agent Roster — org chart + live dossier. Browse-and-pick
  // sibling of the composer's compact AgentSelectorDropdown.
  roster: { id: 'agent-roster', type: 'modal', component: 'agentRoster', size: 'full' },
  changePassword: { id: 'change-password', type: 'modal', component: 'changePassword', size: 'sm' },
};

/**
 * Tab-surfaces: the TopBar's desktop surface pills (Deck, Memory, Projects). They
 * behave like tabs over the page, so navigating to a page dismisses them; left
 * open, they would cover the page it just switched to. Every navigation path
 * goes through `dismissTabSurfaces`: the TopBar's page pills, the palette's
 * navigation commands and ⌘T.
 *
 * Dismissal is a request, not a teardown. An editor holding unsaved work inside a
 * tab-surface (the Projects drawer's document editor, Memory's entity and fact
 * editors) registers a dismissal guard (`hooks/useDismissGuard`), and the
 * surface stays open if the user keeps editing. The navigation then does not
 * happen either: the caller navigates only when this resolves `true`.
 *
 * The other surfaces stay open on navigation. They are modal (Settings,
 * Connectors & Tools, Organization, the Roster, the profile editor, the
 * drawers) and can hold unsaved work behind their own close path, such as
 * Organization's "Discard unsaved changes?".
 */
export const TAB_SURFACES: readonly SurfaceKey[] = ['ambassadorDeck', 'memory', 'workspaces'];

/**
 * Ask every open tab-surface to close. Takes `useModal()`'s `isOpen` and
 * `closeModal`. Resolves `true` when none is left open, `false` when a surface
 * kept itself open (unsaved work the user chose to keep); stops at that one.
 */
export async function dismissTabSurfaces(modals: {
  isOpen: (id: string) => boolean;
  closeModal: (id: string) => Promise<boolean> | void;
}): Promise<boolean> {
  for (const key of TAB_SURFACES) {
    const id = SURFACES[key].id;
    if (id && modals.isOpen(id) && (await modals.closeModal(id)) === false) return false;
  }
  return true;
}
