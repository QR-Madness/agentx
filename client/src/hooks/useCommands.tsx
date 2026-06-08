/**
 * useCommands — the command registry.
 *
 * Assembles the palette's actions from app contexts, decoupled from how the
 * palette renders them. Both the registry and the TopBar's live icons open
 * surfaces through `SURFACES` (see `lib/surfaces.ts`) so the two can't drift.
 *
 * Scope discipline: groups are fixed to Navigation / Conversation / Workspace /
 * Theme / Account. We deliberately do NOT generate per-conversation, per-agent,
 * or per-server entries — those have their own surfaces and would turn the
 * palette into a state browser.
 */

import {
  Home, LayoutDashboard, Bot, Plus, X, Settings, Wrench, Database, ListChecks,
  BookMarked, Languages, BrainCircuit, Eye, EyeOff, Zap, KeyRound, LogOut, Radio,
  ScrollText, Moon, Sun, Contrast, Monitor, MessagesSquare,
} from 'lucide-react';
import { useMemo } from 'react';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { SURFACES, type SurfaceKey } from '../lib/surfaces';
import type { ThemePreference } from '../lib/theme';
import type { PageId } from '../layouts/TopBar';

export type CommandGroup = 'Navigation' | 'Conversation' | 'Workspace' | 'Theme' | 'Account';

export const GROUP_ORDER: CommandGroup[] = [
  'Navigation', 'Conversation', 'Workspace', 'Theme', 'Account',
];

export interface Command {
  id: string;
  group: CommandGroup;
  label: string;
  keywords?: string[];
  icon: React.ReactNode;
  hint?: string;
  run: () => void;
  isActive?: boolean;
}

interface UseCommandsArgs {
  onNavigate: (page: PageId) => void;
  onClose: () => void;
}

export function useCommands({ onNavigate, onClose }: UseCommandsArgs): Command[] {
  const { openModal } = useModal();
  const { addTab, closeTab, activeTabId, activeTab, updateTab } = useConversation();
  const { focusMode, toggleFocusMode } = useUIChrome();
  const { authRequired, isAuthenticated, logout } = useAuth();
  const { preference, setTheme } = useTheme();

  return useMemo<Command[]>(() => {
    const go = (page: PageId) => () => { onNavigate(page); onClose(); };
    const open = (key: SurfaceKey) => () => { openModal(SURFACES[key]); onClose(); };
    const theme = (pref: ThemePreference) => () => { setTheme(pref); onClose(); };
    const memLocked = !!(activeTab && (activeTab.sessionId || activeTab.messages.length > 0));

    const cmds: Command[] = [
      // Navigation
      { id: 'nav-chat', group: 'Navigation', label: 'Go to Chat', icon: <Bot size={16} />, keywords: ['agentx', 'conversation'], run: go('agentx') },
      { id: 'nav-start', group: 'Navigation', label: 'Go to Start', icon: <Home size={16} />, keywords: ['home'], run: go('start') },
      { id: 'nav-dashboard', group: 'Navigation', label: 'Go to Dashboard', icon: <LayoutDashboard size={16} />, keywords: ['metrics', 'status'], run: go('dashboard') },

      // Conversation
      { id: 'conv-new', group: 'Conversation', label: 'New conversation', hint: '⌘T', icon: <Plus size={16} />, keywords: ['tab', 'create', 'chat'], run: () => { addTab(); onNavigate('agentx'); onClose(); } },
      { id: 'open-conversations', group: 'Conversation', label: 'Open conversations', icon: <MessagesSquare size={16} />, keywords: ['tabs', 'switch', 'list', 'sessions', 'sidebar'], run: open('conversations') },

      // Workspace
      { id: 'open-settings', group: 'Workspace', label: 'Open Settings', hint: '⌘,', icon: <Settings size={16} />, keywords: ['config', 'preferences', 'providers'], run: open('settings') },
      { id: 'open-tools', group: 'Workspace', label: 'Open Tools', icon: <Wrench size={16} />, keywords: ['toolkit', 'mcp', 'servers'], run: open('tools') },
      { id: 'open-memory', group: 'Workspace', label: 'Open Memory', icon: <Database size={16} />, keywords: ['facts', 'entities', 'recall', 'memories'], run: open('memory') },
      { id: 'open-sources', group: 'Workspace', label: 'Open Sources', icon: <BookMarked size={16} />, keywords: ['citations', 'bibliography', 'references', 'links'], run: open('sources') },
      { id: 'open-ambassador', group: 'Workspace', label: 'Open Ambassador', icon: <Radio size={16} />, keywords: ['briefing', 'summarize', 'interpret', 'turn', 'parallel', 'ambassadors'], run: open('ambassador') },
      { id: 'open-plans', group: 'Workspace', label: 'Open Plans', icon: <ListChecks size={16} />, keywords: ['tasks', 'subtasks', 'progress'], run: open('plans') },
      { id: 'open-translation', group: 'Workspace', label: 'Open Translation', icon: <Languages size={16} />, keywords: ['translate', 'language', 'nllb', 'translations'], run: open('translation') },
      { id: 'open-logs', group: 'Workspace', label: 'Open Logs', icon: <ScrollText size={16} />, keywords: ['console', 'debug', 'server', 'trace'], run: open('logs') },
      { id: 'open-profile', group: 'Workspace', label: 'Agent profiles', icon: <BrainCircuit size={16} />, keywords: ['profile', 'profiles', 'agent', 'agents', 'persona', 'model', 'temperature', 'prompt', 'edit'], run: open('profileEditor') },
      { id: 'focus', group: 'Workspace', label: focusMode ? 'Exit focus mode' : 'Enter focus mode', hint: 'Zen', icon: focusMode ? <EyeOff size={16} /> : <Eye size={16} />, keywords: ['zen', 'immersive', 'hide', 'chrome'], run: () => { toggleFocusMode(); onClose(); } },

      // Theme
      { id: 'theme-cosmic', group: 'Theme', label: 'Theme: Cosmic', icon: <Moon size={16} />, keywords: ['dark', 'appearance'], isActive: preference === 'cosmic', run: theme('cosmic') },
      { id: 'theme-light', group: 'Theme', label: 'Theme: Light', icon: <Sun size={16} />, keywords: ['appearance', 'bright'], isActive: preference === 'light', run: theme('light') },
      { id: 'theme-professional', group: 'Theme', label: 'Theme: Professional', icon: <Contrast size={16} />, keywords: ['monochrome', 'graphite', 'appearance'], isActive: preference === 'professional', run: theme('professional') },
      { id: 'theme-system', group: 'Theme', label: 'Theme: System', icon: <Monitor size={16} />, keywords: ['auto', 'appearance'], isActive: preference === 'system', run: theme('system') },
    ];

    // Conversation actions that depend on an active tab
    if (activeTabId && !memLocked) {
      cmds.push({
        id: 'toggle-mem', group: 'Conversation',
        label: activeTab?.noMemorization ? 'Enable memorization for this chat' : 'Disable memorization for this chat',
        icon: <Zap size={16} />, keywords: ['memory', 'private', 'temporary', 'no-memorization'],
        run: () => { updateTab(activeTabId, { noMemorization: !activeTab?.noMemorization }); onClose(); },
      });
    }
    if (activeTabId) {
      cmds.push({ id: 'conv-close', group: 'Conversation', label: 'Close conversation', hint: '⌘W', icon: <X size={16} />, keywords: ['tab', 'remove'], run: () => { closeTab(activeTabId); onClose(); } });
    }

    // Account (only when auth is enabled + signed in)
    if (authRequired && isAuthenticated) {
      cmds.push({ id: 'change-password', group: 'Account', label: 'Change Password', icon: <KeyRound size={16} />, keywords: ['account', 'security', 'login'], run: open('changePassword') });
      cmds.push({ id: 'sign-out', group: 'Account', label: 'Sign Out', icon: <LogOut size={16} />, keywords: ['logout', 'exit', 'session', 'account'], run: () => { onClose(); logout().then(() => onNavigate('start')); } });
    }

    return cmds;
  }, [openModal, onNavigate, onClose, addTab, closeTab, activeTabId, activeTab, updateTab, focusMode, toggleFocusMode, preference, setTheme, authRequired, isAuthenticated, logout]);
}
