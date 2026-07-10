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
  ScrollText, Monitor, MessagesSquare, FileStack, FolderOpen, Users,
  Orbit, Telescope,
} from 'lucide-react';
import { useMemo } from 'react';
import { useModal } from '../contexts/ModalContext';
import { useConversation } from '../contexts/ConversationContext';
import { useUIChrome } from '../contexts/UIChromeContext';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useOpenAmbassador } from './useOpenAmbassador';
import { SURFACES, type SurfaceKey } from '../lib/surfaces';
import { RESEARCH_MODE, thinkingModeOf, thinkingModeTabPatch } from '../lib/thinkingModes';
import { THEMES, type ThemePreference } from '../lib/theme';
import { THEME_ICONS } from '../components/common/themeIcons';
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
  const openAmbassador = useOpenAmbassador();

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
      { id: 'open-tools', group: 'Workspace', label: 'Open Connectors & Tools', icon: <Wrench size={16} />, keywords: ['toolkit', 'mcp', 'servers', 'connectors', 'integrations', 'skills', 'registry', 'catalog'], run: open('tools') },
      { id: 'open-memory', group: 'Workspace', label: 'Open Memory', icon: <Database size={16} />, keywords: ['facts', 'entities', 'recall', 'memories'], run: open('memory') },
      { id: 'open-sources', group: 'Workspace', label: 'Open Sources', icon: <BookMarked size={16} />, keywords: ['citations', 'bibliography', 'references', 'links'], run: open('sources') },
      { id: 'open-consolidation', group: 'Workspace', label: 'Memory Consolidation', icon: <Zap size={16} />, keywords: ['consolidate', 'memory', 'extract', 'facts', 'entities', 'lightning'], run: open('consolidation') },
      { id: 'open-ambassador', group: 'Workspace', label: 'Open Ambassador', icon: <Radio size={16} />, keywords: ['briefing', 'summarize', 'interpret', 'turn', 'parallel', 'ambassadors'], run: () => { onNavigate('agentx'); openAmbassador(); onClose(); } },
      { id: 'open-ambassador-deck', group: 'Workspace', label: 'Open Command Deck', icon: <LayoutDashboard size={16} />, keywords: ['ambassador', 'deck', 'agents', 'overview', 'survey', 'roster', 'discovered'], run: open('ambassadorDeck') },
      { id: 'open-plans', group: 'Workspace', label: 'Open Plans', icon: <ListChecks size={16} />, keywords: ['tasks', 'subtasks', 'progress'], run: open('plans') },
      { id: 'open-translation', group: 'Workspace', label: 'Open Translation', icon: <Languages size={16} />, keywords: ['translate', 'language', 'nllb', 'translations'], run: open('translation') },
      { id: 'open-logs', group: 'Workspace', label: 'Open Logs', icon: <ScrollText size={16} />, keywords: ['console', 'debug', 'server', 'trace'], run: open('logs') },
      { id: 'open-tool-outputs', group: 'Workspace', label: 'Open Tool Outputs', icon: <FileStack size={16} />, keywords: ['debug', 'stored', 'cache', 'outputs', 'tool', 'results'], run: open('toolOutputBrowser') },
      { id: 'open-workspaces', group: 'Workspace', label: 'Open Projects', icon: <FolderOpen size={16} />, keywords: ['workspaces', 'projects', 'files', 'documents', 'upload', 'rag', 'pdf', 'corpus', 'knowledge', 'instructions'], run: open('workspaces') },
      { id: 'open-teams', group: 'Workspace', label: 'Manage agent teams', icon: <Users size={16} />, keywords: ['alloy', 'workflow', 'team', 'teams', 'delegation', 'lead', 'members', 'factory'], run: open('teams') },
      { id: 'open-profile', group: 'Workspace', label: 'Agent profiles', icon: <BrainCircuit size={16} />, keywords: ['profile', 'profiles', 'agent', 'agents', 'persona', 'model', 'temperature', 'prompt', 'edit'], run: open('profileEditor') },
      { id: 'focus', group: 'Workspace', label: focusMode ? 'Exit focus mode' : 'Enter focus mode', hint: 'Zen', icon: focusMode ? <EyeOff size={16} /> : <Eye size={16} />, keywords: ['zen', 'immersive', 'hide', 'chrome'], run: () => { toggleFocusMode(); onClose(); } },

      // Theme — registry-driven: adding a theme to THEMES surfaces it here automatically.
      ...Object.values(THEMES).map((t): Command => {
        const Icon = THEME_ICONS[t.icon];
        return {
          id: `theme-${t.name}`,
          group: 'Theme',
          label: `Theme: ${t.displayName}`,
          icon: <Icon size={16} />,
          keywords: ['appearance', ...t.description.toLowerCase().split(/[^a-z]+/).filter(Boolean)],
          isActive: preference === t.name,
          run: theme(t.name as ThemePreference),
        };
      }),
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
      // Solo/Team: per-conversation ad-hoc delegation toggle. Unlike memory it
      // never locks — delegation is per-turn. Ignored while a workflow is active.
      cmds.push({
        id: 'toggle-delegation', group: 'Conversation',
        label: activeTab?.noDelegation ? 'Enable delegation for this chat' : 'Disable delegation for this chat',
        icon: <Users size={16} />, keywords: ['solo', 'team', 'delegate', 'delegation', 'roster'],
        run: () => { updateTab(activeTabId, { noDelegation: !activeTab?.noDelegation }); onClose(); },
      });
      cmds.push({ id: 'conv-close', group: 'Conversation', label: 'Close conversation', hint: '⌘W', icon: <X size={16} />, keywords: ['tab', 'remove'], run: () => { closeTab(activeTabId); onClose(); } });

      // Relay bridge: these reach ChatPanel state via window events (the
      // registry can't hold refs into the composer). Navigate to the chat
      // page first so the panel is mounted to hear the event.
      const fire = (evt: string) => () => {
        onNavigate('agentx');
        onClose();
        setTimeout(() => window.dispatchEvent(new Event(evt)), 80);
      };
      cmds.push({
        id: 'open-relay', group: 'Conversation',
        label: 'Open the Relay',
        icon: <Orbit size={16} />,
        keywords: ['relay', 'command', 'center', 'orbit', 'background', 'attach', 'mode', 'tiles'],
        run: fire('agentx:relay-open'),
      });
      const researchOn = thinkingModeOf(activeTab) === RESEARCH_MODE;
      cmds.push({
        id: 'toggle-research', group: 'Conversation',
        label: researchOn ? 'Turn off Research mode' : 'Turn on Research mode',
        icon: <Telescope size={16} />,
        keywords: ['research', 'deep', 'cited', 'mode', 'thinking'],
        run: () => {
          updateTab(activeTabId, thinkingModeTabPatch(researchOn ? '' : RESEARCH_MODE));
          onClose();
        },
      });
    }

    // Account (only when auth is enabled + signed in)
    if (authRequired && isAuthenticated) {
      cmds.push({ id: 'change-password', group: 'Account', label: 'Change Password', icon: <KeyRound size={16} />, keywords: ['account', 'security', 'login'], run: open('changePassword') });
      cmds.push({ id: 'sign-out', group: 'Account', label: 'Sign Out', icon: <LogOut size={16} />, keywords: ['logout', 'exit', 'session', 'account'], run: () => { onClose(); logout().then(() => onNavigate('start')); } });
    }

    return cmds;
  }, [openModal, openAmbassador, onNavigate, onClose, addTab, closeTab, activeTabId, activeTab, updateTab, focusMode, toggleFocusMode, preference, setTheme, authRequired, isAuthenticated, logout]);
}
