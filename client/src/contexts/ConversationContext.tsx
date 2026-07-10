/**
 * Conversation Context — Manages browser-style conversation tabs
 * with server-side conversation history.
 *
 * Thin composition over four concern hooks in ./conversation:
 *   useConversationTabs    — tab state + persistence + CRUD (source of truth)
 *   useTabMessages         — active-tab message ops
 *   useTabSettings         — per-tab profile / workflow / context-info
 *   useConversationHistory — server history: list, restore, delete
 *
 * The public surface (ConversationProvider, useConversation, the ConversationTab
 * re-export, and the context value shape) is unchanged.
 */

import { createContext, useCallback, useContext, useRef, type ReactNode } from 'react';
import type { ConversationMessage } from '../lib/messages';
import type { ConversationTab } from '../lib/storage';
import type { ActiveChatRun, ConversationSummary } from '../lib/api';
import { useConversationTabs } from './conversation/useConversationTabs';
import { useTabMessages } from './conversation/useTabMessages';
import { useTabSettings } from './conversation/useTabSettings';
import { useConversationHistory } from './conversation/useConversationHistory';
import { useResumeRun } from './conversation/useResumeRun';

// Re-export for convenience
export type { ConversationTab } from '../lib/storage';

interface ConversationContextValue {
  tabs: ConversationTab[];
  activeTabId: string | null;
  activeTab: ConversationTab | null;

  addTab: (profileId?: string | null) => ConversationTab;
  closeTab: (id: string) => void;
  switchTab: (id: string) => void;
  renameTab: (id: string, title: string) => void;
  updateTab: (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => void;

  // Per-tab agent profile selection
  setTabProfile: (tabId: string, profileId: string | null) => void;
  setActiveTabProfile: (profileId: string | null) => void;

  // Per-tab Agent Alloy workflow selection
  setTabWorkflow: (tabId: string, workflowId: string | null) => void;
  setActiveTabWorkflow: (workflowId: string | null) => void;

  // Per-tab model override (null = inherit the profile's model)
  setActiveTabModel: (model: string | null) => void;

  // Per-tab context-window usage indicator
  setTabContextInfo: (tabId: string, info: { window: number; used: number; summarized?: boolean; droppedTurns?: number } | null) => void;

  // Server-side conversation history
  serverConversations: ConversationSummary[];
  isLoadingHistory: boolean;
  restoreConversation: (
    conversationId: string,
    opts?: { activeRun?: { runId: string }; seedUserMessage?: string },
  ) => Promise<string>;
  refreshHistory: () => Promise<void>;

  // Reopen a detached run (recovery surfaces) — restores/seeds a tab + attaches.
  resumeRun: (run: ActiveChatRun) => Promise<void>;

  // Delete conversations (from server)
  deleteConversation: (tabId: string) => Promise<void>;
  deleteServerConversation: (conversationId: string) => Promise<void>;

  // Convenience methods for active tab
  appendMessage: (message: ConversationMessage) => void;
  updateMessage: (messageId: string, updates: Partial<ConversationMessage>) => void;
  setStreaming: (isStreaming: boolean) => void;
  setSessionId: (sessionId: string) => void;

  // Outbound relay seam: ChatPanel owns a tab's send/steer path and registers it
  // here so other surfaces (e.g. the Ambassador panel) can relay a user message
  // into the conversation without duplicating the chat stream. The relayed text
  // becomes a real user turn (or steers the running turn) — the Ambassador stays
  // a non-participant; the user is the author.
  registerRelay: (tabId: string, fn: (text: string) => void) => () => void;
  relayToConversation: (tabId: string, text: string) => boolean;
}

const ConversationContext = createContext<ConversationContextValue | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  // Order matters: the tabs hook's effects (server-load, persistence) must
  // register before the history hook's fetch effect, since the fetch is gated
  // on the shared `initializedServerId` ref being set by the tab-load effect.
  const tabsApi = useConversationTabs();

  const messages = useTabMessages({
    activeTabId: tabsApi.activeTabId,
    setTabs: tabsApi.setTabs,
    updateTab: tabsApi.updateTab,
  });

  const settings = useTabSettings({
    activeTabId: tabsApi.activeTabId,
    updateTab: tabsApi.updateTab,
  });

  const history = useConversationHistory({
    activeServer: tabsApi.activeServer,
    initializedServerId: tabsApi.initializedServerId,
    tabs: tabsApi.tabs,
    setTabs: tabsApi.setTabs,
    setActiveTabId: tabsApi.setActiveTabId,
    closeTab: tabsApi.closeTab,
  });

  const resumeRun = useResumeRun({
    restoreConversation: history.restoreConversation,
    addTab: tabsApi.addTab,
    updateTab: tabsApi.updateTab,
  });

  // Outbound relay registry: tabId -> the live send/steer handler ChatPanel owns.
  const relayHandlers = useRef<Map<string, (text: string) => void>>(new Map());
  const registerRelay = useCallback((tabId: string, fn: (text: string) => void) => {
    relayHandlers.current.set(tabId, fn);
    return () => {
      if (relayHandlers.current.get(tabId) === fn) relayHandlers.current.delete(tabId);
    };
  }, []);
  const relayToConversation = useCallback((tabId: string, text: string) => {
    const fn = relayHandlers.current.get(tabId);
    if (!fn) return false;
    fn(text);
    return true;
  }, []);

  return (
    <ConversationContext.Provider
      value={{
        tabs: tabsApi.tabs,
        activeTabId: tabsApi.activeTabId,
        activeTab: tabsApi.activeTab,
        addTab: tabsApi.addTab,
        closeTab: tabsApi.closeTab,
        switchTab: tabsApi.switchTab,
        renameTab: tabsApi.renameTab,
        updateTab: tabsApi.updateTab,
        setTabProfile: settings.setTabProfile,
        setActiveTabProfile: settings.setActiveTabProfile,
        setTabWorkflow: settings.setTabWorkflow,
        setActiveTabWorkflow: settings.setActiveTabWorkflow,
        setActiveTabModel: settings.setActiveTabModel,
        setTabContextInfo: settings.setTabContextInfo,
        serverConversations: history.serverConversations,
        isLoadingHistory: history.isLoadingHistory,
        restoreConversation: history.restoreConversation,
        refreshHistory: history.refreshHistory,
        resumeRun,
        deleteConversation: history.deleteConversation,
        deleteServerConversation: history.deleteServerConversation,
        appendMessage: messages.appendMessage,
        updateMessage: messages.updateMessage,
        setStreaming: messages.setStreaming,
        setSessionId: messages.setSessionId,
        registerRelay,
        relayToConversation,
      }}
    >
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversation() {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
}
