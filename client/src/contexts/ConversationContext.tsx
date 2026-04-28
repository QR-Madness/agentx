/**
 * Conversation Context — Manages browser-style conversation tabs
 * with server-side conversation history
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from 'react';
import { useServer } from './ServerContext';
import type { ConversationMessage } from '../lib/messages';
import { createMessageId } from '../lib/messages';
import {
  type ConversationTab,
  getConversationTabs,
  saveConversationTabs,
  getActiveTabId,
  saveActiveTabId,
  createDefaultTab,
  generateTabId,
} from '../lib/storage';
import { api, type ConversationSummary, type ServerMessage } from '../lib/api';

// Re-export for convenience
export type { ConversationTab } from '../lib/storage';

function safeParseJson(str: string): Record<string, unknown> {
  try { return JSON.parse(str); } catch { return {}; }
}

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

  // Server-side conversation history
  serverConversations: ConversationSummary[];
  isLoadingHistory: boolean;
  restoreConversation: (conversationId: string) => Promise<void>;
  refreshHistory: () => Promise<void>;

  // Delete conversations (from server)
  deleteConversation: (tabId: string) => Promise<void>;
  deleteServerConversation: (conversationId: string) => Promise<void>;

  // Convenience methods for active tab
  appendMessage: (message: ConversationMessage) => void;
  updateMessage: (messageId: string, updates: Partial<ConversationMessage>) => void;
  setStreaming: (isStreaming: boolean) => void;
  setSessionId: (sessionId: string) => void;
}

const ConversationContext = createContext<ConversationContextValue | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const { activeServer } = useServer();
  const [tabs, setTabs] = useState<ConversationTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [serverConversations, setServerConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Track if we've initialized for this server
  const initializedServerId = useRef<string | null>(null);

  // Debounce save to avoid excessive writes
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load tabs when server changes
  useEffect(() => {
    if (!activeServer) {
      setTabs([]);
      setActiveTabId(null);
      initializedServerId.current = null;
      return;
    }

    // Skip if already initialized for this server
    if (initializedServerId.current === activeServer.id) {
      return;
    }

    const loadedTabs = getConversationTabs(activeServer.id);
    const loadedActiveId = getActiveTabId(activeServer.id);

    if (loadedTabs.length === 0) {
      // Create a default tab
      const defaultTab = createDefaultTab();
      setTabs([defaultTab]);
      setActiveTabId(defaultTab.id);
      saveConversationTabs([defaultTab], activeServer.id);
      saveActiveTabId(defaultTab.id, activeServer.id);
    } else {
      setTabs(loadedTabs);
      // Ensure active tab exists
      const validActiveId = loadedTabs.some(t => t.id === loadedActiveId)
        ? loadedActiveId
        : loadedTabs[0].id;
      setActiveTabId(validActiveId);
      if (validActiveId !== loadedActiveId) {
        saveActiveTabId(validActiveId, activeServer.id);
      }
    }

    initializedServerId.current = activeServer.id;
  }, [activeServer]);

  // Save tabs when they change (debounced)
  useEffect(() => {
    if (!activeServer || initializedServerId.current !== activeServer.id) {
      return;
    }

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveConversationTabs(tabs, activeServer.id);
    }, 300);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [tabs, activeServer]);

  // Save active tab ID immediately (no debounce)
  useEffect(() => {
    if (!activeServer || initializedServerId.current !== activeServer.id) {
      return;
    }
    saveActiveTabId(activeTabId, activeServer.id);
  }, [activeTabId, activeServer]);

  const activeTab = tabs.find(t => t.id === activeTabId) ?? null;

  const addTab = useCallback((profileId?: string | null): ConversationTab => {
    const now = new Date().toISOString();
    const newTab: ConversationTab = {
      id: generateTabId(),
      title: 'New Conversation',
      sessionId: null,
      profileId: profileId ?? null,
      workflowId: null,
      messages: [],
      isStreaming: false,
      createdAt: now,
      lastMessageAt: now,
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);

    return newTab;
  }, []);

  const closeTab = useCallback((id: string) => {
    setTabs(prev => {
      const index = prev.findIndex(t => t.id === id);
      if (index === -1) return prev;

      const newTabs = prev.filter(t => t.id !== id);

      // If closing active tab, switch to adjacent
      if (id === activeTabId && newTabs.length > 0) {
        const newIndex = Math.min(index, newTabs.length - 1);
        setActiveTabId(newTabs[newIndex].id);
      } else if (newTabs.length === 0) {
        // Create a new default tab if closing last one
        const defaultTab = createDefaultTab();
        setActiveTabId(defaultTab.id);
        return [defaultTab];
      }

      return newTabs;
    });
  }, [activeTabId]);

  const switchTab = useCallback((id: string) => {
    setActiveTabId(id);
  }, []);

  const renameTab = useCallback((id: string, title: string) => {
    setTabs(prev =>
      prev.map(t => (t.id === id ? { ...t, title } : t))
    );
  }, []);

  const updateTab = useCallback(
    (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => {
      setTabs(prev =>
        prev.map(t => (t.id === id ? { ...t, ...updates } : t))
      );
    },
    []
  );

  // Convenience methods for active tab
  const appendMessage = useCallback((message: ConversationMessage) => {
    if (!activeTabId) return;

    setTabs(prev =>
      prev.map(t => {
        if (t.id !== activeTabId) return t;

        // Auto-generate title from first user message
        const newTitle =
          t.messages.length === 0 && message.type === 'user'
            ? message.content.slice(0, 40) + (message.content.length > 40 ? '...' : '')
            : t.title;

        return {
          ...t,
          title: newTitle,
          messages: [...t.messages, message],
          lastMessageAt: new Date().toISOString(),
        };
      })
    );
  }, [activeTabId]);

  const updateMessage = useCallback((messageId: string, updates: Partial<ConversationMessage>) => {
    if (!activeTabId) return;

    setTabs(prev =>
      prev.map(t => {
        if (t.id !== activeTabId) return t;

        return {
          ...t,
          messages: t.messages.map(m =>
            m.id === messageId ? ({ ...m, ...updates } as ConversationMessage) : m
          ),
        };
      })
    );
  }, [activeTabId]);

  const setStreaming = useCallback((isStreaming: boolean) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { isStreaming });
  }, [activeTabId, updateTab]);

  const setSessionId = useCallback((sessionId: string) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { sessionId });
  }, [activeTabId, updateTab]);

  // Set the agent profile for a specific tab
  const setTabProfile = useCallback((tabId: string, profileId: string | null) => {
    updateTab(tabId, { profileId });
  }, [updateTab]);

  // Set the agent profile for the active tab
  const setActiveTabProfile = useCallback((profileId: string | null) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { profileId });
  }, [activeTabId, updateTab]);

  // Set the Alloy workflow for a specific tab
  const setTabWorkflow = useCallback((tabId: string, workflowId: string | null) => {
    updateTab(tabId, { workflowId });
  }, [updateTab]);

  // Set the Alloy workflow for the active tab
  const setActiveTabWorkflow = useCallback((workflowId: string | null) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { workflowId });
  }, [activeTabId, updateTab]);

  // Fetch conversation history from server
  const refreshHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const response = await api.listConversations({ limit: 50 });
      setServerConversations(response.conversations);
    } catch {
      // Server might not have memory enabled — silently ignore
      setServerConversations([]);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  // Fetch history when server changes
  useEffect(() => {
    if (!activeServer || initializedServerId.current !== activeServer.id) return;
    refreshHistory();
  }, [activeServer, refreshHistory]);

  // Map server messages to frontend ConversationMessage types.
  // Pairs adjacent tool_call + tool_result rows (matched by tool_call_id)
  // into a single ToolCallMessage, and reconstructs a DelegationMessage for
  // delegate_to pairs so workflow conversations restore as the user saw
  // them live (single delegation card with full specialist output).
  const mapServerMessages = useCallback((messages: ServerMessage[]): ConversationMessage[] => {
    const filtered = messages.filter(m =>
      ['user', 'assistant', 'tool_call', 'tool_result'].includes(m.role)
    );

    // Index tool_result rows by tool_call_id so we can pair them with their
    // tool_call and skip them when iterating.
    const resultsByCallId = new Map<string, ServerMessage>();
    for (const m of filtered) {
      if (m.role === 'tool_result') {
        const id = m.metadata?.tool_call_id as string | undefined;
        if (id) resultsByCallId.set(id, m);
      }
    }
    const consumedResults = new Set<string>();

    const out: ConversationMessage[] = [];
    for (const m of filtered) {
      const base = {
        id: createMessageId(),
        timestamp: m.timestamp || new Date().toISOString(),
      };

      if (m.role === 'user') {
        out.push({ ...base, type: 'user', content: m.content });
        continue;
      }

      if (m.role === 'assistant') {
        // Skip phantom empty assistant turns from older data.
        if (!m.content || !m.content.trim()) continue;
        out.push({
          ...base,
          type: 'assistant',
          content: m.content,
          model: m.metadata?.model as string | undefined,
          thinking: m.metadata?.thinking as string | undefined,
        });
        continue;
      }

      if (m.role === 'tool_call') {
        const toolName = (m.metadata?.tool as string) || 'unknown';
        const toolCallId = (m.metadata?.tool_call_id as string) || base.id;
        const args = safeParseJson(m.content);
        const result = resultsByCallId.get(toolCallId);
        if (result) consumedResults.add(toolCallId);

        if (toolName === 'delegate_to') {
          const delegationMeta = (result?.metadata?.delegation ?? {}) as {
            raw_content?: string;
            target_agent_id?: string;
            task?: string;
          };
          const targetAgentId =
            delegationMeta.target_agent_id ||
            (args.agent_id as string | undefined) ||
            'unknown';
          const task = delegationMeta.task || (args.task as string | undefined) || '';
          const success = (result?.metadata?.success as boolean) ?? true;
          out.push({
            ...base,
            type: 'delegation',
            delegationId: toolCallId,
            targetAgentId,
            task,
            depth: 1,
            status: success ? 'completed' : 'failed',
            content: delegationMeta.raw_content || result?.content || '',
            resultPreview: result?.content,
          });
          continue;
        }

        out.push({
          ...base,
          type: 'tool_call',
          toolName,
          toolCallId,
          arguments: args,
          status: result
            ? ((result.metadata?.success as boolean) ?? true ? 'completed' : 'failed')
            : 'completed',
          result: result
            ? {
                content: result.content,
                success: (result.metadata?.success as boolean) ?? true,
                durationMs: result.metadata?.duration_ms as number | undefined,
              }
            : undefined,
        });
        continue;
      }

      if (m.role === 'tool_result') {
        const toolCallId = (m.metadata?.tool_call_id as string) || base.id;
        if (consumedResults.has(toolCallId)) continue;  // already merged into its tool_call
        // Orphan tool_result (no matching tool_call row) — keep as a standalone card.
        out.push({
          ...base,
          type: 'tool_result',
          toolName: (m.metadata?.tool as string) || 'unknown',
          toolCallId,
          content: m.content,
          success: (m.metadata?.success as boolean) ?? true,
          durationMs: m.metadata?.duration_ms as number | undefined,
        });
        continue;
      }
    }

    return out;
  }, []);

  // Restore a conversation from the server into a new tab
  const restoreConversation = useCallback(async (conversationId: string) => {
    // Check if already open
    const existing = tabs.find(t => t.sessionId === conversationId);
    if (existing) {
      setActiveTabId(existing.id);
      return;
    }

    const response = await api.getConversationMessages(conversationId);
    const messages = mapServerMessages(response.messages);

    const title = messages.find(m => m.type === 'user')?.content.slice(0, 40) || 'Restored Conversation';
    const now = new Date().toISOString();

    const newTab: ConversationTab = {
      id: generateTabId(),
      title: title.length >= 40 ? title + '...' : title,
      sessionId: conversationId,
      profileId: null,
      workflowId: null,
      messages,
      isStreaming: false,
      createdAt: response.messages[0]?.timestamp || now,
      lastMessageAt: response.messages[response.messages.length - 1]?.timestamp || now,
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
  }, [tabs, mapServerMessages]);

  // Delete an open tab and its server conversation
  const deleteConversation = useCallback(async (tabId: string) => {
    const tab = tabs.find(t => t.id === tabId);
    if (!tab) return;

    // Delete from server if it has a sessionId
    if (tab.sessionId) {
      try {
        await api.deleteConversation(tab.sessionId);
      } catch (err) {
        console.error('Failed to delete conversation from server:', err);
        // Still close locally even if server delete fails
      }
    }

    // Close the local tab
    closeTab(tabId);

    // Refresh server list to remove it from past conversations
    refreshHistory();
  }, [tabs, closeTab, refreshHistory]);

  // Delete a past conversation from server (not currently open as a tab)
  const deleteServerConversation = useCallback(async (conversationId: string) => {
    try {
      await api.deleteConversation(conversationId);
      refreshHistory();
    } catch (err) {
      console.error('Failed to delete conversation from server:', err);
      throw err;
    }
  }, [refreshHistory]);

  return (
    <ConversationContext.Provider
      value={{
        tabs,
        activeTabId,
        activeTab,
        addTab,
        closeTab,
        switchTab,
        renameTab,
        updateTab,
        setTabProfile,
        setActiveTabProfile,
        setTabWorkflow,
        setActiveTabWorkflow,
        serverConversations,
        isLoadingHistory,
        restoreConversation,
        refreshHistory,
        deleteConversation,
        deleteServerConversation,
        appendMessage,
        updateMessage,
        setStreaming,
        setSessionId,
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
