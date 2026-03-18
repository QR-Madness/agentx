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

interface ConversationContextValue {
  tabs: ConversationTab[];
  activeTabId: string | null;
  activeTab: ConversationTab | null;

  addTab: (profileId?: string | null) => ConversationTab;
  closeTab: (id: string) => void;
  switchTab: (id: string) => void;
  renameTab: (id: string, title: string) => void;
  updateTab: (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => void;

  // Server-side conversation history
  serverConversations: ConversationSummary[];
  isLoadingHistory: boolean;
  restoreConversation: (conversationId: string) => Promise<void>;
  refreshHistory: () => Promise<void>;

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

  // Map server messages to frontend ConversationMessage types
  const mapServerMessages = useCallback((messages: ServerMessage[]): ConversationMessage[] => {
    return messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => {
        const base = {
          id: createMessageId(),
          timestamp: m.timestamp || new Date().toISOString(),
        };
        if (m.role === 'user') {
          return { ...base, type: 'user' as const, content: m.content };
        }
        return {
          ...base,
          type: 'assistant' as const,
          content: m.content,
          model: m.metadata?.model as string | undefined,
        };
      });
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
      messages,
      isStreaming: false,
      createdAt: response.messages[0]?.timestamp || now,
      lastMessageAt: response.messages[response.messages.length - 1]?.timestamp || now,
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
  }, [tabs, mapServerMessages]);

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
        serverConversations,
        isLoadingHistory,
        restoreConversation,
        refreshHistory,
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
