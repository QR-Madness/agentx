/**
 * Server-side conversation history: the past-conversations list, plus restore
 * (into a new tab) and delete. The fetch is gated on tab-initialization for the
 * current server via the shared `initializedServerId` ref from
 * {@link useConversationTabs}, so it only fires once tabs are ready.
 */

import {
  useState,
  useCallback,
  useEffect,
  type Dispatch,
  type SetStateAction,
  type MutableRefObject,
} from 'react';
import {
  type ConversationTab,
  type ServerConfig,
  generateTabId,
} from '../../lib/storage';
import { api, type ConversationSummary } from '../../lib/api';
import { mapServerMessages } from './mapServerMessages';
import { getTitleOverride } from '../../lib/conversationTitles';
import { createMessageId } from '../../lib/messages';

interface UseConversationHistoryArgs {
  activeServer: ServerConfig | null;
  initializedServerId: MutableRefObject<string | null>;
  tabs: ConversationTab[];
  setTabs: Dispatch<SetStateAction<ConversationTab[]>>;
  setActiveTabId: Dispatch<SetStateAction<string | null>>;
  closeTab: (id: string) => void;
}

export function useConversationHistory({
  activeServer,
  initializedServerId,
  tabs,
  setTabs,
  setActiveTabId,
  closeTab,
}: UseConversationHistoryArgs) {
  const [serverConversations, setServerConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Fetch conversation history from server
  const refreshHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      // Push any pre-Projects local workspace attaches up to the server first
      // (no-op after the first clean pass) so the rows fetched below already
      // carry their workspace_id.
      await import('../../lib/projectSync')
        .then((m) => m.syncLocalProjectLinksOnce())
        .catch(() => undefined);
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

  // Restore a conversation from the server into a new tab. Returns the tab id
  // (existing or newly created). `opts.activeRun` bakes a detached run onto the
  // tab so ChatPanel re-attaches on mount (used by run recovery).
  const restoreConversation = useCallback(async (
    conversationId: string,
    opts?: { activeRun?: { runId: string }; seedUserMessage?: string },
  ): Promise<string> => {
    // Check if already open
    const existing = tabs.find(t => t.sessionId === conversationId);
    if (existing) {
      if (opts?.activeRun) {
        setTabs(prev => prev.map(t =>
          t.id === existing.id ? { ...t, activeRun: opts.activeRun } : t));
      }
      setActiveTabId(existing.id);
      return existing.id;
    }

    const response = await api.getConversationMessages(conversationId);
    const messages = mapServerMessages(response.messages);

    // A still-running turn's user message exists nowhere server-side (turns
    // persist only at turn END) — seed it from the run's stored label so the
    // reopened tab shows what was asked instead of a blank while the attach
    // replay streams the assistant side in.
    const seed = opts?.activeRun ? opts?.seedUserMessage : undefined;
    if (seed && !messages.some(m => m.type === 'user' && m.content.startsWith(seed))) {
      messages.push({
        id: createMessageId(),
        type: 'user',
        content: seed,
        timestamp: new Date().toISOString(),
      });
    }

    const derived = messages.find(m => m.type === 'user')?.content.slice(0, 40) || 'Restored Conversation';
    const fallbackTitle = derived.length >= 40 ? derived + '...' : derived;
    // A client-side rename of this past conversation takes precedence.
    const title = getTitleOverride(conversationId) ?? fallbackTitle;
    const now = new Date().toISOString();

    // Recover the last known context-window snapshot from persisted metadata so
    // older conversations show the usage bar without requiring a fresh stream.
    let restoredContextInfo: ConversationTab['contextInfo'];
    for (let i = response.messages.length - 1; i >= 0; i--) {
      const m = response.messages[i];
      if (m.role !== 'assistant') continue;
      const win = m.metadata?.context_window as number | undefined;
      const used = m.metadata?.context_used as number | undefined;
      if (win && used !== undefined) {
        restoredContextInfo = {
          window: win,
          used,
          updatedAt: m.timestamp ? new Date(m.timestamp).getTime() : Date.now(),
        };
        break;
      }
    }

    const newTab: ConversationTab = {
      id: generateTabId(),
      title,
      sessionId: conversationId,
      profileId: null,
      workflowId: null,
      messages,
      isStreaming: false,
      createdAt: response.messages[0]?.timestamp || now,
      lastMessageAt: response.messages[response.messages.length - 1]?.timestamp || now,
      contextInfo: restoredContextInfo,
      activeRun: opts?.activeRun,
    };

    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
    return newTab.id;
  }, [tabs, setTabs, setActiveTabId]);

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

  return {
    serverConversations,
    isLoadingHistory,
    refreshHistory,
    restoreConversation,
    deleteConversation,
    deleteServerConversation,
  };
}
