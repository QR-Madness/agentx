/**
 * Active-tab message operations: append/update messages and toggle the
 * streaming / session-id fields. Composes on the tab state owned by
 * {@link useConversationTabs}.
 */

import { useCallback, type Dispatch, type SetStateAction } from 'react';
import type { ConversationTab } from '../../lib/storage';
import type { ConversationMessage } from '../../lib/messages';

interface UseTabMessagesArgs {
  activeTabId: string | null;
  setTabs: Dispatch<SetStateAction<ConversationTab[]>>;
  updateTab: (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => void;
}

export function useTabMessages({ activeTabId, setTabs, updateTab }: UseTabMessagesArgs) {
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
  }, [activeTabId, setTabs]);

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
  }, [activeTabId, setTabs]);

  const setStreaming = useCallback((isStreaming: boolean) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { isStreaming });
  }, [activeTabId, updateTab]);

  const setSessionId = useCallback((sessionId: string) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { sessionId });
  }, [activeTabId, updateTab]);

  return { appendMessage, updateMessage, setStreaming, setSessionId };
}
