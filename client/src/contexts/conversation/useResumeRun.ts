/**
 * Resume a detached chat run in a tab so ChatPanel re-attaches to it. Used by
 * the recovery surfaces (Relay inbox, conversation selector) to reopen a run
 * whose owning tab was closed.
 *
 * - With a known `session_id`: restore the conversation from server history and
 *   bake the run onto that tab.
 * - Without one (a brand-new chat that hasn't emitted its session yet): open a
 *   fresh tab, seed the user turn from the run's stored message, and attach so
 *   the replayed assistant events land on it without duplication.
 */

import { useCallback } from 'react';
import type { ActiveChatRun } from '../../lib/api';
import type { ConversationTab } from '../../lib/storage';
import { createMessageId, type UserMessage } from '../../lib/messages';

interface UseResumeRunArgs {
  restoreConversation: (
    conversationId: string,
    opts?: { activeRun?: { runId: string } },
  ) => Promise<string>;
  addTab: (profileId?: string | null) => ConversationTab;
  updateTab: (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => void;
}

export function useResumeRun({ restoreConversation, addTab, updateTab }: UseResumeRunArgs) {
  return useCallback(async (run: ActiveChatRun) => {
    const activeRun = { runId: run.run_id };

    if (run.session_id) {
      await restoreConversation(run.session_id, { activeRun });
      return;
    }

    // No session yet — seed a fresh tab from the run's message label so the
    // re-attach replay rebuilds the assistant side onto a real user turn.
    const tab = addTab();
    const seed: UserMessage[] = run.message
      ? [{
          id: createMessageId(),
          type: 'user',
          content: run.message,
          timestamp: run.created_at || new Date().toISOString(),
        }]
      : [];
    updateTab(tab.id, {
      messages: seed,
      title: run.message ? run.message.slice(0, 40) : 'Resumed Run',
      activeRun,
    });
  }, [restoreConversation, addTab, updateTab]);
}
