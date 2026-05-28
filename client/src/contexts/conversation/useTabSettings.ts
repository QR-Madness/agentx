/**
 * Per-tab settings: agent profile, Agent Alloy workflow, and context-window
 * usage indicator. All thin wrappers over `updateTab` from
 * {@link useConversationTabs}.
 */

import { useCallback } from 'react';
import type { ConversationTab } from '../../lib/storage';

interface UseTabSettingsArgs {
  activeTabId: string | null;
  updateTab: (id: string, updates: Partial<Omit<ConversationTab, 'id'>>) => void;
}

export function useTabSettings({ activeTabId, updateTab }: UseTabSettingsArgs) {
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

  // Set a per-conversation model override for the active tab (null = inherit
  // the profile's model). Passed as `model` on the next stream request.
  const setActiveTabModel = useCallback((model: string | null) => {
    if (!activeTabId) return;
    updateTab(activeTabId, { modelOverride: model });
  }, [activeTabId, updateTab]);

  const setTabContextInfo = useCallback(
    (tabId: string, info: { window: number; used: number } | null) => {
      updateTab(tabId, {
        contextInfo: info
          ? { window: info.window, used: info.used, updatedAt: Date.now() }
          : undefined,
      });
    },
    [updateTab],
  );

  return {
    setTabProfile,
    setActiveTabProfile,
    setTabWorkflow,
    setActiveTabWorkflow,
    setActiveTabModel,
    setTabContextInfo,
  };
}
