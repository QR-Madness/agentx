/**
 * Owns the tab state that every other conversation concern composes on:
 * `tabs` / `activeTabId`, their localStorage persistence (debounced save +
 * active-id save), the server-change load, the mobile single-tab collapse, and
 * the CRUD ops. The returned `setTabs`/`setActiveTabId`/`updateTab`/`closeTab`
 * and the `initializedServerId` ref are wired into the message/history hooks.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useServer } from '../ServerContext';
import {
  type ConversationTab,
  getConversationTabs,
  saveConversationTabs,
  getActiveTabId,
  saveActiveTabId,
  createDefaultTab,
  generateTabId,
} from '../../lib/storage';

export function useConversationTabs() {
  const { activeServer } = useServer();
  const [tabs, setTabs] = useState<ConversationTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);

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

  // Enforce single-tab mode on mobile: when the viewport drops below the
  // mobile breakpoint, collapse to just the active tab so the user doesn't
  // get stranded on a tab they can't see.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 600px)');
    const collapse = () => {
      if (!mq.matches) return;
      setTabs(prev => {
        if (prev.length <= 1) return prev;
        const keep = prev.find(t => t.id === activeTabId) ?? prev[prev.length - 1];
        setActiveTabId(keep.id);
        return [keep];
      });
    };
    collapse();
    mq.addEventListener('change', collapse);
    return () => mq.removeEventListener('change', collapse);
  }, [activeTabId]);

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

    // Mobile clients can only run one conversation at a time — replace
    // any existing tabs so the new one becomes the sole active conversation.
    const isMobile =
      typeof window !== 'undefined' &&
      window.matchMedia('(max-width: 600px)').matches;

    setTabs(prev => (isMobile ? [newTab] : [...prev, newTab]));
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

  return {
    activeServer,
    tabs,
    setTabs,
    activeTabId,
    setActiveTabId,
    activeTab,
    initializedServerId,
    addTab,
    closeTab,
    switchTab,
    renameTab,
    updateTab,
  };
}
