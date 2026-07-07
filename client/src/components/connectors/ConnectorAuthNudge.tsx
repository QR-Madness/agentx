/**
 * ConnectorAuthNudge — a headless watcher that fires a Claude-style toast when
 * the user starts a new conversation and the active agent has OAuth connectors
 * (MCP servers) that still need an interactive sign-in. The toast carries a
 * "Connect" action that opens the Toolkit sign-in.
 *
 * Renders nothing; mount once inside the provider stack (see App.tsx).
 */
import { useEffect, useRef } from 'react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useModal } from '../../contexts/ModalContext';
import { useNotify } from '../../contexts/NotificationContext';
import { useMCPServers } from '../../lib/hooks';
import { SURFACES } from '../../lib/surfaces';
import { connectorsNeedingAuth, connectorAuthMessage } from '../../lib/connectors';

export function ConnectorAuthNudge() {
  const { activeTab } = useConversation();
  const { servers, loading } = useMCPServers();
  const { activeProfile } = useAgentProfile();
  const { openModal } = useModal();
  const { notify } = useNotify();
  // Connectors already surfaced this session — nudge at most once each, so a
  // dismissed-without-connecting toast never nags on every new conversation.
  const nudged = useRef<Set<string>>(new Set());

  // Only nudge at the *start* of a conversation: a fresh, empty, unsaved tab.
  const isFresh = !!activeTab && activeTab.sessionId === null && activeTab.messages.length === 0;

  useEffect(() => {
    if (!isFresh || loading) return;
    const pending = connectorsNeedingAuth(servers, activeProfile?.agentId ?? null)
      .filter(s => !nudged.current.has(s.name));
    if (pending.length === 0) return;
    // Mark before notifying so a StrictMode double-invoke can't double-toast.
    pending.forEach(s => nudged.current.add(s.name));
    notify({
      kind: 'warning',
      duration: 0, // persist until Connect / dismiss
      title: 'Connector sign-in required',
      message: connectorAuthMessage(pending),
      action: { label: 'Connect', onClick: () => openModal(SURFACES.tools) },
    });
  }, [isFresh, loading, servers, activeProfile, notify, openModal]);

  return null;
}
