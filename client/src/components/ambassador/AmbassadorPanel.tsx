/**
 * AmbassadorPanel — the parallel conversation interpreter's surface (Phase 16.6).
 *
 * Subscribed to the *active* conversation (like SourcesPanel): it shows the
 * per-turn briefings the user has CC'd, streaming live and persisted in the
 * server sidecar. Because it reads `activeTab.sessionId`, switching tabs re-aims
 * the ambassador at whichever conversation is in front — the deliberate seam for
 * cross-seeding context from different conversations later.
 *
 * Briefings never enter the chat transcript; this panel is the only place (plus
 * the Redis sidecar) they live.
 */

import { useEffect, useMemo } from 'react';
import { Radio, Loader2, AlertTriangle, X } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import type { AmbassadorBriefing } from '../../lib/api';

function StatusBadge({ briefing }: { briefing: AmbassadorBriefing }) {
  switch (briefing.status) {
    case 'streaming':
      return (
        <span className="inline-flex items-center gap-1 text-xs text-fg-muted">
          <Loader2 size={12} className="animate-spin" /> briefing…
        </span>
      );
    case 'empty_provider':
      return <span className="text-xs text-warning">no provider</span>;
    case 'error':
      return <span className="text-xs text-error">failed</span>;
    case 'cancelled':
      return <span className="text-xs text-fg-muted">cancelled</span>;
    default:
      return null;
  }
}

export function AmbassadorPanel() {
  const { activeTab } = useConversation();
  const { briefingsFor, refresh, cancel } = useAmbassador();
  const conversationId = activeTab?.sessionId;

  // Subscribe: repopulate from the sidecar whenever the active conversation changes.
  useEffect(() => {
    if (conversationId) void refresh(conversationId);
  }, [conversationId, refresh]);

  const briefings = useMemo(
    () => briefingsFor(conversationId),
    [briefingsFor, conversationId],
  );

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <Radio size={18} className="text-accent" />
        <h2 className="text-base font-semibold text-fg">Ambassador</h2>
        <span className="text-sm text-fg-muted">({briefings.length})</span>
      </div>

      {!conversationId ? (
        <p className="text-sm text-fg-muted">
          Open a conversation, then use the CC button on any reply to have the ambassador brief
          that turn here — without touching the conversation itself.
        </p>
      ) : briefings.length === 0 ? (
        <p className="text-sm text-fg-muted">
          No briefings yet. Click the <Radio size={12} className="mx-0.5 inline align-middle" /> CC
          button on an assistant reply and the ambassador will brief that turn here.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {briefings.map((b) => (
            <li
              key={b.message_id}
              className="flex flex-col gap-1.5 rounded-md border border-line bg-surface-raised p-3 text-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <StatusBadge briefing={b} />
                {b.status === 'streaming' && (
                  <button
                    className="inline-flex items-center gap-1 text-xs text-fg-muted hover:text-error"
                    onClick={() => conversationId && cancel(conversationId, b.message_id)}
                    title="Cancel this briefing"
                  >
                    <X size={12} /> cancel
                  </button>
                )}
              </div>
              {b.status === 'error' ? (
                <p className="flex items-start gap-1.5 text-error">
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  <span>{b.error || 'The ambassador could not brief this turn.'}</span>
                </p>
              ) : (
                <p className="whitespace-pre-wrap text-fg">
                  {b.summary || (b.status === 'streaming' ? '…' : '')}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
