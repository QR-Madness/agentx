/**
 * AmbassadorPanel — the parallel conversation interpreter's surface (Phase 16.6).
 *
 * Subscribed to the *active* conversation tab: it lists that conversation's
 * assistant turns and lets you brief any of them (or the latest) on demand —
 * the panel itself is the entry point, no per-message button required. Switching
 * tabs re-aims the ambassador at whichever conversation is in front (the seam for
 * cross-seeding context from different conversations).
 *
 * Briefings never enter the chat transcript; they live only here + the server's
 * `ambassador:` Redis sidecar.
 */

import { useEffect, useMemo } from 'react';
import { Radio, Loader2, AlertTriangle, X, Sparkles, RotateCcw, Ban } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import { isAssistantMessage, type AssistantMessage } from '../../lib/messages';
import { gatherTurnContext, resolveTurnAgentName } from '../../lib/ambassadorTurn';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import type { AmbassadorBriefing } from '../../lib/api';

function snippet(text: string, max = 140): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

/** A small status chip mirroring the briefing's lifecycle. */
function StatusChip({ briefing }: { briefing: AmbassadorBriefing | undefined }) {
  if (!briefing) return null;
  const base =
    'inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide';
  switch (briefing.status) {
    case 'streaming':
      return (
        <span className={`${base} bg-accent-tertiary text-accent`}>
          <Loader2 size={10} className="animate-spin" /> briefing
        </span>
      );
    case 'done':
      return <span className={`${base} bg-surface-sunken text-success`}>briefed</span>;
    case 'cancelled':
      return (
        <span className={`${base} bg-surface-sunken text-fg-muted`}>
          <Ban size={10} /> cancelled
        </span>
      );
    case 'error':
      return <span className={`${base} bg-surface-sunken text-error`}>error</span>;
    case 'empty_provider':
      return <span className={`${base} bg-surface-sunken text-warning`}>no model</span>;
    default:
      return null;
  }
}

function BriefingBody({ briefing }: { briefing: AmbassadorBriefing | undefined }) {
  if (!briefing) return null;
  if (briefing.status === 'error') {
    return (
      <p className="mt-1.5 flex items-start gap-1.5 whitespace-pre-wrap text-sm text-error">
        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
        <span>{briefing.error || 'The ambassador could not brief this turn.'}</span>
      </p>
    );
  }
  if (briefing.status === 'empty_provider') {
    return (
      <p className="mt-1.5 whitespace-pre-wrap text-sm text-warning">
        {briefing.summary || 'No model provider is configured for the ambassador.'}
      </p>
    );
  }
  const streaming = briefing.status === 'streaming';
  if (!briefing.summary && !streaming) return null;
  return (
    <p className="mt-1.5 whitespace-pre-wrap text-sm leading-relaxed text-fg">
      {briefing.summary}
      {streaming && (
        <span className="ml-0.5 inline-block h-3.5 w-px animate-pulse bg-accent align-middle" />
      )}
    </p>
  );
}

export function AmbassadorPanel() {
  const { activeTab } = useConversation();
  const { briefingForMessage, briefingsFor, refresh, ccTurn, cancel } = useAmbassador();
  const { profiles, getAgentName } = useAgentProfile();
  const conversationId = activeTab?.sessionId;

  // Subscribe: repopulate from the sidecar whenever the active conversation changes.
  useEffect(() => {
    if (conversationId) void refresh(conversationId);
  }, [conversationId, refresh]);

  const messages = activeTab?.messages ?? [];
  // Assistant turns (non-empty content), newest first, carrying their index.
  const turns = useMemo(
    () =>
      (messages
        .map((m, i) => ({ m, i }))
        .filter((x) => isAssistantMessage(x.m) && x.m.content.trim().length > 0)
        .reverse() as { m: AssistantMessage; i: number }[]),
    [messages],
  );

  const brief = (m: AssistantMessage) => {
    if (!conversationId) return;
    const { userText, artifacts } = gatherTurnContext(messages, m.id);
    const tabProfileName = activeTab?.profileId
      ? profiles.find((p) => p.id === activeTab.profileId)?.name
      : undefined;
    const agentName = resolveTurnAgentName(m, {
      nameByProfileId: (id) => profiles.find((p) => p.id === id)?.name,
      fallback: tabProfileName ?? getAgentName(),
    });
    ccTurn(conversationId, m, { userText, artifacts, agentName });
  };

  const latest = turns[0];
  const latestStreaming =
    !!latest && briefingForMessage(conversationId, latest.m.id)?.status === 'streaming';
  const anyStreaming = useMemo(
    () => briefingsFor(conversationId).some((b) => b.status === 'streaming'),
    [briefingsFor, conversationId],
  );

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex flex-col gap-2 border-b border-line px-4 pb-3 pt-4">
        <div className="flex items-center gap-2">
          <span className="relative inline-flex">
            <Radio size={18} className="text-accent" />
            {anyStreaming && (
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-ping rounded-full bg-accent" />
            )}
          </span>
          <h2 className="text-base font-semibold text-fg">Ambassador</h2>
          <span className="ml-auto text-xs text-fg-muted">
            {turns.length} turn{turns.length === 1 ? '' : 's'}
          </span>
        </div>
        <p className="text-xs leading-relaxed text-fg-muted">
          A dedicated agent that briefs you on a turn in parallel — without entering the
          conversation. Pick its model in Settings → Ambassador.
        </p>
      </div>

      {/* Body */}
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
        {!conversationId ? (
          <p className="text-sm text-fg-muted">
            Start (or open) a conversation and the ambassador can brief its turns here.
          </p>
        ) : turns.length === 0 ? (
          <p className="text-sm text-fg-muted">No assistant turns in this conversation yet.</p>
        ) : (
          <>
            <button
              type="button"
              className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-3 py-2 text-sm font-medium text-fg-inverse transition-colors hover:bg-accent-secondary disabled:opacity-50"
              onClick={() => latest && brief(latest.m)}
              disabled={latestStreaming}
            >
              {latestStreaming ? (
                <Loader2 size={15} className="animate-spin" />
              ) : (
                <Sparkles size={15} />
              )}
              {latestStreaming ? 'Briefing the latest turn…' : 'Brief the latest turn'}
            </button>

            <ul className="flex flex-col gap-2">
              {turns.map(({ m }, idx) => {
                const briefing = briefingForMessage(conversationId, m.id);
                const turnNo = turns.length - idx; // oldest = 1
                const streaming = briefing?.status === 'streaming';
                const briefed = !!briefing && briefing.status !== 'streaming';
                return (
                  <li
                    key={m.id}
                    className="flex flex-col gap-1 rounded-lg border border-line bg-surface-raised p-3 transition-colors data-[briefed=true]:border-line-strong"
                    data-briefed={briefed || undefined}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-fg-muted">Turn {turnNo}</span>
                      <StatusChip briefing={briefing} />
                      {streaming ? (
                        <button
                          type="button"
                          className="ml-auto inline-flex items-center gap-1 text-xs text-fg-muted transition-colors hover:text-error"
                          onClick={() => cancel(conversationId, m.id)}
                          title="Cancel briefing"
                        >
                          <X size={13} /> cancel
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="ml-auto inline-flex items-center gap-1 text-xs text-accent transition-colors hover:underline"
                          onClick={() => brief(m)}
                          title={briefing ? 'Brief this turn again' : 'Brief this turn'}
                        >
                          {briefing ? <RotateCcw size={12} /> : <Radio size={13} />}
                          {briefing ? 're-brief' : 'brief'}
                        </button>
                      )}
                    </div>
                    <p className="text-xs text-fg-muted">{snippet(m.content)}</p>
                    <BriefingBody briefing={briefing} />
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
