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
import { Radio, Loader2, AlertTriangle, X, Sparkles } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import {
  isAssistantMessage,
  isUserMessage,
  type AssistantMessage,
  type ConversationMessage,
} from '../../lib/messages';
import type { AmbassadorBriefing } from '../../lib/api';

/** Nearest preceding user message — gives the ambassador the turn's prompt. */
function precedingUserText(messages: ConversationMessage[], index: number): string {
  for (let i = index - 1; i >= 0; i -= 1) {
    const m = messages[i];
    if (m && isUserMessage(m)) return m.content;
  }
  return '';
}

function snippet(text: string, max = 140): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

function BriefingBody({ briefing }: { briefing: AmbassadorBriefing | undefined }) {
  if (!briefing) return null;
  if (briefing.status === 'error') {
    return (
      <p className="mt-1 flex items-start gap-1.5 whitespace-pre-wrap text-error">
        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
        <span>{briefing.error || 'The ambassador could not brief this turn.'}</span>
      </p>
    );
  }
  if (briefing.status === 'empty_provider') {
    return (
      <p className="mt-1 whitespace-pre-wrap text-warning">
        {briefing.summary || 'No model provider is configured for the ambassador.'}
      </p>
    );
  }
  if (!briefing.summary && briefing.status !== 'streaming') return null;
  return (
    <p className="mt-1 whitespace-pre-wrap text-fg">
      {briefing.summary || '…'}
    </p>
  );
}

export function AmbassadorPanel() {
  const { activeTab } = useConversation();
  const { briefingForMessage, refresh, ccTurn, cancel } = useAmbassador();
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

  const brief = (m: AssistantMessage, i: number) => {
    if (!conversationId) return;
    ccTurn(conversationId, m, precedingUserText(messages, i));
  };

  const latest = turns[0];
  const latestStreaming =
    !!latest && briefingForMessage(conversationId, latest.m.id)?.status === 'streaming';

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-4">
      <div className="flex items-center gap-2">
        <Radio size={18} className="text-accent" />
        <h2 className="text-base font-semibold text-fg">Ambassador</h2>
        <span className="text-sm text-fg-muted">
          {turns.length} turn{turns.length === 1 ? '' : 's'}
        </span>
      </div>

      <p className="text-sm text-fg-muted">
        A dedicated agent that briefs you on a turn in parallel — without entering the conversation.
        Pick its model in Settings → Ambassador.
      </p>

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
            onClick={() => latest && brief(latest.m, latest.i)}
            disabled={latestStreaming}
          >
            <Sparkles size={15} />
            Brief the latest turn
          </button>

          <ul className="flex flex-col gap-2">
            {turns.map(({ m, i }, idx) => {
              const briefing = briefingForMessage(conversationId, m.id);
              const turnNo = turns.length - idx; // oldest = 1
              const streaming = briefing?.status === 'streaming';
              return (
                <li
                  key={m.id}
                  className="flex flex-col gap-1 rounded-md border border-line bg-surface-raised p-3 text-sm"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-fg-muted">Turn {turnNo}</span>
                    {streaming ? (
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-xs text-fg-muted hover:text-error"
                        onClick={() => cancel(conversationId, m.id)}
                        title="Cancel briefing"
                      >
                        <X size={13} /> cancel
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
                        onClick={() => brief(m, i)}
                        title="Brief this turn"
                      >
                        <Radio size={13} /> {briefing ? 're-brief' : 'brief'}
                      </button>
                    )}
                  </div>
                  <p className="text-xs text-fg-muted">{snippet(m.content)}</p>
                  {streaming && !briefing?.summary && (
                    <span className="inline-flex items-center gap-1 text-xs text-fg-muted">
                      <Loader2 size={12} className="animate-spin" /> briefing…
                    </span>
                  )}
                  <BriefingBody briefing={briefing} />
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
