/**
 * AmbassadorContext — client state for the parallel conversation Ambassador
 * (Phase 16.6). Holds per-turn briefings keyed by conversation id → message id,
 * drives the briefing run (POST → SSE tail), and exposes refresh/replay from the
 * server sidecar so the panel repopulates after reload or tab-switch.
 *
 * The ambassador is a *parallel* channel: nothing here writes into the
 * conversation transcript — briefings live only in this context + the server's
 * `ambassador:` Redis sidecar.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { api } from '../lib/api';
import type { AmbassadorBriefing, AmbassadorStatus } from '../lib/api';
import type { AssistantMessage } from '../lib/messages';

interface AmbassadorContextValue {
  /** All briefings for a conversation (unordered). */
  briefingsFor: (conversationId: string | null | undefined) => AmbassadorBriefing[];
  /** A single message's briefing, if any. */
  briefingForMessage: (
    conversationId: string | null | undefined,
    messageId: string,
  ) => AmbassadorBriefing | undefined;
  /** CC the ambassador to brief one assistant turn. Idempotent per message id. */
  ccTurn: (conversationId: string, message: AssistantMessage, userText: string) => void;
  /** Cancel an in-flight briefing. */
  cancel: (conversationId: string, messageId: string) => void;
  /** Replay persisted briefings from the server sidecar. */
  refresh: (conversationId: string) => Promise<void>;
}

const AmbassadorContext = createContext<AmbassadorContextValue | null>(null);

type ConversationBriefings = Record<string, AmbassadorBriefing>;

export function AmbassadorProvider({ children }: { children: ReactNode }) {
  // conversationId -> (messageId -> briefing)
  const [state, setState] = useState<Record<string, ConversationBriefings>>({});
  // Active SSE controllers, keyed `${conversationId}::${messageId}`, so a live
  // stream isn't clobbered by a refresh and can be aborted on cancel.
  const controllers = useRef<Record<string, { abort: () => void }>>({});

  const ctrlKey = (conversationId: string, messageId: string) =>
    `${conversationId}::${messageId}`;

  const setBriefing = useCallback(
    (conversationId: string, messageId: string, patch: Partial<AmbassadorBriefing>) => {
      setState((prev) => {
        const conv = prev[conversationId] ?? {};
        const existing = conv[messageId] ?? { message_id: messageId, status: 'streaming' as AmbassadorStatus, summary: '' };
        return {
          ...prev,
          [conversationId]: { ...conv, [messageId]: { ...existing, ...patch } },
        };
      });
    },
    [],
  );

  const refresh = useCallback(async (conversationId: string) => {
    if (!conversationId) return;
    try {
      const briefings = await api.fetchAmbassadorBriefings(conversationId);
      setState((prev) => {
        const conv: ConversationBriefings = { ...(prev[conversationId] ?? {}) };
        for (const b of briefings) {
          // Don't clobber a locally-streaming record (a live tail is authoritative).
          if (controllers.current[ctrlKey(conversationId, b.message_id)]) continue;
          conv[b.message_id] = b;
        }
        return { ...prev, [conversationId]: conv };
      });
    } catch {
      /* sidecar unavailable — leave current state */
    }
  }, []);

  const ccTurn = useCallback(
    (conversationId: string, message: AssistantMessage, userText: string) => {
      const messageId = message.id;
      // Idempotent: re-CC refreshes the same record rather than duplicating.
      setBriefing(conversationId, messageId, {
        message_id: messageId,
        status: 'streaming',
        summary: '',
        error: undefined,
      });

      api
        .briefTurn({
          conversation_id: conversationId,
          message_id: messageId,
          assistant_text: message.content,
          user_text: userText,
        })
        .then(({ run_id }) => {
          setBriefing(conversationId, messageId, { run_id });
          const handle = api.streamAmbassador(run_id, {
            onChunk: (text) =>
              setState((prev) => {
                const conv = prev[conversationId] ?? {};
                const existing: AmbassadorBriefing = conv[messageId] ?? {
                  message_id: messageId,
                  status: 'streaming',
                  summary: '',
                };
                return {
                  ...prev,
                  [conversationId]: {
                    ...conv,
                    [messageId]: { ...existing, summary: (existing.summary ?? '') + text },
                  },
                };
              }),
            onDone: (summary, status) => {
              // onDone carries the authoritative full text — replace.
              setBriefing(conversationId, messageId, { summary, status });
              delete controllers.current[ctrlKey(conversationId, messageId)];
            },
            onError: (error) => {
              setBriefing(conversationId, messageId, { status: 'error', error });
              delete controllers.current[ctrlKey(conversationId, messageId)];
            },
            onMissing: () => {
              // Buffer expired — fall back to the persisted sidecar record.
              void refresh(conversationId);
              delete controllers.current[ctrlKey(conversationId, messageId)];
            },
          });
          controllers.current[ctrlKey(conversationId, messageId)] = handle;
        })
        .catch((err: unknown) => {
          setBriefing(conversationId, messageId, {
            status: 'error',
            error: err instanceof Error ? err.message : 'Failed to reach the ambassador',
          });
        });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setBriefing],
  );

  const cancel = useCallback((conversationId: string, messageId: string) => {
    const key = ctrlKey(conversationId, messageId);
    controllers.current[key]?.abort();
    delete controllers.current[key];
    const briefing = state[conversationId]?.[messageId];
    if (briefing?.run_id) void api.cancelChatRun(briefing.run_id).catch(() => {});
    setBriefing(conversationId, messageId, { status: 'cancelled' });
  }, [state, setBriefing]);

  const briefingsFor = useCallback(
    (conversationId: string | null | undefined) =>
      conversationId ? Object.values(state[conversationId] ?? {}) : [],
    [state],
  );

  const briefingForMessage = useCallback(
    (conversationId: string | null | undefined, messageId: string) =>
      conversationId ? state[conversationId]?.[messageId] : undefined,
    [state],
  );

  const value = useMemo(
    () => ({ briefingsFor, briefingForMessage, ccTurn, cancel, refresh }),
    [briefingsFor, briefingForMessage, ccTurn, cancel, refresh],
  );

  return <AmbassadorContext.Provider value={value}>{children}</AmbassadorContext.Provider>;
}

export function useAmbassador(): AmbassadorContextValue {
  const ctx = useContext(AmbassadorContext);
  if (!ctx) throw new Error('useAmbassador must be used within an AmbassadorProvider');
  return ctx;
}
