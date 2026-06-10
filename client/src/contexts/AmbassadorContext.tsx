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
import type {
  AmbassadorActiveConversation,
  AmbassadorBriefing,
  AmbassadorQA,
  AmbassadorStatus,
  AmbassadorToolCall,
  AmbassadorTurnArtifacts,
} from '../lib/api';
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
  ccTurn: (conversationId: string, message: AssistantMessage, opts: CcTurnOptions) => void;
  /** Cancel an in-flight briefing. */
  cancel: (conversationId: string, messageId: string) => void;
  /** All Q&A entries for a conversation (unordered; sort by created_at). */
  qaFor: (conversationId: string | null | undefined) => AmbassadorQA[];
  /** The unified ordered thread (briefings + Q&A interleaved) — the panel's single
   *  conversation ("Inquiry"), oldest-first by created_at. */
  threadFor: (conversationId: string | null | undefined) => ThreadStreamItem[];
  /** The Inquiry's own title (empty → caller falls back to the chat title). */
  titleFor: (conversationId: string | null | undefined) => string;
  /** Rename the Inquiry (persists to the sidecar; empty clears it). */
  renameThread: (conversationId: string, title: string) => void;
  /** Ask the ambassador a free-form question about the conversation. */
  ask: (conversationId: string, question: string, opts?: AskOptions) => void;
  /** Cancel an in-flight Q&A answer. */
  cancelQa: (conversationId: string, qaId: string) => void;
  /** Replay persisted briefings + Q&A from the server sidecar. */
  refresh: (conversationId: string) => Promise<void>;
}

/** One item in the unified ambassador thread — a briefing or a Q&A, tagged + timestamped
 *  so the panel renders both as one ordered conversation. */
export type ThreadStreamItem =
  | { kind: 'briefing'; id: string; createdAt?: string; briefing: AmbassadorBriefing }
  | { kind: 'qa'; id: string; createdAt?: string; qa: AmbassadorQA };

/** Everything a CC needs beyond the message itself (caller-resolved). */
export interface CcTurnOptions {
  /** The preceding user message (turn prompt), for grounding. */
  userText: string;
  /** What the agent did this turn (tools/sources/exhibits). */
  artifacts?: AmbassadorTurnArtifacts;
  /** Resolved display name of the briefed agent (so the briefing names it). */
  agentName?: string;
}

/** Caller-resolved grounding for a free-form question. */
export interface AskOptions {
  /** Resolved display name of the focused conversation's agent. */
  agentName?: string;
  /** Latest-turn substance, as extra grounding. */
  artifacts?: AmbassadorTurnArtifacts;
  /** Where the person currently is (ambient context, distinct from the focus). */
  activeConversation?: AmbassadorActiveConversation;
}

const AmbassadorContext = createContext<AmbassadorContextValue | null>(null);

type ConversationBriefings = Record<string, AmbassadorBriefing>;
type ConversationQA = Record<string, AmbassadorQA>;

function newQaId(): string {
  const rand = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  return `qa_${rand.replace(/-/g, '').slice(0, 32)}`;
}

export function AmbassadorProvider({ children }: { children: ReactNode }) {
  // conversationId -> (messageId -> briefing)
  const [state, setState] = useState<Record<string, ConversationBriefings>>({});
  // conversationId -> (qaId -> Q&A entry)
  const [qaState, setQaState] = useState<Record<string, ConversationQA>>({});
  // conversationId -> the Inquiry's own title ("" → fall back to the chat title)
  const [titles, setTitles] = useState<Record<string, string>>({});
  // Active SSE controllers, keyed so a live stream isn't clobbered by a refresh
  // and can be aborted on cancel. Briefings + Q&A share the ref via disjoint keys.
  const controllers = useRef<Record<string, { abort: () => void }>>({});

  const ctrlKey = (conversationId: string, messageId: string) =>
    `${conversationId}::${messageId}`;
  const qaCtrlKey = (conversationId: string, qaId: string) =>
    `${conversationId}::qa::${qaId}`;

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

  const setQa = useCallback(
    (conversationId: string, qaId: string, patch: Partial<AmbassadorQA>) => {
      setQaState((prev) => {
        const conv = prev[conversationId] ?? {};
        const existing: AmbassadorQA = conv[qaId] ?? {
          qa_id: qaId,
          question: '',
          answer: '',
          status: 'streaming' as AmbassadorStatus,
        };
        return {
          ...prev,
          [conversationId]: { ...conv, [qaId]: { ...existing, ...patch } },
        };
      });
    },
    [],
  );

  // --- Tool-call chips: append on start, mark done on result (live) -----------
  const appendToolCall = (list: AmbassadorToolCall[] | undefined, tool: string, args?: Record<string, unknown>) =>
    [...(list ?? []), { tool, args, done: false }];
  const settleToolCall = (list: AmbassadorToolCall[] | undefined, tool: string) => {
    if (!list?.length) return list;
    const idx = list.findIndex((t) => t.tool === tool && !t.done);
    if (idx < 0) return list;
    const next = list.slice();
    next[idx] = { ...next[idx], done: true };
    return next;
  };

  const updateQaTools = useCallback(
    (conversationId: string, qaId: string, fn: (cur: AmbassadorToolCall[] | undefined) => AmbassadorToolCall[] | undefined) => {
      setQaState((prev) => {
        const conv = prev[conversationId] ?? {};
        const existing = conv[qaId];
        if (!existing) return prev;
        return { ...prev, [conversationId]: { ...conv, [qaId]: { ...existing, toolCalls: fn(existing.toolCalls) } } };
      });
    },
    [],
  );

  const updateBriefingTools = useCallback(
    (conversationId: string, messageId: string, fn: (cur: AmbassadorToolCall[] | undefined) => AmbassadorToolCall[] | undefined) => {
      setState((prev) => {
        const conv = prev[conversationId] ?? {};
        const existing = conv[messageId];
        if (!existing) return prev;
        return { ...prev, [conversationId]: { ...conv, [messageId]: { ...existing, toolCalls: fn(existing.toolCalls) } } };
      });
    },
    [],
  );

  const refresh = useCallback(async (conversationId: string) => {
    if (!conversationId) return;
    try {
      // The unified thread: one ordered list + the Inquiry's title. Split the entries
      // back into the briefing/Q&A maps the streaming logic already drives.
      const { title, entries } = await api.fetchAmbassadorThread(conversationId);
      const briefings: AmbassadorBriefing[] = [];
      const qa: AmbassadorQA[] = [];
      for (const e of entries) {
        if (e.kind === 'briefing') {
          briefings.push({
            message_id: e.message_id ?? e.id,
            status: e.status,
            summary: e.content,
            error: e.error,
            run_id: e.run_id,
            created_at: e.created_at,
            updated_at: e.updated_at,
            toolCalls: e.toolCalls,
          });
        } else {
          qa.push({
            qa_id: e.id,
            question: e.question,
            answer: e.content,
            status: e.status,
            error: e.error,
            run_id: e.run_id,
            created_at: e.created_at,
            updated_at: e.updated_at,
            toolCalls: e.toolCalls,
          });
        }
      }
      setTitles((prev) => ({ ...prev, [conversationId]: title }));
      setState((prev) => {
        const conv: ConversationBriefings = { ...(prev[conversationId] ?? {}) };
        for (const b of briefings) {
          // Don't clobber a locally-streaming record (a live tail is authoritative).
          if (controllers.current[ctrlKey(conversationId, b.message_id)]) continue;
          conv[b.message_id] = b;
        }
        return { ...prev, [conversationId]: conv };
      });
      setQaState((prev) => {
        const conv: ConversationQA = { ...(prev[conversationId] ?? {}) };
        for (const item of qa) {
          if (controllers.current[qaCtrlKey(conversationId, item.qa_id)]) continue;
          conv[item.qa_id] = item;
        }
        return { ...prev, [conversationId]: conv };
      });
    } catch {
      /* sidecar unavailable — leave current state */
    }
  }, []);

  const ccTurn = useCallback(
    (conversationId: string, message: AssistantMessage, opts: CcTurnOptions) => {
      const { userText, artifacts, agentName } = opts;
      const messageId = message.id;
      // Idempotent: re-CC refreshes the same record rather than duplicating.
      setBriefing(conversationId, messageId, {
        message_id: messageId,
        status: 'streaming',
        summary: '',
        error: undefined,
        toolCalls: [],
      });

      api
        .briefTurn({
          conversation_id: conversationId,
          message_id: messageId,
          assistant_text: message.content,
          user_text: userText,
          agent_name: agentName || message.agentName,
          artifacts,
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
            onToolCall: (tool, args) =>
              updateBriefingTools(conversationId, messageId, (cur) => appendToolCall(cur, tool, args)),
            onToolResult: (tool) =>
              updateBriefingTools(conversationId, messageId, (cur) => settleToolCall(cur, tool)),
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

  const ask = useCallback(
    (conversationId: string, question: string, opts?: AskOptions) => {
      const q = question.trim();
      if (!conversationId || !q) return;
      const qaId = newQaId();
      // Optimistic: show the question immediately, streaming its answer.
      setQa(conversationId, qaId, { qa_id: qaId, question: q, answer: '', status: 'streaming', toolCalls: [] });

      api
        .askAmbassador({
          conversation_id: conversationId,
          qa_id: qaId,
          question: q,
          agent_name: opts?.agentName,
          artifacts: opts?.artifacts,
          active_conversation: opts?.activeConversation,
        })
        .then(({ run_id }) => {
          setQa(conversationId, qaId, { run_id });
          const handle = api.streamAmbassador(run_id, {
            onChunk: (text) =>
              setQaState((prev) => {
                const conv = prev[conversationId] ?? {};
                const existing: AmbassadorQA = conv[qaId] ?? {
                  qa_id: qaId,
                  question: q,
                  answer: '',
                  status: 'streaming',
                };
                return {
                  ...prev,
                  [conversationId]: {
                    ...conv,
                    [qaId]: { ...existing, answer: (existing.answer ?? '') + text },
                  },
                };
              }),
            onToolCall: (tool, args) =>
              updateQaTools(conversationId, qaId, (cur) => appendToolCall(cur, tool, args)),
            onToolResult: (tool) =>
              updateQaTools(conversationId, qaId, (cur) => settleToolCall(cur, tool)),
            onDone: (summary, status) => {
              setQa(conversationId, qaId, { answer: summary, status });
              delete controllers.current[qaCtrlKey(conversationId, qaId)];
            },
            onError: (error) => {
              setQa(conversationId, qaId, { status: 'error', error });
              delete controllers.current[qaCtrlKey(conversationId, qaId)];
            },
            onMissing: () => {
              void refresh(conversationId);
              delete controllers.current[qaCtrlKey(conversationId, qaId)];
            },
          });
          controllers.current[qaCtrlKey(conversationId, qaId)] = handle;
        })
        .catch((err: unknown) => {
          setQa(conversationId, qaId, {
            status: 'error',
            error: err instanceof Error ? err.message : 'Failed to reach the ambassador',
          });
        });
    },
    [setQa, refresh, updateQaTools],
  );

  const cancelQa = useCallback(
    (conversationId: string, qaId: string) => {
      const key = qaCtrlKey(conversationId, qaId);
      controllers.current[key]?.abort();
      delete controllers.current[key];
      const entry = qaState[conversationId]?.[qaId];
      if (entry?.run_id) void api.cancelChatRun(entry.run_id).catch(() => {});
      setQa(conversationId, qaId, { status: 'cancelled' });
    },
    [qaState, setQa],
  );

  const qaFor = useCallback(
    (conversationId: string | null | undefined) =>
      conversationId ? Object.values(qaState[conversationId] ?? {}) : [],
    [qaState],
  );

  const threadFor = useCallback(
    (conversationId: string | null | undefined): ThreadStreamItem[] => {
      if (!conversationId) return [];
      const items: ThreadStreamItem[] = [];
      for (const b of Object.values(state[conversationId] ?? {})) {
        items.push({ kind: 'briefing', id: b.message_id, createdAt: b.created_at, briefing: b });
      }
      for (const q of Object.values(qaState[conversationId] ?? {})) {
        items.push({ kind: 'qa', id: q.qa_id, createdAt: q.created_at, qa: q });
      }
      // Oldest first; brand-new local entries (no created_at yet) sort last, next to the input.
      items.sort((a, b) => {
        const ta = a.createdAt ? Date.parse(a.createdAt) : Infinity;
        const tb = b.createdAt ? Date.parse(b.createdAt) : Infinity;
        return ta - tb;
      });
      return items;
    },
    [state, qaState],
  );

  const titleFor = useCallback(
    (conversationId: string | null | undefined) =>
      (conversationId ? titles[conversationId] : '') ?? '',
    [titles],
  );

  const renameThread = useCallback((conversationId: string, title: string) => {
    if (!conversationId) return;
    const t = title.trim();
    setTitles((prev) => ({ ...prev, [conversationId]: t }));
    void api.renameAmbassadorThread(conversationId, t).catch(() => {});
  }, []);

  const value = useMemo(
    () => ({
      briefingsFor, briefingForMessage, ccTurn, cancel, qaFor,
      threadFor, titleFor, renameThread, ask, cancelQa, refresh,
    }),
    [
      briefingsFor, briefingForMessage, ccTurn, cancel, qaFor,
      threadFor, titleFor, renameThread, ask, cancelQa, refresh,
    ],
  );

  return <AmbassadorContext.Provider value={value}>{children}</AmbassadorContext.Provider>;
}

export function useAmbassador(): AmbassadorContextValue {
  const ctx = useContext(AmbassadorContext);
  if (!ctx) throw new Error('useAmbassador must be used within an AmbassadorProvider');
  return ctx;
}
