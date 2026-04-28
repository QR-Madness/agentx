/**
 * Pure state machine for the chat streaming pipeline.
 *
 * Owns transient in-flight stream state — the live token buffer, active
 * tool-call / plan / delegation handles, and the streaming phase. The
 * reducer never mutates the conversation message list; the hook layer
 * (`useChatStream`) calls `appendMessage` / `updateMessage` against
 * `ConversationContext` in response to events. Keeping side effects out
 * of the reducer lets us test it with plain inputs and gives us one
 * obvious place to look when stream state behaves oddly.
 */

import type { DelegationToolEvent, PlanSubtask } from './messages';

export type StreamPhase = 'idle' | 'streaming';

export interface ContextInfo {
  window: number;
  used: number;
}

export interface ActiveToolCall {
  messageId: string;
  toolName: string;
}

export interface ActivePlan {
  messageId: string;
  subtasks: PlanSubtask[];
}

export interface ActiveDelegation {
  messageId: string;
  content: string;
  toolEvents: DelegationToolEvent[];
}

export interface StreamState {
  phase: StreamPhase;
  /** Accumulated supervisor tokens for the in-flight assistant bubble. */
  liveContent: string;
  /** Active tool-call cards keyed by tool_call_id. */
  activeToolCalls: Map<string, ActiveToolCall>;
  /** Active plan execution card (only one at a time today). */
  activePlan: ActivePlan | null;
  /** Active delegation cards keyed by delegation_id. */
  activeDelegations: Map<string, ActiveDelegation>;
}

export const initialStreamState: StreamState = {
  phase: 'idle',
  liveContent: '',
  activeToolCalls: new Map(),
  activePlan: null,
  activeDelegations: new Map(),
};

export type StreamAction =
  | { type: 'send_started' }
  | { type: 'chunk_appended'; content: string }
  | { type: 'live_content_flushed' }
  | { type: 'tool_call_registered'; toolCallId: string; messageId: string; toolName: string }
  | { type: 'tool_call_resolved'; toolCallId: string }
  | { type: 'plan_started'; messageId: string }
  | { type: 'plan_subtasks_updated'; subtasks: PlanSubtask[] }
  | { type: 'plan_finished' }
  | { type: 'delegation_started'; delegationId: string; messageId: string }
  | { type: 'delegation_chunk_appended'; delegationId: string; content: string }
  | { type: 'delegation_tool_event_appended'; delegationId: string; events: DelegationToolEvent[] }
  | { type: 'delegation_finished'; delegationId: string }
  | { type: 'stream_ended' };

export function streamReducer(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case 'send_started':
      // ONE place state is reset. Every active artifact is dropped — if a
      // prior stream left dangling cards in `streaming` status, callers
      // should mark them stale before sending a new prompt.
      return {
        phase: 'streaming',
        liveContent: '',
        activeToolCalls: new Map(),
        activePlan: null,
        activeDelegations: new Map(),
      };

    case 'chunk_appended':
      return { ...state, liveContent: state.liveContent + action.content };

    case 'live_content_flushed':
      // Always clears, regardless of whether content was whitespace-only.
      // Callers decide whether to emit an intermediate AssistantMessage
      // before dispatching this; the reducer just guarantees the buffer
      // is empty afterward.
      return { ...state, liveContent: '' };

    case 'tool_call_registered': {
      const next = new Map(state.activeToolCalls);
      next.set(action.toolCallId, {
        messageId: action.messageId,
        toolName: action.toolName,
      });
      return { ...state, activeToolCalls: next };
    }

    case 'tool_call_resolved': {
      const next = new Map(state.activeToolCalls);
      next.delete(action.toolCallId);
      return { ...state, activeToolCalls: next };
    }

    case 'plan_started':
      return { ...state, activePlan: { messageId: action.messageId, subtasks: [] } };

    case 'plan_subtasks_updated':
      if (!state.activePlan) return state;
      return { ...state, activePlan: { ...state.activePlan, subtasks: action.subtasks } };

    case 'plan_finished':
      return { ...state, activePlan: null };

    case 'delegation_started': {
      const next = new Map(state.activeDelegations);
      next.set(action.delegationId, {
        messageId: action.messageId,
        content: '',
        toolEvents: [],
      });
      return { ...state, activeDelegations: next };
    }

    case 'delegation_chunk_appended': {
      const existing = state.activeDelegations.get(action.delegationId);
      if (!existing) return state;  // unknown delegation — drop silently
      const next = new Map(state.activeDelegations);
      next.set(action.delegationId, {
        ...existing,
        content: existing.content + action.content,
      });
      return { ...state, activeDelegations: next };
    }

    case 'delegation_tool_event_appended': {
      const existing = state.activeDelegations.get(action.delegationId);
      if (!existing) return state;
      const next = new Map(state.activeDelegations);
      next.set(action.delegationId, { ...existing, toolEvents: action.events });
      return { ...state, activeDelegations: next };
    }

    case 'delegation_finished': {
      const next = new Map(state.activeDelegations);
      next.delete(action.delegationId);
      return { ...state, activeDelegations: next };
    }

    case 'stream_ended':
      return {
        ...state,
        phase: 'idle',
        liveContent: '',
      };

    default:
      return state;
  }
}

/** Convenience: the active-delegation count drives the typing-spinner suppression. */
export function activeDelegationCount(state: StreamState): number {
  return state.activeDelegations.size;
}
