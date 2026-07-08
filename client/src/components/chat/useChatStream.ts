/**
 * useChatStream — owns the streaming pipeline state and dispatches events
 * from `api.streamChat` into a pure reducer.
 *
 * Replaces the ~320-line callback web that used to live inline in
 * `ChatPanel.tsx`. The hook surface is small: `send(req)`, `stop()`, plus
 * `state` (the streaming UI shape) and `liveThinking` (the in-flight
 * supervisor tokens for the streaming preview).
 */

import { useCallback, useReducer, useRef } from 'react';
import { api, type ChatRequest, type PlanResumeRequest } from '../../lib/api';
import type { StreamCallbacks } from '../../lib/api/streaming';
import { useNotify } from '../../contexts/NotificationContext';
import {
  type ActiveDelegation,
  type ContextInfo,
  type StreamState,
  initialStreamState,
  streamReducer,
} from '../../lib/streamReducer';
import {
  type AssistantMessage,
  type ConversationMessage,
  type DelegationMessage,
  type DelegationToolEvent,
  type ExhibitMessage,
  type MemoryInjectionMessage,
  type PlanExecutionMessage,
  type PlanStepRef,
  type PlanSubtask,
  type ToolCallMessage,
  createMessageId,
  stripThinkingTags,
} from '../../lib/messages';
import { exhibitFromWire } from '../../lib/exhibits';
import type { PlanRecord } from '../../contexts/PlansContext';

/** Mutators from PlansContext, plumbed in so the global registry tracks plans. */
interface PlansSink {
  upsertPlan: (record: PlanRecord) => void;
  patchPlan: (planId: string, patch: Partial<PlanRecord>) => void;
}

interface UseChatStreamOpts {
  appendMessage: (m: ConversationMessage) => void;
  updateMessage: (id: string, patch: Partial<ConversationMessage>) => void;
  agentName?: string;
  resolveAgentName?: (agentId: string) => string | undefined;
  onSessionId?: (id: string) => void;
  onContextInfo?: (info: ContextInfo) => void;
  /** Owning conversation tab — stamped on global plan records. */
  tabId?: string;
  tabTitle?: string;
  /** Global plan registry sink (drawer + toolbar indicator). */
  plans?: PlansSink;
  /** Fired when the detached run id changes — persist it on the tab (null clears). */
  onRunChanged?: (runId: string | null) => void;
  /** Re-attach found no live run (TTL expired) — caller restores from history. */
  onRunMissing?: () => void;
  /** Fired when the agent autonomously saves a `checkpoint` mid-stream. */
  onCheckpointSaved?: () => void;
  /** Fired when the agent writes conversation state (`update_conversation_state`) mid-stream. */
  onConversationStateSaved?: () => void;
  /** Stored media fell back to a workspace (Home) — the conversation can durably attach. */
  onWorkspaceAttached?: (workspaceId: string) => void;
}

interface UseChatStreamApi {
  state: StreamState;
  send: (req: ChatRequest) => void;
  /** Resume an interrupted plan by id (continues its remaining subtasks). */
  resume: (planId: string, body: PlanResumeRequest) => void;
  /** Re-attach to an in-flight detached run (replays + follows live). */
  attach: (runId: string) => void;
  /** User-initiated cancel: abort + cancel the run server-side. */
  stop: () => void;
  /** Steer the running turn: fold a message in without stopping the run. */
  steer: (message: string) => void;
  /** Tab switch / unmount: drop the connection but leave the run running. */
  detach: () => void;
}

function extractThinking(content: string): string | null {
  const patterns = [
    /<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi,
    /\[think(?:ing)?\]([\s\S]*?)\[\/think(?:ing)?\]/gi,
    /<internal_monologue>([\s\S]*?)<\/internal_monologue>/gi,
  ];
  const thoughts: string[] = [];
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(content)) !== null) {
      if (match[1]?.trim()) thoughts.push(match[1].trim());
    }
  }
  return thoughts.length > 0 ? thoughts.join('\n\n') : null;
}

export function useChatStream(opts: UseChatStreamOpts): UseChatStreamApi {
  const [state, dispatch] = useReducer(streamReducer, initialStreamState);
  const { notify, notifyError } = useNotify();
  const abortRef = useRef<{ abort: () => void } | null>(null);

  // Caller passes a fresh opts literal every render, so capture it in a ref
  // and read through the ref inside callbacks. Lets `send`/`stop`/`flush`
  // stay referentially stable, which matters because consumers put them in
  // useEffect deps.
  const optsRef = useRef(opts);
  optsRef.current = opts;

  // Mirror state into refs so SSE callbacks (which close over the first
  // render's state) always see the latest snapshot. The reducer remains
  // the source of truth for renders; these refs only serve the callbacks.
  const liveContentRef = useRef('');
  const activeToolCallsRef = useRef<Map<string, { messageId: string; toolName: string }>>(new Map());
  const activePlanRef = useRef<{ messageId: string; planId: string; subtasks: PlanSubtask[] } | null>(null);
  const activeDelegationsRef = useRef<Map<string, ActiveDelegation>>(new Map());
  // Exhibit id -> its message id, so re-presenting the same id amends the
  // existing card in place (declarative reconcile) instead of stacking a new one.
  const exhibitMessageIdsRef = useRef<Map<string, string>>(new Map());
  // Subtask affinity stamped onto messages produced inside the current step.
  // Set on subtask_start, cleared on subtask_complete/failed and plan end —
  // so synthesis content (streamed after the last subtask) stays untagged.
  const currentStepRef = useRef<PlanStepRef | null>(null);
  // The detached run currently attached to (for the Stop button's cancel call).
  const currentRunIdRef = useRef<string | null>(null);

  // Helper: flush in-flight supervisor tokens as an intermediate
  // AssistantMessage, then clear the buffer. Always clears — no
  // whitespace guard. Whitespace-only content is simply not appended.
  const flushLiveContent = useCallback(() => {
    const pending = liveContentRef.current;
    liveContentRef.current = '';
    dispatch({ type: 'live_content_flushed' });
    if (!pending.trim()) return;
    const thinking = extractThinking(pending);
    const cleanContent = stripThinkingTags(pending);
    if (!thinking && !cleanContent) return;
    const msg: AssistantMessage = {
      id: createMessageId(),
      type: 'assistant',
      content: cleanContent,
      thinking: thinking ?? undefined,
      timestamp: new Date().toISOString(),
      agentName: optsRef.current.agentName,
      planStep: currentStepRef.current ?? undefined,
    };
    optsRef.current.appendMessage(msg);
  }, []);

  // Any delegation still in activeDelegationsRef when a stream ends/cancels never
  // got its delegation_complete (e.g. a fan-out cut short). Mark those cards as
  // failed-interrupted so they don't sit "streaming" forever, then drop the refs.
  const finalizeDanglingDelegations = useCallback(() => {
    for (const handle of activeDelegationsRef.current.values()) {
      optsRef.current.updateMessage(handle.messageId, {
        status: 'failed',
        error: 'Delegation interrupted',
        completedAt: new Date().toISOString(),
      });
    }
    activeDelegationsRef.current = new Map();
  }, []);

  const reset = useCallback(() => {
    liveContentRef.current = '';
    activeToolCallsRef.current = new Map();
    activePlanRef.current = null;
    activeDelegationsRef.current = new Map();
    exhibitMessageIdsRef.current = new Map();
    currentStepRef.current = null;
    dispatch({ type: 'send_started' });
  }, []);

  // Built once and shared by send() and attach() — a re-attach replays the
  // same event set, so the callback wiring is identical.
  const buildCallbacks = useCallback((): StreamCallbacks => ({
      onRunStarted: (data) => {
        currentRunIdRef.current = data.run_id;
        optsRef.current.onRunChanged?.(data.run_id);
      },

      onStart: (data) => {
        // Persist the session id on the tab at turn *start* (not just on done),
        // so a plan cancelled/interrupted mid-run still has a session to target
        // cancel/resume against — critical on a brand-new conversation's first turn.
        if (data.session_id) optsRef.current.onSessionId?.(data.session_id);
      },

      onRunMissing: () => {
        // The run buffer expired (or was orphaned) — let the caller restore
        // from server history instead of replaying.
        currentRunIdRef.current = null;
        optsRef.current.onRunChanged?.(null);
        dispatch({ type: 'stream_ended' });
        optsRef.current.onRunMissing?.();
      },

      onChunk: (content) => {
        liveContentRef.current += content;
        dispatch({ type: 'chunk_appended', content });
      },

      onStatus: (data) => {
        dispatch({
          type: 'status_changed',
          activity: { label: data.label, group: data.group, progress: data.progress },
        });
      },

      onWorkspaceAttached: (data) => {
        optsRef.current.onWorkspaceAttached?.(data.workspace_id);
      },

      onSteer: (data) => {
        // Close the in-flight assistant bubble first (like onToolCall), so the
        // steer user turn slots in between it and the agent's continuation.
        flushLiveContent();
        optsRef.current.appendMessage({
          id: data.id,
          type: 'user',
          content: data.message,
          timestamp: new Date().toISOString(),
          steered: true,
        });
      },

      onMemoryContext: (data) => {
        const hasAny =
          (data.facts?.length ?? 0) > 0 ||
          (data.entities?.length ?? 0) > 0 ||
          (data.relevant_turns?.length ?? 0) > 0;
        if (!hasAny) return;
        const memMsg: MemoryInjectionMessage = {
          id: createMessageId(),
          type: 'memory_injection',
          timestamp: new Date().toISOString(),
          facts: data.facts,
          entities: data.entities,
          relevantTurns: data.relevant_turns ?? [],
          queryUsed: data.query,
        };
        optsRef.current.appendMessage(memMsg);
      },

      onToolCall: (data) => {
        flushLiveContent();
        const messageId = createMessageId();
        activeToolCallsRef.current.set(data.tool_call_id, {
          messageId,
          toolName: data.tool,
        });
        dispatch({
          type: 'tool_call_registered',
          toolCallId: data.tool_call_id,
          messageId,
          toolName: data.tool,
        });
        const msg: ToolCallMessage = {
          id: messageId,
          type: 'tool_call',
          timestamp: new Date().toISOString(),
          toolName: data.tool,
          toolCallId: data.tool_call_id,
          arguments: data.arguments,
          status: 'running',
          planStep: currentStepRef.current ?? undefined,
        };
        optsRef.current.appendMessage(msg);
      },

      onToolResult: (data) => {
        const handle = activeToolCallsRef.current.get(data.tool_call_id);
        if (!handle) return;
        optsRef.current.updateMessage(handle.messageId, {
          status: data.success ? 'completed' : 'failed',
          result: {
            content: data.content,
            success: data.success,
            durationMs: data.duration_ms,
          },
        });
        if (handle.toolName === 'checkpoint' && data.success) {
          optsRef.current.onCheckpointSaved?.();
        }
        if (handle.toolName === 'update_conversation_state' && data.success) {
          optsRef.current.onConversationStateSaved?.();
        }
        activeToolCallsRef.current.delete(data.tool_call_id);
        dispatch({ type: 'tool_call_resolved', toolCallId: data.tool_call_id });
      },

      onExhibit: (data) => {
        const exhibit = exhibitFromWire(data);
        const existingId = exhibitMessageIdsRef.current.get(exhibit.id);
        if (existingId) {
          // Amend: re-presented with the same id → replace in place and clear any
          // prior answer so a revised choice is answerable again.
          optsRef.current.updateMessage(existingId, { exhibit, answeredValue: undefined });
          return;
        }
        // New exhibit: close any preceding prose, then append the card.
        flushLiveContent();
        const messageId = createMessageId();
        exhibitMessageIdsRef.current.set(exhibit.id, messageId);
        const msg: ExhibitMessage = {
          id: messageId,
          type: 'exhibit',
          timestamp: new Date().toISOString(),
          exhibit,
          planStep: currentStepRef.current ?? undefined,
        };
        optsRef.current.appendMessage(msg);
      },

      onPlanStart: (data) => {
        const messageId = createMessageId();
        activePlanRef.current = { messageId, planId: data.plan_id, subtasks: [] };
        currentStepRef.current = null;
        dispatch({ type: 'plan_started', messageId });
        const msg: PlanExecutionMessage = {
          id: messageId,
          type: 'plan_execution',
          timestamp: new Date().toISOString(),
          planId: data.plan_id,
          task: data.task,
          complexity: data.complexity,
          subtaskCount: data.subtask_count,
          status: 'running',
          subtasks: [],
        };
        optsRef.current.appendMessage(msg);
        optsRef.current.plans?.upsertPlan({
          planId: data.plan_id,
          tabId: optsRef.current.tabId ?? 'unknown',
          tabTitle: optsRef.current.tabTitle,
          task: data.task,
          complexity: data.complexity,
          status: 'running',
          subtaskCount: data.subtask_count,
          completedCount: 0,
          subtasks: [],
          startedAt: msg.timestamp,
        });
      },

      onPlanResumed: (data) => {
        // A resumed run never emitted plan_start; paint the card pre-filled
        // with the snapshot of already-done subtasks, then the normal
        // subtask_start/complete events drive the remaining ones.
        const mapStatus = (s: string): PlanSubtask['status'] =>
          s === 'complete' ? 'completed'
            : s === 'abandoned' ? 'skipped'
              : (s as PlanSubtask['status']);
        const subtasks: PlanSubtask[] = data.subtasks.map(s => ({
          subtaskId: s.subtask_id,
          description: s.description,
          subtaskType: s.type,
          status: mapStatus(s.status),
          resultPreview: s.result_preview || undefined,
        }));
        const messageId = createMessageId();
        activePlanRef.current = { messageId, planId: data.plan_id, subtasks };
        currentStepRef.current = null;
        dispatch({ type: 'plan_started', messageId });
        const msg: PlanExecutionMessage = {
          id: messageId,
          type: 'plan_execution',
          timestamp: new Date().toISOString(),
          planId: data.plan_id,
          task: data.task,
          complexity: data.complexity,
          subtaskCount: data.subtask_count,
          status: 'running',
          completedCount: data.completed_count,
          subtasks,
        };
        optsRef.current.appendMessage(msg);
        optsRef.current.plans?.upsertPlan({
          planId: data.plan_id,
          tabId: optsRef.current.tabId ?? 'unknown',
          tabTitle: optsRef.current.tabTitle,
          task: data.task,
          complexity: data.complexity,
          status: 'running',
          subtaskCount: data.subtask_count,
          completedCount: data.completed_count,
          subtasks,
          startedAt: msg.timestamp,
        });
      },

      onSubtaskStart: (data) => {
        // Each subtask runs its own streaming_tool_loop, so any supervisor
        // text left over from the previous subtask must be flushed before
        // the next round of chunks begins — otherwise preambles from
        // every subtask concatenate into one bubble. Flush happens under the
        // *previous* step's affinity, so set currentStepRef afterward.
        flushLiveContent();
        const plan = activePlanRef.current;
        if (!plan) return;
        const idx = plan.subtasks.findIndex(s => s.subtaskId === data.subtask_id);
        if (idx >= 0) {
          plan.subtasks[idx] = { ...plan.subtasks[idx], status: 'running' };
        } else {
          plan.subtasks.push({
            subtaskId: data.subtask_id,
            description: data.description,
            subtaskType: data.type,
            status: 'running',
          });
        }
        currentStepRef.current = {
          planId: plan.planId,
          subtaskId: data.subtask_id,
          subtaskIndex: data.subtask_id + 1,
          subtaskCount: data.progress?.total ?? plan.subtasks.length,
          subtaskTitle: data.description,
        };
        const subtasks = [...plan.subtasks];
        dispatch({ type: 'plan_subtasks_updated', subtasks });
        optsRef.current.updateMessage(plan.messageId, { subtasks });
        optsRef.current.plans?.patchPlan(plan.planId, { subtasks });
      },

      onSubtaskComplete: (data) => {
        // Flush this subtask's narration as its own (still step-tagged) bubble
        // before clearing affinity, so synthesis output stays unannotated.
        flushLiveContent();
        currentStepRef.current = null;
        const plan = activePlanRef.current;
        if (!plan) return;
        const target = plan.subtasks.find(s => s.subtaskId === data.subtask_id);
        if (target) {
          target.status = 'completed';
          target.resultPreview = data.result_preview;
        }
        const subtasks = [...plan.subtasks];
        const completedCount = subtasks.filter(s => s.status === 'completed').length;
        dispatch({ type: 'plan_subtasks_updated', subtasks });
        optsRef.current.updateMessage(plan.messageId, { subtasks, completedCount });
        optsRef.current.plans?.patchPlan(plan.planId, { subtasks, completedCount });
      },

      onSubtaskFailed: (data) => {
        flushLiveContent();
        currentStepRef.current = null;
        const plan = activePlanRef.current;
        if (!plan) return;
        const target = plan.subtasks.find(s => s.subtaskId === data.subtask_id);
        if (target) {
          target.status = 'failed';
          target.error = data.error;
        }
        const subtasks = [...plan.subtasks];
        dispatch({ type: 'plan_subtasks_updated', subtasks });
        optsRef.current.updateMessage(plan.messageId, { subtasks });
        optsRef.current.plans?.patchPlan(plan.planId, { subtasks });
      },

      onPlanComplete: (data) => {
        currentStepRef.current = null;
        const plan = activePlanRef.current;
        if (!plan) return;
        const status = data.completed_count === data.subtask_count ? 'completed' : 'failed';
        optsRef.current.updateMessage(plan.messageId, {
          status,
          completedCount: data.completed_count,
          totalTimeMs: data.total_time_ms,
        });
        optsRef.current.plans?.patchPlan(plan.planId, {
          status,
          completedCount: data.completed_count,
          totalTimeMs: data.total_time_ms,
        });
        activePlanRef.current = null;
        dispatch({ type: 'plan_finished' });
      },

      onPlanCancelled: (data) => {
        currentStepRef.current = null;
        const plan = activePlanRef.current;
        if (!plan) return;
        optsRef.current.updateMessage(plan.messageId, {
          status: 'cancelled',
          completedCount: data.completed_count,
          totalTimeMs: data.total_time_ms,
        });
        optsRef.current.plans?.patchPlan(plan.planId, {
          status: 'cancelled',
          completedCount: data.completed_count,
          totalTimeMs: data.total_time_ms,
        });
        activePlanRef.current = null;
        dispatch({ type: 'plan_finished' });
      },

      onDelegationStart: (data) => {
        flushLiveContent();
        const messageId = createMessageId();
        activeDelegationsRef.current.set(data.delegation_id, {
          messageId,
          content: '',
          toolEvents: [],
        });
        dispatch({ type: 'delegation_started', delegationId: data.delegation_id, messageId });

        const targetName = optsRef.current.resolveAgentName?.(data.target_agent_id);
        const msg: DelegationMessage = {
          id: messageId,
          type: 'delegation',
          timestamp: new Date().toISOString(),
          delegationId: data.delegation_id,
          targetAgentId: data.target_agent_id,
          targetAgentName: targetName,
          task: data.task,
          depth: data.depth,
          status: 'streaming',
          content: '',
          toolEvents: [],
        };
        optsRef.current.appendMessage(msg);
      },

      onDelegationChunk: (data) => {
        const handle = activeDelegationsRef.current.get(data.delegation_id);
        if (!handle) return;
        const accumulated = handle.content + data.content;
        handle.content = accumulated;
        dispatch({
          type: 'delegation_chunk_appended',
          delegationId: data.delegation_id,
          content: data.content,
        });
        optsRef.current.updateMessage(handle.messageId, { content: accumulated });
      },

      onDelegationComplete: (data) => {
        const key = data.delegation_id ?? '';
        const handle = key ? activeDelegationsRef.current.get(key) : undefined;
        if (handle) {
          optsRef.current.updateMessage(handle.messageId, {
            status: data.status === 'success' ? 'completed' : 'failed',
            content: handle.content || data.result_preview,
            error: data.error ?? undefined,
            resultPreview: data.result_preview,
            tokensInput: data.tokens_input,
            tokensOutput: data.tokens_output,
            costEstimate: data.cost_estimate,
            costCurrency: data.cost_currency,
            durationMs: data.duration_ms,
            completedAt: new Date().toISOString(),
          });
        }
        if (key) {
          activeDelegationsRef.current.delete(key);
          dispatch({ type: 'delegation_finished', delegationId: key });
        }
      },

      onDelegationToolCall: (data) => {
        const handle = activeDelegationsRef.current.get(data.delegation_id);
        if (!handle) return;
        const events: DelegationToolEvent[] = [
          ...handle.toolEvents,
          {
            toolName: data.tool,
            toolCallId: data.tool_call_id,
            status: 'running',
            arguments: data.arguments,
          },
        ];
        handle.toolEvents = events;
        dispatch({
          type: 'delegation_tool_event_appended',
          delegationId: data.delegation_id,
          events,
        });
        optsRef.current.updateMessage(handle.messageId, { toolEvents: events });
      },

      onDelegationToolResult: (data) => {
        const handle = activeDelegationsRef.current.get(data.delegation_id);
        if (!handle) return;
        const events = handle.toolEvents.map(evt =>
          evt.toolCallId === data.tool_call_id
            ? {
                ...evt,
                status: (data.success ? 'completed' : 'failed') as DelegationToolEvent['status'],
                content: data.content,
                success: data.success,
                durationMs: data.duration_ms,
              }
            : evt,
        );
        handle.toolEvents = events;
        dispatch({
          type: 'delegation_tool_event_appended',
          delegationId: data.delegation_id,
          events,
        });
        optsRef.current.updateMessage(handle.messageId, { toolEvents: events });
      },

      onDone: (data) => {
        const finalContent = liveContentRef.current;
        liveContentRef.current = '';
        currentRunIdRef.current = null;
        optsRef.current.onRunChanged?.(null);
        finalizeDanglingDelegations();
        dispatch({ type: 'stream_ended' });

        const cleanContent = stripThinkingTags(finalContent);
        if (cleanContent) {
          const msg: AssistantMessage = {
            id: createMessageId(),
            type: 'assistant',
            content: cleanContent,
            timestamp: new Date().toISOString(),
            thinking: data.thinking,
            latencyMs: data.total_time_ms,
            agentName: data.agent_name ?? optsRef.current.agentName,
            tokensInput: data.tokens_input ?? undefined,
            tokensOutput: data.tokens_output ?? undefined,
            costEstimate: data.cost_estimate ?? undefined,
            costCurrency: data.cost_currency ?? undefined,
            model: data.model ?? undefined,
            truncated: data.truncated === true || undefined,
          };
          optsRef.current.appendMessage(msg);
        }

        if (data.truncated) {
          notify({
            kind: 'warning',
            title: 'Response truncated',
            message: 'The answer hit the model’s token limit and may be incomplete.',
          });
        }

        if (data.session_id) optsRef.current.onSessionId?.(data.session_id);
        if (data.context_window && data.context_used) {
          optsRef.current.onContextInfo?.({
            window: data.context_window,
            used: data.context_used,
            summarized: data.context_summarized,
            droppedTurns: data.context_dropped_turns,
          });
        }
      },

      onError: (error) => {
        liveContentRef.current = '';
        activeToolCallsRef.current = new Map();
        if (activePlanRef.current) {
          optsRef.current.plans?.patchPlan(activePlanRef.current.planId, { status: 'failed' });
        }
        activePlanRef.current = null;
        currentStepRef.current = null;
        currentRunIdRef.current = null;
        optsRef.current.onRunChanged?.(null);
        finalizeDanglingDelegations();
        dispatch({ type: 'stream_ended' });

        const msg: AssistantMessage = {
          id: createMessageId(),
          type: 'assistant',
          content: `Sorry, I encountered an error: ${error}`,
          timestamp: new Date().toISOString(),
        };
        optsRef.current.appendMessage(msg);

        // Also raise a toast so the failure is visible even when the chat is
        // scrolled away or the user has switched tabs.
        notifyError(error);
      },
  }), [flushLiveContent, notify, notifyError, finalizeDanglingDelegations]);

  const send = useCallback((req: ChatRequest) => {
    abortRef.current?.abort();
    reset();
    currentRunIdRef.current = null;
    abortRef.current = api.streamChat(req, buildCallbacks());
  }, [reset, buildCallbacks]);

  const attach = useCallback((runId: string) => {
    // Reset in-flight state first so a full replay-from-0 rebuilds the live
    // bubble cleanly. The caller is responsible for truncating the tab's
    // messages back to the triggering user turn before calling this.
    abortRef.current?.abort();
    reset();
    currentRunIdRef.current = runId;
    abortRef.current = api.attachChatRun(runId, buildCallbacks());
  }, [reset, buildCallbacks]);

  // Resume an interrupted plan: continue its remaining subtasks as a fresh
  // detached run (first event is plan_resumed, which paints the card from the
  // saved snapshot). Mirrors `send` — same callback surface and run tracking.
  const resume = useCallback((planId: string, body: PlanResumeRequest) => {
    abortRef.current?.abort();
    reset();
    currentRunIdRef.current = null;
    abortRef.current = api.resumePlan(planId, body, buildCallbacks());
  }, [reset, buildCallbacks]);

  // Tab switch / unmount: drop the connection but leave the server run alive
  // and the activeRun id intact so it can be re-attached. Does NOT mark the
  // plan cancelled — it really is still running.
  const detach = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    liveContentRef.current = '';
    activeToolCallsRef.current = new Map();
    activePlanRef.current = null;
    currentStepRef.current = null;
    activeDelegationsRef.current = new Map();
    dispatch({ type: 'stream_ended' });
  }, []);

  // User pressed Stop: abort locally AND cancel the run server-side so it
  // stops burning tokens; clear the persisted run.
  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    const runId = currentRunIdRef.current;
    if (runId) {
      api.cancelChatRun(runId).catch(() => { /* best-effort */ });
    }
    currentRunIdRef.current = null;
    optsRef.current.onRunChanged?.(null);
    liveContentRef.current = '';
    activeToolCallsRef.current = new Map();
    if (activePlanRef.current) {
      optsRef.current.plans?.patchPlan(activePlanRef.current.planId, { status: 'cancelled' });
    }
    activePlanRef.current = null;
    currentStepRef.current = null;
    finalizeDanglingDelegations();
    dispatch({ type: 'stream_ended' });
  }, [finalizeDanglingDelegations]);

  // User typed while the turn streams: fold the message into the running run.
  // The steer bubble is appended when the server echoes the `steer` event back
  // (onSteer), so live + re-attached clients stay consistent. No-op until the
  // run_started event has given us a run id.
  const steer = useCallback((message: string) => {
    const runId = currentRunIdRef.current;
    if (!runId || !message.trim()) return;
    api.steerChatRun(runId, message.trim()).catch(() => { /* best-effort */ });
  }, []);

  return { state, send, resume, attach, stop, steer, detach };
}
