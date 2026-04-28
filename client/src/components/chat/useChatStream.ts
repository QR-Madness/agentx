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
import { api, type ChatRequest } from '../../lib/api';
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
  type MemoryInjectionMessage,
  type PlanExecutionMessage,
  type PlanSubtask,
  type ToolCallMessage,
  createMessageId,
} from '../../lib/messages';

interface UseChatStreamOpts {
  appendMessage: (m: ConversationMessage) => void;
  updateMessage: (id: string, patch: Partial<ConversationMessage>) => void;
  agentName?: string;
  resolveAgentName?: (agentId: string) => string | undefined;
  onSessionId?: (id: string) => void;
  onContextInfo?: (info: ContextInfo) => void;
}

interface UseChatStreamApi {
  state: StreamState;
  send: (req: ChatRequest) => void;
  stop: () => void;
}

function stripThinkingTags(content: string): string {
  return content
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/\[thinking\][\s\S]*?\[\/thinking\]/gi, '')
    .replace(/\[think\][\s\S]*?\[\/think\]/gi, '')
    .replace(/<internal_monologue>[\s\S]*?<\/internal_monologue>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
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
  const abortRef = useRef<{ abort: () => void } | null>(null);

  // Mirror state into refs so SSE callbacks (which close over the first
  // render's state) always see the latest snapshot. The reducer remains
  // the source of truth for renders; these refs only serve the callbacks.
  const liveContentRef = useRef('');
  const activeToolCallsRef = useRef<Map<string, { messageId: string; toolName: string }>>(new Map());
  const activePlanRef = useRef<{ messageId: string; subtasks: PlanSubtask[] } | null>(null);
  const activeDelegationsRef = useRef<Map<string, ActiveDelegation>>(new Map());

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
      agentName: opts.agentName,
    };
    opts.appendMessage(msg);
  }, [opts]);

  const reset = useCallback(() => {
    liveContentRef.current = '';
    activeToolCallsRef.current = new Map();
    activePlanRef.current = null;
    activeDelegationsRef.current = new Map();
    dispatch({ type: 'send_started' });
  }, []);

  const send = useCallback((req: ChatRequest) => {
    abortRef.current?.abort();
    reset();

    abortRef.current = api.streamChat(req, {
      onChunk: (content) => {
        liveContentRef.current += content;
        dispatch({ type: 'chunk_appended', content });
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
        opts.appendMessage(memMsg);
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
        };
        opts.appendMessage(msg);
      },

      onToolResult: (data) => {
        const handle = activeToolCallsRef.current.get(data.tool_call_id);
        if (!handle) return;
        opts.updateMessage(handle.messageId, {
          status: data.success ? 'completed' : 'failed',
          result: {
            content: data.content,
            success: data.success,
            durationMs: data.duration_ms,
          },
        });
        activeToolCallsRef.current.delete(data.tool_call_id);
        dispatch({ type: 'tool_call_resolved', toolCallId: data.tool_call_id });
      },

      onPlanStart: (data) => {
        const messageId = createMessageId();
        activePlanRef.current = { messageId, subtasks: [] };
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
        opts.appendMessage(msg);
      },

      onSubtaskStart: (data) => {
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
        const subtasks = [...plan.subtasks];
        dispatch({ type: 'plan_subtasks_updated', subtasks });
        opts.updateMessage(plan.messageId, { subtasks });
      },

      onSubtaskComplete: (data) => {
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
        opts.updateMessage(plan.messageId, { subtasks, completedCount });
      },

      onSubtaskFailed: (data) => {
        const plan = activePlanRef.current;
        if (!plan) return;
        const target = plan.subtasks.find(s => s.subtaskId === data.subtask_id);
        if (target) {
          target.status = 'failed';
          target.error = data.error;
        }
        const subtasks = [...plan.subtasks];
        dispatch({ type: 'plan_subtasks_updated', subtasks });
        opts.updateMessage(plan.messageId, { subtasks });
      },

      onPlanComplete: (data) => {
        const plan = activePlanRef.current;
        if (!plan) return;
        opts.updateMessage(plan.messageId, {
          status: data.completed_count === data.subtask_count ? 'completed' : 'failed',
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

        const targetName = opts.resolveAgentName?.(data.target_agent_id);
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
        opts.appendMessage(msg);
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
        opts.updateMessage(handle.messageId, { content: accumulated });
      },

      onDelegationComplete: (data) => {
        const key = data.delegation_id ?? '';
        const handle = key ? activeDelegationsRef.current.get(key) : undefined;
        if (handle) {
          opts.updateMessage(handle.messageId, {
            status: data.status === 'success' ? 'completed' : 'failed',
            content: handle.content || data.result_preview,
            error: data.error ?? undefined,
            resultPreview: data.result_preview,
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
        opts.updateMessage(handle.messageId, { toolEvents: events });
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
        opts.updateMessage(handle.messageId, { toolEvents: events });
      },

      onDone: (data) => {
        const finalContent = liveContentRef.current;
        liveContentRef.current = '';
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
            agentName: data.agent_name ?? opts.agentName,
            tokensInput: data.tokens_input ?? undefined,
            tokensOutput: data.tokens_output ?? undefined,
          };
          opts.appendMessage(msg);
        }

        if (data.session_id) opts.onSessionId?.(data.session_id);
        if (data.context_window && data.context_used) {
          opts.onContextInfo?.({ window: data.context_window, used: data.context_used });
        }
      },

      onError: (error) => {
        liveContentRef.current = '';
        activeToolCallsRef.current = new Map();
        activePlanRef.current = null;
        activeDelegationsRef.current = new Map();
        dispatch({ type: 'stream_ended' });

        const msg: AssistantMessage = {
          id: createMessageId(),
          type: 'assistant',
          content: `Sorry, I encountered an error: ${error}`,
          timestamp: new Date().toISOString(),
        };
        opts.appendMessage(msg);
      },
    });
  }, [opts, flushLiveContent, reset]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    liveContentRef.current = '';
    activeToolCallsRef.current = new Map();
    activePlanRef.current = null;
    activeDelegationsRef.current = new Map();
    dispatch({ type: 'stream_ended' });
  }, []);

  return { state, send, stop };
}
