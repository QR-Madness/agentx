/**
 * Message Type System — Discriminated union for all conversation message types
 */

import type { Exhibit } from './exhibits';
import type { ChatImageRef } from './api/types';

export type MessageType =
  | 'user'
  | 'assistant'
  | 'system'
  | 'tool_call'
  | 'tool_result'
  | 'memory_injection'
  | 'plan_execution'
  | 'agent_handoff'
  | 'delegation'
  | 'work_order_report'
  | 'exhibit'
  | 'error';

/**
 * Subtask affinity stamped onto messages produced while a plan subtask was
 * executing, so the transcript can badge + group them by step. Rides on the
 * message itself, so it persists to localStorage for free.
 */
export interface PlanStepRef {
  planId: string;
  subtaskId: number;
  /** 1-based for display ("Step 2/5"). */
  subtaskIndex: number;
  subtaskCount: number;
  subtaskTitle: string;
}

interface BaseMessage {
  id: string;
  timestamp: string;
  type: MessageType;
  /** Set when this message was produced inside a plan subtask. */
  planStep?: PlanStepRef;
}

export interface UserMessage extends BaseMessage {
  type: 'user';
  content: string;
  editedAt?: string;
  /** Agent IDs mentioned via @ in the message (for multi-agent routing) */
  targetAgentIds?: string[];
  /** Sent mid-turn to steer a running agent (vs. starting a new turn). */
  steered?: boolean;
  /** Images attached for vision input — rendered on the bubble (live + reload). */
  images?: ChatImageRef[];
}

export interface AssistantMessage extends BaseMessage {
  type: 'assistant';
  content: string;
  thinking?: string;
  model?: string;
  tokensUsed?: number;
  tokensInput?: number;
  tokensOutput?: number;
  costEstimate?: number;
  costCurrency?: string;
  latencyMs?: number;
  profileId?: string;
  agentName?: string;
  branchId?: string;
  /** Turn was cut short by a hard-stop (Stop button); content is partial. */
  interrupted?: boolean;
  /** Answer hit the model's token limit (finish_reason "length"); content is partial. */
  truncated?: boolean;
  /** Thinking pattern the turn ran with (from the done event) — badges the mode. */
  thinkingPattern?: string;
  /** Turn ran in Research Mode (patterns don't apply there — the badge says so). */
  research?: boolean;
}

export interface ToolCallMessage extends BaseMessage {
  type: 'tool_call';
  toolName: string;
  toolCallId: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  parentMessageId?: string;
  // Result fields (populated when complete)
  result?: {
    content: string;
    success: boolean;
    durationMs?: number;
  };
}

export interface ToolResultMessage extends BaseMessage {
  type: 'tool_result';
  toolName: string;
  toolCallId: string;
  content: string;
  success: boolean;
  durationMs?: number;
}

export interface MemoryInjectionMessage extends BaseMessage {
  type: 'memory_injection';
  facts: Array<{ claim: string; confidence: number; source?: string }>;
  entities: Array<{ name: string; type: string }>;
  relevantTurns: Array<{ timestamp: string; role: string; content: string }>;
  queryUsed: string;
}

export interface PlanSubtask {
  subtaskId: number;
  description: string;
  subtaskType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  resultPreview?: string;
  error?: string;
}

export interface PlanExecutionMessage extends BaseMessage {
  type: 'plan_execution';
  planId: string;
  task: string;
  complexity: string;
  subtaskCount: number;
  status: 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  subtasks: PlanSubtask[];
  totalTimeMs?: number;
  completedCount?: number;
}

export interface SystemMessage extends BaseMessage {
  type: 'system';
  content: string;
}

export interface ErrorMessage extends BaseMessage {
  type: 'error';
  content: string;
  recoverable: boolean;
}

/** A nested tool invocation that occurred inside a delegation. */
export interface DelegationToolEvent {
  toolName: string;
  toolCallId: string;
  status: 'running' | 'completed' | 'failed';
  arguments?: Record<string, unknown>;
  content?: string;
  success?: boolean;
  durationMs?: number;
}

/** Delegation message — supervisor delegating a task to a specialist via Agent Alloy */
export interface DelegationMessage extends BaseMessage {
  type: 'delegation';
  delegationId: string;
  targetAgentId: string;
  targetAgentName?: string;
  task: string;
  depth: number;
  status: 'streaming' | 'completed' | 'failed' | 'cancelled';
  content: string;
  error?: string;
  resultPreview?: string;
  /** 'background' = a delegate_start work order (receipt now, report later). */
  mode?: 'await' | 'background';
  /** Parent work order in the delegation tree (null/absent at depth 0). */
  parentDelegationId?: string | null;
  /** Set when this work order's report folded into the turn. */
  reportDelivered?: boolean;
  /** Specialist tool calls made during the delegation, rendered inside the card. */
  toolEvents?: DelegationToolEvent[];
  // Per-delegation metrics (populated on completion; persisted into the
  // delegate_to tool_result metadata so they survive a reload). `timestamp`
  // is the delegation start; `completedAt` is stamped on completion.
  tokensInput?: number;
  tokensOutput?: number;
  costEstimate?: number | null;
  costCurrency?: string | null;
  durationMs?: number;
  completedAt?: string;
  pricingSnapshot?: Record<string, unknown> | null;
}

/**
 * Hairline transcript marker: a background work order's report folded into
 * the turn here. Preserves narrative causality (the assistant reacts right
 * after it) without duplicating the report content — the Work Order card and
 * the trace console carry the content.
 */
export interface WorkOrderReportMarkerMessage extends BaseMessage {
  type: 'work_order_report';
  delegationId: string;
  targetAgentId: string;
  targetAgentName?: string;
  status: string;
}

/**
 * Exhibit message — a typed, declarative artifact the agent presented via the
 * `present_exhibit` tool (e.g. a Mermaid diagram). Rendered in one bubble by
 * `ExhibitBubble` via the element registry. Keyed by `exhibit.id` so an amend
 * (re-presenting the same id) replaces in place.
 */
export interface ExhibitMessage extends BaseMessage {
  type: 'exhibit';
  exhibit: Exhibit;
  /**
   * Set once the user picks an option from a `choice` element in this exhibit —
   * renders the choice resolved (disabled, selection marked) and persists via
   * localStorage. Cleared if the exhibit is amended (re-presented).
   */
  answeredValue?: string;
}

/** Agent handoff message - displayed when an agent transfers conversation to another */
export interface AgentHandoffMessage extends BaseMessage {
  type: 'agent_handoff';
  fromAgent: {
    id: string;
    name: string;
  };
  toAgent: {
    id: string;
    name: string;
  };
  reason?: string;
}

/** Union of all conversation message types */
export type ConversationMessage =
  | UserMessage
  | AssistantMessage
  | ToolCallMessage
  | ToolResultMessage
  | MemoryInjectionMessage
  | PlanExecutionMessage
  | AgentHandoffMessage
  | DelegationMessage
  | WorkOrderReportMarkerMessage
  | ExhibitMessage
  | SystemMessage
  | ErrorMessage;

/** Create a unique message ID */
export function createMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/** Type guard helpers */
export function isUserMessage(msg: ConversationMessage): msg is UserMessage {
  return msg.type === 'user';
}

export function isAssistantMessage(msg: ConversationMessage): msg is AssistantMessage {
  return msg.type === 'assistant';
}

export function isToolCallMessage(msg: ConversationMessage): msg is ToolCallMessage {
  return msg.type === 'tool_call';
}

export function isToolResultMessage(msg: ConversationMessage): msg is ToolResultMessage {
  return msg.type === 'tool_result';
}

export function isMemoryInjectionMessage(msg: ConversationMessage): msg is MemoryInjectionMessage {
  return msg.type === 'memory_injection';
}

export function isPlanExecutionMessage(msg: ConversationMessage): msg is PlanExecutionMessage {
  return msg.type === 'plan_execution';
}

export function isSystemMessage(msg: ConversationMessage): msg is SystemMessage {
  return msg.type === 'system';
}

export function isErrorMessage(msg: ConversationMessage): msg is ErrorMessage {
  return msg.type === 'error';
}

export function isAgentHandoffMessage(msg: ConversationMessage): msg is AgentHandoffMessage {
  return msg.type === 'agent_handoff';
}

export function isDelegationMessage(msg: ConversationMessage): msg is DelegationMessage {
  return msg.type === 'delegation';
}

export function isWorkOrderReportMarkerMessage(
  msg: ConversationMessage,
): msg is WorkOrderReportMarkerMessage {
  return msg.type === 'work_order_report';
}

export function isExhibitMessage(msg: ConversationMessage): msg is ExhibitMessage {
  return msg.type === 'exhibit';
}

/**
 * Strip `<thinking>` / `<think>` / `[thinking]` / `<internal_monologue>`
 * blocks from a model response. With `isStreaming = true`, also strips
 * still-unclosed opening tags so the live preview doesn't leak the
 * reasoning content while the model is mid-thought.
 *
 * Lives next to the message types because it is consumed by both the
 * streaming hook ({@link useChatStream}) and the streaming preview in
 * ChatPanel — keep them on the same regex set.
 */
export function stripThinkingTags(content: string, isStreaming = false): string {
  let result = content
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/\[thinking\][\s\S]*?\[\/thinking\]/gi, '')
    .replace(/\[think\][\s\S]*?\[\/think\]/gi, '')
    .replace(/<internal_monologue>[\s\S]*?<\/internal_monologue>/gi, '');

  if (isStreaming) {
    result = result
      .replace(/<thinking>[\s\S]*$/gi, '')
      .replace(/<think>[\s\S]*$/gi, '')
      .replace(/\[thinking\][\s\S]*$/gi, '')
      .replace(/\[think\][\s\S]*$/gi, '')
      .replace(/<internal_monologue>[\s\S]*$/gi, '');
  }

  return result.replace(/\n{3,}/g, '\n\n').trim();
}
