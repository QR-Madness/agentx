/**
 * Message Type System — Discriminated union for all conversation message types
 */

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
  | 'error';

interface BaseMessage {
  id: string;
  timestamp: string;
  type: MessageType;
}

export interface UserMessage extends BaseMessage {
  type: 'user';
  content: string;
  editedAt?: string;
  /** Agent IDs mentioned via @ in the message (for multi-agent routing) */
  targetAgentIds?: string[];
}

export interface AssistantMessage extends BaseMessage {
  type: 'assistant';
  content: string;
  thinking?: string;
  model?: string;
  tokensUsed?: number;
  tokensInput?: number;
  tokensOutput?: number;
  latencyMs?: number;
  profileId?: string;
  agentName?: string;
  branchId?: string;
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
  status: 'running' | 'completed' | 'failed';
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

/** Delegation message — supervisor delegating a task to a specialist via Agent Alloy */
export interface DelegationMessage extends BaseMessage {
  type: 'delegation';
  delegationId: string;
  targetAgentId: string;
  targetAgentName?: string;
  task: string;
  depth: number;
  status: 'streaming' | 'completed' | 'failed';
  content: string;
  error?: string;
  resultPreview?: string;
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
