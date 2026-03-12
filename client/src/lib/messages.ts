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
  status: 'pending' | 'approved' | 'rejected' | 'completed';
  parentMessageId?: string;
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
  queryUsed: string;
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

/** Union of all conversation message types */
export type ConversationMessage =
  | UserMessage
  | AssistantMessage
  | ToolCallMessage
  | ToolResultMessage
  | MemoryInjectionMessage
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

export function isSystemMessage(msg: ConversationMessage): msg is SystemMessage {
  return msg.type === 'system';
}

export function isErrorMessage(msg: ConversationMessage): msg is ErrorMessage {
  return msg.type === 'error';
}
