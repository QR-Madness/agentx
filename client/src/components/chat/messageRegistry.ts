/**
 * Message-type registry — maps a `ConversationMessage.type` discriminant to
 * the React component that knows how to render it.
 *
 * Adding a new message type:
 *   1. Add the variant to `ConversationMessage` in `lib/messages.ts`.
 *   2. Drop a renderer file in `chat/bubbles/`.
 *   3. Add one entry below.
 *
 * Types without a renderer fall through to `UnknownBubble`. This keeps the
 * UI safe when the backend sends new event kinds before the client ships
 * the matching renderer.
 */

import type { ComponentType } from 'react';
import type { MessageType } from '../../lib/messages';
import type { BubbleProps } from './bubbles/types';

import { UserBubble } from './bubbles/UserBubble';
import { AssistantBubble } from './bubbles/AssistantBubble';
import { ToolCallBubble } from './bubbles/ToolCallBubble';
import { ToolResultBubble } from './bubbles/ToolResultBubble';
import { MemoryInjectionBubble } from './bubbles/MemoryInjectionBubble';
import { PlanExecutionBubble } from './bubbles/PlanExecutionBubble';
import { DelegationBubble } from './bubbles/DelegationBubble';
import { ExhibitBubble } from './bubbles/ExhibitBubble';
import { SystemBubble } from './bubbles/SystemBubble';
import { ErrorBubble } from './bubbles/ErrorBubble';

// `Partial` so unmapped types (agent_handoff today) gracefully degrade to
// UnknownBubble in the dispatcher instead of failing to compile.
export type MessageRegistry = {
  [K in MessageType]?: ComponentType<BubbleProps<K>>;
};

export const messageRegistry: MessageRegistry = {
  user: UserBubble,
  assistant: AssistantBubble,
  tool_call: ToolCallBubble,
  tool_result: ToolResultBubble,
  memory_injection: MemoryInjectionBubble,
  plan_execution: PlanExecutionBubble,
  delegation: DelegationBubble,
  exhibit: ExhibitBubble,
  system: SystemBubble,
  error: ErrorBubble,
};
