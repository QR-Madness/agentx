/**
 * MessageBubble — dispatches a `ConversationMessage` to its registered
 * renderer. New message types plug in via {@link messageRegistry}; this
 * file does not need to change.
 *
 * Memoized: `useTabMessages` replaces the message object on every update,
 * so shallow equality on the props is both correct and effective at
 * stopping the full message list from re-rendering on each stream chunk.
 */

import React from 'react';
import type { AssistantMessage, ConversationMessage } from '../../lib/messages';
import { messageRegistry } from './messageRegistry';
import { UnknownBubble } from './bubbles/UnknownBubble';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: ConversationMessage;
  agentName?: string;
  avatarId?: string;
  onRegenerate?: () => void;
  onEdit?: (content: string) => void;
  onSubmitChoice?: (value: string, messageId: string) => void;
  onAmbassador?: (message: AssistantMessage) => void;
  ambassadorStatus?: 'idle' | 'streaming' | 'done' | 'error';
  busy?: boolean;
}

function MessageBubbleImpl(props: MessageBubbleProps) {
  const { message } = props;
  // The registry is keyed by the discriminant; TS can't narrow the
  // component type across the lookup, so we cast at the boundary. The
  // BubbleProps<K> contract guarantees this is safe for every entry.
  const Renderer = messageRegistry[message.type] as
    | React.ComponentType<MessageBubbleProps>
    | undefined;

  if (!Renderer) return <UnknownBubble message={message} />;
  return <Renderer {...props} />;
}

export const MessageBubble = React.memo(MessageBubbleImpl);
