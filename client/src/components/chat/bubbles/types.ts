/**
 * Shared prop shape for every entry in the message-type registry.
 *
 * Every bubble accepts the same surface area; individual renderers ignore
 * the props they don't need. This keeps the registry uniformly typed and
 * lets the dispatcher stay a one-liner.
 */

import type { ConversationMessage, MessageType } from '../../../lib/messages';

export interface BubbleProps<K extends MessageType = MessageType> {
  message: Extract<ConversationMessage, { type: K }>;
  agentName?: string;
  avatarId?: string;
  onRegenerate?: () => void;
  onEdit?: (content: string) => void;
  /** Submit a choice-element selection (sent as the next user turn). */
  onSubmitChoice?: (value: string, messageId: string) => void;
  /** CC the Ambassador to brief this assistant turn (16.6). */
  onAmbassador?: (message: Extract<ConversationMessage, { type: 'assistant' }>) => void;
  /** Current ambassador-briefing status for this turn (drives the button state). */
  ambassadorStatus?: 'idle' | 'streaming' | 'done' | 'error';
  /** A turn is in flight — interactive elements (choice) render inert. */
  busy?: boolean;
}
