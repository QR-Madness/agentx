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
}
