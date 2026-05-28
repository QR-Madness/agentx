import { MessageContent } from '../MessageContent';
import { ThinkingBubble } from '../ThinkingBubble';
import { MessageActions } from '../MessageActions';
import { MetadataBar } from '../MetadataBar';
import { getAvatarIcon } from '../../../lib/avatars';
import type { BubbleProps } from './types';

export function AssistantBubble({ message, agentName, avatarId, onRegenerate }: BubbleProps<'assistant'>) {
  const displayName = message.agentName || agentName || 'Assistant';
  const AvatarIcon = getAvatarIcon(avatarId);

  return (
    <div className="message-bubble assistant">
      <div className="message-avatar assistant-avatar">
        <AvatarIcon size={16} />
      </div>
      <div className="message-body">
        <div className="assistant-header">
          <span className="assistant-name">{displayName}</span>
        </div>

        {message.thinking && (
          <ThinkingBubble thinking={message.thinking} defaultExpanded={true} />
        )}

        <MessageContent content={message.content} />

        <MetadataBar
          model={message.model}
          tokensInput={message.tokensInput}
          tokensOutput={message.tokensOutput}
          tokensUsed={message.tokensUsed}
          costEstimate={message.costEstimate}
          costCurrency={message.costCurrency}
          latencyMs={message.latencyMs}
        />

        <MessageActions
          content={message.content}
          isAssistant={true}
          timestamp={new Date(message.timestamp)}
          onRegenerate={onRegenerate}
        />
      </div>
    </div>
  );
}
