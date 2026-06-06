import { MessageContent } from '../MessageContent';
import { ThinkingBubble } from '../ThinkingBubble';
import { MessageActions } from '../MessageActions';
import { MetadataBar } from '../MetadataBar';
import { getAvatarIcon } from '../../../lib/avatars';
import type { BubbleProps } from './types';

export function AssistantBubble({ message, agentName, avatarId, onRegenerate, onAmbassador, ambassadorStatus }: BubbleProps<'assistant'>) {
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
          {message.interrupted && (
            <span className="assistant-stopped-tag" title="Generation was stopped; this response is partial">
              stopped
            </span>
          )}
        </div>

        {/* CoT is shown expanded *while streaming* (ChatPanel's live preview); once
            the turn lands, it collapses to a quiet affordance so the result — not the
            process — leads. It's session-only; not persisted, gone on reload. */}
        {message.thinking && (
          <ThinkingBubble thinking={message.thinking} defaultExpanded={false} />
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
          onAmbassador={onAmbassador ? () => onAmbassador(message) : undefined}
          ambassadorStatus={ambassadorStatus}
        />
      </div>
    </div>
  );
}
