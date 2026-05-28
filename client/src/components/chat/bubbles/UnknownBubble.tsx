import { Info } from 'lucide-react';
import type { ConversationMessage } from '../../../lib/messages';

export function UnknownBubble({ message }: { message: ConversationMessage }) {
  return (
    <div className="message-bubble unknown">
      <div className="message-avatar">
        <Info size={16} />
      </div>
      <div className="message-body">
        <div className="message-text">Unknown message type: {message.type}</div>
      </div>
    </div>
  );
}
