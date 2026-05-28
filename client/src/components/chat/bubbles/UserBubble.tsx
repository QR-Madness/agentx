import { User, Edit3 } from 'lucide-react';
import type { BubbleProps } from './types';

export function UserBubble({ message, onEdit }: BubbleProps<'user'>) {
  return (
    <div className="message-bubble user">
      <div className="message-avatar user-avatar">
        <User size={16} />
      </div>
      <div className="message-body">
        <div className="user-header">
          <span className="user-name">You</span>
        </div>
        <div className="message-text">{message.content}</div>
        <div className="message-meta">
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {message.editedAt && <span className="message-edited">(edited)</span>}
        </div>
        {onEdit && (
          <button
            className="edit-button"
            onClick={() => onEdit(message.content)}
            title="Edit message"
          >
            <Edit3 size={12} />
          </button>
        )}
      </div>
    </div>
  );
}
