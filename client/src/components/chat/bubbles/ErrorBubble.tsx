import { AlertTriangle } from 'lucide-react';
import type { BubbleProps } from './types';

export function ErrorBubble({ message }: BubbleProps<'error'>) {
  return (
    <div className="message-bubble error">
      <div className="error-content">
        <div className="error-header">
          <AlertTriangle size={16} />
          <span>Error</span>
          {message.recoverable && (
            <span className="recoverable-badge">Recoverable</span>
          )}
        </div>
        <div className="error-message">{message.content}</div>
      </div>
    </div>
  );
}
