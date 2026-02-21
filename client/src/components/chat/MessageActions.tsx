import React, { useState } from 'react';
import { Copy, RotateCcw, Check } from 'lucide-react';
import './MessageActions.css';

interface MessageActionsProps {
  content: string;
  isAssistant: boolean;
  timestamp: Date;
  onRegenerate?: () => void;
}

/**
 * Formats a timestamp to a relative time string
 */
function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) {
    return 'Just now';
  } else if (diffMins < 60) {
    return `${diffMins} min${diffMins !== 1 ? 's' : ''} ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  } else if (diffDays === 1) {
    return 'Yesterday';
  } else if (diffDays < 7) {
    return `${diffDays} days ago`;
  } else {
    return date.toLocaleDateString();
  }
}

export const MessageActions: React.FC<MessageActionsProps> = ({
  content,
  isAssistant,
  timestamp,
  onRegenerate,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="message-actions-container">
      <span className="message-timestamp-hover">
        {formatRelativeTime(timestamp)}
      </span>
      <div className="message-actions">
        <button
          className="message-action-btn"
          onClick={handleCopy}
          title={copied ? 'Copied!' : 'Copy message'}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
        {isAssistant && onRegenerate && (
          <button
            className="message-action-btn"
            onClick={onRegenerate}
            title="Regenerate response"
          >
            <RotateCcw size={14} />
          </button>
        )}
      </div>
    </div>
  );
};
