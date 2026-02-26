import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Lightbulb } from 'lucide-react';
import './ThinkingBubble.css';

interface ThinkingBubbleProps {
  thinking: string;
  defaultExpanded?: boolean;
  isStreaming?: boolean;
}

export const ThinkingBubble: React.FC<ThinkingBubbleProps> = ({
  thinking,
  defaultExpanded = false,
  isStreaming = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded || isStreaming);

  if (!thinking) return null;

  // Count approximate thinking steps (paragraphs or numbered items)
  const stepCount = thinking.split(/\n\n+/).filter(s => s.trim()).length;

  return (
    <div className={`thinking-bubble ${isExpanded ? 'expanded' : ''} ${isStreaming ? 'streaming' : ''}`}>
      <button
        className="thinking-header"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className="thinking-header-left">
          {isExpanded ? (
            <ChevronDown size={16} className="thinking-chevron" />
          ) : (
            <ChevronRight size={16} className="thinking-chevron" />
          )}
          <Lightbulb size={16} className={`thinking-icon ${isStreaming ? 'pulse' : ''}`} />
          <span className="thinking-label">{isStreaming ? 'Thinking...' : 'Thinking'}</span>
        </div>
        {!isStreaming && (
          <span className="thinking-meta">
            {stepCount} {stepCount === 1 ? 'step' : 'steps'}
          </span>
        )}
      </button>
      
      {isExpanded && (
        <div className="thinking-content">
          {thinking.split(/\n\n+/).map((paragraph, idx) => (
            <p key={idx} className="thinking-paragraph">
              {paragraph}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};

export default ThinkingBubble;
