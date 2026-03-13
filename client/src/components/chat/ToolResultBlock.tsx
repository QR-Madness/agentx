/**
 * ToolResultBlock — Renders a tool result message with success/failure indicator
 */

import { useState } from 'react';
import { Check, X, ChevronDown, ChevronRight, Clock } from 'lucide-react';
import './ToolResultBlock.css';

export interface ToolResultBlockProps {
  toolName: string;
  toolCallId: string;
  content: string;
  success: boolean;
  durationMs?: number;
}

export function ToolResultBlock({ toolName, toolCallId, content, success, durationMs }: ToolResultBlockProps) {
  const [expanded, setExpanded] = useState(content.length > 200);

  const preview = content.length > 200 ? content.slice(0, 200) + '...' : content;
  const needsExpansion = content.length > 200;

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className={`tool-result-block ${success ? 'success' : 'failure'}`}>
      <div className="tool-result-header">
        <div className={`tool-result-status ${success ? 'success' : 'failure'}`}>
          {success ? <Check size={12} /> : <X size={12} />}
        </div>
        <div className="tool-result-info">
          <span className="tool-result-name">{toolName}</span>
          <span className="tool-result-label">{success ? 'completed' : 'failed'}</span>
        </div>
        {durationMs !== undefined && (
          <div className="tool-result-duration">
            <Clock size={10} />
            <span>{formatDuration(durationMs)}</span>
          </div>
        )}
        {needsExpansion && (
          <button className="tool-result-toggle" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        )}
      </div>

      <div className="tool-result-content">
        <pre>{expanded ? content : preview}</pre>
        {needsExpansion && !expanded && (
          <button className="expand-button" onClick={() => setExpanded(true)}>
            Show full result
          </button>
        )}
      </div>

      <div className="tool-result-footer">
        <span className="tool-result-id">
          Call: <code>{toolCallId.slice(0, 12)}...</code>
        </span>
      </div>
    </div>
  );
}
