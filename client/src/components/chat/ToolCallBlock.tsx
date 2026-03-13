/**
 * ToolCallBlock — Renders a tool call message with collapsible arguments
 */

import { useState } from 'react';
import { Wrench, ChevronDown, ChevronRight, Clock, Check, X, Loader2 } from 'lucide-react';
import './ToolCallBlock.css';

export interface ToolCallBlockProps {
  toolName: string;
  toolCallId: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'approved' | 'rejected' | 'completed';
}

const STATUS_CONFIG = {
  pending: { icon: Clock, label: 'Pending', className: 'status-pending' },
  approved: { icon: Loader2, label: 'Running', className: 'status-approved' },
  rejected: { icon: X, label: 'Rejected', className: 'status-rejected' },
  completed: { icon: Check, label: 'Done', className: 'status-completed' },
};

export function ToolCallBlock({ toolName, toolCallId, arguments: args, status }: ToolCallBlockProps) {
  const [expanded, setExpanded] = useState(false);
  const statusInfo = STATUS_CONFIG[status];
  const StatusIcon = statusInfo.icon;

  const argCount = Object.keys(args).length;
  const argsPreview = argCount > 0
    ? Object.entries(args).slice(0, 2).map(([k, v]) => `${k}: ${JSON.stringify(v).slice(0, 20)}`).join(', ')
    : 'No arguments';

  return (
    <div className={`tool-call-block ${statusInfo.className}`}>
      <div className="tool-call-header" onClick={() => setExpanded(!expanded)}>
        <div className="tool-call-icon">
          <Wrench size={14} />
        </div>
        <div className="tool-call-info">
          <span className="tool-call-name">{toolName}</span>
          {!expanded && (
            <span className="tool-call-preview">({argsPreview})</span>
          )}
        </div>
        <div className={`tool-call-status ${statusInfo.className}`}>
          <StatusIcon size={12} className={status === 'approved' ? 'spin' : ''} />
          <span>{statusInfo.label}</span>
        </div>
        <button className="tool-call-toggle">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {expanded && (
        <div className="tool-call-details">
          <div className="tool-call-id">
            <span className="label">Call ID:</span>
            <code>{toolCallId}</code>
          </div>
          <div className="tool-call-args">
            <span className="label">Arguments:</span>
            <pre className="args-json">{JSON.stringify(args, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
