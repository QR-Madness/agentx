/**
 * ToolExecutionBlock — Unified component for tool call + result display
 */

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Loader2,
  CheckCircle,
  XCircle,
  ExternalLink,
} from 'lucide-react';
import { useModal } from '../../contexts/ModalContext';
import './ToolExecutionBlock.css';

export interface ToolExecutionBlockProps {
  toolName: string;
  toolCallId: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: {
    content: string;
    success: boolean;
    durationMs?: number;
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

const STATUS_CONFIG = {
  pending: { label: 'Pending', className: 'status-pending' },
  running: { label: 'Running', className: 'status-running' },
  completed: { label: 'Done', className: 'status-completed' },
  failed: { label: 'Failed', className: 'status-failed' },
};

export function ToolExecutionBlock({
  toolName,
  toolCallId,
  arguments: args,
  status,
  result,
}: ToolExecutionBlockProps) {
  const [expanded, setExpanded] = useState(false);
  const { openModal } = useModal();
  const statusInfo = STATUS_CONFIG[status];

  const argCount = Object.keys(args).length;
  const argsPreview =
    argCount > 0
      ? Object.entries(args)
          .slice(0, 2)
          .map(([k, v]) => `${k}: ${JSON.stringify(v).slice(0, 20)}`)
          .join(', ')
      : 'No arguments';

  const contentSize = result?.content ? formatBytes(result.content.length) : null;

  const handleViewOutput = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (result?.content) {
      openModal({
        type: 'drawer',
        component: 'toolOutput',
        size: 'lg',
        props: {
          toolName,
          content: result.content,
          success: result.success,
          durationMs: result.durationMs,
        },
      });
    }
  };

  return (
    <div className={`tool-execution-block ${statusInfo.className}`}>
      <div className="tool-execution-header" onClick={() => setExpanded(!expanded)}>
        <div className="tool-execution-icon">
          {status === 'pending' && <Loader2 size={14} className="animate-pulse" />}
          {status === 'running' && <Loader2 size={14} className="animate-spin" />}
          {status === 'completed' && <CheckCircle size={14} />}
          {status === 'failed' && <XCircle size={14} />}
        </div>
        <div className="tool-execution-info">
          <span className="tool-execution-name">{toolName}</span>
          {!expanded && <span className="tool-execution-preview">({argsPreview})</span>}
        </div>
        <div className={`tool-execution-status ${statusInfo.className}`}>
          <span>{statusInfo.label}</span>
        </div>
        <button className="tool-execution-toggle">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      {expanded && (
        <div className="tool-execution-details">
          <div className="tool-execution-id">
            <span className="label">Call ID:</span>
            <code>{toolCallId.slice(0, 16)}...</code>
          </div>
          <div className="tool-execution-args">
            <span className="label">Arguments:</span>
            <pre className="args-json">{JSON.stringify(args, null, 2)}</pre>
          </div>
        </div>
      )}

      {(status === 'completed' || status === 'failed') && result && (
        <div className="tool-execution-result">
          <div className="result-summary">
            {result.success ? (
              <span className="result-success">Success</span>
            ) : (
              <span className="result-failure">Failed</span>
            )}
            {result.durationMs !== undefined && (
              <span className="result-duration">{formatDuration(result.durationMs)}</span>
            )}
            {contentSize && <span className="result-size">{contentSize}</span>}
          </div>
          <button className="view-output-btn" onClick={handleViewOutput}>
            <ExternalLink size={12} />
            <span>View Output</span>
          </button>
        </div>
      )}
    </div>
  );
}
