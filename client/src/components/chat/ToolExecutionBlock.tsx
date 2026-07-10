/**
 * ToolExecutionBlock — Unified component for tool call + result display
 */

import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Search,
  Globe,
  Flag,
  StickyNote,
  History,
  MessageSquare,
  Languages,
  Brain,
  Trash2,
  LayoutGrid,
  FileText,
  Users,
  Wrench,
  type LucideIcon,
} from 'lucide-react';
import { useModal } from '../../contexts/ModalContext';
import { UserHistoryView } from '../memory/UserHistoryView';
import type { UserHistoryFact, UserHistoryTurn } from '../../lib/api';
import './ToolExecutionBlock.css';

/** Lead icon per internal tool — a glanceable type cue on the collapsed row. */
const TOOL_ICONS: Record<string, LucideIcon> = {
  web_search: Search,
  web_research: Globe,
  web_extract: Globe,
  web_map: Globe,
  web_crawl: Globe,
  checkpoint: Flag,
  scratchpad_note: StickyNote,
  recall_user_history: History,
  read_user_message: MessageSquare,
  translate_text: Languages,
  detect_language: Languages,
  remember_this: Brain,
  forget: Trash2,
  present_exhibit: LayoutGrid,
  read_stored_output: FileText,
  list_stored_outputs: FileText,
  tool_output_query: FileText,
  tool_output_section: FileText,
  tool_output_path: FileText,
  delegate_to: Users,
};

export function toolIconFor(toolName: string): LucideIcon {
  return TOOL_ICONS[toolName] ?? Wrench;
}

/** Tools whose `query` arg is the meaningful preview (shown verbatim, quoted). */
const QUERY_TOOLS = new Set(['web_search', 'web_research']);

/** The single most informative arg to show on the collapsed row. */
function primaryPreview(toolName: string, args: Record<string, unknown>): string {
  if (QUERY_TOOLS.has(toolName) && typeof args.query === 'string' && args.query.trim()) {
    return `"${args.query}"`;
  }
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  const [k, v] = entries[0];
  return `${k}: ${JSON.stringify(v).slice(0, 40)}`;
}

/** Result count for a web_search/web_research result, or null when unparseable. */
function resultCount(content: string | undefined): number | null {
  if (!content) return null;
  try {
    const data = JSON.parse(content);
    return Array.isArray(data?.results) ? data.results.length : null;
  } catch {
    return null;
  }
}

/**
 * Parse a `recall_user_history` tool result into turns + facts, or null when
 * the payload isn't the expected success shape (so we fall back to the generic
 * tool display).
 */
function parseUserHistory(content: string): {
  turns: UserHistoryTurn[];
  facts: UserHistoryFact[];
  topic: string | null;
  summary: string | null;
} | null {
  try {
    const data = JSON.parse(content);
    if (!data || data.success === false || !Array.isArray(data.user_turns)) return null;
    return {
      turns: data.user_turns as UserHistoryTurn[],
      facts: Array.isArray(data.facts) ? (data.facts as UserHistoryFact[]) : [],
      topic: data.topic ?? null,
      summary: data.summary ?? null,
    };
  } catch {
    return null;
  }
}

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

  // Render `recall_user_history` results as a readable card rather than a raw
  // JSON blob. Falls through to the generic block on parse failure.
  const userHistory =
    toolName === 'recall_user_history' && status === 'completed' && result?.content
      ? parseUserHistory(result.content)
      : null;

  if (userHistory) {
    return (
      <div className="tool-execution-block status-completed tool-recall-card">
        <div className="tool-execution-header" onClick={() => setExpanded(!expanded)}>
          <div className="tool-execution-icon">
            <History size={14} />
          </div>
          <div className="tool-execution-info">
            <span className="tool-execution-name">User history recall</span>
            <span className="tool-execution-meta">
              <span>
                {userHistory.turns.length} message{userHistory.turns.length === 1 ? '' : 's'}
              </span>
              {userHistory.facts.length > 0 && <span>{userHistory.facts.length} facts</span>}
            </span>
          </div>
          <button className="tool-execution-toggle">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        </div>
        {expanded && (
          <div className="tool-execution-details">
            <UserHistoryView
              turns={userHistory.turns}
              facts={userHistory.facts}
              topic={userHistory.topic}
              summary={userHistory.summary}
              compact
            />
          </div>
        )}
      </div>
    );
  }

  const preview = primaryPreview(toolName, args);
  const ToolIcon = toolIconFor(toolName);
  const isActive = status === 'pending' || status === 'running';
  const results = QUERY_TOOLS.has(toolName) ? resultCount(result?.content) : null;
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
          <ToolIcon size={14} className={isActive ? 'animate-pulse' : undefined} />
        </div>
        <div className="tool-execution-info">
          <span className="tool-execution-name">{toolName}</span>
          {preview && <span className="tool-execution-preview">{preview}</span>}
          <span className="tool-execution-meta">
            {isActive && <span>{statusInfo.label}</span>}
            {status === 'failed' && <span className="meta-failed">Failed</span>}
            {results !== null && <span>{results} result{results === 1 ? '' : 's'}</span>}
            {result?.durationMs !== undefined && <span>{formatDuration(result.durationMs)}</span>}
          </span>
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
      )}
    </div>
  );
}
