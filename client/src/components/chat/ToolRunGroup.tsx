/**
 * ToolRunGroup — a stack of consecutive tool calls rendered as ONE quiet
 * affordance. Collapsed: a single pill ("Ran 6 tools · 1 failed") with the
 * distinct tool icons overlapped. Expanded: the individual chips.
 *
 * Expansion policy (Claude-style, non-invasive unless something's wrong):
 * open while any call is still running or any failed; auto-collapses once the
 * run settles clean. The user's manual toggle always wins afterwards.
 */
import { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Loader2 } from 'lucide-react';
import type { ConversationMessage, ToolCallMessage } from '../../lib/messages';
import { ToolExecutionBlock, toolIconFor } from './ToolExecutionBlock';

export function ToolRunGroup({ messages }: { messages: ConversationMessage[] }) {
  const calls = messages.filter((m): m is ToolCallMessage => m.type === 'tool_call');
  const activeCount = calls.filter(m => m.status === 'pending' || m.status === 'running').length;
  const failedCount = calls.filter(m => m.status === 'failed').length;
  const busy = activeCount > 0;

  const [expanded, setExpanded] = useState(busy || failedCount > 0);
  const [userToggled, setUserToggled] = useState(false);
  const wasBusy = useRef(busy);

  // Auto-collapse when the run settles clean; auto-expand if a new call starts
  // — unless the user has taken over the toggle.
  useEffect(() => {
    if (userToggled) { wasBusy.current = busy; return; }
    if (wasBusy.current && !busy && failedCount === 0) setExpanded(false);
    if (!wasBusy.current && busy) setExpanded(true);
    wasBusy.current = busy;
  }, [busy, failedCount, userToggled]);

  const distinctIcons = [...new Set(calls.map(m => m.toolName))].slice(0, 4);
  const summary = busy
    ? `Running tools… ${calls.length - activeCount}/${calls.length}`
    : `Ran ${calls.length} tools`;

  return (
    <div className="tool-run-group">
      <button
        type="button"
        className="tool-run-group-header"
        aria-expanded={expanded}
        onClick={() => { setUserToggled(true); setExpanded(e => !e); }}
      >
        <span className="group-icons">
          {distinctIcons.map(name => {
            const Icon = toolIconFor(name);
            return <Icon key={name} size={13} />;
          })}
        </span>
        <span>{summary}</span>
        {failedCount > 0 && (
          <span className="group-failed">{failedCount} failed</span>
        )}
        {busy
          ? <Loader2 size={12} className="animate-spin" />
          : expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {expanded && (
        <div className="tool-run-group-body">
          {calls.map(m => (
            <ToolExecutionBlock
              key={m.id}
              toolName={m.toolName}
              toolCallId={m.toolCallId}
              arguments={m.arguments}
              status={m.status}
              result={m.result}
            />
          ))}
        </div>
      )}
    </div>
  );
}
