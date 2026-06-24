/**
 * AlloyRunTraceModal — inspect a completed/live Agent Alloy run.
 *
 * Static run *tracing* (not replay/re-run): groups the active conversation's
 * delegation messages into runs (see lib/alloyTrace) and renders a per-run
 * breakdown — supervisor, each delegated specialist, timing, tokens, cost, and
 * (for live runs) the specialist's tool calls.
 *
 * Tool-level detail is only available for live runs: the backend persists one
 * rollup turn per delegation, so a restored run shows delegation-level metrics
 * but no tool sub-list.
 */

import { useMemo, useState } from 'react';
import {
  X,
  Crown,
  Clock,
  Coins,
  ArrowRightLeft,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Wrench,
} from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { AgentAvatar } from '../common/AgentAvatar';
import { groupRunsFromMessages, type AlloyRun } from '../../lib/alloyTrace';
import type { AgentProfile } from '../../lib/api';
import type { DelegationMessage, DelegationToolEvent } from '../../lib/messages';
import './AlloyRunTraceModal.css';

interface AlloyRunTraceModalProps {
  onClose: () => void;
  runId?: string;
}

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatCost(cost: number | null | undefined, currency?: string | null): string | null {
  if (typeof cost !== 'number') return null;
  const sym = currency && currency !== 'USD' ? `${currency} ` : '$';
  // Small costs need more precision than two decimals.
  const digits = cost > 0 && cost < 0.01 ? 4 : cost < 1 ? 3 : 2;
  return `${sym}${cost.toFixed(digits)}`;
}

function formatTokens(n: number | undefined): string {
  if (!n) return '0';
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

export function AlloyRunTraceModal({ onClose, runId }: AlloyRunTraceModalProps) {
  const { activeTab } = useConversation();
  const { profiles } = useAgentProfile();
  const { getWorkflowById } = useAlloyWorkflow();

  const runs = useMemo(
    () => groupRunsFromMessages(activeTab?.messages ?? []),
    [activeTab?.messages],
  );

  const profilesByAgentId = useMemo(() => {
    const m = new Map<string, AgentProfile>();
    for (const p of profiles) m.set(p.agentId, p);
    return m;
  }, [profiles]);

  const workflow = activeTab?.workflowId ? getWorkflowById(activeTab.workflowId) : null;

  const [selectedId, setSelectedId] = useState<string>(
    () => runId ?? runs[runs.length - 1]?.id ?? '',
  );
  const selected = runs.find(r => r.id === selectedId) ?? runs[runs.length - 1] ?? null;

  return (
    <div className="alloy-trace-modal">
      <div className="alloy-trace-header">
        <div className="alloy-trace-title-group">
          <div className="alloy-trace-title-icon">
            <ArrowRightLeft size={18} />
          </div>
          <div>
            <h2>Run Trace</h2>
            <div className="alloy-trace-subtitle">
              {workflow ? `Workflow: ${workflow.name}` : 'Multi-agent delegation breakdown'}
            </div>
          </div>
        </div>
        <button type="button" className="alloy-trace-close" onClick={onClose} title="Close">
          <X size={18} />
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="alloy-trace-empty">
          <ArrowRightLeft size={40} />
          <h3>No delegations in this conversation</h3>
          <p>Run trace appears once a supervisor delegates to a specialist.</p>
        </div>
      ) : (
        <div className="alloy-trace-body">
          {runs.length > 1 && (
            <div className="alloy-trace-run-tabs">
              {runs.map((r, i) => (
                <button
                  key={r.id}
                  className={`alloy-trace-run-tab ${r.id === selected?.id ? 'active' : ''}`}
                  onClick={() => setSelectedId(r.id)}
                >
                  Run {i + 1}
                  <span className="run-tab-count">{r.totals.count}</span>
                </button>
              ))}
            </div>
          )}
          {selected && (
            <RunDetail
              run={selected}
              profilesByAgentId={profilesByAgentId}
              supervisorName={
                workflow
                  ? profilesByAgentId.get(workflow.supervisorAgentId)?.name ??
                    workflow.supervisorAgentId
                  : selected.supervisorAgentName
              }
            />
          )}
        </div>
      )}
    </div>
  );
}

function RunDetail({
  run,
  profilesByAgentId,
  supervisorName,
}: {
  run: AlloyRun;
  profilesByAgentId: Map<string, AgentProfile>;
  supervisorName?: string;
}) {
  const t = run.totals;
  const totalCost = formatCost(t.costEstimate, t.costCurrency);

  return (
    <div className="alloy-trace-run">
      {/* Run summary */}
      <div className="alloy-trace-summary">
        <div className="summary-supervisor">
          <Crown size={14} />
          <span>{supervisorName ?? 'Supervisor'}</span>
        </div>
        <div className="summary-stats">
          <span className="summary-stat" title="Delegations">
            <ArrowRightLeft size={13} />
            {t.count} delegation{t.count === 1 ? '' : 's'}
          </span>
          <span className="summary-stat" title="Tokens in / out">
            {formatTokens(t.tokensInput)} in · {formatTokens(t.tokensOutput)} out
          </span>
          {totalCost && (
            <span className="summary-stat cost" title="Estimated cost">
              <Coins size={13} />
              {totalCost}
              {t.costPartial && <span className="partial-flag" title="Mixed currencies">~</span>}
            </span>
          )}
          <span className="summary-stat" title="Wall-clock duration">
            <Clock size={13} />
            {formatDuration(t.wallClockMs)}
          </span>
        </div>
      </div>

      {/* Delegation cards */}
      <div className="alloy-trace-delegations">
        {run.delegations.map(d => (
          <DelegationTraceCard
            key={d.id}
            delegation={d}
            profile={profilesByAgentId.get(d.targetAgentId)}
          />
        ))}
      </div>
    </div>
  );
}

function DelegationTraceCard({
  delegation: d,
  profile,
}: {
  delegation: DelegationMessage;
  profile?: AgentProfile;
}) {
  const [showResult, setShowResult] = useState(false);
  const [showTools, setShowTools] = useState(false);

  const name = profile?.name ?? d.targetAgentName ?? d.targetAgentId;
  const cost = formatCost(d.costEstimate, d.costCurrency);
  const tools = d.toolEvents ?? [];

  return (
    <div className={`trace-delegation status-${d.status}`}>
      <div className="trace-delegation-head">
        <div className="trace-specialist">
          <span className="trace-specialist-avatar">
            <AgentAvatar avatar={profile?.avatar} size={15} />
          </span>
          <div>
            <div className="trace-specialist-name">{name}</div>
            <div className="trace-specialist-id">{d.targetAgentId}</div>
          </div>
        </div>
        <StatusBadge status={d.status} />
      </div>

      <div className="trace-task">{d.task}</div>

      <div className="trace-metrics">
        <span className="trace-metric" title="Duration">
          <Clock size={12} />
          {formatDuration(d.durationMs)}
        </span>
        <span className="trace-metric" title="Tokens in / out">
          {formatTokens(d.tokensInput)} / {formatTokens(d.tokensOutput)} tok
        </span>
        {cost && (
          <span className="trace-metric cost" title="Estimated cost">
            <Coins size={12} />
            {cost}
          </span>
        )}
        {tools.length > 0 && (
          <span className="trace-metric" title="Tool calls">
            <Wrench size={12} />
            {tools.length} tool{tools.length === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {d.error && <div className="trace-error">{d.error}</div>}

      {tools.length > 0 && (
        <div className="trace-tools">
          <button className="trace-toggle" onClick={() => setShowTools(v => !v)}>
            {showTools ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            Tool calls ({tools.length})
          </button>
          {showTools && (
            <div className="trace-tools-list">
              {tools.map((te, i) => (
                <ToolEventRow key={`${te.toolCallId}-${i}`} event={te} />
              ))}
            </div>
          )}
        </div>
      )}

      {(d.content || d.resultPreview) && (
        <div className="trace-result">
          <button className="trace-toggle" onClick={() => setShowResult(v => !v)}>
            {showResult ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            Result
          </button>
          {showResult && (
            <pre className="trace-result-body">{d.content || d.resultPreview}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function ToolEventRow({ event }: { event: DelegationToolEvent }) {
  return (
    <div className={`trace-tool-row status-${event.status}`}>
      <Wrench size={12} />
      <span className="trace-tool-name">{event.toolName}</span>
      <StatusBadge status={event.status} compact />
      {event.durationMs != null && (
        <span className="trace-tool-duration">{formatDuration(event.durationMs)}</span>
      )}
    </div>
  );
}

function StatusBadge({
  status,
  compact,
}: {
  status: 'streaming' | 'running' | 'completed' | 'failed';
  compact?: boolean;
}) {
  const map = {
    streaming: { icon: <Loader2 size={12} className="spin" />, label: 'Running', cls: 'running' },
    running: { icon: <Loader2 size={12} className="spin" />, label: 'Running', cls: 'running' },
    completed: { icon: <CheckCircle2 size={12} />, label: 'Done', cls: 'completed' },
    failed: { icon: <XCircle size={12} />, label: 'Failed', cls: 'failed' },
  } as const;
  const s = map[status];
  return (
    <span className={`trace-status-badge ${s.cls} ${compact ? 'compact' : ''}`}>
      {s.icon}
      {!compact && <span>{s.label}</span>}
    </span>
  );
}
