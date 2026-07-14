/**
 * AlloyRunTraceModal — the Work Console: inspect a completed/live Agent Alloy
 * run. User-facing name: "Agent Teams" (internal: Alloy).
 *
 * Master–detail: a left rail lists the run's work orders as a tree (the
 * "filesystem" of delegated work — flat today, nesting ships with the chain of
 * command), the right pane shows the focused delegation in depth. Deep-links
 * via the `delegationId` prop (Work Order cards in the transcript open the
 * console focused on themselves). Static run *tracing* (not replay/re-run) —
 * pure over the conversation messages, identical live and restored.
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
  Ban,
  Copy,
  Loader2,
  ChevronDown,
  ChevronRight,
  Wrench,
  Undo2,
} from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { AgentAvatar } from '../common/AgentAvatar';
import {
  buildDelegationTree,
  groupRunsFromMessages,
  type AlloyRun,
  type DelegationNode,
} from '../../lib/alloyTrace';
import type { AgentProfile } from '../../lib/api';
import type { DelegationMessage, DelegationToolEvent } from '../../lib/messages';
import './AlloyRunTraceModal.css';

interface AlloyRunTraceModalProps {
  onClose: () => void;
  runId?: string;
  /** Deep-link: open focused on this work order (resolves its run itself). */
  delegationId?: string;
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

/** Short work-order ticket id, e.g. `wo·3f2a`. */
function ticketId(delegationId: string): string {
  return `wo·${delegationId.slice(0, 4)}`;
}

export function AlloyRunTraceModal({ onClose, runId, delegationId }: AlloyRunTraceModalProps) {
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

  const [selectedId, setSelectedId] = useState<string>(() => {
    if (delegationId) {
      const owner = runs.find(r => r.delegations.some(d => d.delegationId === delegationId));
      if (owner) return owner.id;
    }
    return runId ?? runs[runs.length - 1]?.id ?? '';
  });
  const selected = runs.find(r => r.id === selectedId) ?? runs[runs.length - 1] ?? null;
  const selectedIndex = selected ? runs.indexOf(selected) : -1;

  // Focused work order within the selected run; falls back to the first.
  const [focusedDelegationId, setFocusedDelegationId] = useState<string | null>(
    delegationId ?? null,
  );
  const focused =
    selected?.delegations.find(d => d.delegationId === focusedDelegationId) ??
    selected?.delegations[0] ??
    null;

  const selectRun = (id: string) => {
    setSelectedId(id);
    setFocusedDelegationId(null); // fall back to the run's first work order
  };

  return (
    <div className="alloy-trace-modal">
      <div className="alloy-trace-header">
        <div className="alloy-trace-title-group">
          <div className="alloy-trace-title-icon">
            <ArrowRightLeft size={18} />
          </div>
          <div>
            <h2>Team Run Trace</h2>
            <div className="alloy-trace-subtitle">
              {workflow ? `Team: ${workflow.name}` : 'Work orders & delegations for this conversation'}
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
          <h3>No work orders in this conversation yet</h3>
          <p>The trace fills in once an agent delegates work to a teammate.</p>
        </div>
      ) : (
        <div className="alloy-trace-body">
          {runs.length > 1 && (
            <div className="alloy-trace-run-tabs">
              {runs.map((r, i) => (
                <button
                  key={r.id}
                  className={`alloy-trace-run-tab ${r.id === selected?.id ? 'active' : ''}`}
                  onClick={() => selectRun(r.id)}
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
              runNumber={selectedIndex + 1}
              focused={focused}
              onFocus={setFocusedDelegationId}
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
  runNumber,
  focused,
  onFocus,
  profilesByAgentId,
  supervisorName,
}: {
  run: AlloyRun;
  runNumber: number;
  focused: DelegationMessage | null;
  onFocus: (delegationId: string) => void;
  profilesByAgentId: Map<string, AgentProfile>;
  supervisorName?: string;
}) {
  const t = run.totals;
  const totalCost = formatCost(t.costEstimate, t.costCurrency);
  const tree = useMemo(() => buildDelegationTree(run.delegations), [run.delegations]);

  return (
    <div className="alloy-trace-run">
      {/* Run summary */}
      <div className="alloy-trace-summary">
        <div className="summary-supervisor">
          <Crown size={14} />
          <span>{supervisorName ?? 'Lead'}</span>
        </div>
        <div className="summary-stats">
          <span className="summary-stat" title="Work orders / delegations">
            <ArrowRightLeft size={13} />
            {t.count} work order{t.count === 1 ? '' : 's'}
          </span>
          <span className="summary-stat" title="Tokens in / out">
            {formatTokens(t.tokensInput)} in · {formatTokens(t.tokensOutput)} out
          </span>
          {totalCost ? (
            <span className="summary-stat cost" title="Estimated cost">
              <Coins size={13} />
              {totalCost}
              {t.costPartial && (
                <span className="partial-flag" title="Mixed currencies — approximate sum">~</span>
              )}
            </span>
          ) : (
            <span className="summary-stat unavailable" title="No pricing data for the models used">
              <Coins size={13} />
              Pricing unavailable
            </span>
          )}
          <span className="summary-stat" title="Wall-clock duration">
            <Clock size={13} />
            {formatDuration(t.wallClockMs)}
          </span>
        </div>
      </div>

      {/* Master–detail: work-order rail + focused delegation */}
      <div className="alloy-trace-master">
        <div className="alloy-trace-rail" role="listbox" aria-label="Work orders">
          {tree.map(node => (
            <RailNode
              key={node.delegation.delegationId}
              node={node}
              depth={0}
              focusedId={focused?.delegationId}
              onFocus={onFocus}
              profilesByAgentId={profilesByAgentId}
            />
          ))}
        </div>
        <div className="alloy-trace-detail">
          {focused && (
            <>
              <div className="trace-breadcrumb">
                <span className="trace-breadcrumb-path">
                  Run {runNumber} / <span className="trace-ticket">{ticketId(focused.delegationId)}</span>
                  {' — '}
                  {profilesByAgentId.get(focused.targetAgentId)?.name ??
                    focused.targetAgentName ?? focused.targetAgentId}
                </span>
                <button
                  type="button"
                  className="trace-copy-id"
                  title="Copy work order id"
                  onClick={() => void navigator.clipboard?.writeText(focused.delegationId)}
                >
                  <Copy size={12} />
                  {focused.delegationId}
                </button>
              </div>
              <DelegationTraceCard
                key={focused.delegationId}
                delegation={focused}
                profile={profilesByAgentId.get(focused.targetAgentId)}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function RailNode({
  node,
  depth,
  focusedId,
  onFocus,
  profilesByAgentId,
}: {
  node: DelegationNode;
  depth: number;
  focusedId?: string;
  onFocus: (delegationId: string) => void;
  profilesByAgentId: Map<string, AgentProfile>;
}) {
  const d = node.delegation;
  const profile = profilesByAgentId.get(d.targetAgentId);
  const name = profile?.name ?? d.targetAgentName ?? d.targetAgentId;
  const cost = formatCost(d.costEstimate, d.costCurrency);
  return (
    <>
      <button
        type="button"
        role="option"
        aria-selected={d.delegationId === focusedId}
        className={`alloy-trace-rail-item status-${d.status} ${d.delegationId === focusedId ? 'active' : ''}`}
        style={depth > 0 ? { marginLeft: `${depth * 14}px` } : undefined}
        onClick={() => onFocus(d.delegationId)}
      >
        <span className="rail-avatar">
          <AgentAvatar avatar={profile?.avatar} size={13} />
        </span>
        <span className="rail-main">
          <span className="rail-name">
            {name}
            {d.mode === 'background' && <span className="rail-mode-tag">background</span>}
          </span>
          <span className="rail-ticket">{ticketId(d.delegationId)}</span>
        </span>
        <span className="rail-meta">
          {cost && <span className="rail-cost">{cost}</span>}
          <StatusBadge status={d.status} compact />
        </span>
      </button>
      {node.children.map(child => (
        <RailNode
          key={child.delegation.delegationId}
          node={child}
          depth={depth + 1}
          focusedId={focusedId}
          onFocus={onFocus}
          profilesByAgentId={profilesByAgentId}
        />
      ))}
    </>
  );
}

function DelegationTraceCard({
  delegation: d,
  profile,
}: {
  delegation: DelegationMessage;
  profile?: AgentProfile;
}) {
  // The focused detail pane opens with everything visible; the collapsibles
  // remain for taming very long output.
  const [showResult, setShowResult] = useState(true);
  const [showTools, setShowTools] = useState(true);

  const name = profile?.name ?? d.targetAgentName ?? d.targetAgentId;
  const cost = formatCost(d.costEstimate, d.costCurrency);
  const hasTokens = (d.tokensInput ?? 0) > 0 || (d.tokensOutput ?? 0) > 0;
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
        <div className="trace-head-badges">
          {d.reportDelivered && (
            <span className="trace-report-chip" title="This work order's report was delivered to the delegating agent">
              <Undo2 size={11} />
              Report delivered
            </span>
          )}
          <StatusBadge status={d.status} />
        </div>
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
        {cost ? (
          <span
            className="trace-metric cost"
            title={d.pricingSnapshot ? JSON.stringify(d.pricingSnapshot) : 'Estimated cost'}
          >
            <Coins size={12} />
            {cost}
          </span>
        ) : hasTokens ? (
          <span className="trace-metric unavailable" title="No pricing data for this model">
            <Coins size={12} />
            Pricing unavailable
          </span>
        ) : null}
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
  status: DelegationMessage['status'] | DelegationToolEvent['status'];
  compact?: boolean;
}) {
  const map = {
    streaming: { icon: <Loader2 size={12} className="spin" />, label: 'Running', cls: 'running' },
    running: { icon: <Loader2 size={12} className="spin" />, label: 'Running', cls: 'running' },
    completed: { icon: <CheckCircle2 size={12} />, label: 'Done', cls: 'completed' },
    failed: { icon: <XCircle size={12} />, label: 'Failed', cls: 'failed' },
    cancelled: { icon: <Ban size={12} />, label: 'Cancelled', cls: 'cancelled' },
  } as const;
  // Unknown/future statuses degrade to the running treatment.
  const s = map[status as keyof typeof map] ?? map.running;
  return (
    <span className={`trace-status-badge ${s.cls} ${compact ? 'compact' : ''}`}>
      {s.icon}
      {!compact && <span>{s.label}</span>}
    </span>
  );
}
