/**
 * WorkOrderCard — the holographic transcript artifact for a delegated task.
 *
 * The conversation is a report stream; the work lives elsewhere. This card is
 * the *receipt*: one compact ToolExecutionBlock-family row per work order
 * (blocking `delegate_to` and background `delegate_start` alike) showing who,
 * status, and — once terminal — the metrics strip. Clicking the row opens the
 * trace console focused on this work order; the chevron expands an inline
 * preview (task, specialist tool calls, live output tail).
 *
 * Background orders live-update in place: dispatched → working (shimmer) →
 * report delivered (one-time glow pulse) — the same card, no new bubbles.
 */

import { useState } from 'react';
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Coins,
  Loader2,
  Undo2,
  Users,
  XCircle,
} from 'lucide-react';
import { useModal } from '../../../contexts/ModalContext';
import { formatCost } from '../../../lib/format';
import { MessageContent } from '../MessageContent';
import { ToolExecutionBlock } from '../ToolExecutionBlock';
import type { BubbleProps } from './types';
import './WorkOrderCard.css';

function formatDuration(ms: number | null | undefined): string | null {
  if (ms == null) return null;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTokens(n: number | undefined): string {
  if (!n) return '0';
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;
}

/** Short work-order ticket id, e.g. `wo·3f2a`. */
export function ticketId(delegationId: string): string {
  return `wo·${delegationId.slice(0, 4)}`;
}

const STATUS_META = {
  streaming: { icon: Loader2, label: 'Working', cls: 'running', spin: true },
  completed: { icon: CheckCircle2, label: 'Done', cls: 'completed', spin: false },
  failed: { icon: XCircle, label: 'Failed', cls: 'failed', spin: false },
  cancelled: { icon: Ban, label: 'Cancelled', cls: 'cancelled', spin: false },
} as const;

export function WorkOrderCard({ message }: BubbleProps<'delegation'>) {
  const [expanded, setExpanded] = useState(false);
  const { openModal } = useModal();

  const targetName = message.targetAgentName || message.targetAgentId;
  const meta = STATUS_META[message.status] ?? STATUS_META.streaming;
  const StatusIcon = meta.icon;
  const terminal = message.status !== 'streaming';
  const duration = formatDuration(message.durationMs);
  const cost = typeof message.costEstimate === 'number'
    ? formatCost(message.costEstimate, message.costCurrency ?? 'USD')
    : null;
  const hasTokens = (message.tokensInput ?? 0) > 0 || (message.tokensOutput ?? 0) > 0;

  const openConsole = () => {
    openModal({
      id: 'alloy-run-trace',
      type: 'modal',
      component: 'alloyRunTrace',
      size: 'full',
      props: { delegationId: message.delegationId },
    });
  };

  return (
    <div className="message-bubble delegation">
      <div
        className={[
          'work-order-card',
          `status-${message.status}`,
          message.reportDelivered ? 'report-delivered' : '',
        ].join(' ')}
      >
        <div
          className="work-order-row"
          role="button"
          tabIndex={0}
          title="Open in the trace console"
          onClick={openConsole}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') openConsole(); }}
        >
          <span className="work-order-icon"><Users size={13} /></span>
          <span className="work-order-label">Work Order</span>
          <span className="work-order-ticket">{ticketId(message.delegationId)}</span>
          <span className="work-order-target">→ {targetName}</span>
          {message.mode === 'background' && (
            <span className="work-order-mode-tag">background</span>
          )}
          {message.depth > 1 && (
            <span className="work-order-depth">depth {message.depth}</span>
          )}
          <span className="work-order-meta">
            {terminal && duration && (
              <span className="work-order-metric" title="Duration">
                <Clock size={11} />{duration}
              </span>
            )}
            {terminal && hasTokens && (
              <span className="work-order-metric" title="Tokens in / out">
                {formatTokens(message.tokensInput)}/{formatTokens(message.tokensOutput)} tok
              </span>
            )}
            {terminal && (cost ? (
              <span className="work-order-metric cost" title="Estimated cost">
                <Coins size={11} />{cost}
              </span>
            ) : hasTokens ? (
              <span className="work-order-metric unavailable" title="No pricing data for this model">
                <Coins size={11} />n/a
              </span>
            ) : null)}
            {message.reportDelivered && (
              <span
                className="work-order-report-chip"
                title="Report delivered to the delegating agent"
              >
                <Undo2 size={11} />
              </span>
            )}
            <span className={`work-order-status ${meta.cls}`}>
              <StatusIcon size={11} className={meta.spin ? 'spin' : ''} />
              <span>{meta.label}</span>
            </span>
          </span>
          <button
            type="button"
            className="work-order-toggle bg-transparent"
            title={expanded ? 'Collapse preview' : 'Expand preview'}
            onClick={e => { e.stopPropagation(); setExpanded(v => !v); }}
          >
            {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          </button>
        </div>

        {expanded && (
          <div className="work-order-details">
            {message.task && <div className="work-order-task">{message.task}</div>}
            {message.toolEvents && message.toolEvents.length > 0 && (
              <div className="work-order-tools">
                {message.toolEvents.map(evt => (
                  <ToolExecutionBlock
                    key={evt.toolCallId}
                    toolName={evt.toolName}
                    toolCallId={evt.toolCallId}
                    arguments={evt.arguments ?? {}}
                    status={evt.status}
                    result={evt.content !== undefined ? {
                      content: evt.content,
                      success: evt.success ?? true,
                      durationMs: evt.durationMs,
                    } : undefined}
                  />
                ))}
              </div>
            )}
            {message.content && (
              <div className="work-order-body">
                <MessageContent content={message.content} />
              </div>
            )}
            {message.error && (
              <div className="work-order-error">
                <AlertTriangle size={12} /> {message.error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
