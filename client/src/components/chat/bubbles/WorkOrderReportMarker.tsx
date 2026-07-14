/**
 * WorkOrderReportMarker — hairline transcript row marking the moment a
 * background work order's report folded into the turn.
 *
 * Preserves narrative causality (the assistant reacts right after this point)
 * without duplicating content: the Work Order card and the trace console
 * carry the report itself. Clicking jumps to the console focused on the order.
 */

import { Undo2 } from 'lucide-react';
import { useModal } from '../../../contexts/ModalContext';
import type { BubbleProps } from './types';
import { ticketId } from './WorkOrderCard';
import './WorkOrderCard.css';

export function WorkOrderReportMarker({ message }: BubbleProps<'work_order_report'>) {
  const { openModal } = useModal();
  const name = message.targetAgentName || message.targetAgentId;
  const failed = message.status !== 'completed';

  return (
    <button
      type="button"
      className={`work-order-report-marker bg-transparent ${failed ? 'failed' : ''}`}
      title="Open this work order in the trace console"
      onClick={() =>
        openModal({
          id: 'alloy-run-trace',
          type: 'modal',
          component: 'alloyRunTrace',
          size: 'full',
          props: { delegationId: message.delegationId },
        })
      }
    >
      <span className="marker-line" aria-hidden="true" />
      <span className="marker-text">
        <Undo2 size={11} />
        {ticketId(message.delegationId)} report {failed ? `(${message.status})` : 'delivered'} — {name}
      </span>
      <span className="marker-line" aria-hidden="true" />
    </button>
  );
}
