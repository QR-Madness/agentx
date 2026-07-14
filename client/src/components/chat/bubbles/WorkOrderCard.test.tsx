import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { WorkOrderCard } from './WorkOrderCard';
import { WorkOrderReportMarker } from './WorkOrderReportMarker';
import type { ConversationMessage } from '../../../lib/messages';

const openModal = vi.fn();
vi.mock('../../../contexts/ModalContext', () => ({
  useModal: () => ({ openModal }),
}));

type Delegation = Extract<ConversationMessage, { type: 'delegation' }>;
type Marker = Extract<ConversationMessage, { type: 'work_order_report' }>;

function delegation(p: Partial<Delegation> = {}): Delegation {
  return {
    id: 'm1',
    type: 'delegation',
    timestamp: new Date().toISOString(),
    delegationId: 'abc12345',
    targetAgentId: 'beta',
    targetAgentName: 'Beta',
    task: 'research the topic',
    depth: 1,
    status: 'completed',
    content: 'findings',
    ...p,
  };
}

beforeEach(() => openModal.mockClear());

describe('WorkOrderCard', () => {
  it('renders the work-order row: label, ticket, target, status', () => {
    render(<WorkOrderCard message={delegation()} />);
    expect(screen.getByText('Work Order')).toBeInTheDocument();
    expect(screen.getByText('wo·abc1')).toBeInTheDocument();
    expect(screen.getByText('→ Beta')).toBeInTheDocument();
    expect(screen.getByText('Done')).toBeInTheDocument();
  });

  it('tags background work orders and shows the metrics strip when terminal', () => {
    render(<WorkOrderCard message={delegation({
      mode: 'background',
      durationMs: 2500,
      tokensInput: 1200,
      tokensOutput: 400,
      costEstimate: 0.0123,
      costCurrency: 'USD',
    })} />);
    expect(screen.getByText('background')).toBeInTheDocument();
    expect(screen.getByText('2.5s')).toBeInTheDocument();
    expect(screen.getByText('1.2k/400 tok')).toBeInTheDocument();
    expect(screen.getByText(/\$0\.012/)).toBeInTheDocument();
  });

  it('shows an honest pricing-unavailable state instead of omitting cost', () => {
    render(<WorkOrderCard message={delegation({
      tokensInput: 100, tokensOutput: 50, costEstimate: null,
    })} />);
    expect(screen.getByText('n/a')).toBeInTheDocument();
  });

  it('renders the cancelled state', () => {
    const { container } = render(
      <WorkOrderCard message={delegation({ status: 'cancelled' })} />,
    );
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
    expect(container.querySelector('.work-order-card.status-cancelled')).not.toBeNull();
  });

  it('opens the trace console focused on this work order on row click', () => {
    render(<WorkOrderCard message={delegation()} />);
    fireEvent.click(screen.getByTitle('Open in the trace console'));
    expect(openModal).toHaveBeenCalledWith(
      expect.objectContaining({
        component: 'alloyRunTrace',
        props: { delegationId: 'abc12345' },
      }),
    );
  });

  it('expands an inline preview without opening the console', () => {
    render(<WorkOrderCard message={delegation()} />);
    fireEvent.click(screen.getByTitle('Expand preview'));
    expect(screen.getByText('research the topic')).toBeInTheDocument();
    expect(openModal).not.toHaveBeenCalled();
  });
});

describe('WorkOrderReportMarker', () => {
  const marker: Marker = {
    id: 'mk1',
    type: 'work_order_report',
    timestamp: new Date().toISOString(),
    delegationId: 'abc12345',
    targetAgentId: 'beta',
    targetAgentName: 'Beta',
    status: 'completed',
  };

  it('renders the hairline delivered line', () => {
    render(<WorkOrderReportMarker message={marker} />);
    expect(screen.getByText(/wo·abc1 report delivered — Beta/)).toBeInTheDocument();
  });

  it('names the status when the order did not complete', () => {
    render(<WorkOrderReportMarker message={{ ...marker, status: 'failed' }} />);
    expect(screen.getByText(/wo·abc1 report \(failed\) — Beta/)).toBeInTheDocument();
  });

  it('clicks through to the trace console', () => {
    render(<WorkOrderReportMarker message={marker} />);
    fireEvent.click(screen.getByRole('button'));
    expect(openModal).toHaveBeenCalledWith(
      expect.objectContaining({ props: { delegationId: 'abc12345' } }),
    );
  });
});
