import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EntryModules, ENTRY_MODULES, type EntryModuleCtx } from './entryModules';
import type { AmbassadorToolCall } from '../../lib/api';

const ctx = (over: Partial<EntryModuleCtx> = {}): EntryModuleCtx => ({
  isLatest: true,
  resolutionVersion: 0,
  onConfirmProposal: vi.fn(),
  onDismissProposal: vi.fn(),
  ...over,
});

describe('entryModules', () => {
  it('registers tool chips and proposal strips, in that order', () => {
    expect(ENTRY_MODULES.map((m) => m.id)).toEqual(['tool-chips', 'proposal-strips']);
  });

  it('renders tool chips for plain read calls', () => {
    const calls: AmbassadorToolCall[] = [
      { tool: 'survey_conversations', done: true },
      { tool: 'read_conversation_results', done: false },
    ];
    render(<EntryModules entry={{ toolCalls: calls }} ctx={ctx()} />);
    expect(screen.getByText(/surveying your conversations/)).toBeInTheDocument();
    expect(screen.getByText(/checking what it produced/)).toBeInTheDocument();
  });

  it('renders a live confirm strip for a proposal on the latest entry', () => {
    const calls: AmbassadorToolCall[] = [
      {
        tool: 'dispatch_task',
        done: true,
        proposal: {
          proposal_id: 'prop_test1',
          action: 'dispatch',
          agent_id: 'bold-atlas',
          agent_name: 'Atlas',
          task: 'Survey the archive of open questions.',
        },
      },
    ];
    render(<EntryModules entry={{ toolCalls: calls }} ctx={ctx()} />);
    expect(screen.getByText(/Dispatch to Atlas/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dispatch' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dismiss' })).toBeInTheDocument();
  });

  it('collapses a stale proposal on an older entry to a passive line', () => {
    const calls: AmbassadorToolCall[] = [
      {
        tool: 'rename_conversation',
        done: true,
        proposal: {
          proposal_id: 'prop_test2',
          action: 'rename',
          conversation_id: 'c1',
          title: 'First Principles Review',
        },
      },
    ];
    render(<EntryModules entry={{ toolCalls: calls }} ctx={ctx({ isLatest: false })} />);
    expect(screen.getByText(/proposal expired/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Confirm' })).toBeNull();
  });

  it('renders nothing for an entry with no module payloads', () => {
    const { container } = render(<EntryModules entry={{}} ctx={ctx()} />);
    expect(container.textContent).toBe('');
  });
});
