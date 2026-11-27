import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { MemoryEntity, MemoryFact } from '../../lib/api';
import { MemoryWorkbench } from './MemoryWorkbench';

// The workbench's data hooks stay real and only the API beneath them is stubbed,
// so each refetch shows up as a second call to the matching list endpoint.
const api = vi.hoisted(() => ({
  consolidateNow: vi.fn(),
  getMemoryStats: vi.fn(),
  listMemoryChannels: vi.fn(),
  listMemoryEntities: vi.fn(),
  listMemoryFacts: vi.fn(),
  listMemoryStrategies: vi.fn(),
  listMemoryProcedures: vi.fn(),
}));

vi.mock('../../lib/api', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  api,
}));

vi.mock('../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

vi.mock('../ui/ConfirmDialog', () => ({ useConfirm: () => vi.fn().mockResolvedValue(false) }));

const entity: MemoryEntity = {
  id: 'e1', name: 'Ada Lovelace', type: 'person', channel: '_default',
  salience: 0.8, last_accessed: '2026-11-26T12:00:00Z', access_count: 1,
};

const fact: MemoryFact = {
  id: 'f1', claim: 'Rivers carve canyons over millennia', confidence: 0.9, source: 'conversation',
  channel: '_default', created_at: '2026-11-26T12:00:00Z', entity_ids: ['e1'],
};

/** The four lists the workbench keeps in state whichever tab is open. */
const LIST_FETCHERS = [
  api.listMemoryEntities, api.listMemoryFacts, api.listMemoryStrategies, api.listMemoryProcedures,
];

function page<K extends string, T>(key: K, items: T[]) {
  return { [key]: items, total: items.length, has_next: false };
}

beforeEach(() => {
  vi.resetAllMocks();
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false, media: query, onchange: null,
    addEventListener: vi.fn(), removeEventListener: vi.fn(),
    addListener: vi.fn(), removeListener: vi.fn(), dispatchEvent: vi.fn(),
  }));
  api.getMemoryStats.mockResolvedValue({
    totals: { entities: 0, facts: 0, strategies: 0, turns: 4 }, by_channel: {},
  });
  api.listMemoryChannels.mockResolvedValue({ channels: [] });
  // Empty when the workbench opens; afterwards the server holds what the run extracted.
  api.listMemoryEntities
    .mockResolvedValueOnce(page('entities', []))
    .mockResolvedValue(page('entities', [entity]));
  api.listMemoryFacts
    .mockResolvedValueOnce(page('facts', []))
    .mockResolvedValue(page('facts', [fact]));
  api.listMemoryStrategies.mockResolvedValue(page('strategies', []));
  api.listMemoryProcedures.mockResolvedValue(page('procedures', []));
});

async function openWorkbench() {
  const user = userEvent.setup();
  render(<MemoryWorkbench onClose={vi.fn()} />);
  await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(1)));
  return user;
}

describe('MemoryWorkbench — Consolidate', () => {
  it('refetches facts, entities and the other lists after a successful run', async () => {
    api.consolidateNow.mockResolvedValue({
      success: true, duration_ms: 40, errors: [],
      results: { consolidate: { entities: 1, facts: 1, relationships: 0 } },
    });
    const user = await openWorkbench();

    await user.click(screen.getByRole('button', { name: 'Consolidate' }));
    await screen.findByText('Extracted 1 entities, 1 facts, 0 relationships');

    // One fetch on open and one refetch. Nothing else re-requests a list, because
    // the page, filters and selection are left as they were.
    await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2)));
    expect(api.getMemoryStats).toHaveBeenCalledTimes(2);
    expect(api.listMemoryChannels).toHaveBeenCalledTimes(2);

    // The reported symptom: the Facts tab said "No facts found" until a manual refresh.
    const tabs = within(screen.getByRole('navigation', { name: 'Memory areas' }));
    await user.click(tabs.getByRole('button', { name: 'Facts' }));
    expect(await screen.findByText(fact.claim)).toBeInTheDocument();
    expect(screen.queryByText('No facts found')).not.toBeInTheDocument();

    await user.click(tabs.getByRole('button', { name: 'Entities' }));
    expect(await screen.findByText(entity.name)).toBeInTheDocument();
  });

  it('refetches after a failed run too, since the other jobs may still have written', async () => {
    // The pipeline runs every job and answers 400 if any one failed, which the
    // client reports with this generic message.
    api.consolidateNow.mockRejectedValue(new Error('API request failed: Bad Request'));
    const user = await openWorkbench();

    await user.click(screen.getByRole('button', { name: 'Consolidate' }));
    await screen.findByText('Consolidation failed: API request failed: Bad Request');

    await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2)));
  });
});
