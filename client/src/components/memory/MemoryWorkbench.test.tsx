import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { MemoryEntity, MemoryFact, MemoryStrategy } from '../../lib/api';
import { MemoryWorkbench } from './MemoryWorkbench';

// The workbench's data hooks stay real and only the API beneath them is stubbed,
// so each refetch shows up as another call to the matching list endpoint.
const api = vi.hoisted(() => ({
  consolidateNow: vi.fn(),
  importMemory: vi.fn(),
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

const strategy: MemoryStrategy = {
  id: 's1', description: 'Gather sources, then compare their claims', tool_sequence: ['web_search', 'web_extract'],
  success_count: 4, failure_count: 1, success_rate: 0.8, channel: '_default',
};

/** A small model of the server: memory is empty until something writes to it. */
const server = { written: false };

/** The four lists the workbench keeps in state whichever tab is open. */
const LIST_FETCHERS = [
  api.listMemoryEntities, api.listMemoryFacts, api.listMemoryStrategies, api.listMemoryProcedures,
];

function page<K extends string, T>(key: K, items: T[]) {
  return { [key]: items, total: items.length, has_next: false };
}

function areaTab(name: string) {
  return within(screen.getByRole('navigation', { name: 'Memory areas' })).getByRole('button', { name });
}

beforeEach(() => {
  vi.resetAllMocks();
  server.written = false;
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false, media: query, onchange: null,
    addEventListener: vi.fn(), removeEventListener: vi.fn(),
    addListener: vi.fn(), removeListener: vi.fn(), dispatchEvent: vi.fn(),
  }));
  api.getMemoryStats.mockResolvedValue({
    totals: { entities: 0, facts: 0, strategies: 0, turns: 4 }, by_channel: {},
  });
  api.listMemoryChannels.mockResolvedValue({ channels: [] });
  api.listMemoryEntities.mockImplementation(async () => page('entities', server.written ? [entity] : []));
  api.listMemoryFacts.mockImplementation(async () => page('facts', server.written ? [fact] : []));
  api.listMemoryStrategies.mockImplementation(async () => page('strategies', server.written ? [strategy] : []));
  api.listMemoryProcedures.mockResolvedValue(page('procedures', []));
  api.consolidateNow.mockImplementation(async () => {
    server.written = true;
    return {
      success: true, duration_ms: 40, errors: [],
      results: { consolidate: { entities: 1, facts: 1, relationships: 0 } },
    };
  });
});

async function openWorkbench() {
  const user = userEvent.setup();
  render(<MemoryWorkbench onClose={vi.fn()} />);
  await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(1)));
  // Refresh stays disabled until the stats have loaded.
  await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh' })).toBeEnabled());
  return user;
}

describe('MemoryWorkbench — reloading after memory changes', () => {
  it('refetches facts, entities and the other lists after a successful Consolidate', async () => {
    const user = await openWorkbench();

    await user.click(screen.getByRole('button', { name: 'Consolidate' }));
    await screen.findByText('Extracted 1 entities, 1 facts, 0 relationships');

    // One fetch on open and one refetch. Nothing else re-requests a list, because
    // the page, filters and selection are left as they were.
    await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2)));
    expect(api.getMemoryStats).toHaveBeenCalledTimes(2);
    expect(api.listMemoryChannels).toHaveBeenCalledTimes(2);

    // The reported symptom: the Facts tab said "No facts found" until a manual refresh.
    await user.click(areaTab('Facts'));
    expect(await screen.findByText(fact.claim)).toBeInTheDocument();
    expect(screen.queryByText('No facts found')).not.toBeInTheDocument();

    await user.click(areaTab('Entities'));
    expect(await screen.findByText(entity.name)).toBeInTheDocument();
  });

  it('refetches after a failed Consolidate too, since the other jobs may still have written', async () => {
    // The pipeline runs every job and answers 400 if any one failed, which the
    // client reports with this generic message; the jobs that succeeded have written.
    api.consolidateNow.mockImplementation(async () => {
      server.written = true;
      throw new Error('API request failed: Bad Request');
    });
    const user = await openWorkbench();

    await user.click(screen.getByRole('button', { name: 'Consolidate' }));
    await screen.findByText('Consolidation failed: API request failed: Bad Request');

    await waitFor(() => LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2)));
    await user.click(areaTab('Facts'));
    expect(await screen.findByText(fact.claim)).toBeInTheDocument();
  });

  it('reloads strategies and procedures from the header Refresh as well', async () => {
    const user = await openWorkbench();
    await user.click(areaTab('Strategies'));
    expect(screen.queryByText(strategy.description)).not.toBeInTheDocument();

    server.written = true; // a scheduled consolidation wrote while the workbench was open
    await user.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(await screen.findByText(strategy.description)).toBeInTheDocument();
    LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it('reloads every list after an import', async () => {
    api.importMemory.mockImplementation(async () => {
      server.written = true;
      return {
        imported: {
          mode: 'merge', channel: null, recomputed_embeddings: 0,
          imported: { strategies: { created: 1, total: 1 } },
          pg_conversation_logs: 0, pg_tool_invocations: 0,
        },
      };
    });
    const user = await openWorkbench();
    await user.click(areaTab('Strategies'));

    const input = document.querySelector('input[type="file"]');
    if (!(input instanceof HTMLInputElement)) throw new Error('file input not rendered');
    const snapshot = new File([JSON.stringify({ schema_version: 1 })], 'snapshot.json', { type: 'application/json' });
    await user.upload(input, snapshot);
    await user.click(await screen.findByRole('button', { name: 'Merge Import' }));

    expect(await screen.findByText(strategy.description)).toBeInTheDocument();
    LIST_FETCHERS.forEach(fetch => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it('reloads the Explore graph in place after Consolidate', async () => {
    const user = await openWorkbench();
    await user.click(areaTab('Explore'));
    expect(await screen.findByText('Select an entity or procedure to explore')).toBeInTheDocument();
    // The graph runs its own entity query: opening it adds exactly one fetch.
    await waitFor(() => expect(api.listMemoryEntities).toHaveBeenCalledTimes(2));

    await user.click(screen.getByRole('button', { name: 'Consolidate' }));

    // Without leaving the tab, the graph's entity list picks up the new entity:
    // one refetch for the workbench's list and one for the graph's.
    expect(await screen.findByText(entity.name)).toBeInTheDocument();
    expect(api.listMemoryEntities).toHaveBeenCalledTimes(4);
  });
});
