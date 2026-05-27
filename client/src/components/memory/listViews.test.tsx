import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EntityListView } from './EntityListView';
import { FactListView } from './FactListView';
import { StrategyListView } from './StrategyListView';
import type { MemoryEntity, MemoryFact, MemoryStrategy, ApiError } from '../../lib/api';

const apiError = { message: 'boom', kind: 'server', status: 500 } as ApiError;

const entity: MemoryEntity = {
  id: 'e1', name: 'Ada Lovelace', type: 'person', channel: '_global',
  salience: 0.8, last_accessed: new Date().toISOString(), access_count: 3,
};

const fact: MemoryFact = {
  id: 'f1', claim: 'The sky is blue', confidence: 0.9, source: 'observation',
  channel: '_global', created_at: new Date().toISOString(), entity_ids: [],
};

const strategy: MemoryStrategy = {
  id: 's1', description: 'Search then summarize', tool_sequence: ['search', 'summarize'],
  success_count: 5, failure_count: 1, success_rate: 0.83, channel: '_global',
};

describe('EntityListView', () => {
  it('renders rows and footer count', () => {
    render(
      <EntityListView
        entities={[entity]} total={1} loading={false} error={null}
        selectedEntityId={null} onSelectEntity={vi.fn()}
      />
    );
    expect(screen.getByText('Ada Lovelace')).toBeInTheDocument();
    expect(screen.getByText('Showing 1 of 1 entities')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    render(
      <EntityListView
        entities={[]} total={0} loading error={null}
        selectedEntityId={null} onSelectEntity={vi.fn()}
      />
    );
    expect(screen.getByText('Loading entities...')).toBeInTheDocument();
  });

  it('shows the error message', () => {
    render(
      <EntityListView
        entities={[]} total={0} loading={false} error={apiError}
        selectedEntityId={null} onSelectEntity={vi.fn()}
      />
    );
    expect(screen.getByText(/Failed to load entities: boom/)).toBeInTheDocument();
  });

  it('shows the empty state', () => {
    render(
      <EntityListView
        entities={[]} total={0} loading={false} error={null}
        selectedEntityId={null} onSelectEntity={vi.fn()}
      />
    );
    expect(screen.getByText('No entities found')).toBeInTheDocument();
  });
});

describe('FactListView', () => {
  it('renders the claim and footer', () => {
    render(
      <FactListView
        facts={[fact]} total={1} loading={false} error={null}
        selectedFactId={null} onSelectFact={vi.fn()}
      />
    );
    expect(screen.getByText('The sky is blue')).toBeInTheDocument();
    expect(screen.getByText('Showing 1 of 1 facts')).toBeInTheDocument();
  });

  it('shows the empty state', () => {
    render(
      <FactListView
        facts={[]} total={0} loading={false} error={null}
        selectedFactId={null} onSelectFact={vi.fn()}
      />
    );
    expect(screen.getByText('No facts found')).toBeInTheDocument();
  });
});

describe('StrategyListView', () => {
  it('renders description, tool chips, and footer', () => {
    render(
      <StrategyListView
        strategies={[strategy]} total={1} loading={false} error={null}
        selectedStrategyId={null} onSelectStrategy={vi.fn()}
      />
    );
    expect(screen.getByText('Search then summarize')).toBeInTheDocument();
    expect(screen.getByText('search')).toBeInTheDocument();
    expect(screen.getByText('Showing 1 of 1 strategies')).toBeInTheDocument();
  });

  it('shows the empty state with the learning hint', () => {
    render(
      <StrategyListView
        strategies={[]} total={0} loading={false} error={null}
        selectedStrategyId={null} onSelectStrategy={vi.fn()}
      />
    );
    expect(screen.getByText('No strategies found')).toBeInTheDocument();
    expect(screen.getByText(/learned from successful tool usage/)).toBeInTheDocument();
  });
});
