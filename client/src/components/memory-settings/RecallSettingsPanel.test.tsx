import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RecallSettingsPanel } from './RecallSettingsPanel';
import { SettingsManifestProvider } from '../unified-settings/SettingsManifestContext';

vi.mock('../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

// The model pickers open a catalog modal; stand them in with a plain readout.
vi.mock('../common/ModelPickerField', () => ({
  ModelPickerField: ({ label, value }: { label: string; value: string }) => (
    <div data-testid={`model-${label}`}>{value || 'inherit'}</div>
  ),
}));

const SHIPPED = {
  recall_enable_hybrid: true,
  recall_enable_entity_centric: true,
  recall_enable_query_expansion: true,
  recall_enable_hyde: true,
  recall_enable_self_query: true,
  cross_encoder_enabled: true,
  cross_encoder_model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
  recall_candidate_pool: 50,
  recall_ce_max_demotion: 2,
  recall_hybrid_bm25_weight: 0.3,
  recall_hybrid_vector_weight: 0.7,
  recall_entity_similarity_threshold: 0.65,
  recall_entity_max_entities: 5,
  recall_expansion_max_variants: 3,
  recall_hyde_model: '',
  recall_hyde_temperature: 0.7,
  recall_hyde_max_tokens: 150,
  recall_self_query_model: '',
  recall_self_query_temperature: 0.2,
  recall_self_query_max_tokens: 200,
  recall_min_confidence: 0.5,
  recall_hybrid_rrf_k: 60,
  recall_entity_graph_depth: 1,
  recall_first_person_guard: false,
  recall_first_person_penalty: 0.5,
};

const mockGetRecall = vi.fn();
const mockUpdateRecall = vi.fn().mockResolvedValue({ success: true, updated: [] });
const mockManifest = vi.fn();

vi.mock('../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: {
      getRecallSettings: () => mockGetRecall(),
      updateRecallSettings: (p: unknown) => mockUpdateRecall(p),
      getSettingsManifest: () => mockManifest(),
    },
  };
});

/** A manifest entry per recall key, so the panel can bind every control. */
function manifestFor(overrides: Record<string, unknown> = {}) {
  const values = { ...SHIPPED, ...overrides };
  return {
    version: 2,
    generated_at: '2026-08-02T00:00:00Z',
    counts: { total: Object.keys(SHIPPED).length },
    entries: Object.entries(SHIPPED).map(([key, shipped]) => ({
      key,
      store: 'memory' as const,
      type: typeof shipped,
      default: shipped,
      value: (values as Record<string, unknown>)[key],
      secret: false,
      writable_via: '/api/memory/recall-settings',
      ui_section: 'memory-recall',
      ...(key === 'recall_candidate_pool'
        ? { constraints: { min: 10, max: 200, step: 1 }, tier: 'essential' as const,
            help: { summary: 'How many candidates the reranker scores.' } }
        : {}),
    })),
  };
}

function renderPanel() {
  return render(
    <SettingsManifestProvider>
      <RecallSettingsPanel />
    </SettingsManifestProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockGetRecall.mockResolvedValue({ ...SHIPPED });
  mockManifest.mockResolvedValue(manifestFor());
});

describe('RecallSettingsPanel', () => {
  it('gives each technique its own knobs instead of splitting them', async () => {
    // HyDE's model and temperature used to live under "HyDE Settings" while its
    // token budget sat in "Advanced" — tuning one technique meant hunting in two
    // places. Self-Query was split the same way. Each now renders exactly one
    // Max Tokens, inside its own group.
    renderPanel();
    await screen.findByText('Retrieval Techniques');

    expect(screen.getAllByRole('spinbutton', { name: 'Max Tokens' })).toHaveLength(2);

    const advanced = screen.getByRole('button', { name: /Advanced/ });
    expect(advanced).toHaveAttribute('aria-expanded', 'false');
  });

  it('gates the Advanced knobs behind a disclosure without losing them', async () => {
    // "Gated, not lost": collapsed content is correctly out of the a11y tree
    // (hence `hidden: true` to see it), and opening the disclosure brings it
    // back as real, reachable controls.
    renderPanel();
    await screen.findByText('Advanced');

    const trigger = screen.getByRole('button', { name: /Advanced/ });
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('spinbutton', { name: 'Hybrid RRF k', hidden: true }))
      .toBeInTheDocument();

    await userEvent.click(trigger);

    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('spinbutton', { name: 'Hybrid RRF k' })).toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: 'Entity Graph Depth' })).toBeInTheDocument();
  });

  it('takes bounds and help from the manifest, not from local literals', async () => {
    renderPanel();
    const pool = await screen.findByRole('spinbutton', { name: 'Candidate Pool' });
    expect(pool).toHaveAttribute('min', '10');
    expect(pool).toHaveAttribute('max', '200');
    expect(screen.getByRole('button', { name: 'About Candidate Pool' })).toBeInTheDocument();
  });

  it('marks a changed setting and resets it to the shipped default', async () => {
    mockGetRecall.mockResolvedValue({ ...SHIPPED, recall_candidate_pool: 120 });
    mockManifest.mockResolvedValue(manifestFor({ recall_candidate_pool: 120 }));
    renderPanel();

    await screen.findByRole('spinbutton', { name: 'Candidate Pool' });
    await waitFor(() =>
      expect(screen.getAllByLabelText('Changed from the default').length).toBeGreaterThan(0)
    );

    await userEvent.click(
      screen.getByRole('button', { name: 'Reset Candidate Pool to default' })
    );
    await waitFor(() =>
      expect(screen.getByRole('spinbutton', { name: 'Candidate Pool' })).toHaveValue(50)
    );
  });

  it('offers a retry when the settings fail to load', async () => {
    mockGetRecall.mockRejectedValue(new Error('backend down'));
    renderPanel();
    expect(await screen.findByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });

  it('renders without a manifest at all', async () => {
    // The manifest is metadata: if it never arrives, the controls still work.
    mockManifest.mockRejectedValue(new Error('no manifest'));
    renderPanel();

    expect(await screen.findByRole('spinbutton', { name: 'Candidate Pool' })).toHaveValue(50);
    expect(screen.queryByRole('button', { name: /^About/ })).not.toBeInTheDocument();
  });
});
