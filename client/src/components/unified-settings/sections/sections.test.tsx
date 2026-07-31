import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type React from 'react';

// The two rebuilt sections depend on server/notification contexts + the api
// client; mock them so the components render in isolation. The hook return
// values are stable singletons — ProvidersSection's effect keys off
// `activeMetadata`, so a fresh object per render would loop infinitely.
vi.mock('../../../contexts/ServerContext', () => {
  const value = {
    activeServer: { id: 's1', name: 'Local' },
    activeMetadata: {},
    updateMetadata: vi.fn(),
  };
  return { useServer: () => value };
});

// ProvidersSection's Supply Line reads the active agent's model to resolve a
// route. Stable singleton — a fresh object per render would re-fire the effect.
vi.mock('../../../contexts/AgentProfileContext', () => {
  const value = {
    activeProfile: { id: 'p1', name: 'X1', defaultModel: 'openrouter:test-model' },
    profiles: [],
  };
  return { useAgentProfile: () => value };
});

vi.mock('../../../contexts/NotificationContext', () => {
  const value = {
    notify: vi.fn(),
    notifyError: vi.fn(),
    notifySuccess: vi.fn(),
    dismiss: vi.fn(),
  };
  return { useNotify: () => value };
});

const mockGetModelRoles = vi.fn().mockResolvedValue({
  roles: {
    fast_utility: {
      label: 'Fast Utility',
      description: 'Quick classification & extraction — speed first.',
      model: '',
    },
    deep_reasoning: {
      label: 'Deep Reasoning',
      description: 'Consolidation & distillation — quality and cost-efficiency.',
      model: 'openrouter:nvidia/nemotron-3-ultra-550b-a55b',
    },
    summarizer: {
      label: 'Summarizer',
      description: 'Cheap, reliable compression & recaps.',
      model: '',
    },
  },
  members: [
    {
      member: 'extraction', label: 'Extraction', role: 'fast_utility',
      kind: 'memory', source: 'extraction_model',
      explicit: 'lmstudio:google/gemma-3-4b',
      role_model: '', effective: 'lmstudio:google/gemma-3-4b', following: 'explicit',
    },
    {
      member: 'combined_extraction', label: 'Combined extraction', role: 'deep_reasoning',
      kind: 'memory', source: 'combined_extraction_model',
      explicit: '', role_model: 'openrouter:nvidia/nemotron-3-ultra-550b-a55b',
      effective: 'openrouter:nvidia/nemotron-3-ultra-550b-a55b', following: 'role',
    },
    {
      member: 'compression', label: 'Tool-output compression', role: 'summarizer',
      kind: 'config', source: 'compression.model',
      explicit: '', role_model: '', effective: '', following: 'fallback',
    },
  ],
});
const mockUpdateModelRoles = vi.fn().mockResolvedValue({ status: 'ok' });

vi.mock('../../../lib/api', () => ({
  api: {
    getConfig: vi.fn().mockResolvedValue({ preferences: { default_model: '' } }),
    getContextLimits: vi.fn().mockResolvedValue({
      lmstudio: { context_window: 8192, max_output_tokens: 2048 },
      models: {},
    }),
    updateConfig: vi.fn().mockResolvedValue({}),
    updateContextLimits: vi.fn().mockResolvedValue({}),
    getModelRoles: (...args: unknown[]) => mockGetModelRoles(...args),
    updateModelRoles: (...args: unknown[]) => mockUpdateModelRoles(...args),
    // ProvidersSection's on-device tiles read /api/health via useHealth.
    health: vi.fn().mockResolvedValue({
      status: 'healthy',
      compute: { device: 'cpu', cuda_available: false },
      embeddings: { provider: 'local', model: 'BAAI/bge-m3', dimensions: 1024 },
      translation: { status: 'not_loaded', models: {} },
    }),
    // The rebuilt Providers surface is catalog-driven: the list, the key
    // fingerprints and the reachability all come from the server.
    getProviderCatalog: vi.fn().mockResolvedValue({
      count: 5,
      configured: 1,
      providers: [
        { id: 'openrouter', kind: 'openrouter', label: 'OpenRouter', base_url: null,
          key_fingerprint: '····4991', header_names: [], enabled: true, builtin: true,
          credential: 'api_key', configured: true },
        { id: 'anthropic', kind: 'anthropic', label: 'Anthropic', base_url: null,
          key_fingerprint: null, header_names: [], enabled: true, builtin: true,
          credential: 'api_key', configured: false },
        { id: 'openai', kind: 'openai_compatible', label: 'OpenAI', base_url: null,
          key_fingerprint: null, header_names: [], enabled: true, builtin: true,
          credential: 'api_key', configured: false },
        { id: 'vercel', kind: 'vercel', label: 'Vercel AI Gateway', base_url: null,
          key_fingerprint: null, header_names: [], enabled: true, builtin: true,
          credential: 'api_key', configured: false },
        { id: 'lmstudio', kind: 'lmstudio', label: 'LM Studio', base_url: null,
          key_fingerprint: null, header_names: [], enabled: true, builtin: true,
          credential: 'base_url', configured: false },
      ],
    }),
    checkProvidersHealth: vi.fn().mockResolvedValue({
      status: 'degraded',
      providers: { openrouter: { status: 'healthy', models_available: 364 } },
    }),
    getProviderRoute: vi.fn().mockResolvedValue({
      requested: 'openrouter:test-model',
      resolved: {
        model: 'openrouter:test-model', provider: 'openrouter', provider_label: 'OpenRouter',
        model_id: 'test-model', configured: true, healthy: true, known: true,
        context_window: 1_000_000, max_output_tokens: 64_000,
        cost_per_1k_input: 0.003, cost_per_1k_output: 0.015,
      },
      substituted: false,
      candidates: [],
      fallback_enabled: true,
    }),
    testProvider: vi.fn().mockResolvedValue({
      reachable: true, base_url: 'https://x.test/v1', elapsed_ms: 12,
      models_available: 3, models: [], error: null,
    }),
    saveCustomProvider: vi.fn().mockResolvedValue({ provider: {} }),
    deleteCustomProvider: vi.fn().mockResolvedValue({
      status: 'deleted', id: 'groq', referencing_profiles: [],
    }),
  },
  apiErrorMessage: (error: unknown) => String(error),
}));

// ModelsSection now embeds ModelPickerField, whose effect fetches the model
// catalog — stub it so the section renders without hitting the network.
vi.mock('../../common/modelCatalog', () => ({
  fetchModelsOnce: vi.fn().mockResolvedValue([]),
}));

// ProvidersSection replaced window.confirm with the app dialog; stub the hook
// so the section renders without a ConfirmProvider at the root.
vi.mock('../../ui/ConfirmDialog', () => ({
  useConfirm: () => vi.fn().mockResolvedValue(true),
}));

import ProvidersSection from './ProvidersSection';
import { OpenRouterLink } from '../providers/OpenRouterLink';
import ModelsSection from './ModelsSection';
import ModelRolesSection from './ModelRolesSection';
import { SECTION_HIERARCHY, getAllSections, findSectionById } from './index';

describe('ProvidersSection', () => {
  it('renders every catalog entry with its badge and a save action', async () => {
    render(<ProvidersSection />);
    // The list is catalog-driven now, so all five arrive from the server mock.
    for (const name of ['LM Studio', 'Anthropic', 'OpenAI', 'OpenRouter', 'Vercel AI Gateway']) {
      expect(await screen.findByRole('button', { name: `${name} connection settings` })).toBeInTheDocument();
    }
    expect(screen.getByText('Recommended')).toBeInTheDocument();
    // Tier eyebrows are gone — the tier lives only on the card badge now, so
    // these counts are exact rather than "at least".
    expect(screen.getAllByText('Beta')).toHaveLength(3);
    expect(screen.getAllByText('Local')).toHaveLength(1);
    // Secrets save explicitly; nothing is dirty on first paint.
    const save = screen.getByRole('button', { name: /save to server/i });
    expect(save).toBeDisabled();
    expect(screen.queryByText('Unsaved changes')).toBeNull();
  });

  it('leads with connection state rather than an identical box per provider', async () => {
    render(<ProvidersSection />);
    // Reachable provider reports its catalog size; the rest say they're not set up.
    expect(await screen.findByText(/364 models available/)).toBeInTheDocument();
    expect(screen.getAllByText(/Not connected\. Add a key/).length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText(/Not connected\. Add a server URL/)).toBeInTheDocument();
    // The tally summarizes the same truth.
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('never renders a stored key, only its fingerprint', async () => {
    render(<ProvidersSection />);
    // Expand OpenRouter — the credential field is behind the summary toggle.
    await userEvent.click(
      await screen.findByRole('button', { name: 'OpenRouter connection settings' })
    );
    expect(await screen.findByText(/on file ····4991/)).toBeInTheDocument();
    // The input starts EMPTY: typing is an explicit replacement, not an edit of
    // a pre-filled secret (which the client no longer possesses).
    const field = screen.getByLabelText(/API key/i) as HTMLInputElement;
    expect(field.value).toBe('');
    expect(field.placeholder).toMatch(/replace the stored one/i);
  });

  it('states the live route in the supply line', async () => {
    render(<ProvidersSection />);
    // The signature element: which provider resolves the active agent's model,
    // and the model's REAL context window (not a provider default).
    expect(await screen.findByText('Supply line')).toBeInTheDocument();
    expect(screen.getByText('test-model')).toBeInTheDocument();
    expect(screen.getByText(/1M context/)).toBeInTheDocument();
    expect(screen.getByText('reachable')).toBeInTheDocument();
  });

  it('offers the custom-endpoint affordance', async () => {
    render(<ProvidersSection />);
    expect(await screen.findByText('Connect an endpoint')).toBeInTheDocument();
  });
});

describe('SECTION_HIERARCHY (settings overhaul S4)', () => {
  it('has the reorganized group/section layout', () => {
    // Snapshot the shape so moving/renaming a section is a visible diff.
    const shape = Object.fromEntries(
      Object.entries(SECTION_HIERARCHY).map(([key, group]) => [
        key,
        group.sections.map(s => s.id),
      ])
    );
    expect(shape).toEqual({
      infrastructure: ['providers', 'models', 'model-roles', 'search', 'images'],
      intelligence: ['planner', 'thinking', 'alloy', 'ambassador', 'research'],
      prompts: ['prompt-stack', 'prompts', 'prompt-templates', 'feature-prompts'],
      memory: ['memory-overview', 'context', 'memory-recall', 'memory-consolidation'],
      tools: ['translation'],
      interface: ['appearance'],
    });
  });

  it('renames the alloy section to Agent Teams (user-facing only)', () => {
    // Internal id stays `alloy` (Workspaces→Projects precedent) — only the
    // label changed. `teams` keyword routes palette/search to it.
    const alloy = findSectionById('alloy');
    expect(alloy?.label).toBe('Agent Teams');
    expect(alloy?.keywords).toContain('teams');
  });

  it('every section carries search keywords', () => {
    for (const section of getAllSections()) {
      expect(section.keywords?.length, `${section.id} has no keywords`).toBeGreaterThan(0);
    }
  });

  it('search keywords route to the moved sections', () => {
    const byKeyword = (kw: string) =>
      getAllSections().filter(s => s.keywords?.some(k => k.includes(kw))).map(s => s.id);
    expect(byKeyword('cross-encoder')).toContain('memory-recall');
    expect(byKeyword('consolidation')).toContain('memory-consolidation');
    expect(byKeyword('summarizer')).toContain('model-roles');
    expect(byKeyword('template')).toContain('prompt-templates');
    expect(byKeyword('extraction prompt')).toContain('feature-prompts');
    // The retired combined section is gone.
    expect(findSectionById('memory-settings')).toBeNull();
  });
});

describe('ModelRolesSection', () => {
  it('renders the three roles with member resolution chips', async () => {
    render(<ModelRolesSection />);
    expect(await screen.findByText('Fast Utility')).toBeInTheDocument();
    expect(screen.getByText('Deep Reasoning')).toBeInTheDocument();
    expect(screen.getByText('Summarizer')).toBeInTheDocument();
    // The global default model now lives here (moved from Model Limits).
    expect(screen.getByText('Global Default Model')).toBeInTheDocument();
    // Member chips reflect the live resolution chain.
    expect(screen.getByText('following role')).toBeInTheDocument();
    expect(screen.getByText('custom')).toBeInTheDocument();
    expect(screen.getByText('fallback chain')).toBeInTheDocument();
    // The set role shows its concrete model on the picker trigger.
    expect(
      screen.getAllByText(/nemotron-3-ultra-550b-a55b/).length
    ).toBeGreaterThan(0);
  });
});

describe('ModelsSection', () => {
  it('renders the context-limits header and the LM Studio limits card', async () => {
    render(<ModelsSection />);
    // The global default model moved to Model Roles — it is no longer here.
    expect(screen.queryByText('Global default model')).toBeNull();
    expect(screen.getByText('Model Context Limits')).toBeInTheDocument();
    // getContextLimits resolves → the limits card appears with kit NumberFields.
    expect(await screen.findByText('Local')).toBeInTheDocument();
    expect(screen.getByText('Context Window (tokens)')).toBeInTheDocument();
    expect(screen.getByText('Max Output Tokens')).toBeInTheDocument();
    // The explicit Save button is gone — limits autosave (SaveStatusChip).
    expect(screen.queryByRole('button', { name: /save limits/i })).toBeNull();
  });
});

describe('OpenRouterLink', () => {
  const renderLink = (props: Partial<React.ComponentProps<typeof OpenRouterLink>> = {}) =>
    render(<OpenRouterLink link={null} hasKey={false} onChanged={vi.fn()} {...props} />);

  it('offers one-click linking when nothing is stored', () => {
    renderLink();
    expect(screen.getByRole('button', { name: /link account/i })).toBeInTheDocument();
    expect(screen.getByText(/no copying and pasting/i)).toBeInTheDocument();
  });

  it('distinguishes a linked account from a hand-pasted key', () => {
    // The distinction is honest, not cosmetic: only an OAuth link has a user id,
    // and claiming one for a pasted key would be a lie about where it came from.
    const { unmount } = renderLink({
      link: { method: 'oauth', user_id: 'user_42', linked_at: '2026-07-31T00:00:00Z' },
      hasKey: true,
    });
    expect(screen.getByText('Linked account')).toBeInTheDocument();
    expect(screen.getByText('user_42')).toBeInTheDocument();
    unmount();

    renderLink({ hasKey: true });
    expect(screen.getByText('Key added by hand')).toBeInTheDocument();
  });

  it('always offers a route to real revocation', () => {
    // Forgetting the key is local — AgentX holds no management key — so the UI
    // must also point at where the user can actually revoke it.
    renderLink({ hasKey: true });
    expect(screen.getByRole('button', { name: /forget key/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /manage keys/i })).toBeInTheDocument();
  });
});
