import { describe, it, expect, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
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
    // The OpenRouter card's account strip. `available: false` is the quiet
    // path — the strip renders nothing, which keeps these tests about the card.
    getOpenRouterAccount: vi.fn().mockResolvedValue({ available: false, reason: 'not_configured' }),
    scanOpenRouterAliases: vi.fn().mockResolvedValue({
      count: 0, catalog_available: true, refs: [],
    }),
    repairOpenRouterAliases: vi.fn().mockResolvedValue({ applied: [], failed: [], count: 0 }),
    unlinkOpenRouter: vi.fn().mockResolvedValue({
      status: 'unlinked', had_key: true, revoke_url: 'https://openrouter.ai/settings/keys',
    }),
    cancelOpenRouterLink: vi.fn().mockResolvedValue({ status: 'not_pending' }),
    deleteCustomProvider: vi.fn().mockResolvedValue({
      status: 'deleted', id: 'groq', referencing_profiles: [],
    }),
    // Only the ModelRolesSection anchor test mounts a manifest provider. One
    // entry is enough for the Global Default Model control to bind.
    getSettingsManifest: vi.fn().mockResolvedValue({
      version: 3,
      generated_at: '2026-11-27T00:00:00Z',
      counts: { total: 1 },
      sections: [],
      entries: [{
        key: 'preferences.default_model', store: 'config', type: 'str',
        default: null, value: null, secret: false,
        writable_via: '/api/config/update', ui_section: 'model-roles',
        help: { summary: 'The model everything falls back to.' },
      }],
    }),
  },
  apiErrorMessage: (error: unknown) => String(error),
}));

// ModelsSection now embeds ModelPickerField, whose effect fetches the model
// catalog — stub it so the section renders without hitting the network. The
// recents helpers are for the picker dialog, which one Model Roles test opens.
vi.mock('../../common/modelCatalog', () => ({
  fetchModelsOnce: vi.fn().mockResolvedValue([]),
  invalidateModelCache: vi.fn(),
  readRecentModels: () => [],
  pushRecentModel: (recent: string[]) => recent,
  writeRecentModel: vi.fn(),
}));

// ProvidersSection replaced window.confirm with the app dialog; stub the hook
// so the section renders without a ConfirmProvider at the root.
vi.mock('../../ui/ConfirmDialog', () => ({
  useConfirm: () => vi.fn().mockResolvedValue(true),
}));

import ProvidersSection from './ProvidersSection';
import { OpenRouterLink } from '../providers/OpenRouterLink';
import { OpenRouterAccount } from '../providers/OpenRouterAccount';
import ModelsSection from './ModelsSection';
import ModelRolesSection from './ModelRolesSection';
import { SECTION_HIERARCHY, getAllSections, findSectionById } from './index';
import { SettingsManifestProvider } from '../SettingsManifestContext';
import { api } from '../../../lib/api';

/** Typed handle on a mocked api method, for per-test overrides. */
const mocked = (fn: unknown) => fn as ReturnType<typeof vi.fn>;

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

describe('ProvidersSection — Global Default Model nudge', () => {
  // Connected but no default: an agent with no model of its own has nothing to
  // run on (short of LM Studio), so its first conversation would stop with "No
  // model chosen". The section points at the control that fixes it, and never
  // picks a model on the user's behalf.
  const NUDGE = 'Next, choose a Global Default Model';

  /** A config read the test can await, so "no nudge" is asserted after it lands. */
  function configRead(defaultModel: string) {
    const read = Promise.resolve({ preferences: { default_model: defaultModel } });
    mocked(api.getConfig).mockReturnValueOnce(read);
    return read;
  }

  it('offers to open Model Roles once a provider is connected and no default is set', async () => {
    mocked(api.updateConfig).mockClear();
    mockUpdateModelRoles.mockClear();
    const onFocusSetting = vi.fn();
    render(<ProvidersSection onFocusSetting={onFocusSetting} />);

    expect(await screen.findByText(NUDGE)).toBeInTheDocument();
    expect(
      screen.getByText(/can't start a conversation unless LM Studio is connected/)
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /open model roles/i }));
    expect(onFocusSetting).toHaveBeenCalledWith('model-roles', 'config:preferences.default_model');
    // It offers; it never chooses.
    expect(api.updateConfig).not.toHaveBeenCalled();
    expect(mockUpdateModelRoles).not.toHaveBeenCalled();
  });

  it('stays quiet once a Global Default Model is set', async () => {
    const read = configRead('openrouter:acme/model');
    render(<ProvidersSection onFocusSetting={vi.fn()} />);
    expect(await screen.findByText(/of 5 connected/)).toBeInTheDocument();
    await act(async () => { await read; });
    expect(screen.queryByText(NUDGE)).not.toBeInTheDocument();
  });

  it('stays quiet while nothing is connected', async () => {
    const { providers } = await api.getProviderCatalog();
    mocked(api.getProviderCatalog).mockResolvedValueOnce({
      count: providers.length,
      configured: 0,
      providers: providers.map(entry => ({ ...entry, configured: false, key_fingerprint: null })),
    });
    const read = configRead('');
    render(<ProvidersSection onFocusSetting={vi.fn()} />);
    // Wait for the empty catalog to land: the tally reads 0 only once it has.
    await waitFor(() =>
      expect(screen.getByText(/of 5 connected/)).toHaveTextContent('0 of 5 connected')
    );
    await act(async () => { await read; });
    expect(screen.queryByText(NUDGE)).not.toBeInTheDocument();
  });

  it("stays quiet when the default model can't be read", async () => {
    const read = Promise.reject(new Error('offline'));
    read.catch(() => {});
    mocked(api.getConfig).mockReturnValueOnce(read);
    render(<ProvidersSection onFocusSetting={vi.fn()} />);
    expect(await screen.findByText(/of 5 connected/)).toBeInTheDocument();
    await act(async () => { await read.catch(() => {}); });
    // Unknown is not "unset": the nudge would be a guess.
    expect(screen.queryByText(NUDGE)).not.toBeInTheDocument();
  });

  it('still explains itself where nothing can navigate', async () => {
    render(<ProvidersSection />);
    expect(await screen.findByText(NUDGE)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /open model roles/i })).not.toBeInTheDocument();
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
      home: ['overview'],
      infrastructure: ['providers', 'models', 'model-roles', 'search', 'images'],
      intelligence: ['planner', 'thinking', 'alloy', 'ambassador', 'research'],
      prompts: ['prompt-stack', 'prompts', 'prompt-templates', 'feature-prompts'],
      memory: ['memory-overview', 'context', 'memory-recall', 'memory-consolidation'],
      tools: ['translation'],
      interface: ['appearance'],
    });
  });

  it('opens on Overview, and home is the first group', () => {
    // Settings used to land on Model Providers — an API-key admin page as the
    // answer to "where do I start". Overview leads instead, and leads visually.
    expect(Object.keys(SECTION_HIERARCHY)[0]).toBe('home');
    expect(findSectionById('overview')).not.toBeNull();
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

describe('ModelRolesSection — Global Default Model', () => {
  const NO_DEFAULT = /can't start a conversation until you choose one, unless LM Studio is connected/;

  it('says what an empty default means for agents with no model of their own', async () => {
    render(<ModelRolesSection />);
    expect(await screen.findByText(NO_DEFAULT)).toBeInTheDocument();
    expect(screen.getByText('Not set — choose a model')).toBeInTheDocument();
    // The old copy claimed every agent had a profile model to fall back on.
    expect(screen.queryByText(/each agent uses its profile model/)).not.toBeInTheDocument();
  });

  it('drops the warning once a default is set', async () => {
    mocked(api.getConfig).mockResolvedValueOnce({
      preferences: { default_model: 'openrouter:acme/model' },
    });
    render(<ModelRolesSection />);
    expect(
      await screen.findByText('Used when an agent profile has no model of its own.')
    ).toBeInTheDocument();
    // Not in the (empty) catalog, so the trigger falls back to the bare id.
    expect(screen.getByText('acme/model')).toBeInTheDocument();
    expect(screen.queryByText(NO_DEFAULT)).not.toBeInTheDocument();
  });

  it("doesn't claim \"Not set\" before the current value has been read", async () => {
    let settle: (value: unknown) => void = () => {};
    mocked(api.getConfig).mockReturnValueOnce(new Promise(resolve => { settle = resolve; }));
    render(<ModelRolesSection />);
    expect(screen.getByText('Loading…')).toBeInTheDocument();
    expect(screen.queryByText(NO_DEFAULT)).not.toBeInTheDocument();
    await act(async () => { settle({ preferences: { default_model: '' } }); });
    expect(await screen.findByText(NO_DEFAULT)).toBeInTheDocument();
  });

  it('labels the picker’s empty row as no default, not a system default', async () => {
    render(<ModelRolesSection />);
    await userEvent.click(
      await screen.findByRole('button', { name: /Not set — choose a model/ })
    );
    expect(await screen.findByText('No global default')).toBeInTheDocument();
    expect(screen.queryByText('System default')).not.toBeInTheDocument();
  });

  it('carries the anchor the Overview step and the Providers nudge land on', async () => {
    const { container } = render(
      <SettingsManifestProvider>
        <ModelRolesSection />
      </SettingsManifestProvider>
    );
    await waitFor(() =>
      expect(
        container.querySelector('[data-setting="config:preferences.default_model"]')
      ).not.toBeNull()
    );
    // An unset default is the shipped default, so it isn't flagged as changed.
    expect(
      container.querySelector('[data-setting="config:preferences.default_model"] .setting-modified-dot')
    ).toBeNull();
  });

  it('re-reads the settings manifest after a model is chosen, so the Overview step ticks', async () => {
    // The Overview's setup list reads this setting from the manifest, fetched
    // once per Settings session: without a re-read, the required step would stay
    // unticked until Settings was reopened.
    const { fetchModelsOnce } = await import('../../common/modelCatalog');
    mocked(fetchModelsOnce).mockResolvedValue([
      { id: 'openrouter:acme/model', name: 'Acme Model', provider: 'openrouter', context_length: 128_000 },
    ]);
    mocked(api.updateConfig).mockClear();
    mocked(api.getSettingsManifest).mockClear();
    try {
      const { container } = render(
        <SettingsManifestProvider>
          <ModelRolesSection />
        </SettingsManifestProvider>
      );
      const anchor = '[data-setting="config:preferences.default_model"]';
      await waitFor(() => expect(container.querySelector(anchor)).not.toBeNull());
      expect(api.getSettingsManifest).toHaveBeenCalledTimes(1);

      await userEvent.click(container.querySelector<HTMLElement>(`${anchor} .mpf-trigger`)!);
      await waitFor(() =>
        expect(document.querySelector('[data-model-id="openrouter:acme/model"]')).not.toBeNull()
      );
      await userEvent.click(document.querySelector<HTMLElement>('[data-model-id="openrouter:acme/model"]')!);
      await userEvent.click(screen.getByRole('button', { name: /use this model/i }));

      await waitFor(() =>
        expect(api.updateConfig).toHaveBeenCalledWith({
          preferences: { default_model: 'openrouter:acme/model' },
        })
      );
      await waitFor(() => expect(api.getSettingsManifest).toHaveBeenCalledTimes(2));
    } finally {
      mocked(fetchModelsOnce).mockResolvedValue([]);
    }
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

describe('OpenRouterAccount', () => {
  it('shows the balance and this-month spend', async () => {
    const { api } = await import('../../../lib/api');
    (api.getOpenRouterAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      available: true, limit: null, limit_remaining: null,
      usage_monthly: 8.46, total_credits: 35, total_usage: 28.7, is_free_tier: false,
    });
    render(<OpenRouterAccount />);
    // $35 bought − $28.70 used = $6.30 left, which is the number a user thinks in.
    expect(await screen.findByText('$6.30')).toBeInTheDocument();
    expect(screen.getByText(/of \$35\.00/)).toBeInTheDocument();
    expect(screen.getByText('$8.46')).toBeInTheDocument();
  });

  it('says an uncapped key has no cap instead of drawing an empty gauge', async () => {
    // `limit: null` means "no cap", not zero — a meter here would read as
    // "nothing left" and send the user hunting for a billing problem.
    const { api } = await import('../../../lib/api');
    (api.getOpenRouterAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      available: true, limit: null, limit_remaining: null, usage_monthly: 1.2,
      total_credits: null, total_usage: null,
    });
    render(<OpenRouterAccount />);
    expect(await screen.findByText('None')).toBeInTheDocument();
    expect(screen.getByText(/on this key/)).toBeInTheDocument();
  });

  it('renders nothing when the account is unreadable', async () => {
    // A supporting strip must never be the reason the page looks broken.
    const { api } = await import('../../../lib/api');
    (api.getOpenRouterAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      available: false, reason: 'not_configured',
    });
    const { container } = render(<OpenRouterAccount />);
    await new Promise((r) => setTimeout(r, 0));
    expect(container.querySelector('.openrouter-facts')).toBeNull();
  });

  it('offers the alias repair only when something is actually stale', async () => {
    const { api } = await import('../../../lib/api');
    (api.scanOpenRouterAliases as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      count: 2, catalog_available: true,
      refs: [
        { store: 'config', location: 'preferences.default_model', label: 'Default model',
          current: 'openrouter:anthropic/claude-sonnet-latest',
          suggested: 'openrouter:~anthropic/claude-sonnet-latest' },
        { store: 'profile', location: 'p1.default_model', label: 'X1 — model',
          current: 'openrouter:google/gemini-flash-latest',
          suggested: 'openrouter:~google/gemini-flash-latest' },
      ],
    });
    render(<OpenRouterAccount />);
    expect(await screen.findByRole('button', { name: /fix 2 references/i })).toBeInTheDocument();
  });
});
