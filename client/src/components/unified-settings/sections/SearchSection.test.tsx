import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchSection from './SearchSection';
import { SettingsManifestProvider } from '../SettingsManifestContext';

vi.mock('../../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

const mockGetConfig = vi.fn();
const mockUpdateConfig = vi.fn().mockResolvedValue({ success: true });
const mockManifest = vi.fn();

vi.mock('../../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: {
      getConfig: () => mockGetConfig(),
      updateConfig: (p: unknown) => mockUpdateConfig(p),
      getSettingsManifest: () => mockManifest(),
      searchHealth: vi.fn(),
    },
  };
});

const POLICY_DEFAULT = { trusted: [], blocked: [], goggle: '' };

function config(over: Record<string, unknown> = {}) {
  return {
    search: {
      backend: 'tavily',
      max_results: 5,
      per_turn_limit: 8,
      brave_context_max_tokens: 4096,
      brave_context_max_tokens_per_url: 1024,
      source_policy: { ...POLICY_DEFAULT },
      ...over,
    },
  };
}

function entry(key: string, dflt: unknown, over: Record<string, unknown> = {}) {
  return {
    key, store: 'config' as const, type: typeof dflt,
    default: dflt, value: dflt, secret: false,
    writable_via: '/api/config/update', ui_section: 'search',
    ...over,
  };
}

function manifest() {
  return {
    version: 3,
    generated_at: '2026-08-07T00:00:00Z',
    counts: { total: 5 },
    sections: [{ id: 'search', label: 'Infrastructure → Web Search', writable_count: 21 }],
    entries: [
      entry('search.max_results', 5, {
        constraints: { min: 1, max: 20, step: 1, unit: 'results per search' },
        tier: 'essential',
        help: { summary: 'How many results one search returns.' },
      }),
      entry('search.per_turn_limit', 8, { tier: 'essential' }),
      entry('search.brave_context_max_tokens_per_url', 1024, { tier: 'advanced' }),
      entry('search.source_policy', POLICY_DEFAULT, {
        write_mode: 'whole',
        tier: 'essential',
        help: { summary: 'Which corners of the web may ground an answer.' },
      }),
      entry('search.tavily_api_key', '', {
        secret: true,
        help: { summary: 'Your Tavily key.', what: 'Credentials for tavily.com.' },
      }),
    ],
  };
}

function renderSection() {
  return render(
    <SettingsManifestProvider>
      <SearchSection />
    </SettingsManifestProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockGetConfig.mockResolvedValue(config());
  mockManifest.mockResolvedValue(manifest());
});

describe('SearchSection — manifest binding', () => {
  it('takes bounds and help from the manifest', async () => {
    const { container } = renderSection();
    const input = await screen.findByRole('spinbutton', { name: 'Max Results' });
    expect(input).toHaveAttribute('min', '1');
    expect(input).toHaveAttribute('max', '20');
    expect(screen.getByRole('button', { name: 'About Max Results' })).toBeInTheDocument();
    expect(container.querySelector('[data-setting="config:search.max_results"]'))
      .toBeInTheDocument();
  });

  it('surfaces the per-page grounding budget, which had no control before', async () => {
    renderSection();
    expect(await screen.findByRole('spinbutton', { name: 'Per-page budget (tokens)' }))
      .toHaveValue(1024);
  });

  it('offers help on the API key, which carries no other chrome', async () => {
    // Secrets have no default to compare and nothing to reset to, so the help
    // popover is the only way the authored billing detail reaches the screen.
    renderSection();
    expect(await screen.findByRole('button', { name: 'About Tavily API Key' }))
      .toBeInTheDocument();
  });
});

describe('SearchSection — source policy binds once', () => {
  it('is unchanged when the three inputs match the shipped dict', async () => {
    renderSection();
    await screen.findByRole('textbox', { name: 'Preferred sources' });
    // Comparing a comma-string to a dict would mark this changed forever; the
    // binding reconstructs the object before comparing.
    expect(screen.queryByLabelText('Changed from the default')).not.toBeInTheDocument();
  });

  it('marks changed once a domain is entered, and resets all three fields', async () => {
    mockGetConfig.mockResolvedValue(config({
      source_policy: { trusted: ['arxiv.org'], blocked: ['quora.com'], goggle: '' },
    }));
    renderSection();

    const preferred = await screen.findByRole('textbox', { name: 'Preferred sources' });
    expect(preferred).toHaveValue('arxiv.org');
    expect(screen.getByLabelText('Changed from the default')).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: 'Reset Preferred sources to default' })
    );
    await waitFor(() => expect(preferred).toHaveValue(''));
    expect(screen.getByRole('textbox', { name: 'Blocked sources' })).toHaveValue('');
  });

  it('sends the whole policy dict when one part changes', async () => {
    renderSection();
    const blocked = await screen.findByRole('textbox', { name: 'Blocked sources' });
    await userEvent.type(blocked, 'pinterest.com');

    await waitFor(() => expect(mockUpdateConfig).toHaveBeenCalled(), { timeout: 3000 });
    const payload = mockUpdateConfig.mock.calls.at(-1)![0] as {
      search: { source_policy?: Record<string, unknown> };
    };
    // Patching one leaf server-side would drop the siblings, so all three ride.
    expect(payload.search.source_policy).toEqual({
      trusted: [], blocked: ['pinterest.com'], goggle: '',
    });
  });
});
