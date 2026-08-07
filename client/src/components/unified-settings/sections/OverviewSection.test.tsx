import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import OverviewSection from './OverviewSection';
import { SettingsManifestProvider } from '../SettingsManifestContext';
import { SettingsSearchProvider } from '../SettingsSearchContext';

const mockManifest = vi.fn();

vi.mock('../../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: { getSettingsManifest: () => mockManifest() },
  };
});

type EntryOverrides = Partial<{
  key: string; store: 'config' | 'memory'; type: string;
  default: unknown; value: unknown;
  secret: boolean; writable_via: string | null; ui_section: string;
  help: { summary?: string };
}>;

function entry(key: string, over: EntryOverrides = {}) {
  return {
    key,
    store: 'config' as const,
    type: 'str',
    default: '',
    value: '',
    secret: false,
    writable_via: '/api/config/update',
    ...over,
  };
}

/**
 * A small manifest with one changed setting, one configured provider key, and
 * section metadata — enough to exercise every branch the Overview has.
 */
function manifest(over: { entries?: ReturnType<typeof entry>[] } = {}) {
  return {
    version: 3,
    generated_at: '2026-08-07T00:00:00Z',
    counts: { total: 3 },
    sections: [
      {
        id: 'overview',
        label: 'Overview',
        writable_count: 0,
        help: { summary: "What you've changed, and where everything lives." },
      },
      {
        id: 'search',
        label: 'Infrastructure → Web Search',
        writable_count: 22,
        help: { summary: 'How the agent searches the web — and what that costs.' },
      },
      { id: 'providers', label: 'Infrastructure → Model Providers', writable_count: 12 },
      { id: 'model-roles', label: 'Infrastructure → Model Roles', writable_count: 3 },
    ],
    entries: over.entries ?? [
      entry('search.max_results', {
        type: 'int', default: 5, value: 12, ui_section: 'search',
        help: { summary: 'How many results a single search returns.' },
      }),
      entry('providers.anthropic.api_key', {
        secret: true, value: '***', ui_section: 'providers',
      }),
      entry('models.roles.summarizer', { ui_section: 'model-roles' }),
      entry('search.tavily_api_key', { secret: true, value: '', ui_section: 'search' }),
    ],
  };
}

function renderOverview(props: Parameters<typeof OverviewSection>[0] = {}) {
  return render(
    <SettingsManifestProvider>
      <SettingsSearchProvider>
        <OverviewSection {...props} />
      </SettingsSearchProvider>
    </SettingsManifestProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockManifest.mockResolvedValue(manifest());
});

describe('OverviewSection — what you have changed', () => {
  it('lists changed settings under their section and lands on the control', async () => {
    const onFocusSetting = vi.fn();
    renderOverview({ onFocusSetting });

    const row = await screen.findByRole('button', { name: /Search · Max results/ });
    expect(row).toHaveTextContent('5');
    expect(row).toHaveTextContent('12');

    await userEvent.click(row);
    expect(onFocusSetting).toHaveBeenCalledWith('search', 'config:search.max_results');
  });

  it('never counts a secret as changed — its value arrives redacted', async () => {
    renderOverview();
    await screen.findByRole('button', { name: /Search · Max results/ });
    // '***' !== '' would otherwise read as "changed from the default".
    expect(screen.queryByText(/Anthropic · Api key/)).not.toBeInTheDocument();
    expect(screen.getByText('Changed from defaults').parentElement)
      .toHaveTextContent('1');
  });
});

describe('OverviewSection — getting set up', () => {
  it('ticks a step whose key is configured and flags the ones that are not', async () => {
    renderOverview();
    const provider = await screen.findByRole('button', { name: /Connect a model provider/ });
    expect(provider).toHaveTextContent('Configured');

    // A secret that is present-but-empty is not configured.
    expect(screen.getByRole('button', { name: /Add a web-search key/ }))
      .not.toHaveTextContent('Configured');
  });

  it('disappears once every step is done', async () => {
    mockManifest.mockResolvedValue(manifest({
      entries: [
        entry('providers.anthropic.api_key', { secret: true, value: '***', ui_section: 'providers' }),
        entry('models.roles.summarizer', { value: 'anthropic:opus', ui_section: 'model-roles' }),
        entry('search.tavily_api_key', { secret: true, value: '***', ui_section: 'search' }),
      ],
    }));
    renderOverview();
    await screen.findByText('Changed from defaults');
    expect(screen.queryByText('Getting set up')).not.toBeInTheDocument();
  });

  it('navigates to the section a step belongs to', async () => {
    const onNavigate = vi.fn();
    renderOverview({ onNavigate });
    await userEvent.click(
      await screen.findByRole('button', { name: /Pick a model for each role/ })
    );
    expect(onNavigate).toHaveBeenCalledWith('model-roles');
  });
});

describe('OverviewSection — tiles', () => {
  it('carries each screen’s blurb and its changed count', async () => {
    renderOverview();
    const tiles = await screen.findByText('Everything else');
    const region = tiles.parentElement!;

    const webSearch = within(region).getByRole('button', { name: /Web Search/ });
    expect(webSearch).toHaveTextContent('How the agent searches the web');
    // One changed setting lives on that screen.
    expect(webSearch).toHaveTextContent('1');
  });

  it('still lists a screen the manifest says nothing about', async () => {
    renderOverview();
    const tiles = await screen.findByText('Everything else');
    expect(within(tiles.parentElement!).getByRole('button', { name: /Appearance/ }))
      .toBeInTheDocument();
  });
});

describe('OverviewSection — search', () => {
  it('finds a setting by its humanised name and lands on it', async () => {
    const onFocusSetting = vi.fn();
    renderOverview({ onFocusSetting });
    await screen.findByText('Changed from defaults');

    await userEvent.type(screen.getByRole('searchbox', { name: 'Search every setting' }), 'max results');

    const hit = await screen.findByRole('button', { name: /Search · Max results/ });
    await userEvent.click(hit);
    expect(onFocusSetting).toHaveBeenCalledWith('search', 'config:search.max_results');
  });

  it('replaces the landing content while searching', async () => {
    renderOverview();
    await screen.findByText('Changed from defaults');
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search every setting' }), 'max results');
    await waitFor(() =>
      expect(screen.queryByText('Everything else')).not.toBeInTheDocument()
    );
  });

  it('says so when nothing matches', async () => {
    renderOverview();
    await screen.findByText('Changed from defaults');
    await userEvent.type(
      screen.getByRole('searchbox', { name: 'Search every setting' }),
      'zzzznotasetting'
    );
    expect(await screen.findByText(/Nothing matches/)).toBeInTheDocument();
  });
});
