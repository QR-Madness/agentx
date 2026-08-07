import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import ThinkingSection from './ThinkingSection';
import PlannerSection from './PlannerSection';
import AlloySection from './AlloySection';
import { SettingsManifestProvider } from '../SettingsManifestContext';

vi.mock('../../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

// The model pickers open a catalog modal; the stub forwards the anchor so a
// dropped binding fails here rather than passing silently.
vi.mock('../../common/ModelPickerField', () => ({
  ModelPickerField: ({ label, value, binding }: {
    label: string; value: string;
    binding?: { entry: { store: string; key: string } } | null;
  }) => (
    <div
      data-testid={`model-${label}`}
      data-setting={binding ? `${binding.entry.store}:${binding.entry.key}` : undefined}
    >
      {value || 'inherit'}
    </div>
  ),
}));

const mockGetConfig = vi.fn();
const mockManifest = vi.fn();

vi.mock('../../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: {
      getConfig: () => mockGetConfig(),
      updateConfig: vi.fn().mockResolvedValue({ success: true }),
      getSettingsManifest: () => mockManifest(),
    },
  };
});

/** Every key these three screens own, with the section each declares. */
const ENTRIES: [string, unknown, string][] = [
  ['reasoning.chat_patterns_enabled', true, 'thinking'],
  ['reasoning.step_back_timeout_seconds', 20, 'thinking'],
  ['reasoning.sc_k', 3, 'thinking'],
  ['reasoning.classifier_model', '', 'thinking'],
  ['reasoning.step_back_model', '', 'thinking'],
  ['reasoning.sc_model', '', 'thinking'],
  ['planner.enabled', true, 'planner'],
  ['planner.model', '', 'planner'],
  ['planner.max_subtasks', 6, 'planner'],
  // Declared onto Feature Prompts, where it is actually edited.
  ['planner.prompt_override', '', 'feature-prompts'],
  ['alloy.allow_adhoc_delegation', true, 'alloy'],
  ['alloy.non_blocking_delegations', true, 'alloy'],
];

function manifest() {
  return {
    version: 3,
    generated_at: '2026-08-07T00:00:00Z',
    counts: { total: ENTRIES.length },
    sections: [],
    entries: ENTRIES.map(([key, dflt, ui_section]) => ({
      key, store: 'config' as const, type: typeof dflt,
      default: dflt, value: dflt, secret: false,
      writable_via: '/api/config/update', ui_section,
      help: { summary: `About ${key}.` },
    })),
  };
}

function renderSection(Component: () => React.ReactElement) {
  return render(
    <SettingsManifestProvider>
      <Component />
    </SettingsManifestProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockGetConfig.mockResolvedValue({ reasoning: {}, planner: {}, alloy: {} });
  mockManifest.mockResolvedValue(manifest());
});

describe('Thinking Patterns', () => {
  it('surfaces the step-back timeout, which had no control before', async () => {
    const { container } = renderSection(ThinkingSection);
    expect(await screen.findByRole('spinbutton', { name: 'Step-back timeout (seconds)' }))
      .toHaveValue(20);
    expect(container.querySelector('[data-setting="config:reasoning.step_back_timeout_seconds"]'))
      .toBeInTheDocument();
  });

  it('binds its model pickers', async () => {
    const { container } = renderSection(ThinkingSection);
    await screen.findByRole('spinbutton', { name: 'Consensus samples (k)' });
    expect(container.querySelector('[data-setting="config:reasoning.classifier_model"]'))
      .toBeInTheDocument();
  });

  it('takes bounds from the manifest and offers help', async () => {
    renderSection(ThinkingSection);
    await screen.findByRole('spinbutton', { name: 'Consensus samples (k)' });
    expect(screen.getByRole('button', { name: 'About Enable thinking patterns' }))
      .toBeInTheDocument();
  });
});

describe('Task Planner', () => {
  it('surfaces the subtask cap beside the threshold it pairs with', async () => {
    renderSection(PlannerSection);
    expect(await screen.findByRole('spinbutton', { name: 'Max subtasks' })).toHaveValue(6);
  });

  it('does not render the prompt override — Feature Prompts owns it', async () => {
    // The screen says so in prose; the declaration now agrees.
    const { container } = renderSection(PlannerSection);
    await screen.findByRole('spinbutton', { name: 'Max subtasks' });
    expect(container.querySelector('[data-setting="config:planner.prompt_override"]')).toBeNull();
  });
});

describe('Agent Teams', () => {
  it('surfaces non-blocking dispatch, which had no control before', async () => {
    const { container } = renderSection(AlloySection);
    expect(await screen.findByRole('checkbox', { name: /Non-blocking dispatch/ }))
      .toBeInTheDocument();
    expect(container.querySelector('[data-setting="config:alloy.non_blocking_delegations"]'))
      .toBeInTheDocument();
  });
});
