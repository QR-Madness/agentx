import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ConsolidationSettingsPanel } from './ConsolidationSettingsPanel';

vi.mock('../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

vi.mock('../ui/ConfirmDialog', () => ({ useConfirm: () => vi.fn().mockResolvedValue(false) }));

// The model pickers open a catalog modal; a plain readout is enough here.
vi.mock('../common/ModelPickerField', () => ({
  ModelPickerField: ({ label, value }: { label: string; value: string }) => (
    <div data-testid={`model-${label}`}>{value || 'inherit'}</div>
  ),
}));

const mockGetSettings = vi.fn();

vi.mock('../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: {
      getConsolidationSettings: () => mockGetSettings(),
      updateConsolidationSettings: vi.fn().mockResolvedValue({ success: true, updated: [] }),
      getModelRoles: vi.fn().mockResolvedValue({ roles: {}, members: [] }),
      resetMemory: vi.fn(),
      clearStuckJobs: vi.fn(),
    },
  };
});

const NOTE = /No effect on Anthropic models/;

/** The stages with a Temperature slider, in the order the panel renders them. */
const STAGES = ['extraction', 'relevance_filter', 'combined_extraction', 'contradiction', 'correction'] as const;
type Stage = (typeof STAGES)[number];

function settingsWith(models: Partial<Record<Stage, string>>) {
  return {
    feature_default_model: '',
    relevance_filter_enabled: true,
    contradiction_detection_enabled: true,
    correction_detection_enabled: true,
    ...Object.fromEntries(STAGES.map(stage => [`${stage}_model`, models[stage] ?? ''])),
  };
}

/** Whether a slider's `aria-describedby` resolves to the note text. */
function isNoted(slider: HTMLElement): boolean {
  const ids = (slider.getAttribute('aria-describedby') ?? '').split(' ').filter(Boolean);
  return ids.some(id => NOTE.test(document.getElementById(id)?.textContent ?? ''));
}

async function notedStages(): Promise<boolean[]> {
  render(<ConsolidationSettingsPanel onConsolidate={vi.fn().mockResolvedValue(undefined)} />);
  const sliders = await screen.findAllByRole('slider', { name: 'Temperature' });
  expect(sliders).toHaveLength(STAGES.length);
  return sliders.map(isNoted);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('ConsolidationSettingsPanel — temperature notes', () => {
  it('notes each stage whose own model is a direct Anthropic model', async () => {
    mockGetSettings.mockResolvedValue(settingsWith({
      extraction: 'anthropic:claude-3-5-haiku-latest',
      relevance_filter: '',
      combined_extraction: 'openrouter:anthropic/claude-3.5-haiku',
      contradiction: 'role:deep_reasoning',
      correction: 'anthropic:claude-3-5-haiku-latest',
    }));
    expect(await notedStages()).toEqual([true, false, false, false, true]);
  });

  it('follows each stage to its own model, not a neighbour', async () => {
    mockGetSettings.mockResolvedValue(settingsWith({
      extraction: 'inherit',
      relevance_filter: 'anthropic:claude-3-5-haiku-latest',
      combined_extraction: 'anthropic:claude-3-5-haiku-latest',
      contradiction: 'anthropic:claude-3-5-haiku-latest',
      correction: 'vercel:anthropic/claude-3.5-haiku',
    }));
    expect(await notedStages()).toEqual([false, true, true, true, false]);
  });

  it('shows no note while every stage inherits, even under an Anthropic bulk default', async () => {
    // The bulk default applies only after the stage's role, which resolves server-side.
    mockGetSettings.mockResolvedValue({
      ...settingsWith({}),
      feature_default_model: 'anthropic:claude-3-5-haiku-latest',
    });
    expect(await notedStages()).toEqual([false, false, false, false, false]);
  });
});
