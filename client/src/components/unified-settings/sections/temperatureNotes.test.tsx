/**
 * Temperature notes on the config-backed sections (Task Planner, Prompt
 * Enhancement): each slider follows its own section's model setting.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import PlannerSection from './PlannerSection';
import PromptsSection from './PromptsSection';

vi.mock('../../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

// The model pickers open a catalog modal; a plain readout is enough here.
vi.mock('../../common/ModelPickerField', () => ({
  ModelPickerField: ({ label, value }: { label: string; value: string }) => (
    <div data-testid={`model-${label}`}>{value || 'inherit'}</div>
  ),
}));

const mockGetConfig = vi.fn();

vi.mock('../../../lib/api', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    api: {
      getConfig: () => mockGetConfig(),
      updateConfig: vi.fn().mockResolvedValue({}),
    },
  };
});

const NOTE = /No effect on Anthropic models/;
const DIRECT = 'anthropic:claude-3-5-haiku-latest';
const NOT_DIRECT = [
  'role:summarizer',
  'openrouter:anthropic/claude-3.5-haiku',
  'vercel:anthropic/claude-3.5-haiku',
];

beforeEach(() => {
  vi.clearAllMocks();
});

describe('PlannerSection — temperature note', () => {
  it('notes a direct Anthropic planner model', async () => {
    mockGetConfig.mockResolvedValue({ planner: { model: DIRECT } });
    render(<PlannerSection />);
    expect(await screen.findByRole('slider', { name: 'Temperature' })).toHaveAccessibleDescription(NOTE);
  });

  it.each([null, ...NOT_DIRECT])('shows no note for planner.model %j', async model => {
    mockGetConfig.mockResolvedValue({ planner: { model } });
    render(<PlannerSection />);
    expect(await screen.findByRole('slider', { name: 'Temperature' })).not.toHaveAccessibleDescription(NOTE);
    expect(screen.queryByText(NOTE)).not.toBeInTheDocument();
  });
});

describe('PromptsSection — temperature note', () => {
  it('notes a direct Anthropic enhancement model', async () => {
    mockGetConfig.mockResolvedValue({ prompt_enhancement: { model: DIRECT } });
    render(<PromptsSection />);
    expect(await screen.findByRole('slider', { name: 'Temperature' })).toHaveAccessibleDescription(NOTE);
  });

  it('shows an unset model as inherited, with no note', async () => {
    // Empty follows the summarizer role server-side; the section must not
    // stand a literal Anthropic model in for it.
    mockGetConfig.mockResolvedValue({ prompt_enhancement: { model: '' } });
    render(<PromptsSection />);
    expect(await screen.findByRole('slider', { name: 'Temperature' })).not.toHaveAccessibleDescription(NOTE);
    expect(screen.getByTestId('model-Enhancement Model')).toHaveTextContent('inherit');
  });

  it.each(NOT_DIRECT)('shows no note for prompt_enhancement.model %j', async model => {
    mockGetConfig.mockResolvedValue({ prompt_enhancement: { model } });
    render(<PromptsSection />);
    expect(await screen.findByRole('slider', { name: 'Temperature' })).not.toHaveAccessibleDescription(NOTE);
    expect(screen.queryByText(NOTE)).not.toBeInTheDocument();
  });
});
