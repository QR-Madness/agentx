import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// The two rebuilt sections depend on server/notification contexts + the api
// client; mock them so the components render in isolation. The hook return
// values are stable singletons — ProvidersSection's effect keys off
// `activeMetadata`, so a fresh object per render would loop infinitely.
vi.mock('../../../contexts/ServerContext', () => {
  const value = {
    activeServer: { id: 's1', name: 'Local' },
    activeMetadata: { apiKeys: {} },
    updateMetadata: vi.fn(),
  };
  return { useServer: () => value };
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

vi.mock('../../../lib/api', () => ({
  api: {
    getConfig: vi.fn().mockResolvedValue({ preferences: { default_model: '' } }),
    getContextLimits: vi.fn().mockResolvedValue({
      lmstudio: { context_window: 8192, max_output_tokens: 2048 },
      models: {},
    }),
    updateConfig: vi.fn().mockResolvedValue({}),
    updateContextLimits: vi.fn().mockResolvedValue({}),
  },
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
import ModelsSection from './ModelsSection';

describe('ProvidersSection', () => {
  it('renders the provider cards, badges and a save action', () => {
    render(<ProvidersSection />);
    // All five providers present.
    for (const name of ['LM Studio', 'Anthropic', 'OpenAI', 'OpenRouter', 'Vercel AI Gateway']) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    // Tinted badge pills survived the primitive swap.
    expect(screen.getByText('High-Reasoning')).toBeInTheDocument();
    expect(screen.getByText('Cloud Router')).toBeInTheDocument();
    // Save button is a real <button> via the Button primitive, and starts
    // disabled because the form matches the last-loaded state (not dirty).
    const save = screen.getByRole('button', { name: /save to server/i });
    expect(save).toBeInTheDocument();
    expect(save).toBeDisabled();
    expect(screen.queryByText('Unsaved changes')).toBeNull();
  });
});

describe('ModelsSection', () => {
  it('renders the default-model picker, header and the LM Studio limits card', async () => {
    render(<ModelsSection />);
    expect(screen.getByText('Default Model')).toBeInTheDocument();
    expect(screen.getByText('Global default model')).toBeInTheDocument();
    expect(screen.getByText('Model Context Limits')).toBeInTheDocument();
    // getContextLimits resolves → the limits card appears with kit NumberFields.
    expect(await screen.findByText('Local')).toBeInTheDocument();
    expect(screen.getByText('Context Window (tokens)')).toBeInTheDocument();
    expect(screen.getByText('Max Output Tokens')).toBeInTheDocument();
    // The explicit Save button is gone — limits autosave (SaveStatusChip).
    expect(screen.queryByRole('button', { name: /save limits/i })).toBeNull();
  });
});
