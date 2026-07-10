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
import ModelRolesSection from './ModelRolesSection';
import { SECTION_HIERARCHY, getAllSections, findSectionById } from './index';

describe('ProvidersSection', () => {
  it('renders the provider cards, badges and a save action', () => {
    render(<ProvidersSection />);
    // All five providers present.
    for (const name of ['LM Studio', 'Anthropic', 'OpenAI', 'OpenRouter', 'Vercel AI Gateway']) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    // Capability-tier badges (OpenRouter primary, Anthropic/OpenAI/Vercel beta,
    // LM Studio local). "Beta"/"Local" also appear as tier group eyebrows, so
    // count tolerantly (3 beta badges + eyebrow; 1 local badge + eyebrow).
    expect(screen.getByText('Recommended')).toBeInTheDocument();
    expect(screen.getAllByText('Beta').length).toBeGreaterThanOrEqual(3);
    expect(screen.getAllByText('Local').length).toBeGreaterThanOrEqual(1);
    // Save button is a real <button> via the Button primitive, and starts
    // disabled because the form matches the last-loaded state (not dirty).
    const save = screen.getByRole('button', { name: /save to server/i });
    expect(save).toBeInTheDocument();
    expect(save).toBeDisabled();
    expect(screen.queryByText('Unsaved changes')).toBeNull();
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
