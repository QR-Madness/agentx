import { describe, it, expect, vi, beforeEach } from 'vitest';

const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock('../../lib/api', () => ({
  api: { listModels },
}));

import { fetchModelsOnce, invalidateModelCache } from './modelCatalog';

const MODELS = [
  { id: 'anthropic:claude', name: 'Claude', provider: 'anthropic' },
  { id: 'openai:gpt', name: 'GPT', provider: 'openai' },
];

describe('modelCatalog', () => {
  beforeEach(() => {
    invalidateModelCache();
    listModels.mockReset();
    listModels.mockResolvedValue({ models: MODELS });
  });

  it('fetches once and caches across calls', async () => {
    const a = await fetchModelsOnce();
    const b = await fetchModelsOnce();
    expect(a).toEqual(MODELS);
    expect(b).toBe(a); // same cached array
    expect(listModels).toHaveBeenCalledTimes(1);
  });

  it('refetches after invalidateModelCache', async () => {
    await fetchModelsOnce();
    invalidateModelCache();
    await fetchModelsOnce();
    expect(listModels).toHaveBeenCalledTimes(2);
  });

  it('returns an empty list (and clears the promise) on failure', async () => {
    listModels.mockRejectedValueOnce(new Error('boom'));
    const first = await fetchModelsOnce();
    expect(first).toEqual([]);
    // The failed promise was cleared, so the next call retries rather than
    // resolving the cached failure forever.
    const second = await fetchModelsOnce();
    expect(second).toEqual(MODELS);
    expect(listModels).toHaveBeenCalledTimes(2);
  });
});
