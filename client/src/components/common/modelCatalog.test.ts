import { describe, it, expect, vi, beforeEach } from 'vitest';

const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock('../../lib/api', () => ({
  api: { listModels },
}));

import {
  fetchModelsOnce, invalidateModelCache,
  pushRecentModel, readRecentModels, writeRecentModel, RECENT_MODELS_KEY,
} from './modelCatalog';

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

describe('recent models', () => {
  beforeEach(() => {
    localStorage.removeItem(RECENT_MODELS_KEY);
  });

  it('pushRecentModel prepends, de-dupes, and caps at 5', () => {
    expect(pushRecentModel([], 'a')).toEqual(['a']);
    expect(pushRecentModel(['a', 'b'], 'b')).toEqual(['b', 'a']);
    expect(pushRecentModel(['a', 'b', 'c', 'd', 'e'], 'f')).toEqual(['f', 'a', 'b', 'c', 'd']);
    // Empty id is ignored (System default is never a "recent")
    expect(pushRecentModel(['a'], '')).toEqual(['a']);
  });

  it('pushRecentModel does not mutate its input', () => {
    const input = ['a', 'b'];
    pushRecentModel(input, 'c');
    expect(input).toEqual(['a', 'b']);
  });

  it('write/read round-trips most-recent-first through localStorage', () => {
    writeRecentModel('anthropic:claude');
    writeRecentModel('openai:gpt');
    writeRecentModel('anthropic:claude');
    expect(readRecentModels()).toEqual(['anthropic:claude', 'openai:gpt']);
  });

  it('readRecentModels tolerates garbage storage', () => {
    localStorage.setItem(RECENT_MODELS_KEY, 'not json{');
    expect(readRecentModels()).toEqual([]);
    localStorage.setItem(RECENT_MODELS_KEY, JSON.stringify({ nope: true }));
    expect(readRecentModels()).toEqual([]);
    localStorage.setItem(RECENT_MODELS_KEY, JSON.stringify(['ok', 42, null, 'also-ok']));
    expect(readRecentModels()).toEqual(['ok', 'also-ok']);
  });
});
