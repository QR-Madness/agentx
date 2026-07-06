/**
 * modelCatalog — shared model list cache.
 *
 * A single in-memory cache so the many model pickers across the app
 * (ModelPickerModal, ModelPickerField, chat metadata) don't each fire a
 * `/api/providers/models` request. Lives apart from any component so it can be
 * imported without pulling in React UI.
 */

import { api, type ModelInfo } from '../../lib/api';

let modelCache: ModelInfo[] | null = null;
let modelCachePromise: Promise<ModelInfo[]> | null = null;

export function fetchModelsOnce(): Promise<ModelInfo[]> {
  if (modelCache) return Promise.resolve(modelCache);
  if (!modelCachePromise) {
    modelCachePromise = api
      .listModels()
      .then(({ models }) => {
        modelCache = models;
        return models;
      })
      .catch((err) => {
        console.error('Failed to fetch models:', err);
        modelCachePromise = null;
        return [];
      });
  }
  return modelCachePromise;
}

/** Invalidate the shared cache so the next mount re-fetches */
export function invalidateModelCache() {
  modelCache = null;
  modelCachePromise = null;
}

// ---------------------------------------------------------------------------
// Recently-picked models (ModelPickerModal "Recent" group)
// ---------------------------------------------------------------------------

export const RECENT_MODELS_KEY = 'agentx:models:recent';
export const RECENT_MODELS_MAX = 5;

/** Pure MRU update: put `id` first, de-duped, capped at `max`. Empty ids are ignored. */
export function pushRecentModel(recent: readonly string[], id: string, max = RECENT_MODELS_MAX): string[] {
  if (!id) return recent.slice(0, max);
  return [id, ...recent.filter(r => r !== id)].slice(0, max);
}

/** Read the recent-model ids from localStorage (most-recent-first). Never throws. */
export function readRecentModels(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(RECENT_MODELS_KEY) ?? '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x): x is string => typeof x === 'string').slice(0, RECENT_MODELS_MAX);
  } catch {
    return [];
  }
}

/** Record a confirmed pick into the recent-model list. Never throws. */
export function writeRecentModel(id: string): void {
  if (!id) return;
  try {
    localStorage.setItem(RECENT_MODELS_KEY, JSON.stringify(pushRecentModel(readRecentModels(), id)));
  } catch {
    // Storage unavailable (private mode, quota) — recents are a nicety, not a requirement.
  }
}
