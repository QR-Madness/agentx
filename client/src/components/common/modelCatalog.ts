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
