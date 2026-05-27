import { request as apiRequest } from './core';
import type { LanguageDetectResponse, TranslateRequest, TranslateResponse } from './types';

export const translationApi = {
  // === Translation ===

  async translate(request: TranslateRequest): Promise<TranslateResponse> {
    return apiRequest('/api/tools/translate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async detectLanguage(text: string): Promise<LanguageDetectResponse> {
    return apiRequest('/api/tools/language-detect-20', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },
};
