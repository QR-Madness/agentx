import { api } from "../lib/api";

export interface TranslationRequest {
    text: string;
    sourceLanguage?: string;
    targetLanguage: string;
}

export interface TranslationResponse {
    translatedText: string;
}

export const postTranslation = async (request: TranslationRequest): Promise<TranslationResponse> => {
    return api.translate(request);
}