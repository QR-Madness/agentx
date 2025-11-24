import {api} from "../lib/api";

const urls = {
    translate: () => `${api.baseUrl}/tools/translate`
}

export interface TranslationRequest {
    text: string;
    sourceLanguage: string;
    targetLanguage: string;
}

export interface TranslationResponse {
    translatedText: string;
}

export const postTranslation = async (request: TranslationRequest): Promise<TranslationResponse> => {
    const response = await fetch(urls.translate(), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
    });
    return await response.json();
}