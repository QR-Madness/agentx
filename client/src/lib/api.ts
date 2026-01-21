class ApiClient {
    constructor(baseUrl: string) {
        this._baseUrl = baseUrl;
    }

    private _baseUrl: string;

    get baseUrl(): string {
        return this._baseUrl;
    }
}

// Use environment variable for API URL, with fallback for development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:12319/api';
export const api = new ApiClient(API_URL);