class ApiClient {
    constructor(baseUrl: string) {
        this._baseUrl = baseUrl;
    }

    private _baseUrl: string;

    get baseUrl(): string {
        return this._baseUrl;
    }
}

// TODO make API URL dynamic
export const api = new ApiClient('http://localhost:12319/api');