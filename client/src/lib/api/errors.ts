/**
 * API error contract.
 *
 * The backend (see `api/agentx_ai/exceptions.py` + `utils/responses.py`) returns
 * a flat `{"error": "<message>"}` body and encodes the *type* of failure in the
 * HTTP status code — there are no error codes. These helpers normalize anything
 * caught from an API/stream call into a stable {@link ApiError} shape and derive
 * a coarse, human-meaningful {@link ApiErrorKind} from the status.
 */

/**
 * Coarse classification of an API failure, derived from the HTTP status:
 *   400 bad_request · 401 auth · 404 not_found · 502 upstream (provider) ·
 *   503 unavailable (MCP / memory store) · 0/timeout network · else server.
 */
export type ApiErrorKind =
  | 'bad_request'
  | 'auth'
  | 'not_found'
  | 'upstream'
  | 'unavailable'
  | 'network'
  | 'server';

export interface ApiError {
  message: string;
  status: number;
  kind: ApiErrorKind;
  details?: unknown;
}

/** Map an HTTP status to its coarse {@link ApiErrorKind}. */
export function classifyStatus(status: number): ApiErrorKind {
  switch (status) {
    case 400: return 'bad_request';
    case 401: return 'auth';
    case 404: return 'not_found';
    case 408: return 'network';
    case 502: return 'upstream';
    case 503: return 'unavailable';
    default:
      return status === 0 ? 'network' : 'server';
  }
}

/** Type guard for the {@link ApiError} shape thrown by the request layer. */
export function isApiError(err: unknown): err is ApiError {
  return (
    typeof err === 'object' &&
    err !== null &&
    'status' in err &&
    'message' in err &&
    'kind' in err
  );
}

/**
 * Normalize anything caught from an API/stream call into an {@link ApiError}.
 *
 * Handles three shapes: an already-normalized `ApiError` (passed through), a
 * network-layer `TypeError`/`Error` (→ `network`), and a bare string (as
 * emitted by SSE `error` events → `server`).
 */
export function toApiError(err: unknown): ApiError {
  if (isApiError(err)) return err;
  if (typeof err === 'string') {
    return { message: err, status: 0, kind: 'server' };
  }
  if (err instanceof Error) {
    // `fetch` rejects with a TypeError when the host is unreachable.
    const isNetwork = err.name === 'TypeError';
    return {
      message: isNetwork ? 'Cannot reach server' : err.message,
      status: 0,
      kind: isNetwork ? 'network' : 'server',
      details: err,
    };
  }
  return { message: 'Unknown error', status: 0, kind: 'server', details: err };
}

/** Extract a user-facing message from any caught error. */
export function apiErrorMessage(err: unknown): string {
  return toApiError(err).message;
}

/** Pull the backend's `{"error": "..."}` message out of a parsed body, if present. */
export function backendErrorMessage(details: unknown): string | undefined {
  if (typeof details === 'object' && details !== null && 'error' in details) {
    const msg = (details as { error?: unknown }).error;
    if (typeof msg === 'string' && msg.trim()) return msg;
  }
  return undefined;
}
