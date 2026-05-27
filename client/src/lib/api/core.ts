/**
 * Request layer shared by every domain module.
 *
 * Holds the base-URL resolver, the `request()` helper (auth/gateway headers,
 * 401 handling, backend-error normalization), and the SSE stream registry that
 * guarantees in-flight streams are aborted on unload / server switch.
 *
 * These exports are internal to `lib/api/` — only `setAuthRequired` is part of
 * the public surface (re-exported by `index.ts`).
 */

import {
  getActiveServer,
  getActiveServerId,
  getAuthToken,
  clearAuthToken,
  getActiveGatewayToken,
} from '../storage';
import { toApiError, classifyStatus, backendErrorMessage, type ApiError } from './errors';

// === Active stream registry ===
//
// Streams (SSE fetches with AbortControllers) register themselves here so the
// page can guarantee they are torn down on unload — including the hard reload
// triggered by `ServerContext.switchServer`. Without this, a server switch
// could leave a long-running SSE connection open against the previous host
// just long enough to overlap with the new one.

const activeStreamControllers = new Set<AbortController>();

export function registerStreamController(controller: AbortController): () => void {
  activeStreamControllers.add(controller);
  const cleanup = () => activeStreamControllers.delete(controller);
  controller.signal.addEventListener('abort', cleanup, { once: true });
  return cleanup;
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    for (const c of activeStreamControllers) {
      try { c.abort(); } catch { /* ignore */ }
    }
    activeStreamControllers.clear();
  });
}

/**
 * Whether the active server requires authentication. Set by AuthContext
 * after /api/auth/status resolves so request() can decide whether a missing
 * token is fatal (auth enabled) or fine (auth disabled — server ignores it).
 */
let _authRequired = false;

export function setAuthRequired(value: boolean): void {
  _authRequired = value;
}

export function getBaseUrl(): string {
  const server = getActiveServer();
  if (!server) {
    // Fallback to environment variable or default
    return import.meta.env.VITE_API_URL || 'http://localhost:12319';
  }
  return server.url;
}

export async function request<T>(
  path: string,
  options: RequestInit = {},
  skipAuth = false
): Promise<T> {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}${path}`;

  const defaultHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  // Cluster gateway header — required when the API is fronted by the
  // Nginx gateway in docker-compose.cluster.yml. Absent for local/LAN
  // servers, in which case the header simply isn't sent.
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) {
    defaultHeaders['AgentX-Gateway-Token'] = gatewayToken;
  }

  // Add auth token if available (unless skipped for auth endpoints)
  if (!skipAuth) {
    const token = getAuthToken();
    if (token) {
      defaultHeaders['X-Auth-Token'] = token;
    } else if (_authRequired) {
      // Diagnostic: surface where the token went missing so we can tell apart
      // "no active server" vs "active server but no token" vs "raw fetch bypassing this path".
      // Remove once root cause of memory-settings 401 cascade is confirmed.
      const activeId = getActiveServerId();
      console.warn('[api] no auth token at request time', {
        path,
        method: options.method ?? 'GET',
        activeServerId: activeId,
        hasActiveServer: !!getActiveServer(),
        knownServerKeys: Object.keys(localStorage).filter(k => k.startsWith('agentx:server:')),
      });

      // Fail fast only when the server is known to require auth — otherwise the
      // request would have been served fine without a token. Dispatching
      // auth-required lets ModalPortal close any open modals so their hooks
      // stop firing further unauthenticated requests.
      window.dispatchEvent(new CustomEvent('agentx:auth-required'));
      throw {
        message: 'Auth required: no token available',
        status: 401,
        kind: 'auth',
        details: { path, reason: 'missing-token' },
      } as ApiError;
    }
    // If auth is not required server-side, fall through and let the request fire
    // without a token — the server will accept it.
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });
  } catch (err) {
    // `fetch` rejects (TypeError) when the host is unreachable — surface a
    // clean network ApiError instead of leaking an opaque rejection.
    throw toApiError(err);
  }

  if (!response.ok) {
    // Handle 401 Unauthorized - clear token and notify app
    if (response.status === 401) {
      clearAuthToken();
      window.dispatchEvent(new CustomEvent('agentx:auth-required'));
    }

    let details: unknown;
    try {
      details = await response.json();
    } catch {
      details = await response.text();
    }

    // Prefer the backend's `{"error": "..."}` message; the status carries the
    // error type (see ApiErrorKind), so fall back to statusText only when the
    // body has no usable message.
    const message =
      backendErrorMessage(details) ??
      (response.statusText
        ? `API request failed: ${response.statusText}`
        : `API request failed (${response.status})`);

    throw {
      message,
      status: response.status,
      kind: classifyStatus(response.status),
      details,
    } as ApiError;
  }

  return response.json();
}
