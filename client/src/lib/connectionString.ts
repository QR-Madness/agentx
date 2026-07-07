/**
 * connectionString — shareable "connect to this server" links.
 *
 * A host copies a link; the recipient opens it and only enters a password. The
 * link carries the server URL (+ optional gateway token + display name) in the
 * URL **fragment** (`#connect=…`), which browsers never send to the server, so
 * the shared secret stays off the wire / out of access logs.
 *
 * The payload is NOT encrypted — treat a connection link as a bearer credential.
 * The receiving app shows a confirmation gate before adding the server (see
 * components/ConnectGate.tsx).
 */

/** Optional public web origin the desktop app should point share links at. */
export const PUBLIC_APP_URL: string =
  (import.meta.env.VITE_PUBLIC_APP_URL as string | undefined)?.replace(/\/+$/, '') ?? '';

const PAYLOAD_VERSION = 1;
const FRAGMENT_KEY = 'connect';

export interface ConnectionPayload {
  url: string;
  gatewayToken?: string;
  name?: string;
}

/** Compact wire shape: keep keys short since this rides in a URL. */
interface WirePayload {
  v: number;
  u: string;
  g?: string;
  n?: string;
}

// --- base64url (unicode-safe) ------------------------------------------------

function toBase64Url(input: string): string {
  const b64 = btoa(unescape(encodeURIComponent(input)));
  return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(input: string): string {
  const b64 = input.replace(/-/g, '+').replace(/_/g, '/');
  const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4);
  return decodeURIComponent(escape(atob(padded)));
}

/** Validate an http(s) URL and return it trimmed of trailing slashes, else null. */
function normalizeUrl(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  return trimmed.replace(/\/+$/, '');
}

// --- encode / decode ---------------------------------------------------------

export function encodeConnection(payload: ConnectionPayload): string {
  const wire: WirePayload = { v: PAYLOAD_VERSION, u: payload.url.replace(/\/+$/, '') };
  if (payload.gatewayToken) wire.g = payload.gatewayToken;
  if (payload.name) wire.n = payload.name;
  return toBase64Url(JSON.stringify(wire));
}

/** Decode a bare code (the value after `#connect=`). Returns null on anything invalid. */
export function parseConnectCode(code: string): ConnectionPayload | null {
  let obj: unknown;
  try {
    obj = JSON.parse(fromBase64Url(code));
  } catch {
    return null;
  }
  if (!obj || typeof obj !== 'object') return null;
  const wire = obj as Partial<WirePayload>;
  if (wire.v !== PAYLOAD_VERSION) return null;
  const url = normalizeUrl(wire.u);
  if (!url) return null;
  return {
    url,
    gatewayToken: typeof wire.g === 'string' && wire.g ? wire.g : undefined,
    name: typeof wire.n === 'string' && wire.n ? wire.n : undefined,
  };
}

/** Extract + decode a connection payload from a URL hash (`#connect=…`). */
export function parseConnectFragment(hash: string): ConnectionPayload | null {
  const raw = hash.replace(/^#/, '');
  if (!raw) return null;
  const code = new URLSearchParams(raw).get(FRAGMENT_KEY);
  return code ? parseConnectCode(code) : null;
}

/**
 * Build a shareable link. Uses `appBase` if given, else the configured
 * `PUBLIC_APP_URL`, else the current origin (fine for the web build; a desktop
 * host should set `VITE_PUBLIC_APP_URL` since `tauri://localhost` isn't shareable).
 */
export function buildConnectUrl(payload: ConnectionPayload, appBase?: string): string {
  const base =
    (appBase?.replace(/\/+$/, '') || PUBLIC_APP_URL) ||
    (typeof window !== 'undefined' ? window.location.origin : '');
  return `${base}/#${FRAGMENT_KEY}=${encodeConnection(payload)}`;
}

// --- boot-time consumption ---------------------------------------------------

let pending: ConnectionPayload | null = null;
let consumed = false;

/**
 * Read + strip a `#connect=` fragment. Call ONCE, synchronously, at boot before
 * React renders: it stashes the payload for ConnectGate and immediately removes
 * the fragment from the URL/history so the token doesn't linger. Idempotent.
 */
export function consumeConnectFragment(): ConnectionPayload | null {
  if (consumed) return pending;
  consumed = true;
  if (typeof window === 'undefined') return null;

  const payload = parseConnectFragment(window.location.hash);
  if (payload) {
    pending = payload;
    const clean = window.location.pathname + window.location.search;
    window.history.replaceState(null, '', clean || '/');
  }
  return pending;
}

export function getPendingConnect(): ConnectionPayload | null {
  return pending;
}

export function clearPendingConnect(): void {
  pending = null;
}
