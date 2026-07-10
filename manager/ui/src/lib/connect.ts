/* Connection-link encoder — wire format v1.
 *
 * Deliberate copy of the CANONICAL implementation in
 * client/src/lib/connectionString.ts (which both encodes and decodes; this
 * copy only encodes — the manager never consumes links). If the wire shape
 * ever changes, bump PAYLOAD_VERSION in BOTH files in the same change: the
 * client rejects unknown versions, so drift fails closed instead of silently
 * mis-connecting. A parity vector produced by this encoder is asserted in
 * client/src/lib/connectionString.test.ts.
 *
 * A connection link is a bearer credential (it can carry the gateway token):
 * build it on demand, never log it.
 */

const PAYLOAD_VERSION = 1;
const FRAGMENT_KEY = "connect";

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

function toBase64Url(input: string): string {
  const b64 = btoa(unescape(encodeURIComponent(input)));
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function encodeConnection(payload: ConnectionPayload): string {
  const wire: WirePayload = { v: PAYLOAD_VERSION, u: payload.url.replace(/\/+$/, "") };
  if (payload.gatewayToken) wire.g = payload.gatewayToken;
  if (payload.name) wire.n = payload.name;
  return toBase64Url(JSON.stringify(wire));
}

/**
 * Build a shareable link against an explicit web-app base. Unlike the client's
 * variant there is no fallback to the current origin — the manager's own
 * origin is never a valid app base, so the caller must supply one.
 */
export function buildConnectUrl(payload: ConnectionPayload, appBase: string): string {
  return `${appBase.replace(/\/+$/, "")}/#${FRAGMENT_KEY}=${encodeConnection(payload)}`;
}
