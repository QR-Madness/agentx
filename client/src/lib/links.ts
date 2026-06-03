/**
 * Link safety helper. Mirrors the gate `MessageContent` uses for markdown
 * links: only `http(s)` URLs become real anchors (opened in a new tab); every
 * other scheme (`javascript:`, `data:`, …) is treated as inert text. This is
 * the URL sanitization for agent-authored content (citations, etc.).
 */

export function isHttpUrl(url: string | undefined | null): url is string {
  return typeof url === 'string' && /^https?:\/\//i.test(url.trim());
}

/** Host portion of a URL for compact display, or the raw string on parse failure. */
export function urlHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
