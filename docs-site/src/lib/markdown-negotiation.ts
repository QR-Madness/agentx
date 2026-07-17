// Pure helpers for Accept-header content negotiation. Kept free of any Vercel imports so
// they can be exercised directly; `middleware.ts` holds the platform wiring.

/**
 * Parse an Accept header into `media-range -> q`.
 *
 * Ignores accept-extension params other than `q` (`text/html;level=1;q=0.5`), which is
 * what RFC 9110 §12.5.1 asks for. A malformed or absent `q` falls back to 1.
 */
export function parseAccept(header: string | null): Map<string, number> {
  const ranges = new Map<string, number>();
  if (!header) return ranges;

  for (const part of header.split(',')) {
    const [rawType, ...params] = part.split(';');
    const type = rawType.trim().toLowerCase();
    if (!type) continue;

    let q = 1;
    for (const param of params) {
      const [key, value] = param.split('=');
      if (key?.trim().toLowerCase() !== 'q') continue;
      const parsed = Number.parseFloat(value ?? '');
      if (Number.isFinite(parsed)) q = Math.min(Math.max(parsed, 0), 1);
    }

    // A repeated range keeps its strongest weight.
    ranges.set(type, Math.max(ranges.get(type) ?? 0, q));
  }
  return ranges;
}

/** Best q for `type`, honouring the `type/*` and `*​/*` wildcards that may cover it. */
function wildcardQ(ranges: Map<string, number>, type: string): number {
  const [group] = type.split('/');
  const candidates = [ranges.get(type), ranges.get(`${group}/*`), ranges.get('*/*')];
  return Math.max(0, ...candidates.filter((q): q is number => q !== undefined));
}

/**
 * Should this request be answered with Markdown instead of HTML?
 *
 * HTML is the default and must stay the default for browsers, so `text/markdown` only
 * wins when the client asks for it *by name* and weights it at least as highly as HTML.
 * Matching it through a wildcard would be wrong: `Accept: *​/*` — what curl, most crawlers
 * and plenty of HTTP clients send — would then silently get Markdown.
 *
 *   text/markdown                          -> true  (md 1   vs html 0)
 *   text/markdown, text/html;q=0.9         -> true  (md 1   vs html 0.9)
 *   text/markdown;q=0.9, text/html         -> false (md 0.9 vs html 1)
 *   text/html,application/xhtml+xml,...    -> false (browser: md never named)
 *   *​/*                                    -> false (md never named)
 *   text/markdown;q=0                      -> false (explicitly refused)
 */
export function prefersMarkdown(acceptHeader: string | null): boolean {
  const ranges = parseAccept(acceptHeader);
  const markdownQ = ranges.get('text/markdown');
  if (markdownQ === undefined || markdownQ === 0) return false;
  return markdownQ >= wildcardQ(ranges, 'text/html');
}

/**
 * Map a page URL to the URL of its Markdown twin, or null when the path isn't a page that
 * could have one.
 *
 * The twins are laid out by the build: `/` -> `/index.md` (src/pages/index.md.ts) and
 * `/docs/<id>` -> `/docs/<id>.md` (src/pages/docs/[...slug].md.ts). Astro emits pages as
 * directories, so both `/docs/x` and `/docs/x/` are live and must map alike.
 *
 * Returning a path here does not promise the twin exists — the caller checks it against
 * the generated manifest, which is what keeps twin-less pages (the interactive API
 * explorer) on HTML instead of rewriting them into a 404.
 */
export function twinPathFor(pathname: string): string | null {
  const path = pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;

  if (path === '' || path === '/') return '/index.md';
  if (path === '/docs') return '/docs/index.md';
  if (!path.startsWith('/docs/')) return null;

  // Anything already carrying an extension is a file (`.md` twins, images), not a page.
  const lastSegment = path.slice(path.lastIndexOf('/') + 1);
  if (lastSegment.includes('.')) return null;

  return `${path}.md`;
}
