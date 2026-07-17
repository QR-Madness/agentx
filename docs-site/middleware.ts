import { next, rewrite } from '@vercel/functions';
import { MARKDOWN_TOKENS } from './src/generated/markdown-manifest';
import { prefersMarkdown, twinPathFor } from './src/lib/markdown-negotiation';

// Markdown for Agents — true `Accept` content negotiation.
//
// Agents that send `Accept: text/markdown` get the page's Markdown twin; everything else
// (browsers, crawlers, `Accept: */*`) keeps getting HTML. The site is a static Astro build
// on Vercel, which has no origin server to negotiate at — Routing Middleware is the
// documented way to do this, because it "runs globally before the cache", so a rewrite
// lands on a *different* cache key (`/docs/x.md`) and can never poison the HTML entry.
//
// Not Cloudflare's Content Converter: agentx.thejpnet.net is CNAME'd straight to Vercel
// (DNS-only), so Cloudflare's edge never sees this traffic and cannot convert it. We also
// serve the real authored Markdown source rather than HTML converted back to Markdown.
//
// This is `middleware.ts`, not `proxy.ts`: the middleware -> proxy rename is a Next.js 16
// file convention (nextjs.org/docs/messages/middleware-to-proxy). Vercel's convention for
// non-Next frameworks is still a root `middleware.ts` with `@vercel/functions` helpers, so
// a `proxy.ts` here would simply never be invoked.
export const config = {
  // Only pages can negotiate. Assets, /llms.txt and /openapi.yaml never reach this.
  matcher: ['/', '/docs/:path*'],
};

export default function middleware(request: Request): Response {
  // `Vary: Accept` rides every negotiable response, including the HTML one — a shared
  // cache that saw only the HTML must not serve it to an agent asking for Markdown.
  const vary = { Vary: 'Accept' };

  if (request.method !== 'GET' && request.method !== 'HEAD') return next();
  if (!prefersMarkdown(request.headers.get('accept'))) return next({ headers: vary });

  const twin = twinPathFor(new URL(request.url).pathname);
  // No twin (the interactive API explorer, a stale manifest) -> HTML remains correct.
  if (!twin || !(twin in MARKDOWN_TOKENS)) return next({ headers: vary });

  return rewrite(new URL(twin, request.url), {
    headers: {
      // Load-bearing. vercel.json's `/(.*).md` rule does NOT cover us: its `headers`
      // match the *requested* path (`/docs/x`), not the path we rewrite to — proved on a
      // preview by tagging that rule with a probe header, which a direct hit on
      // `/docs/x.md` echoed back and a negotiated `/docs/x` did not.
      //
      // The same rule's `X-Robots-Tag: noindex` therefore doesn't reach here either, which
      // is exactly right: it exists to keep twin *URLs* out of search as duplicates, while
      // this response is the canonical page and should stay as indexable as its HTML.
      // Don't "helpfully" set X-Robots-Tag here — on preview deploys that would override
      // Vercel's own noindex.
      'Content-Type': 'text/markdown; charset=utf-8',
      'x-markdown-tokens': String(MARKDOWN_TOKENS[twin]),
      ...vary,
    },
  });
}
