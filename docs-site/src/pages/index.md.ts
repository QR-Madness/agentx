import type { APIRoute } from 'astro';
// `?raw` keeps the twin a plain authored file on disk, which is what lets
// scripts/gen-markdown-manifest.mjs estimate its tokens without executing TypeScript.
import homepage from '../content/homepage.md?raw';

// Markdown-for-agents: the home page's Markdown twin at `/index.md`.
//
// The landing page (index.astro) is bespoke composed sections, not a content-collection
// entry, so it has no generated twin the way `/docs/*` does — but it is the first thing a
// crawler or readiness scanner hits, so it needs one. The prose is authored for agents
// (facts and links) rather than copied from the marketing sections.
//
// `middleware.ts` rewrites `/` here when the request prefers `text/markdown`; the URL is
// also advertised from index.astro via `<link rel="alternate" type="text/markdown">`.
export const GET: APIRoute = () =>
  new Response(homepage, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'X-Robots-Tag': 'noindex',
    },
  });
