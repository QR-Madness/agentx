import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';

// Markdown-for-agents: serve every docs page's raw Markdown at `<page>.md` so agents
// requesting `text/markdown` get clean source instead of rendered HTML. This is the
// static-host form of content negotiation (Astro static + Vercel has no server to do
// true Accept-header negotiation) — the URL is advertised via a per-page
// `<link rel="alternate" type="text/markdown">` (DocsLayout) and a site-wide `/llms.txt`.
//
// The route sits alongside `[...slug].astro`; their URLs never collide (`/docs/x` vs
// `/docs/x.md`). The glob loader's `id` is the file path sans extension, so `index`
// becomes `/docs/index.md` and `features/chat` becomes `/docs/features/chat.md`.
export const getStaticPaths: GetStaticPaths = async () => {
  const entries = await getCollection('docs');
  return entries.map((entry) => ({
    params: { slug: entry.id },
    props: { entry },
  }));
};

export const GET: APIRoute = ({ props }) => {
  const { entry } = props as { entry: Awaited<ReturnType<typeof getCollection>>[number] };
  const body = entry.body ?? '';
  return new Response(body, {
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'X-Robots-Tag': 'noindex',
    },
  });
};
