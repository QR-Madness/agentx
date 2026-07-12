import type { APIRoute } from 'astro';
import { nav, isGroup, slugToHref } from '../config/nav';

// Hand-rolled sitemap (no @astrojs/sitemap dependency) built from the docs nav —
// the single source of truth for every documentation route — plus the marketing
// home and docs landing.
export const GET: APIRoute = ({ site }) => {
  const base = (site?.href ?? 'https://agentx.thejpnet.net/').replace(/\/$/, '');

  const paths = new Set<string>(['/', '/docs']);
  for (const entry of nav) {
    if (isGroup(entry)) {
      for (const item of entry.items) paths.add(slugToHref(item.slug));
    } else {
      paths.add(slugToHref(entry.slug));
    }
  }

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${[...paths].map((p) => `  <url><loc>${base}${p}</loc></url>`).join('\n')}
</urlset>
`;

  return new Response(body, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
};
