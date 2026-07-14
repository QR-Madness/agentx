import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { nav, isGroup } from '../config/nav';
import { site } from '../config/site';

// llms.txt — an agent-oriented index of the documentation (https://llmstxt.org).
// Built from the same nav tree the sidebar/sitemap use, but every link targets the
// Markdown copy (`/docs/<slug>.md`, served by `docs/[...slug].md.ts`) and we only
// emit entries that actually exist in the `docs` collection (so nav-only pages like
// the interactive explorer don't produce dead `.md` links).
export const GET: APIRoute = async ({ site: astroSite }) => {
  const base = (astroSite?.href ?? 'https://agentx.thejpnet.net/').replace(/\/$/, '');

  const entries = await getCollection('docs');
  const ids = new Set(entries.map((e) => e.id));
  const descFor = new Map(entries.map((e) => [e.id, e.data.description]));

  const idFor = (slug: string) => slug || 'index';
  const has = (slug: string) => ids.has(idFor(slug));
  const mdUrl = (slug: string) => `${base}/docs/${idFor(slug)}.md`;
  const line = (label: string, slug: string) => {
    const d = descFor.get(idFor(slug));
    return `- [${label}](${mdUrl(slug)})${d ? `: ${d}` : ''}`;
  };

  const lines: string[] = [
    `# ${site.name}`,
    '',
    `> ${site.description}`,
    '',
    'Agent-oriented index of the AgentX documentation. Every link below points to a Markdown copy of the page.',
    '',
  ];

  // Top-level single links (Home, Roadmap) collect into an Overview section.
  const overview = nav.filter((e) => !isGroup(e) && has((e as { slug: string }).slug));
  if (overview.length) {
    lines.push('## Overview');
    for (const e of overview) {
      const link = e as { label: string; slug: string };
      lines.push(line(link.label, link.slug));
    }
    lines.push('');
  }

  for (const entry of nav) {
    if (!isGroup(entry)) continue;
    const items = entry.items.filter((it) => has(it.slug));
    if (!items.length) continue;
    lines.push(`## ${entry.label}`);
    for (const it of items) lines.push(line(it.label, it.slug));
    lines.push('');
  }

  lines.push('## API');
  lines.push(`- [OpenAPI specification](${base}/openapi.yaml): machine-readable REST API contract (self-hosted at http://localhost:12319/api).`);
  lines.push(`- [API catalog](${base}/.well-known/api-catalog): RFC 9727 linkset pointing at the spec and docs.`);
  lines.push('');

  return new Response(lines.join('\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
};
