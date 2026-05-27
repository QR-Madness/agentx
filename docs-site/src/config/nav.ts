// Docs sidebar structure — ported verbatim from mkdocs.yml `nav`.
// Single source of truth for the docs section's navigation.
//
// Slugs are docs-relative (no `/docs` prefix, no `.md`). An empty slug ('') maps to the
// docs landing page (/docs). Helpers below resolve slugs to full routes.
import { site } from './site';

export type NavLink = { label: string; slug: string };
export type NavGroup = { label: string; items: NavLink[] };
export type NavEntry = NavLink | NavGroup;

export const nav: NavEntry[] = [
  { label: 'Home', slug: '' },
  {
    label: 'Getting Started',
    items: [
      { label: 'Installation', slug: 'getting-started/installation' },
      { label: 'Quick Start', slug: 'getting-started/quickstart' },
      { label: 'Configuration', slug: 'getting-started/configuration' },
    ],
  },
  {
    label: 'Architecture',
    items: [
      { label: 'Overview', slug: 'architecture/overview' },
      { label: 'API Layer', slug: 'architecture/api' },
      { label: 'Client Layer', slug: 'architecture/client' },
      { label: 'Database Stack', slug: 'architecture/databases' },
      { label: 'Memory Architecture', slug: 'architecture/memory' },
    ],
  },
  {
    label: 'Features',
    items: [
      { label: 'Chat', slug: 'features/chat' },
      { label: 'Reasoning', slug: 'features/reasoning' },
      { label: 'Drafting', slug: 'features/drafting' },
      { label: 'MCP Client', slug: 'features/mcp' },
      { label: 'Providers', slug: 'features/providers' },
      { label: 'Prompts', slug: 'features/prompts' },
      { label: 'Memory', slug: 'features/memory' },
      { label: 'Translation', slug: 'features/translation' },
    ],
  },
  {
    label: 'Development',
    items: [
      { label: 'Setup', slug: 'development/setup' },
      { label: 'Task Commands', slug: 'development/tasks' },
      { label: 'Testing', slug: 'development/testing' },
      { label: 'Memory Setup', slug: 'development/memory-setup' },
      { label: 'Contributing', slug: 'development/contributing' },
    ],
  },
  { label: 'Roadmap', slug: 'roadmap' },
  {
    label: 'API Reference',
    items: [
      { label: 'Endpoints', slug: 'api/endpoints' },
      { label: 'Models', slug: 'api/models' },
    ],
  },
  {
    label: 'Deployment',
    items: [
      { label: 'Docker', slug: 'deployment/docker' },
      { label: 'Database Migration', slug: 'deployment/migration' },
    ],
  },
];

export function isGroup(entry: NavEntry): entry is NavGroup {
  return (entry as NavGroup).items !== undefined;
}

/** Resolve a docs-relative slug to a full site route (empty slug → docs landing). */
export function slugToHref(slug: string): string {
  return slug ? `${site.docsBasePath}/${slug}` : site.docsBasePath;
}

/** Look up the nav label for a slug, used as the page <title>. */
export function labelForSlug(slug: string): string | undefined {
  for (const entry of nav) {
    if (isGroup(entry)) {
      const match = entry.items.find((item) => item.slug === slug);
      if (match) return match.label;
    } else if (entry.slug === slug) {
      return entry.label;
    }
  }
  return undefined;
}
