// Docs sidebar structure — ported from mkdocs.yml `nav`, reorganized to match the
// design handoff (Development moved last). Single source of truth for the docs
// section's navigation.
//
// Slugs are docs-relative (no `/docs` prefix, no `.md`). An empty slug ('') maps to the
// docs landing page (/docs). Helpers below resolve slugs to full routes.
import { site } from './site';

export type NavLink = { label: string; slug: string; color?: string };
export type NavGroup = { label: string; items: NavLink[] };
export type NavEntry = NavLink | NavGroup;

export const nav: NavEntry[] = [
  { label: 'Home', slug: '' },
  {
    label: 'Getting Started',
    items: [
      { label: 'Installation', slug: 'getting-started/installation' },
      { label: 'Windows Setup', slug: 'getting-started/windows' },
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
    // Feature pages map to runtime subsystems; each carries a semantic accent
    // (the --c-* tokens) shown as a color dot in the sidebar.
    label: 'Features',
    items: [
      { label: 'Chat', slug: 'features/chat', color: 'var(--c-agent)' },
      { label: 'Reasoning', slug: 'features/reasoning', color: 'var(--c-reasoning)' },
      { label: 'Drafting', slug: 'features/drafting', color: 'var(--c-drafting)' },
      { label: 'MCP Client', slug: 'features/mcp', color: 'var(--c-mcp)' },
      { label: 'Multi-Agent', slug: 'features/multi-agent', color: 'var(--c-agent)' },
      { label: 'Providers', slug: 'features/providers', color: 'var(--c-providers)' },
      { label: 'Prompts', slug: 'features/prompts', color: 'var(--c-prompts)' },
      { label: 'Memory', slug: 'features/memory', color: 'var(--c-memory)' },
      { label: 'Translation', slug: 'features/translation', color: 'var(--c-translation)' },
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
      { label: 'Production', slug: 'deployment/production' },
      { label: 'Clusters & Gateway', slug: 'deployment/clusters' },
      { label: 'Authentication', slug: 'deployment/authentication' },
      { label: 'Database Migration', slug: 'deployment/migration' },
    ],
  },
  {
    label: 'Development',
    items: [
      { label: 'Setup', slug: 'development/setup' },
      { label: 'Task Commands', slug: 'development/tasks' },
      { label: 'GPU Acceleration', slug: 'development/gpu' },
      { label: 'Mobile (Android)', slug: 'development/mobile' },
      { label: 'Testing', slug: 'development/testing' },
      { label: 'Memory Setup', slug: 'development/memory-setup' },
      { label: 'Contributing', slug: 'development/contributing' },
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

/** The label of the group containing a slug (for breadcrumbs); undefined for top-level pages. */
export function groupForSlug(slug: string): string | undefined {
  for (const entry of nav) {
    if (isGroup(entry) && entry.items.some((item) => item.slug === slug)) {
      return entry.label;
    }
  }
  return undefined;
}

/** The markdown source path (within the repo) for a docs slug. */
export function contentPathForSlug(slug: string): string {
  const id = slug || 'index';
  return `docs-site/src/content/docs/${id}.md`;
}

/** GitHub "edit this page" URL for a docs slug. */
export function editUrlForSlug(slug: string): string {
  return `${site.repoUrl}/edit/master/${contentPathForSlug(slug)}`;
}

/** GitHub "view source" URL for a docs slug. */
export function sourceUrlForSlug(slug: string): string {
  return `${site.repoUrl}/blob/master/${contentPathForSlug(slug)}`;
}
